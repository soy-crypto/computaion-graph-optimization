"""Time full BERT-Tiny CPU inference for the original and I-K-J MLIR files."""

import math
import os
import re
import subprocess
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import numpy as np


BUILD = Path(__file__).resolve().parents[1] / "build"
LLVM = Path(os.environ.get("MLIR_BUILD", Path.home() / "llvm-project/build")).resolve()
MODELS = {
    "original": BUILD / "bert-tiny-encoder.mlir",
    "optimized": BUILD / "bert-tiny-encoder-ikj.mlir",
}

# Keep the timing driver separate from the JIT-compiled model so every call
# runs, even though each repetition uses the same sentence.
DRIVER = r"""
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { tokens = 64, features = 128, repeats = 20 };

typedef struct {
    int64_t *allocated, *aligned;
    int64_t offset, size[2], stride[2];
} Input;

typedef struct {
    float *allocated, *aligned;
    int64_t offset, size[3], stride[3];
} Output;

typedef void (*Bert)(Output *, Input *, Input *, Input *);

static void read_array(const char *name, void *dest, size_t bytes) {
    FILE *f = fopen(name, "rb");
    if (!f || fread(dest, 1, bytes, f) != bytes) {
        fprintf(stderr, "Could not read %s\n", name);
        exit(1);
    }
    fclose(f);
}

static double seconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return now.tv_sec + now.tv_nsec * 1e-9;
}

static Input input_descriptor(int64_t *data) {
    return (Input){data, data, 0, {1, tokens}, {tokens, 1}};
}

void time_bert(Bert bert) {
    int64_t ids[tokens], mask[tokens], types[tokens];
    float reference[tokens][features];
    read_array("input_ids.bin", ids, sizeof ids);
    read_array("attention_mask.bin", mask, sizeof mask);
    read_array("token_type_ids.bin", types, sizeof types);
    read_array("expected.bin", reference, sizeof reference);
    Input input_ids = input_descriptor(ids);
    Input attention_mask = input_descriptor(mask);
    Input token_types = input_descriptor(types);
    Output out;

    bert(&out, &input_ids, &attention_mask, &token_types);
    if (out.size[0] != 1 || out.size[1] != tokens || out.size[2] != features) {
        fputs("Unexpected BERT output shape\n", stderr);
        exit(1);
    }
    for (int i = 0; i < tokens; ++i) {
        for (int j = 0; j < features; ++j) {
            double got = out.aligned[out.offset + i*out.stride[1] + j*out.stride[2]];
            double ref = reference[i][j];
            if (!isfinite(got) || fabs(got - ref) > 1e-5 + 1e-4*fabs(ref)) {
                fprintf(stderr, "Warm-up mismatch at [0,%d,%d]: %.9g vs %.9g\n", i, j, got, ref);
                exit(1);
            }
        }
    }
    free(out.allocated);

    double start = seconds();
    for (int i = 0; i < repeats; ++i) {
        bert(&out, &input_ids, &attention_mask, &token_types);
        free(out.allocated);
    }
    double elapsed = seconds() - start;
    printf("%.9f\n", elapsed * 1000.0 / repeats);
}
"""

ENTRY = """
  llvm.func @time_bert(!llvm.ptr)
  llvm.func @bench_entry() {
    %model = llvm.mlir.addressof @_mlir_ciface_bert_kernel : !llvm.ptr
    llvm.call @time_bert(%model) : (!llvm.ptr) -> ()
    llvm.return
  }
"""


def lower(source, work):
    print(f"Lowering {source.name}...", flush=True)
    text, count = re.subn(
        r"\bfunc\.func\s+@main\s*(\([^)]*\)\s*->\s*tensor<1x64x128xf32>)\s*\{",
        r"func.func @bert_kernel\1 attributes {llvm.emit_c_interface} {",
        source.read_text(),
    )
    if count != 1:
        raise ValueError(f"Expected one BERT @main returning tensor<1x64x128xf32> in {source}")
    lowered = work / (source.stem + "-bench-llvm.mlir")
    subprocess.run([
        BUILD / "my-opt", "--empty-tensor-to-alloc-tensor",
        "--one-shot-bufferize=bufferize-function-boundaries",
        "--convert-linalg-to-loops", "--buffer-deallocation-pipeline",
        "--expand-strided-metadata", "--lower-affine", "--convert-scf-to-cf",
        "--convert-math-to-llvm", "--convert-math-to-libm", "--convert-arith-to-llvm",
        "--finalize-memref-to-llvm", "--convert-func-to-llvm", "--convert-cf-to-llvm",
        "--reconcile-unrealized-casts", "-o", lowered,
    ], input=text, text=True, check=True)
    ir = lowered.read_text()
    marker = "llvm.func @_mlir_ciface_bert_kernel("
    if ir.count(marker) != 1:
        raise ValueError(f"Missing or ambiguous C entry point in {lowered.name}")
    lowered.write_text(ir.replace(marker, ENTRY + "\n  " + marker, 1))
    return lowered


def benchmark(work):
    with np.load(BUILD / "bert-tiny-encoder-reference.npz") as saved:
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            array = saved[key]
            if array.dtype != np.int64 or array.shape != (1, 64):
                raise ValueError(f"Expected {key} to be int64[1, 64]")
            array.tofile(work / f"{key}.bin")
        key = "expected_output" if "expected_output" in saved else "last_hidden_state"
        expected = saved[key]
        if expected.shape != (1, 64, 128) or not np.isfinite(expected).all():
            raise ValueError("Invalid PyTorch reference output")
        expected.astype(np.float32, copy=False).tofile(work / "expected.bin")

    driver = work / "time_bert.c"
    library = work / "time_bert.so"
    driver.write_text(DRIVER)
    subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC", driver, "-o", library,
                    "-Wl,--no-as-needed", "-lm"], check=True)
    modules = {name: lower(source, work) for name, source in MODELS.items()}
    runtime = LLVM / "lib/libmlir_c_runner_utils.so"
    times = {name: [] for name in MODELS}
    for trial in range(6):
        order = ("original", "optimized") if trial % 2 == 0 else ("optimized", "original")
        for name in order:
            raw = subprocess.check_output([
                LLVM / "bin/mlir-runner", modules[name], "-O3", "-e", "bench_entry",
                "-entry-point-result=void", f"-shared-libs={library},{runtime}",
            ], cwd=work, text=True)
            ms = float(raw.strip())
            if not math.isfinite(ms) or ms <= 0:
                raise ValueError(f"Invalid timing for {name}: {raw!r}")
            times[name].append(ms)
            print(f"{name} run {trial + 1}: {ms:.4f} ms/call", flush=True)
    baseline = median(times["original"])
    optimized = median(times["optimized"])
    print(f"Median original:  {baseline:.4f} ms/call")
    print(f"Median optimized: {optimized:.4f} ms/call")
    print(f"Full-model speedup: {baseline / optimized:.3f}x")


if __name__ == "__main__":
    with TemporaryDirectory(prefix="bert-encoder-bench-") as temp:
        benchmark(Path(temp))
