"""Profile both matmuls inside the original BERT-Tiny layer 0 FFN on CPU.

Place this script beside bench_bert_ffn.py in tools/ and run it from the
project root with the same Python environment as the FFN benchmark.
"""
from pathlib import Path
import re
from statistics import median
import subprocess
import tempfile

import numpy as np

import bench_bert_ffn as bench


# The printer emits one line per bufferized named matmul. Match the original
# FFN's two-input, one-output form. The captured types may have strided layouts.
MATMUL = re.compile(
    r"^(?P<indent> *)linalg\.matmul\s+"
    r"ins\((?P<lhs>%[\w.$-]+),\s*(?P<rhs>%[\w.$-]+)\s*:\s*"
    r"(?P<lhs_type>memref<.+>),\s*(?P<rhs_type>memref<.+>)\)\s*"
    r"outs\((?P<dst>%[\w.$-]+)\s*:\s*(?P<dst_type>memref<.+>)\)\s*$",
    re.MULTILINE,
)
SHAPES = [
    ("64x128xf32", "128x512xf32", "64x512xf32"),
    ("64x512xf32", "512x128xf32", "64x128xf32"),
]
REGIONS = ("128 -> 512", "512 -> 128")

# The existing driver performs one correctness-checked warm-up, then 100
# timed calls. Discard the first observation from each matmul in each process.
TIMERS = r"""
static double started[2], sum_seconds[2];
static unsigned completed[2];

void profile_mark(int32_t event, void *lhs, void *rhs, void *dst) {
    (void)lhs; (void)rhs; (void)dst;
    unsigned op = (unsigned)event / 2;
    if ((event & 1) == 0) {
        started[op] = seconds();
    } else {
        double stop = seconds();
        if (completed[op] > 0)
            sum_seconds[op] += stop - started[op];
        ++completed[op];
    }
}

void profile_report(void) {
    for (unsigned op = 0; op < 2; ++op) {
        if (completed[op] != repeats + 1) {
            fprintf(stderr, "Matmul %u ran %u times, expected %d\n",
                    op + 1, completed[op], repeats + 1);
            exit(1);
        }
        printf("%.9f\n", 1000.0 * sum_seconds[op] / repeats);
    }
}
"""


def instrument(text):
    matches = list(MATMUL.finditer(text))
    if len(matches) != 2:
        raise ValueError(f"Expected two bufferized linalg.matmul ops, found {len(matches)}")

    # Insert backwards to keep the original offsets valid.
    for op in (1, 0):
        match = matches[op]
        indent = match["indent"]
        prefix = f"%profile_mm{op + 1}"
        lines = []
        pointers = []
        for operand, shape in zip(("lhs", "rhs", "dst"), SHAPES[op]):
            ty = match[f"{operand}_type"]
            if not (ty.startswith(f"memref<{shape}>") or ty.startswith(f"memref<{shape},")):
                raise ValueError(f"Unexpected matmul {op + 1} {operand} type: {ty}")
            ptr = f"{prefix}_{operand}"
            lines.extend([
                f"{ptr}_idx = memref.extract_aligned_pointer_as_index {match[operand]} : {ty} -> index",
                f"{ptr}_i64 = arith.index_cast {ptr}_idx : index to i64",
                f"{ptr} = llvm.inttoptr {ptr}_i64 : i64 to !llvm.ptr",
            ])
            pointers.append(ptr)

        # The external calls see all three pointers, which anchors reads and
        # writes at the timer boundaries. The timer never touches the buffers.
        params = ", ".join(pointers)
        signature = "(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()"
        for event, tag in ((2 * op, "start"), (2 * op + 1, "stop")):
            lines.append(f"{prefix}_{tag} = arith.constant {event} : i32")
        lines.extend([
            f"llvm.call @profile_mark({prefix}_start, {params}) : {signature}",
            match[0].strip(),
            f"llvm.call @profile_mark({prefix}_stop, {params}) : {signature}",
        ])
        text = text[:match.start()] + "\n".join(indent + line for line in lines) + text[match.end():]

    return text.replace("func.func @ffn_kernel(",
                        "llvm.func @profile_mark(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr)\n"
                        "  func.func @ffn_kernel(", 1)


def prepare(directory):
    source = (bench.BUILD / "bert-tiny-layer0-ffn.mlir").read_text()
    signature = r"\bfunc\.func\s+@main\s*(\([^)]*\)\s*->\s*tensor<1x64x128xf32>)\s*\{"
    source, count = re.subn(signature, r"func.func @ffn_kernel\1 attributes {llvm.emit_c_interface} {", source)
    if count != 1:
        raise ValueError("Expected one tensor-to-tensor @main in the original FFN")

    opt = bench.BUILD / "my-opt"
    split = bench.PASSES.index("--convert-linalg-to-loops")
    buffered = subprocess.check_output([opt, *bench.PASSES[:split]], input=source, text=True)
    llvm = subprocess.check_output([opt, *bench.PASSES[split:]], input=instrument(buffered), text=True)
    wrapper = "llvm.func @_mlir_ciface_ffn_kernel("
    if wrapper not in llvm:
        raise ValueError("Lowered FFN is missing its C wrapper")
    entry = bench.ENTRY.replace("llvm.return", "llvm.call @profile_report() : () -> ()\n    llvm.return")
    entry = "llvm.func @profile_report()\n  " + entry
    module = directory / "profile.mlir"
    module.write_text(llvm.replace(wrapper, entry + "\n  " + wrapper, 1))
    subprocess.run([opt, module, "-o", "/dev/null"], check=True)
    return module


def run(directory):
    with np.load(bench.BUILD / "bert-tiny-layer0-ffn.npz") as saved:
        for key in ("X", "Y"):
            values = np.ascontiguousarray(saved[key], dtype=np.float32)
            if values.shape != bench.SHAPE or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key} in the saved reference")
            values.tofile(directory / f"{key}.f32")
    c_file = directory / "driver.c"
    library = directory / "driver.so"
    c_file.write_text(bench.DRIVER + TIMERS)
    subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC", c_file, "-o",
                    library, "-Wl,--no-as-needed", "-lm"], check=True)
    module = prepare(directory)
    runtime = bench.LLVM / "lib/libmlir_c_runner_utils.so"
    command = [bench.LLVM / "bin/mlir-runner", module, "-O3", "-e", "bench_entry",
               "-entry-point-result=void", f"-shared-libs={library},{runtime}"]
    samples = []
    for trial in range(1, 6):
        output = subprocess.check_output(command, cwd=directory, text=True)
        sample = tuple(map(float, output.splitlines()))
        if len(sample) != 3 or any(not np.isfinite(x) or x <= 0 for x in sample):
            raise ValueError(f"Unexpected runner output: {output!r}")
        samples.append(sample)
        total, first, second = sample
        print(f"Trial {trial}: total {total:.4f}, matmul 1 {first:.4f}, matmul 2 {second:.4f} ms/call", flush=True)
    print("Medians (ms/call)")
    for label, times in zip(("Instrumented FFN", *REGIONS), zip(*samples)):
        print(f"  {label}: {median(times):.4f}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="bert-ffn-profile-") as work:
        run(Path(work))
