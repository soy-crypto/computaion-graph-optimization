"""Compare two loop orders for BERT-Tiny layer 0's second FFN matmul.

Run from the project root, with bench_bert_ffn.py alongside this script.
"""
import math
import re
import subprocess
import tempfile
from pathlib import Path
from statistics import median

import numpy as np

import bench_bert_ffn as bench


def interchange_second_matmul(source):
    ssa = r"%[\w.$-]+"
    pattern = re.compile(
        rf"^(?P<indent>[ \t]*)(?P<result>{ssa}) = linalg\.matmul\s+"
        rf"ins\(\s*(?P<lhs>{ssa}),\s*(?P<rhs>{ssa})\s*:\s*"
        r"tensor<64x512xf32>,\s*tensor<512x128xf32>\)\s+"
        rf"outs\(\s*(?P<init>{ssa})\s*:\s*tensor<64x128xf32>\)\s*"
        r"->\s*tensor<64x128xf32>", re.MULTILINE,
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise ValueError(f"Expected one 64x512 by 512x128 matmul, found {len(matches)}")
    match = matches[0]
    indent, result, lhs, rhs, init = match.groups()
    # d0 is token i, d1 is reduction k, and d2 is output feature j.
    lines = [
        f"{result} = linalg.generic {{",
        "  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>,",
        "                   affine_map<(d0, d1, d2) -> (d1, d2)>,",
        "                   affine_map<(d0, d1, d2) -> (d0, d2)>],",
        '  iterator_types = ["parallel", "reduction", "parallel"]',
        f"}} ins({lhs}, {rhs} : tensor<64x512xf32>, tensor<512x128xf32>)",
        f"  outs({init} : tensor<64x128xf32>) {{",
        "^bb0(%mm2_a: f32, %mm2_b: f32, %mm2_acc: f32):",
        "  %mm2_product = arith.mulf %mm2_a, %mm2_b : f32",
        "  %mm2_sum = arith.addf %mm2_acc, %mm2_product : f32",
        "  linalg.yield %mm2_sum : f32",
        "} -> tensor<64x128xf32>",
    ]
    replacement = "\n".join(indent + line for line in lines)
    return source[:match.start()] + replacement + source[match.end():]


def matmul_bounds(ir):
    """Read the static 64/128/512 loop nests in function order."""
    constants = dict(re.findall(r"(%[\w.$-]+) = arith\.constant (\d+) : index", ir))
    loops = []
    nests = []
    for line in ir.splitlines():
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        while loops and loops[-1][0] >= indent:
            loops.pop()
        loop = re.match(r"scf\.for\s+%\S+\s+=\s+%\S+\s+to\s+(%[\w.$-]+)\s+step\b", stripped)
        if loop:
            loops.append((indent, constants.get(loop[1])))
            bounds = tuple(value for _, value in loops)
            if len(bounds) == 3 and set(bounds) == {"64", "128", "512"}:
                nests.append(bounds)
    return nests


def inspect_loops(source, expected):
    # Canonicalize this inspection copy to expose literal bounds. Timing uses
    # the unmodified benchmark pipeline on the original tensor-level source.
    passes = bench.PASSES[:bench.PASSES.index("--convert-linalg-to-loops") + 1]
    scf = bench.BUILD / (source.stem + "-inspect-scf.mlir")
    subprocess.run([bench.BUILD / "my-opt", source, *passes, "--canonicalize", "-o", scf],
                   check=True)
    nests = matmul_bounds(scf.read_text())
    if len(nests) != 2:
        print(f"Could not identify both static matmul nests. Inspect {scf}", flush=True)
        return
    print(f"{source.name}: second matmul loop bounds {' -> '.join(nests[1])}", flush=True)
    if nests != [("64", "512", "128"), expected]:
        raise ValueError(f"Unexpected loop order in {scf}: {nests}")


def compare(directory, sources):
    with np.load(bench.BUILD / "bert-tiny-layer0-ffn.npz") as reference:
        for name in ("X", "Y"):
            values = reference[name]
            if values.shape != bench.SHAPE or not np.isfinite(values).all():
                raise ValueError(f"Invalid reference array {name}")
            np.ascontiguousarray(values, dtype=np.float32).tofile(directory / f"{name}.f32")

    driver = directory / "driver.c"
    library = directory / "driver.so"
    driver.write_text(bench.DRIVER)
    subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC", driver, "-o",
                    library, "-Wl,--no-as-needed", "-lm"], check=True)
    modules = {name: bench.lower(source, directory) for name, source in sources.items()}
    runner = bench.LLVM / "bin/mlir-runner"
    runtime = bench.LLVM / "lib/libmlir_c_runner_utils.so"
    timings = {name: [] for name in modules}
    for trial in range(6):
        order = list(modules) if trial % 2 == 0 else list(reversed(modules))
        for name in order:
            output = subprocess.check_output(
                [runner, modules[name], "-O3", "-e", "bench_entry", "-entry-point-result=void",
                 f"-shared-libs={library},{runtime}"], cwd=directory, text=True,
            )
            ms = float(output.strip())
            if not math.isfinite(ms) or ms <= 0:
                raise ValueError(f"Invalid timing for {name}: {output!r}")
            timings[name].append(ms)
            print(f"{name} run {trial + 1}: {ms:.4f} ms/call", flush=True)

    original, ikj = median(timings["original"]), median(timings["ikj"])
    print(f"Median original: {original:.4f} ms/call")
    print(f"Median ikj:      {ikj:.4f} ms/call")
    print(f"Speedup (original/ikj): {original / ikj:.3f}x")


if __name__ == "__main__":
    original = bench.SOURCES["original"]
    ikj = bench.BUILD / "bert-tiny-layer0-ffn-mm2-ikj.mlir"
    ikj.write_text(interchange_second_matmul(original.read_text()))
    inspect_loops(original, ("64", "128", "512"))
    inspect_loops(ikj, ("64", "512", "128"))
    with tempfile.TemporaryDirectory(prefix="bert-mm2-") as temp:
        compare(Path(temp), {"original": original, "ikj": ikj})
