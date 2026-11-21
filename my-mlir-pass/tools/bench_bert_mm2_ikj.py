"""Compare matmul loop orders in BERT-Tiny's layer 0 FFN.

Run from the project root, with bench_bert_ffn.py alongside this script.
Default: original vs. second matmul reordered.
--first: second matmul reordered vs. both matmuls reordered.
"""
import argparse
import math
import re
import subprocess
import tempfile
from pathlib import Path
from statistics import median

import numpy as np

import bench_bert_ffn as bench


def interchange_matmul(source, projection):
    k, n = {"mm1": (128, 512), "mm2": (512, 128)}[projection]
    lhs_type = f"tensor<64x{k}xf32>"
    rhs_type = f"tensor<{k}x{n}xf32>"
    out_type = f"tensor<64x{n}xf32>"
    ssa = r"%[\w.$-]+"
    pattern = re.compile(
        rf"^(?P<indent>[ \t]*)(?P<result>{ssa}) = linalg\.matmul\s+"
        rf"ins\(\s*(?P<lhs>{ssa}),\s*(?P<rhs>{ssa})\s*:\s*"
        rf"{lhs_type},\s*{rhs_type}\)\s+"
        rf"outs\(\s*(?P<init>{ssa})\s*:\s*{out_type}\)\s*"
        rf"->\s*{out_type}", re.MULTILINE,
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise ValueError(f"Expected one {lhs_type} by {rhs_type} matmul, found {len(matches)}")
    match = matches[0]
    indent, result, lhs, rhs, init = match.groups()
    # d0 is token i, d1 is reduction k, and d2 is output feature j.
    lines = [
        f"{result} = linalg.generic {{",
        "  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>,",
        "                   affine_map<(d0, d1, d2) -> (d1, d2)>,",
        "                   affine_map<(d0, d1, d2) -> (d0, d2)>],",
        '  iterator_types = ["parallel", "reduction", "parallel"]',
        f"}} ins({lhs}, {rhs} : {lhs_type}, {rhs_type})",
        f"  outs({init} : {out_type}) {{",
        f"^bb0(%{projection}_a: f32, %{projection}_b: f32, %{projection}_acc: f32):",
        f"  %{projection}_product = arith.mulf %{projection}_a, %{projection}_b : f32",
        f"  %{projection}_sum = arith.addf %{projection}_acc, %{projection}_product : f32",
        f"  linalg.yield %{projection}_sum : f32",
        f"}} -> {out_type}",
    ]
    replacement = "\n".join(indent + line for line in lines)
    return source[:match.start()] + replacement + source[match.end():]


def interchange_second_matmul(source):
    return interchange_matmul(source, "mm2")


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


def inspect_loops(source, expected, *, first_expected=("64", "512", "128")):
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
    print(f"{source.name}: first {' -> '.join(nests[0])}, second {' -> '.join(nests[1])}",
          flush=True)
    if nests != [first_expected, expected]:
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

    baseline, candidate = sources
    baseline_ms = median(timings[baseline])
    candidate_ms = median(timings[candidate])
    print(f"Median {baseline}: {baseline_ms:.4f} ms/call")
    print(f"Median {candidate}: {candidate_ms:.4f} ms/call")
    print(f"Speedup ({baseline}/{candidate}): {baseline_ms / candidate_ms:.3f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", action="store_true",
                        help="measure reordering the first matmul with the second already reordered")
    args = parser.parse_args()
    original = bench.SOURCES["original"]
    ikj = bench.BUILD / "bert-tiny-layer0-ffn-mm2-ikj.mlir"
    second_only = interchange_second_matmul(original.read_text())
    ikj.write_text(second_only)
    if args.first:
        both = bench.BUILD / "bert-tiny-layer0-ffn-both-ikj.mlir"
        both.write_text(interchange_matmul(second_only, "mm1"))
        inspect_loops(ikj, ("64", "512", "128"))
        inspect_loops(both, ("64", "512", "128"), first_expected=("64", "128", "512"))
        sources = {"second-only": ikj, "both": both}
    else:
        inspect_loops(original, ("64", "128", "512"))
        inspect_loops(ikj, ("64", "512", "128"))
        sources = {"original": original, "ikj": ikj}
    with tempfile.TemporaryDirectory(prefix="bert-mm2-") as temp:
        compare(Path(temp), sources)
