"""Check the CPU MLIR output of BERT-Tiny against the saved PyTorch result."""

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np


BUILD = Path(__file__).resolve().parents[1] / "build"
LLVM_BUILD = Path.home() / "llvm-project" / "build"
INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids")


def add_test_entry(source, inputs):
    # Use saved inputs in the same order as the exported function signature.
    constants = []
    for name in INPUT_NAMES:
        value = inputs[name]
        if value.shape != (1, 64) or value.dtype != np.int64:
            raise ValueError(f"{name}: expected int64[1, 64], got {value.dtype}{value.shape}")
        constants.append(f"    %{name} = arith.constant dense<{value.tolist()}> : tensor<1x64xi64>")

    wrapper = """
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func @check_bert() {
INPUT_CONSTANTS
    %y = func.call @main(%input_ids, %attention_mask, %token_type_ids)
        : (tensor<1x64xi64>, tensor<1x64xi64>, tensor<1x64xi64>) -> tensor<1x64x128xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    scf.for %token = %c0 to %c64 step %c1 {
      scf.for %feature = %c0 to %c128 step %c1 {
        %v = tensor.extract %y[%c0, %token, %feature] : tensor<1x64x128xf32>
        // Print the exact f32 bits to avoid decimal rounding.
        %bits = arith.bitcast %v : f32 to i32
        %wide = arith.extui %bits : i32 to i64
        func.call @printI64(%wide) : (i64) -> ()
        func.call @printNewline() : () -> ()
      }
    }
    return
  }
""".replace("INPUT_CONSTANTS", "\n".join(constants))
    entry = "func.func @main("
    if source.count(entry) != 1:
        raise ValueError("Expected one func.func @main in the tensor-level baseline")
    return source.replace(entry, wrapper + "\n  " + entry, 1)


def check_output(stdout, expected):
    if expected.shape != (1, 64, 128):
        raise ValueError(f"Unexpected reference shape: {expected.shape}")
    bits = np.array(stdout.split(), dtype=np.uint32)
    if bits.size != expected.size:
        raise ValueError(f"Expected {expected.size} output values, got {bits.size}")
    actual = bits.view(np.float32).reshape(expected.shape)
    if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        raise ValueError("The MLIR output or PyTorch reference contains NaN or infinity")
    print("Max absolute difference:", np.abs(actual - expected).max())
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    print(f"PASS: all {actual.size} BERT-Tiny outputs match PyTorch within tolerance.")


def main():
    source = (BUILD / "bert-tiny-encoder-ikj.mlir").read_text()
    with np.load(BUILD / "bert-tiny-encoder-reference.npz") as saved:
        expected_key = "expected_output" if "expected_output" in saved else "last_hidden_state"
        expected = saved[expected_key]
        test_module = add_test_entry(source, saved)

    with TemporaryDirectory(dir=BUILD) as work:
        test_path = Path(work) / "check.mlir"
        lowered = Path(work) / "check-llvm.mlir"
        test_path.write_text(test_module)
        subprocess.run([
            BUILD / "my-opt", test_path,
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=bufferize-function-boundaries",
            "--convert-linalg-to-loops", "--expand-strided-metadata", "--lower-affine",
            "--convert-scf-to-cf", "--convert-math-to-llvm", "--convert-math-to-libm",
            "--convert-arith-to-llvm", "--finalize-memref-to-llvm", "--convert-func-to-llvm",
            "--convert-cf-to-llvm", "--reconcile-unrealized-casts", "-o", lowered,
        ], check=True)
        run = subprocess.run([
            LLVM_BUILD / "bin/mlir-runner", lowered, "-e", "check_bert",
            "-entry-point-result=void",
            f"-shared-libs={LLVM_BUILD}/lib/libmlir_c_runner_utils.so",
        ], stdout=subprocess.PIPE, text=True, check=True)
    check_output(run.stdout, expected)


if __name__ == "__main__":
    main()
