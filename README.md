# Neural Network Inference Optimization with MLIR

`my-mlir-pass` implements MLIR compiler optimizations for neural network inference, progressing from individual kernels to a pretrained BERT-Tiny encoder.

## Optimizations

- **MLP:** Fuse GEMM, bias addition, and ReLU with C++ rewrite patterns.
- **Attention:** Fuse QKᵀ with scaling and shifted exponentials with row-sum reduction; run the exponential and reduction kernel on an NVIDIA GPU.
- **Transformer:** Apply the optimization passes to a handwritten block with attention, an MLP, residual connections, and layer normalization.
- **BERT-Tiny:** Export from PyTorch, import into MLIR, and reorder four FFN matmuls from I-J-K to I-K-J using the Transform dialect. Lower to LLVM for CPU execution.

## Results

| CPU workload | Original | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| BERT-Tiny layer-0 FFN | 7.1040 ms | 0.7678 ms | 9.252× |
| Full BERT-Tiny encoder | 23.0308 ms | 10.0454 ms | 2.293× |
| Handwritten Transformer block | 3.1824 ms | 2.7546 ms | 1.155× |

BERT-Tiny was tested with batch size 1 and sequence length 64. Both encoder versions matched PyTorch across all 8,192 output values within `rtol=1e-4` and `atol=1e-5`; maximum absolute error was `4.10e-6`. Full-encoder timings used six alternating trials with 20 calls per trial after warm-up. Results reflect the tested inputs and environment.

## Usage

Run from the configured repository root. Requires an LLVM/MLIR build, CMake, a C compiler, MLIR runner utilities, and a Python environment with PyTorch, Transformers, NumPy, and Torch-MLIR. The commands below assume the attention input and exported BERT-Tiny MLIR files are already prepared in `build/`.

```bash
# Build the compiler passes
cmake --build build

# Fuse the handwritten attention workload
build/my-opt --toy-attention-fusion build/attention-unfused.mlir \
  -o build/attention-fused.mlir

# Check BERT-Tiny against the saved PyTorch reference
.venv/bin/python tools/check_bert_baseline.py build/bert-tiny-encoder.mlir

# Benchmark the prepared original and optimized encoders
.venv/bin/python tools/bench_bert_encoder.py
```
