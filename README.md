Neural Network Inference Optimization with MLIR
`my-mlir-pass` explores neural network inference optimization with C++ rewrite passes, the MLIR Transform dialect, and CPU and GPU lowering. The workloads progress from MLP and attention kernels to a handwritten Transformer block and a pretrained BERT-Tiny encoder.
The largest workload tested was the full pretrained BERT-Tiny encoder. Changing the loop order of four feed-forward matrix multiplications reduced median CPU latency from 23.03 to 10.05 ms per inference (2.293×). Both versions matched PyTorch on all 8,192 output values for the test input.
Implemented optimizations
Workload	What was implemented	How it was checked
MLP	A C++ pattern fuses GEMM, bias addition, and ReLU. It computes each dot product, adds the column bias, and applies ReLU before storing the output. The matcher checks uses of intermediate buffers before removing them.	Compared 4,096 output values on nonuniform inputs. Tested rejection when another operation used an intermediate.
Attention	The `--toy-attention-fusion` pass fuses QKᵀ with scaling, then fuses the shifted exponential with the row-sum reduction. Each exponential feeds the sum directly and is also stored for normalization.	Compared fused and unfused 64×64 results. A GPU version of the exponential and row sum also matched the CPU reference on the test input.
Handwritten Transformer block	Combined self-attention projections, the MLP, residual additions, and layer normalization. Ran the existing optimization passes on the composed block.	All 4,096 outputs matched the unfused block, with a maximum absolute difference of zero.
Pretrained BERT-Tiny	Exported the model from PyTorch, imported it as Linalg-on-tensors, and used the MLIR Transform dialect to interchange the loops in both FFN matmuls in each of two layers.	Ran the original and optimized encoders through LLVM on CPU and compared both with the same saved PyTorch output.
The C++ pass work also included an arithmetic rewrite for integer self-subtraction and a shared-memory GEMM tiling pattern.
Attention GPU execution

In the handwritten attention workload, the outer row loop of the fused exponential and row sum was mapped to the GPU. Kernel outlining produced `gpu.func` and `gpu.launch_func`. NVVM lowering and serialization produced PTX for `sm_86`. The kernel used 64 blocks with one thread per block. Host-shared buffers connected it to the CPU stages, including normalization and the final matrix multiplication. The CPU and hybrid runs agreed on the test input.
Performance and correctness

The model was `google/bert_uncased_L-2_H-128_A-2`. The test used one sequence, `I like cats`, padded to 64 positions. Five positions were active. The reference was the final hidden state of shape `1×64×128`.
CPU workload	Original median	Optimized median	Speedup
BERT-Tiny layer-0 FFN alone	7.1040 ms	0.7678 ms	9.252×
Full pretrained BERT-Tiny encoder	23.0308 ms	10.0454 ms	2.293×
Handwritten 64×64 Transformer block	3.1824 ms	2.7546 ms	1.155×
The BERT transform matches the `128 → 512` expansion and `512 → 128` projection. It changes I-J-K to I-K-J, so the innermost loop visits adjacent output features while reusing one input activation. Attention and normalization also contribute to full-encoder latency, which is why the isolated FFN and full encoder are measured separately.
A separate layer-0 FFN experiment compared the original, upstream fusion, and targeted bias–GELU fusion in the same run. Their medians were 7.1480, 34.1795, and 7.1268 ms, respectively. The upstream-fused IR placed the GELU `erf` inside the second matmul’s reduction body, where it could be evaluated again for each output feature.
The full-encoder benchmark used six alternating CPU trials with 20 timed calls per trial after warm-up. Both encoder variants had a maximum absolute error of 4.0978193e-06 against PyTorch on this input, using `rtol=1e-4` and `atol=1e-5`.
Reproduce the experiments
BERT-Tiny encoder

Run these commands from `my-mlir-pass` with a built `build/my-opt`, an LLVM/MLIR build, and a Python environment containing PyTorch, Transformers, NumPy, and Torch-MLIR. The CPU benchmark also uses a C compiler and the MLIR C runner utilities.
1. Export the pretrained model and reference tensors, then import the graph into MLIR.
```bash
.venv/bin/python tools/bert_full_export.py
.venv/bin/python - <<'PY'
from pathlib import Path
import torch
from torch_mlir import fx

program = torch.export.load("build/bert-tiny-encoder.pt2")
module = fx.export_and_import(program, output_type="linalg-on-tensors")
Path("build/bert-tiny-encoder.mlir").write_text(str(module))
PY
build/my-opt build/bert-tiny-encoder.mlir -o /dev/null
```
The `.pt2` holds the graph and weights. The reference `.npz` holds the token IDs, attention mask, token-type IDs, and expected output. On the tested Torch-MLIR installation, importing BERT’s `embeddings.position_ids` required the frozen importer to resolve buffers through `prog.named_buffers()`.
2. Reorder the four FFN matmuls in a separate copy.
```bash
"$HOME/llvm-project/build/bin/mlir-opt" build/bert-tiny-encoder.mlir \
  --transform-preload-library="transform-library-paths=tools/bert_ffn_ikj.mlir" \
  --transform-interpreter -o build/bert-tiny-encoder-ikj.mlir
```
3. Check both models against PyTorch, then benchmark them. The checker bufferizes the model, lowers Linalg to loops and LLVM, and executes the saved inputs with `mlir-runner`.
```bash
.venv/bin/python tools/check_bert_baseline.py build/bert-tiny-encoder.mlir
.venv/bin/python tools/check_bert_baseline.py build/bert-tiny-encoder-ikj.mlir
.venv/bin/python tools/bench_bert_encoder.py
```
The filename-aware checker prints the model filename before running. With the original two-script setup, run `tools/check_bert_baseline.py` and `tools/check_bert_optimized.py` separately.
Inspect attention fusion
Run the following from the configured `my-mlir-pass` checkout with the handwritten input in `build/`:
```bash
cmake --build build
build/my-opt --toy-attention-fusion build/attention-unfused.mlir \
  -o build/attention-fused.mlir
rg -n 'scf\.for|math\.exp|linalg\.matmul' build/attention-fused.mlir
```
Source guide
Files	Role
`lib/MyPass.cpp`	C++ rewrite patterns, pass registration, and shared-memory tiling. CMake compiles this copy.
`tools/bert_full_export.py`, `tools/bert_ffn_ikj.mlir`	Export the PyTorch reference and apply the loop schedule.
`tools/check_bert_baseline.py`, `tools/bench_bert_encoder.py`	Validate the complete output and measure full-encoder CPU latency.
