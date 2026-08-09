# mllm QNN AOT Rotation Prototype

This context defines the shared language for evaluating offline R1/R2 weight rotation on a standalone Qwen3 Transformer block running through the QNN AOT path.

## Language

**Offline rotation**:
An orthogonal basis change folded into static model parameters before LPBQ G32 quantization and QNN compilation, with no rotation operator executed at runtime.
_Avoid_: Online rotation, runtime rotation

**R1**:
The normalized 2048-dimensional Hadamard rotation applied to the residual-stream basis and folded into the Layer 5 projection weights.
_Avoid_: Hidden-state runtime transform

**R2**:
The normalized 128-dimensional Hadamard rotation shared by every Layer 5 value head and its corresponding attention-output head block.
_Avoid_: Q/K rotation, R3

**Rotated-basis boundary**:
The standalone rotated block contract in which hidden-state input and output use the R1 basis, while past and current value-cache tensors use the per-head R2 basis. Key-cache tensors, RoPE tensors, and the causal mask retain their baseline basis.
_Avoid_: Drop-in unrotated boundary

**Original baseline (A)**:
The standalone Layer 5 graph exported with the current unrotated parameters and existing QDQ configuration.
_Avoid_: Full-model baseline

**Identity-folding control (B)**:
The standalone Layer 5 graph with RMSNorm gamma folded into downstream projections using identity R1/R2 matrices, isolating the folding and export path from rotation.
_Avoid_: Rotated candidate

**Rotated candidate (C)**:
The standalone Layer 5 graph with RMSNorm gamma folding and fixed Hadamard R1/R2 applied before fresh LPBQ G32 quantization.
_Avoid_: Learned rotation

**s1 decode**:
The single-token Layer 5 workload with sequence length 1 and a 1023-token past KV cache at context length 1024.
_Avoid_: s1 prefill

**s32 prefill/chunk**:
The 32-token Layer 5 workload with sequence length 32 and a 992-token past KV cache at context length 1024.
_Avoid_: s32 decode

**Block execute latency**:
Host wall-clock time from calling `QnnGraph_execute` until it returns after context loading, input preparation, and warmup have completed.
_Avoid_: Context-load time, full-model latency

**Performance non-regression**:
A rotated-candidate median block execute latency increase of at most 3% relative to the original baseline, with 3% to 5% treated as inconclusive and more than 5% treated as failure.
_Avoid_: Accuracy acceptance, model-quality acceptance
