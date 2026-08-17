# W4A8G32 LPBQ MLP Operator-Expression Experiment

Status: stopped at the operator warm-performance gate on 2026-08-18. See
[`w4a8g32_lpbq_mlp_gate_result.md`](w4a8g32_lpbq_mlp_gate_result.md).

This experiment starts from commit `987a7156` on the accepted native-U8
RMSNorm baseline. It asks whether expressing Qwen3 MLP projections as QNN
`FullyConnected` or `MatMul` selects a faster physical LPBQ lowering than the
current 1x1 `Conv2D`, without changing the model's deployed mathematics.

## Fixed scope

- Target hardware/backend: the existing SM8750/V79 device and QAIRT
  2.47.0.260601 configuration.
- Target operations: the 28 layers' `gate_proj`, `up_proj`, and `down_proj`
  only (84 projections).
- Preserved operations: attention projections, `lm_head`, native-U8 RMSNorm,
  Softmax, RoPE, KV cache, and every non-MLP quantization recipe.
- Source artifact: the accepted native-U8 RMSNorm `.mllm`, identified by
  digest. Candidate artifacts only relayout target weight carriers.
- No recalibration, requantization, accuracy optimization, or mixed
  FC/MatMul selection is permitted.

Source code and compact contracts belong in git. Model/intermediate artifacts
belong under
`D:\llm_exp\models\qwen3_sm8750_v79\g32\w4a8_rmsnorm_u8_lpbq_mlp\<run-id>`;
results belong under
`D:\llm_exp\results\qwen3_sm8750_v79_w4a8_rmsnorm_u8_lpbq_mlp_<run-id>`.
Small-file-intensive work and disposable builds run in the native WSL
filesystem. No accepted RMSNorm-A8 artifact or result may be overwritten.

## Projection and LPBQ contracts

Let `W[O,K]` be the one canonical mathematical weight matrix. All candidates
must decode exactly to:

```text
W_hat[o,k] = signed_w4[o,k]
             * scale1[o,floor(k/32)]
             * scale2[o]
```

The immutable quantization contract is signed W4 in `[-7,7]`, group size 32,
UInt4 block scales carried in UInt8, one Float32 level-2 scale per output
channel, and asymmetric UInt8 input/output activations with the baseline's
exact scale and zero point.

| Expression | Physical weight | Channel axis | Blocks/channel | Required parameter |
|---|---:|---:|---:|---|
| Conv2D reference | `[1,1,K,O]` HWIO | 3 | `K/32` | existing contract unchanged |
| FullyConnected | `[O,K]` OI | 0 | `K/32` | static weight |
| MatMul | `[K,O]` IO | last | `K/32` | static input 1; `transpose_in1=false` |

The framework must select these contracts explicitly by operation kind. It
must not infer a universal LPBQ layout from rank-relative last dimensions.
The existing Linear `[-8,7]/UInt4` recipe is not equivalent and must not be
used for this experiment.

The logical activation boundary remains `[1,S,K] -> [1,S,O]`. Metadata-only
views needed to express rank are allowed. A physical Transpose, Copy,
ForceFormat, datatype Convert, or host/CPU fallback fails the candidate.

The real gate cases are:

| Projection | K | O | Sequence lengths |
|---|---:|---:|---:|
| layer-14 gate | 2048 | 6144 | 1, 32 |
| layer-14 up | 2048 | 6144 | 1, 32 |
| layer-14 down | 6144 | 2048 | 1, 32 |

Gate and up are separate cases despite equal shapes because their weights and
activation encodings differ.

## Staged advancement

Each candidate advances independently and in this order:

1. Host layout, shape, canonical W4 decode, qparam, and reference-math tests.
2. QAIRT context finalization plus manifest/schematic structural audit.
3. On-device output correctness over all six real cases.
4. On-device cold-weight and warm timing over all six real cases.
5. An independently timed complete layer-14 MLP:
   `gate -> SiLU(gate) * up -> down`.
6. An otherwise unchanged full s1/s32 graph in which only layer 14 uses the
   candidate, proving compatibility with real neighboring operations.

A failed candidate is not timed after a correctness/finalization failure. The
other candidate may continue. If neither uniform candidate survives, or the
selected candidate fails the complete-single-layer gate, work stops before
full-model migration and the result is discussed with the user.

If both candidates pass, select the one with the lowest worst-case normalized
latency across all six cases. A difference within one percent is a tie and
selects FullyConnected.

## Mathematical correctness

Host and device tests use identical canonical weights and activation qparams.
Inputs cover:

- the input zero point (real zero);
- all qmin and all qmax;
- alternating qmin/qmax;
- fixed-seed random UInt8 values;
- digest-pinned real layer-14 s1 and s32 activation fixtures generated from
  the experiment's fixed prompt/calibration data.

Fixtures are stored with model artifacts on `D:`; git stores their generator,
metadata, and digest. Outputs must be deterministic. Weight decode, logical
shape, input/output qparams, and the real-zero behavior are exact gates.

For operator output, compare dequantized QNN output with a high-precision
calculation using `W_hat`. For max error, p99 error, and RMSE, a candidate may
not be worse than the paired Conv2D reference by more than one output LSB.
Candidate and Conv output codes need not be bit-identical because backend
accumulation and rounding order may differ.

## Performance protocol

All speed gates are profiling-off paired measurements on the same device.
The baseline and candidate use the same contexts, inputs, runner behavior, and
thermal logging.

Warm timing uses 10 untimed executions followed by 100 timed executions in
each of five fresh processes. Cold-weight timing uses seven fresh processes;
an unrelated, weight-free HTP graph primes device startup, after which the
previously untouched target graph is executed exactly once. Conv/candidate
order is alternated.

Each of the six cases must independently satisfy:

```text
candidate median latency <= Conv2D median latency * 1.01
```

Warm and cold results are separate hard gates. Optrace is collected in fresh
processes to prove HTP/HMX execution and explain physical kernels, W4
expansion, DRAM/VTCM traffic, DMA wait, and HMX cycles; it is not the sole
latency measurement.

A set is invalid when thermal state is abnormal, process dispersion exceeds
one percent, or paired direction is inconsistent. It may be completely
recollected once after cooldown. Repeated instability is a stop condition,
not permission to retry until a pass appears.

The layer-14 full-graph integration run reports end-to-end performance but
does not gate on it because one changed layer is below whole-model resolution.
Its hard checks are the independently timed layer-14 MLP and the physical
integration audit.

## Full-model acceptance

Only the selected, gate-passing expression replaces all 84 MLP projections.
Both s1 and s32 graphs must show:

- 84 selected MLP expressions and no residual MLP Conv2D;
- the other 925 LPBQ projections unchanged as Conv2D;
- all 1009 target projections retaining W4G32/A8;
- 729 native-U8 RMSNorm operations and zero RMSNorm bridges;
- no new Convert, Transpose, Copy, ForceFormat, or CPU fallback.

The canonical profile retains the fixed prompt, AR=32, 64-token generation,
three profiling-off fresh processes, informational 100-case sanity suite, and
fresh-process s1/s32 Optrace. The native-U8 RMSNorm result is the primary
reference and the archived W4A16 result is secondary.

Because a hard one-percent threshold is finer than the standard three-run
report, acceptance additionally uses five same-session paired RMS-A8/candidate
runs with alternating A-B/B-A order. The median paired ratio must be no more
than 1.01 for both prefill and decode. The standard three-run summary remains
in the report for historical comparison.

The final result includes the canonical
`qwen3-sm8750-v79-g32-e2e-critical-path.html` and an MLP gate table containing
all six cold/warm ratios, physical kernel, expansion work, DRAM/VTCM traffic,
DMA wait, HMX cycles, and pass state.

If a hard gate fails, retain only compact CSV/JSON summaries, manifest
extracts, decisive logs, source revisions, and a failure conclusion. Delete
large failed contexts, raw traces, temporary models, and build caches.
