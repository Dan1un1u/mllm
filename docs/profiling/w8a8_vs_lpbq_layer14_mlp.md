# Layer-14 per-tensor W8A8 versus LPBQ W4A8

## Outcome

On SM8750/V79 with QAIRT 2.47.0.260601, the complete middle-layer Qwen3 MLP is
substantially faster with the accepted LPBQ W4A8 encoding than with symmetric
per-tensor W8A8.  W8 reduces the matrix-engine compute cycles slightly, but its
approximately doubled packed-weight traffic turns `weights_to_vtcm` into the
dominant serial path.

This is a diagnostic result, not a proposed replacement for the accepted
W4A8/RMSNorm-U8 baseline.

## Controlled contract

- Source: accepted RMSNorm-U8 artifact, SHA-256
  `de6c789a684f3276d16b9673f13f4c69aa9eb4a68ab8b9fca2a76be3fbb75a3e`.
- Scope: complete MLP in layer 14: `gate_proj`, `up_proj`, sigmoid/gating, and
  `down_proj`.
- Shapes: hidden 2048, intermediate 6144, sequence lengths 1 and 32.
- Common path: QNN `Conv2d`, NHWC activation, HWIO weight, identical asymmetric
  U8 activation qparams and identical graph boundaries.
- LPBQ weight: signed W4 `[-7, 7]`, G32, UInt4 block scale, FP32 channel scale.
- W8 weight: native QNN `SFIXED_POINT_8`, symmetric per-tensor scale, range
  `[-128, 127]`, zero offset.
- W8 values are deterministically requantized from the deployed effective LPBQ
  weights.  No original floating-point model or calibration change is mixed
  into the timing comparison.
- Timing: profiling off, five paired fresh-process rounds in alternating order,
  20 warmups plus 500 measured executions per case.  Device thermal status was
  0 before and after the run.
- Trace: one fresh-process Optrace capture per variant and sequence length.

The pre-finalize manifests contain 11 Qualcomm `qti.aisw` operations, three
projection ops and 12 quantized activation tensors in every case.  The
activation signatures, shapes, scales, zero points, producers and consumers are
identical between variants.  There is no custom-op or CPU fallback.

## Correctness evidence

Both variants finalize, execute on the phone, return the expected output size,
and reproduce byte-identical output on a second fresh process for s1 and s32.

The host reference explicitly dequantizes the deployed weights and activations,
executes the full MLP, and requantizes at every original QDQ boundary.  Compared
with QNN output, the remaining code differences are consistent with fixed-point
rounding and the hardware sigmoid approximation:

| Graph | Variant | Mean absolute U8 code delta | P99 | Maximum |
|---|---:|---:|---:|---:|
| s1 | LPBQ | 0.358 | 3 | 4 |
| s1 | W8A8 | 0.332 | 3 | 4 |
| s32 | LPBQ | 0.347 | 3 | 5 |
| s32 | W8A8 | 0.335 | 3 | 5 |

W8 requantization has projection-weight NMSE between 0.00122 and 0.00261
relative to the deployed effective LPBQ weights.  The full s1 MLP output NMSE
between the two weight variants is 0.02068.  Accuracy is informational in this
hardware experiment and is not a gate.

## Profiling-off speed

| Graph | LPBQ W4A8 median | Per-tensor W8A8 median | W8 latency change | W8 throughput change |
|---|---:|---:|---:|---:|
| s1 | 437 us | 615 us | +40.7% | -28.9% |
| s32 | 432 us | 619 us | +43.3% | -30.2% |

The five process medians span only 434-439 us for LPBQ s1, 612-619 us for W8
s1, 426-433 us for LPBQ s32, and 616-621 us for W8 s32.  The separation is much
larger than the observed run-to-run variation.

## Optrace explanation

| Graph | Variant | DRAM read | `weights_to_vtcm` work | its dominant-path cycles | W4 expand work | W4 expand dominant | HMX Conv work |
|---|---:|---:|---:|---:|---:|---:|---:|
| s1 | LPBQ | 20.51 MB | 660,765 | 374,923 | 2,581,783 | 339,388 | 152,854 |
| s1 | W8A8 | 38.21 MB | 1,017,162 | 947,666 | 0 | 0 | 132,235 |
| s32 | LPBQ | 20.58 MB | 624,144 | 314,689 | 2,647,919 | 353,787 | 159,280 |
| s32 | W8A8 | 38.27 MB | 1,034,076 | 973,375 | 0 | 0 | 139,562 |

The W8 HMX Conv work is about 13% lower, so the matrix multiplication itself is
not the regression.  W8 reads about 86% more data from DRAM and its
`weights_to_vtcm` operation consumes about 2.5-3.1 times the dominant-path
cycles.  Per-projection traces report maximum parallelism 7 for LPBQ but mostly
1 for W8.  LPBQ's apparently large expand work runs across six HVX lanes and
overlaps weight DMA and HMX consumption; only a small fraction remains on the
dominant path.  W8 removes the expand stage but exposes the larger weight DMA as
the serialized bottleneck.

The finalized W8 contexts are 37,961,728 bytes versus 20,299,776 bytes (s1) and
20,303,872 bytes (s32) for LPBQ, matching the runtime traffic explanation.

## Reproduction

Large artifacts and results remain outside Git:

- Models and contexts:
  `/mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/w8a8_vs_lpbq_layer14_mlp/20260818`
- Results:
  `/mnt/d/llm_exp/results/qwen3_sm8750_v79_layer14_mlp_w8a8_vs_lpbq_20260818`
- Preserved rejected UInt8-carrier diagnostic:
  `/mnt/d/llm_exp/results/qwen3_sm8750_v79_layer14_mlp_w8a8_vs_lpbq_ufixed_weight_diagnostic_20260818`

Run the artifact generator with the accepted source model, then:

```bash
scripts/build_qnn_w8a8_lpbq_layer14_contexts.sh
scripts/run_qnn_w8a8_lpbq_layer14_wsl.sh
scripts/decode_qnn_w8a8_lpbq_layer14_optrace.sh
```

Generate the fail-closed manifest, correctness and timing summary with:

```bash
python3 scripts/qnn_w8a8_lpbq_layer14_summary.py \
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/w8a8_vs_lpbq_layer14_mlp/20260818 \
  /mnt/d/llm_exp/results/qwen3_sm8750_v79_layer14_mlp_w8a8_vs_lpbq_20260818
```

The checked result is `summary.json` in the result directory.  The Python
environment used for artifact generation and host-reference checking must have
NumPy available.
