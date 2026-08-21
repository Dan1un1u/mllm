# LPBQ W4G32 A16 versus A8 projection pipeline experiment

Date: 2026-08-21
Target: SM8750 / HTP V79
QAIRT: 2.47.0.260601
Branch: `codex/w4a8g32-lpbq-a16-vs-a8-projections`

## Outcome

The experiment gate passes.  The 16 standalone projection graphs compile and
execute deterministically on HTP, use one byte-identical common W4G32 LPBQ
payload, and differ only in their asymmetric A16/A8 graph I/O contract and the
corresponding formal activation qparams.  Sampled host references agree with
device output within one output code for every case.

The central result is narrower than the earlier full-model attribution:

- A8 does **not** universally slow the MLP.  At `s32`, gate, up, and down are
  all faster than A16.  At `s1`, gate/up are slower but down is faster.
- `lm_head` is consistently slower with A8 at both `s1` and `s32`.
- W4 expansion and HMX MAC are not the source of the regressions.  A8 usually
  lowers their work.
- The regressing cases are dominated by a slower `weights_to_vtcm` schedule.
  The compressed-weight DRAM byte count is unchanged, but its work and
  dominant-path contribution increase sharply.  A8 therefore changes the
  physical DMA/tiling schedule and damages DMA--expand--HMX overlap.

## Strict single-variable construction

Directly comparing the archived A16 and A8 model files would not be a strict
activation-width experiment: the W4 carrier, block scales, or channel scales
are not byte-identical for any of the four selected projections.  The test
therefore uses one compact artifact with static tensors copied from the
accepted native-U8 RMSNorm model:

`D:\llm_exp\models\qwen3_sm8750_v79\g32\lpbq_a16_vs_a8_projections\20260821\qwen3-lpbq-a16-a8-projections.mllm`

Artifact SHA-256:
`3a5ac1fff05e01c50fb216861b21564a45ceb52290c602f2713d5a2943c49515`

Both variants use:

- `qti.aisw::Conv2d`;
- NHWC activation and HWIO static weight layout;
- signed W4 codes in `[-7, 7]`;
- G32 blockwise expansion;
- UInt4 block scales and FP32 per-output-channel scales;
- the exact same carrier/scale payload.

The only runtime variable is:

- A16: `UFIXED_POINT_16` asymmetric input/output, with qparams copied from the
  archived W4A16 model;
- A8: `UFIXED_POINT_8` asymmetric input/output, with qparams copied from the
  accepted W4A8 native-U8 RMSNorm model.

Paired inputs are quantizations of the same deterministic FP32 fixture.  No
fixture saturates.  This experiment is a hardware-pipeline comparison, not an
accuracy comparison between the two formal models.

## Timing result

Each number is the median of five fresh-process medians.  Every process uses
20 warmups and 500 measured `QnnGraph_execute` calls with profiling disabled;
variant order alternates by round.

| Projection | Seq | A16 (us) | A8 (us) | A8 / A16 | A8 speed change |
|---|---:|---:|---:|---:|---:|
| gate | 1 | 174 | 191 | 1.098x | -8.9% |
| up | 1 | 171 | 190 | 1.111x | -10.0% |
| down | 1 | 152 | 143 | 0.941x | +6.3% |
| lm_head | 1 | 3415 | 4061 | 1.189x | -15.9% |
| gate | 32 | 210 | 188 | 0.895x | +11.7% |
| up | 32 | 207 | 187 | 0.903x | +10.7% |
| down | 32 | 165 | 153 | 0.927x | +7.8% |
| lm_head | 32 | 3541 | 4111 | 1.161x | -13.9% |

The five process medians are stable and preserve the direction in every case.

## Physical pipeline ratios

Ratios below are A8/A16 from one fresh-process Optrace per case.  `wall` is the
projection wall span, `weight work/dom` is `weights_to_vtcm`, `expand` is W4
block expansion, and `HMX` is the physical projection MAC kernel.

| Projection | Seq | wall | weight work | weight dom | expand work | HMX work |
|---|---:|---:|---:|---:|---:|---:|
| gate | 1 | 1.158x | 1.482x | 1.927x | 0.991x | 0.853x |
| up | 1 | 1.143x | 1.452x | 1.838x | 1.012x | 0.870x |
| down | 1 | 0.947x | 1.035x | 1.005x | 0.981x | 0.991x |
| lm_head | 1 | 1.199x | 1.665x | 2.168x | 0.982x | 0.956x |
| gate | 32 | 0.967x | 0.985x | 0.996x | 0.935x | 0.906x |
| up | 32 | 0.942x | 0.946x | 0.925x | 0.943x | 0.921x |
| down | 32 | 0.888x | 0.943x | 0.824x | 0.959x | 1.001x |
| lm_head | 32 | 1.185x | 1.289x | 1.644x | 1.004x | 0.936x |

### Why this identifies scheduling rather than weight volume

For every A16/A8 pair, `weights_to_vtcm` reads exactly the same compressed
weight bytes:

- gate/up/down: 6,684,672 bytes;
- lm_head: 165,232,640 bytes.

Instance counts are also identical for gate/up/down.  Nevertheless:

- gate s1 weight transfer grows from 173,621 to 257,328 work cycles and from
  74,064 to 142,738 dominant cycles;
- up s1 grows from 172,473 to 250,424 work cycles and from 72,955 to 134,081
  dominant cycles;
- lm_head s1 grows from 3,910,284 to 6,508,818 work cycles and from 1,878,909
  to 4,072,814 dominant cycles;
- lm_head s32 grows from 4,659,395 to 6,004,932 work cycles and from 1,343,078
  to 2,207,755 dominant cycles.

The data volume did not increase.  The same DMA traffic takes longer and lands
more heavily on the critical path.  This is consistent with changed tiling,
checkpoint placement, and reduced overlap with expansion/MAC, not with a more
expensive W4 decode algorithm.

### W4 expansion and HMX behave as expected

W4 expansion work stays within roughly +/-7% and is lower in six of eight
cases.  HMX projection work is lower for A8 in seven cases and essentially
flat for down s32.  Thus the intended per-tensor INT8 HMX computation is being
used; the regression occurs around it.

### Graph I/O does not explain the regressions

A8 reduces graph-input reads as expected.  It also reduces or preserves the
input-formatting work.  There is no explicit physical `requant` kernel in any
case; output quantization is fused into the projection/output-format boundary.

However, HTP output DMA often does not shrink even though host output is half
the size:

- gate/up output writes are 393,216 bytes for both A16 and A8;
- down output writes are 131,072 bytes for both;
- lm_head s32 output writes are 9,723,904 bytes for both;
- only lm_head s1 approximately halves, from 305,152 to 153,600 bytes.

These counts match minimum/padded Crouton/output tiles rather than logical host
tensor bytes.  Activation-width savings are therefore partly hidden by
physical tiling.  This removes an expected A8 benefit, but it does not create
the s1/lm_head regression: the decisive increase is still weight DMA time.

### DMA checkpoints reflect a different schedule

The physical graph changes checkpoint/synchronization structure even with the
same static weight bytes.  For example:

- gate/up s1: `DmaCheckpointSet` instances change from 140 to 116, while
  weight-transfer dominant cycles nearly double;
- lm_head s1: checkpoints change 3276 -> 2393 and `SyncOp` instances
  314 -> 657;
- lm_head s32: checkpoints change 1460 -> 1962 and `SyncOp` instances
  1323 -> 919.

The direction is not a simple "more checkpoints is slower" rule.  It is
evidence that A8 selects a different tiling/runlist, which changes when the
same weight DMA can overlap expansion and HMX.

## Correctness and integrity evidence

- All 16 contexts contain only Qualcomm `qti.aisw` operations; no custom or CPU
  fallback is present.
- All finalized static weights report identical W4G32 blockwise signatures
  between A16 and A8.
- All non-static quantized tensors are uniformly U16 in the A16 graphs and U8
  in the A8 graphs.
- First/repeat device outputs are byte-exact in all 16 cases.
- The sampled software reference differs by at most one output code in every
  case; mean absolute code error is at most 0.188 for A16 and 0.031 for A8.

No model-accuracy threshold was applied, per the experimental baseline scope.

## Interpretation for the full W4A8 model

This experiment refines the earlier statement that "MLP is slower":

1. Decode-like `s1` gate/up do regress, so repeated layer MLP projections can
   contribute to the W4A8 decode loss.
2. Down projection is not a regression source in either shape.
3. Prefill-like `s32` MLP projections are individually faster, so the full
   prefill loss should not be assigned to MLP as a whole.  Softmax and
   lm_head remain stronger candidates, plus full-graph scheduling interactions.
4. `lm_head` is a separate, large, consistent A8 regression and should be
   investigated independently from layer MLP.

The next optimization target should therefore be the native LPBQ
`weights_to_vtcm` schedule for gate/up s1 and lm_head, while preserving the
existing W4 expansion and HMX kernels.  Rewriting W4 decode or replacing LPBQ
with W8 would attack the wrong component and would lose the compressed-weight
bandwidth benefit already demonstrated by the W8A8 comparison.

## Artifacts

- Compact model and 16 contexts:
  `D:\llm_exp\models\qwen3_sm8750_v79\g32\lpbq_a16_vs_a8_projections\20260821`
- Raw timing, Optrace, decoded traces, and machine-readable summary:
  `D:\llm_exp\results\qwen3_sm8750_v79_lpbq_a16_vs_a8_projections_20260821`
- Machine-readable gate report:
  `summary.json`
