# LPBQ gate/up S=1 output-channel split micrograph

## Decision

The experiment failed its performance gate and stops before full-model integration.
The candidate is mathematically exact, but it is not consistently at least as fast
as the accepted native-U8 RMSNorm baseline projection. In particular, layer-14
`gate_proj` regressed beyond the agreed 1% measurement-noise allowance.

## Controlled change

The reference is one LPBQ Conv2D projection:

```text
[1,1,1,2048] U8 x [1,1,2048,6144] W4G32 -> [1,1,1,6144] U8
```

The candidate splits only the output-channel axis:

```text
oc0: [1,1,1,2048] U8 x [1,1,2048,3072] W4G32 -> [1,1,1,3072] U8
oc1: [1,1,1,2048] U8 x [1,1,2048,3072] W4G32 -> [1,1,1,3072] U8
concat(oc0, oc1) -> [1,1,1,6144] U8
```

The experiment uses Qwen3-1.7B layer 14, SM8750/V79, QAIRT
`2.47.0.260601`, W4 signed codes `[-7,7]`, group size 32, and the accepted
asymmetric-U8 input/output qparams. The compact artifact is derived only from
the accepted native-U8 RMSNorm model. Each split weight, block-scale, and
channel-scale pair reconstructs its full tensor byte-for-byte.

## Correctness and graph audit

- Four contexts finalize: full/split2 for `gate_proj` and `up_proj`.
- All projection inputs and outputs are `UFIXED_POINT_8` with unchanged
  asymmetric per-tensor encodings.
- All weights retain signed W4G32 LPBQ blockwise expansion, HWIO layout,
  UInt4 block scales, and per-output-channel Float32 scales.
- Six deterministic inputs are run for both projections. All 12 split outputs
  are byte-exact with the corresponding full projection outputs.

The durable audits are `manifest_audit.json` and `correctness_summary.json` in
the result directory.

## Profiling-off gate

Each entry is the median of five fresh-process round medians. Each process uses
20 warmups and 500 measured graph executions; reference/candidate order
alternates by round.

| Projection | Full | Split2 | Split2 / full | Split2 slower rounds | Gate |
|---|---:|---:|---:|---:|---|
| gate_proj | 203 us | 207 us | 1.0197 | 4 / 5 | fail |
| up_proj | 206 us | 204 us | 0.9903 | 3 / 5 | pass, but unstable |

The paired-ratio medians are 1.0488 for gate and 1.0200 for up. Consequently,
the apparent aggregate up improvement is not a robust per-round win.

## Optrace diagnosis

QAIRT lowers both expressions to the same physical amount of projection work.
For every full/split case, the target projection has:

| Physical stage | Instances |
|---|---:|
| weight DMA transfer | 192 |
| weight DMA wait | 96 |
| W4-to-QInt8 expand | 576 |
| HMX MAC tile | 96 |
| checkpoint/sync | 140 |
| total target HTP instances | 1215 |
| target DRAM read | 6,881,280 bytes |

The HMX output tile remains `[1,1,1,64]`; the split does not make the physical
tile or expanded working set smaller. The pre-finalize Concat does not appear
as an additional standalone QNN operator in Optrace, and total target HTP work
count is unchanged. QAIRT already tiles the 6144-channel reference internally.

The main single-capture differences are:

| Projection | Metric | Full | Split2 | Delta |
|---|---|---:|---:|---:|
| gate | projection envelope | 352,288 | 359,141 | +1.95% |
| gate | weight-wait direct cycles | 136,964 | 141,821 | +3.55% |
| gate | W4-expand direct cycles | 143,653 | 144,871 | +0.85% |
| gate | HMX direct cycles | 49,665 | 49,235 | -0.87% |
| gate | HMX idle gaps | 255,022 | 260,749 | +2.25% |
| up | projection envelope | 366,372 | 351,581 | -4.04% |
| up | weight-wait direct cycles | 149,636 | 134,833 | -9.89% |
| up | W4-expand direct cycles | 146,320 | 145,245 | -0.73% |
| up | HMX direct cycles | 48,641 | 49,415 | +1.59% |
| up | HMX idle gaps | 268,233 | 252,667 | -5.80% |

No HMX interval overlaps a weight DMA transfer in any case. DMA overlaps HVX
expansion instead. The candidate overlaps its two QNN-node envelopes by about
67.7k cycles for gate and 65.7k cycles for up, but this does not reduce the
fixed tile/checkpoint counts. The speed movement is dominated by weight-wait
schedule variation: it becomes worse for gate and better for up.

## Conclusion

Graph-level 3072+3072 output splitting is the wrong control surface for this
bottleneck. It preserves the arithmetic and avoids a large Concat penalty, but
QAIRT canonicalizes both forms into the same 64-channel LPBQ tiles. It therefore
cannot restore a different DMA-expand-HMX double-buffer pipeline reliably.

Do not apply this candidate to all MLP layers. A subsequent experiment should
remain a micrograph and act on a control that can actually change physical
lowering, such as an asymmetric partition sweep that triggers a different
tiling specialization, or a pinned HTP compiler/VTCM/HVX-thread configuration
sweep. Such experiments must retain the same byte-exact and per-projection
performance gate.

## Evidence locations

- Model namespace: `D:\llm_exp\models\qwen3_sm8750_v79\g32\w4a8_rmsnorm_u8_lpbq_gateup_oc_split\20260818_layer14_s1`
- Result namespace: `D:\llm_exp\results\qwen3_sm8750_v79_w4a8_rmsnorm_u8_lpbq_gateup_oc_split_gate_20260818`
- Compact gate result: `gate_summary.json`
- Raw fresh-process timings: `profiling_off/`
- Raw and decoded traces: `optrace/`
