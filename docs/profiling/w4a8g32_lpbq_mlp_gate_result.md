# W4A8G32 LPBQ MLP Operator Gate Result

Status: **failed; stopped before cold timing, layer integration, and full-model
migration**.

This experiment used QAIRT 2.47.0.260601 on SM8750/V79 and started from the
accepted native-U8 RMSNorm baseline at commit `987a7156`. It evaluated the
real layer-14 `gate_proj`, `up_proj`, and `down_proj` shapes for sequence
lengths 1 and 32. It did not alter attention, RMSNorm, Softmax, RoPE, KV cache,
activation calibration, or the logical W4G32 projection mathematics.

## Candidate disposition

`FullyConnected` did not survive context finalization. Its pre-finalize
manifest contained `FullyConnected_w_blk_exp_scale`, but V79 graph preparation
aborted with:

```text
no properties registered for q::GenPad
Graph prepare failed with err:-1
```

No FC context was produced, so the candidate was not run or timed.

`MatMul` passed host layout/decode, QAIRT finalization, structural audit, and
on-device correctness. However, its finalized schematic retained the same
physical `ConvLayer`, `QNN_CastInt4ToInt8`, weight-expansion, weight-to-VTCM,
and HMX families as the Conv2D reference. `SpecialMatmul` appeared in addition.
This is front-end MatMul acceptance, not evidence of a distinct faster LPBQ
kernel.

## Correctness

All 72 paired device cases were deterministic. MatMul and Conv2D output codes
were bit-for-bit identical for all six shape cases and six fixtures. Against
the float32 reference decoded from the exact deployed signed W4 `[-7,7]`, G32,
scale1, scale2, and A8 qparams, maximum error was at most one output LSB; the
real-zero fixture was exact.

The fixture named `calibration_qparam_replay` is deliberately disclosed as a
fixed-seed Gaussian real-value replay through the pinned calibration qparams.
It is not a captured layer-14 activation. Therefore the intended real captured
activation fixture was not obtained. This limitation does not reverse the
candidate's decisive hardware performance failure.

## Warm performance gate

Each process used 10 warmups and 100 measured executions. Five fresh processes
were paired with alternating Conv/MatMul order. The first collection exceeded
the one-percent stability bound, so the protocol's single permitted complete
recollection was performed after cooldown. Thermal status remained zero.

| Layer-14 projection | Seq | Conv median | MatMul median | Ratio | Result |
|---|---:|---:|---:|---:|---|
| gate | 1 | 207 us | 206 us | 0.9952 | invalid: dispersion/direction |
| gate | 32 | 207 us | 229 us | 1.1063 | fail |
| up | 1 | 205 us | 207 us | 1.0098 | invalid: dispersion/direction |
| up | 32 | 207 us | 228 us | 1.1014 | fail |
| down | 1 | 162 us | 163 us | 1.0062 | invalid: dispersion/direction |
| down | 32 | 167 us | 205 us | 1.2275 | fail |

The s32 direction was consistent in every paired process. Paired ratios were
1.069--1.118 for gate, 1.087--1.117 for up, and 1.220--1.234 for down. The
recollection also remained invalid under the predeclared stability rule, which
is independently a mandatory stop condition. Sampling was not repeated until
a favorable result appeared.

## Conclusion and advancement decision

Neither uniform candidate survived the operator gate: FC cannot finalize and
MatMul is materially slower for every s32 MLP projection. The most defensible
interpretation is that QAIRT 2.47 canonicalizes this MatMul LPBQ expression to
the existing Conv physical implementation while retaining extra MatMul
handling; the front-end expression alone does not unlock a better LPBQ
lowering.

Per the accepted protocol, cold-weight timing was not collected after the
decisive warm failure, and no complete layer-14 MLP, neighboring full-graph,
84-projection model, end-to-end profile, or Optrace was built. The accepted
native-U8 RMSNorm baseline remains the current successful model branch.

Compact external evidence is stored in:

```text
D:\llm_exp\results\qwen3_sm8750_v79_w4a8_rmsnorm_u8_lpbq_mlp_gate_20260818
```

The authoritative files are `gate-conclusion.json`, `correctness-report.json`,
`structure-audit.json`, `warm-summary.json`,
`recollect/warm-summary.json`, the manifest extracts, and the FC finalization
log. Large failed model artifacts, contexts, raw outputs, and disposable build
caches are intentionally not retained.
