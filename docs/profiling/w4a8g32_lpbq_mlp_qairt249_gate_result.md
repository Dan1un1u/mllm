# W4A8G32 LPBQ MLP Gate with QAIRT 2.49

Status: **failed; stopped before cold timing, layer integration, and full-model
migration**.

This follow-up repeated the layer-14 LPBQ operator-expression gate with QAIRT
2.49.0.260730 on SM8750/V79. It started from source commit
`274c2f5a3f68b3f4fd375577d040fe0225e9b7c0`, whose accepted model input is the
native-U8 RMSNorm baseline from commit `987a7156`. The target cases remained
the real `gate_proj`, `up_proj`, and `down_proj` shapes at sequence lengths 1
and 32. Weight values, signed W4 `[-7,7]`, G32 scales, A8 activation encodings,
and all non-MLP computation were unchanged.

## SDK isolation

The supplied SDK archive was used without replacing the pinned QAIRT 2.47
installation:

```text
C:\Users\35961\Downloads\v2.49.0.260730.zip
SHA256 32DE9B5B2B069AEB93BA090071E777FF464349B3296358C4D3B35040DCCBD159
```

Only the required headers and host/device libraries were extracted into a
temporary WSL-native directory. Separate x86 AOT and Android builds were made
under `build-qnn-aot-qairt249-exp` and
`build-android-arm64-v8a-qairt249-exp`. Their compile commands referenced the
2.49 headers and did not reference the 2.47 SDK. The device used the separate
exact directory `/data/local/tmp/mllm_w4a8_lpbq_mlp_qairt249`.

An initial FC compile was rejected as invalid evidence after discovering that
the context-build script put the default mllm build directory ahead of the
selected compiler's runtime directory. The script was corrected to resolve
all mllm shared libraries beside the selected compiler, the invalid case was
deleted, and every reported context was regenerated with a wholly 2.49
runtime stack.

## Candidate disposition

`FullyConnected` still failed V79 graph preparation at the same internal
operation as QAIRT 2.47:

```text
no properties registered for q::GenPad
Graph prepare failed with err:-1
```

No FC context was produced, so FC could not enter device correctness or
timing.

`MatMul` finalized for all six cases and retained exact logical W4G32 and A8
contracts. QAIRT 2.49 changed one schematic detail: the `SpecialMatmul` marker
seen in 2.47 disappeared. It did **not** select a distinct LPBQ hardware
family. Each paired MatMul and Conv context still contained the same counts of
`ConvLayer`, `QNN_CastInt4ToInt8`,
`expand_block_quant_to_pc_int8_weights`, `weights_to_vtcm`, and HMX work.

| Projection family | ConvLayer | W4 expansion | Weight-to-VTCM | SpecialMatmul 2.47 | SpecialMatmul 2.49 |
|---|---:|---:|---:|---:|---:|
| gate / up | 1344 | 289 | 576 | 481 | 0 |
| down | 448 | 97 | 192 | 161 | 0 |

The 2.49 lowering therefore remains a front-end MatMul expression
canonicalized to the existing Conv physical LPBQ path.

## Correctness

All 72 device rows passed. Both executions of every row were deterministic,
all 36 paired Conv/MatMul output hashes were identical, and the maximum error
against the float32 reference decoded from the exact deployed weights and
qparams was one output LSB. The zero fixture was exact.

The fixture named `calibration_qparam_replay` remains a fixed-seed Gaussian
real-value replay through the pinned layer-14 qparams, not a captured
activation tensor. This disclosed limitation does not affect the decisive
hardware speed failure.

## Warm performance gate

Each process performed 10 warmups followed by 100 measured executions. Five
fresh processes used alternating Conv/MatMul order. The first set was invalid
under the predeclared one-percent process-dispersion rule, so the single
permitted complete recollection was run after cooldown.

The recollection was:

| Layer-14 projection | Seq | Conv median | MatMul median | Ratio | Gate state |
|---|---:|---:|---:|---:|---|
| gate | 1 | 208 us | 208 us | 1.0000 | invalid: dispersion/direction |
| gate | 32 | 207 us | 228 us | 1.1014 | invalid: dispersion; slower in 5/5 pairs |
| up | 1 | 208 us | 208 us | 1.0000 | invalid: dispersion/direction |
| up | 32 | 208 us | 228 us | 1.0962 | **valid fail** |
| down | 1 | 162 us | 163 us | 1.0062 | invalid: direction |
| down | 32 | 168 us | 204 us | 1.2143 | invalid: dispersion; slower in 5/5 pairs |

`up_proj` s32 is independently decisive: both layouts satisfied the
one-percent dispersion bound, the direction was consistent in all five
paired processes, and MatMul was 9.62% slower than Conv, exceeding the
1.01 non-regression threshold. The other s32 cases corroborated the same
direction, with 10.14% and 21.43% median regressions.

The result is effectively unchanged from QAIRT 2.47 despite removal of the
`SpecialMatmul` marker. The corresponding 2.47 recollection ratios were
1.1063, 1.1014, and 1.2275 for gate, up, and down s32; the 2.49 ratios were
1.1014, 1.0962, and 1.2143.

## Conclusion and advancement decision

QAIRT 2.49 does not rescue either uniform candidate. FullyConnected still
cannot finalize, while MatMul remains materially slower for real s32 MLP
shapes and continues to use the ConvLayer/W4-expansion physical path. Removing
the `SpecialMatmul` marker alone is not a performance-relevant native LPBQ
MatMul implementation.

Per the accepted gate, no cold-weight timing, complete layer-14 MLP,
neighboring full graph, 84-projection model, end-to-end profile, or Optrace was
built after the decisive warm failure. The accepted native-U8 RMSNorm branch
remains the successful baseline and is unchanged.

Compact evidence is stored in:

```text
D:\llm_exp\results\qwen3_sm8750_v79_w4a8_rmsnorm_u8_lpbq_mlp_qairt249_gate_20260818
```

The authoritative files are `gate-conclusion.json`,
`correctness-report.json`, `structure-audit.json`, `warm-summary.json`,
`recollect/warm-summary.json`, `manifest-extracts/`, the two transform
summaries, and `fc-gate-proj-s1-finalize.log`. Failed candidate models,
contexts, raw outputs, temporary QAIRT files, device files, and isolated build
caches are intentionally deleted after compact evidence publication.
