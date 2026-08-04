# W4A8O8 symmetric V-boundary requantization results (2026-08-05)

This is the execution record for
`w4a8o8_vsym_gpu_requant_plan_20260805.md`.

## Contract and sources

- BF16 teacher: `/home/daniuniu/llm_exp/models/Qwen3-origin`
- fixed G32 code and initial scales:
  `/mnt/d/llm_exp/models/Qwen3-1.7B-G32-base/model.safetensors`
- base SHA-256:
  `6caa0d36ef6846ad2ab138335204699d9abd44c7dd30df25a5846a6b3e5e6bae`
- no rotation and no W4 code regeneration;
- 200-step W4A16 LPBQ stage followed by 400-step symmetric-A8/LPBQ stage;
- prefix-streaming, one block autograd graph at a time.

The explicit recipe is `sym128_vsym`: signed int8 values are stored in a UInt8
tensor with fixed integer `zero_point=128`, with `scale=alpha/127`. For a V-A8
layer, the V projection output, V cache, and attention value input share one
manifest scale identity. Sensitive input tensors retain the selected A16
fallback map.

## Leakage-controlled software oracle

The fixed calibration split contains 4 train prompts and 6 held-out prompts.
The first `sym128-vsym-allrisk` and `sym128-vsym-mixed` runs selected candidates
on held-out data; their approximately `0.9997` held-out cosine is selection
leakage (independent train top-1 was only 25%), so they are diagnostics only.
All gate decisions below use `selection_split=train` and the independent
no-grad evaluator.

| artifact | V fallback policy | held-out logits cosine | held-out top-1 | held-out NMSE |
| --- | --- | ---: | ---: | ---: |
| W4A16 reference | 196/196 A16 | 0.916866 | 100.0% | 0.159378 |
| `sym128-vsym-allrisk-trainselect` | V output A8 in all 28 layers | 0.915688 | 66.7% | 0.165766 |
| `sym128-vsym-mixed-trainselect` | V output A8 in all 28 layers | 0.924268 | 66.7% | 0.154201 |
| `sym128-vsym-vfallback-allrisk-trainselect` | 8 V-A16, 20 V-A8 layers | 0.897785 | 66.7% | 0.204772 |
| `sym128-vsym-vfallback-mixed-trainselect` | 4 V-A16, 24 V-A8 layers | 0.921037 | 66.7% | 0.163312 |

The `vfallback` rows implement the strict layer/tensor interpretation of the
roadmap: when `v_proj` is selected A16, its output/cache boundary is also kept
A16. They are the deployment-shaped candidates. The strict mixed candidate is
the best current symmetric candidate by held-out cosine, but neither reaches
W4A16 top-1 parity and this six-example held-out result is not an accuracy GO.

## Offline gates

All four symmetric manifests pass:

1. packed HWIO and logical OI INT4 code hashes: `0 mismatch` for all `196`
   projections;
2. root and per-tensor `recipe=sym128_vsym`, `storage_dtype=UInt8`, and
   integer `zero_point=128`;
3. every V scale identity is either tied exactly or explicitly A16 fallback;
4. merged safetensors replace only LPBQ `scale1/scale2`; merge reports
   `checked 196 LPBQ scale pairs; codes_unchanged=true`.

Reusable checks:

```bash
python scripts/verify_lpbq_base_alignment.py \
  --base-checkpoint /mnt/d/llm_exp/models/Qwen3-1.7B-G32-base/model.safetensors \
  --training-dir artifacts/p1/sym128-vsym-vfallback-allrisk-trainselect \
  --training-dir artifacts/p1/sym128-vsym-vfallback-mixed-trainselect

python scripts/verify_sym128_vsym_manifest.py \
  --training-dir artifacts/p1/sym128-vsym-vfallback-allrisk-trainselect \
  --training-dir artifacts/p1/sym128-vsym-vfallback-mixed-trainselect
```

Each strict artifact directory contains 28 layer scale files, a complete
`streaming-train.json`, and a local merged
`qwen3_w4a8o8_vsym_g32.safetensors` (~2.4 GB). The large merged files are kept
as VM handoff artifacts rather than committed to Git.

## VM/QNN next gate

Do not build the old `mapzp` directories or the held-out-selected symmetric
directories as final candidates. In the VM, use one of the two strict merged
files, regenerate the V79/QNN context with QAIRT `2.47.0.260601`, and check
that V projection output, cache append, and attention value input are all
`UInt8PerTensorSym(zp=128)` for V-A8 layers, with no second
`.to(kUInt8PerTensorSym)` requant. Run accuracy sanity and speed/optrace as
separate gates.

If the strict mixed candidate still shows a large device accuracy gap, first
expand the calibration prompt set and retrain with `selection_split=train`; do
not broaden to full-model symmetric projection or add rotation until the
V-only software/QNN contracts agree.
