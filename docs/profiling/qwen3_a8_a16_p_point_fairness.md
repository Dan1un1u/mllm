# Qwen3 W4A8/W4A16 finalize P-point fairness study

Date: 2026-08-22
Target: SM8750 / HTP V79
QAIRT: 2.47.0.260601
Branch: `codex/w4a8g32-lpbq-p-point-search`

## Decision

The fastest validated full-model configurations are:

- W4A8 native-U8 RMSNorm: explicit P19 for both s1 and s32;
- archived W4A16: the original compiler-default selection, with no explicit P override.

The W4A16 check is important for a fair comparison. P6 was the strongest s1
micrograph candidate, but it did not improve the full model. In the decisive
10-round paired run, P6/default reduced prefill throughput by 3.33% and was
decode-neutral within run-to-run noise. It is therefore rejected rather than
used to make the A8 comparison easier.

With each model using its own fastest validated configuration, W4A8 remains
4.91% slower in prefill and 8.50% slower in decode than W4A16.

| Final configuration | Prefill median | Decode median |
|---|---:|---:|
| W4A16, compiler default | 824.672 tok/s | 45.610 tok/s |
| W4A8, global P19 | 784.180 tok/s | 41.734 tok/s |
| W4A8 gap | -4.91% | -8.50% |

P19 remains a useful W4A8 result: relative to the accepted RMSNorm-A8
default in the same paired run, decode improves by 8.54% while prefill is
effectively flat (-0.22% by ratio of medians, +0.49% by paired-round median).

## What a P point changes

`P` is a QAIRT HTP graph-finalize search point. It chooses a compiler
lowering/scheduling solution for an already fixed QNN graph. It is not an
activation precision, tensor shape, quantization group size, or runtime knob.

The experimental controls are:

```text
MLLM_QNN_AOT_FINALIZE_P=<point>      # both graphs; compatibility fallback
MLLM_QNN_AOT_FINALIZE_P_S1=<point>   # decode graph only
MLLM_QNN_AOT_FINALIZE_P_S32=<point>  # prefill graph only
```

Leaving all three unset preserves the original compiler-default path. If only
one graph-specific variable is set, the other graph remains on QAIRT's default
selection. QAIRT 2.47 accepts:

`0,1,2,3,4,5,6,8,13,15,16,17,19,20,21,22,23`

## Search method

All 17 explicit points were first screened for W4A8 and W4A16 on byte-exact
gate, up, down, and lm_head LPBQ micrographs at s1 and s32. This produced 272
explicit contexts, plus defaults. All 288 device-output hashes matched their
respective defaults.

The weighted micrograph leaders were then compiled as full-model candidates:

- W4A8: global P19, P19/default, P19/P17, and P19/P16;
- W4A16: P6/default, P6/P0, and P6/P20.

P8 was not promoted for W4A16: it ranked below P6 in the s1 micrograph screen
and its full-model offline compile made only about 12% progress in 12 minutes.
It was pruned as a pathological compile candidate, not counted as a runtime
failure.

Short full-model screens were treated only as candidate selection. The final
decision used a fresh four-way run containing A8 default/P19 and A16
default/P6-default:

- 10 rounds;
- odd rounds in forward order and even rounds in reverse order;
- five seconds of cooldown between rounds;
- identical runner, prompt, tokenizer, QAIRT runtime, device, and runtime
  configuration within each activation contract;
- profiling disabled for timing;
- thermal status 0 before the run and 1 after the run, with no severe thermal
  status.

| Candidate vs its default | Ratio-of-medians prefill | Paired-median prefill | Ratio-of-medians decode | Paired-median decode |
|---|---:|---:|---:|---:|
| A8 global P19 | -0.22% | +0.49% | +8.54% | +8.17% |
| A16 P6/default | -3.33% | -3.70% | -0.37% | +0.24% |

The A16 decode result is noise-level and the prefill regression is consistent.
The original W4A16 compiler default is therefore the fastest validated
full-model configuration from this search.

## Quantization and numerical controls

The A16 candidate was built with the archived W4A16 source contract and
matching backend libraries. Its s1 and s32 manifests are byte-identical to the
archived baseline manifests:

```text
s1  eb51f99540caf00a947d318418bc353c2b82d2e86814aff49083992144f8f463
s32 9c26f31a3cd15c37114dc0b4f2a662839c2681316b54e1e7b25e2a1dad503d27
```

Thus P6 changed graph finalization only; it did not change weights, activation
encodings, calibration parameters, shapes, or W4G32 LPBQ metadata.

The 100-question deterministic W4A16 sanity suite gives:

| Context | Passed | CSV SHA-256 |
|---|---:|---|
| archived W4A16 default | 77/100 | `6c77abea821bb6de6337a1957392a0e0cf90026ee81f1226d2ea0ec2085cb1eb` |
| current W4A16 default | 77/100 | same |
| current W4A16 P6/default | 77/100 | same |

The three CSV files are byte-identical, not merely score-identical. This is a
minimum sanity test rather than a formal accuracy benchmark, but it directly
shows that the tested compiler scheduling change did not alter any generated
answer. The W4A8 P19 experiment had already passed the corresponding
byte-identical output and manifest checks documented in
`lpbq_p19_vtcm_dma_pipeline.md`.

## Interpretation

The micrograph winner is not automatically the full-model winner. P6 improves
isolated W4A16 s1 projection scheduling, but the complete prompt/decode
pipeline adds graph transitions, non-projection operations, resource overlap,
and mixed s32/s1 execution. Those effects erase the isolated gain and expose a
prefill regression. This is why the final gate is full-model E2E rather than
micrograph cycles alone.

For W4A8, P19 is different: the decode benefit has repeated across the
original P19 experiment, the candidate screens, and the final paired run. Its
benefit comes from improved DMA/expand/HMX overlap, not from changing the
model's mathematics. Even after that repair, the remaining A8 regression
relative to W4A16 is real; it is not an unfair comparison against an
untuned A16 context.

## Artifacts

- exhaustive micrograph contexts:
  `D:\llm_exp\models\qwen3_sm8750_v79\g32\lpbq_p_point_fairness\20260822`
- exhaustive micrograph results:
  `D:\llm_exp\results\qwen3_sm8750_v79_lpbq_p_point_fairness_micrographs_20260822`
- full-model candidate contexts:
  `D:\llm_exp\models\qwen3_sm8750_v79\g32\p_point_fairness_full\20260822`
- broad full-model screen:
  `D:\llm_exp\results\qwen3_sm8750_v79_p_point_fairness_full_screen_20260822`
- focused finalist screen:
  `D:\llm_exp\results\qwen3_sm8750_v79_p_point_fairness_finalists_20260822`
- decisive paired run:
  `D:\llm_exp\results\qwen3_sm8750_v79_p_point_fairness_best_20260822`
- W4A16 accuracy equivalence:
  `D:\llm_exp\results\qwen3_sm8750_v79_a16_p_point_accuracy_20260822`
