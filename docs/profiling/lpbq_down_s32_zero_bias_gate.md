# LPBQ down-projection S32 explicit-zero-bias gate

## Question

The accepted RMSNorm-U8 baseline omits the logical bias input for Qwen3 MLP
projections, while HTP Optrace still exposes a physical
`ConvLayer.opt.bias_to_vtcm` phase. This experiment tests whether supplying an
explicit static zero bias removes that phase or lowers its cost.

Only `model.layers.14.mlp.down_proj` at sequence length 32 is tested. Both
variants use the accepted RMSNorm-U8 model, the same `[1,32,6144]` U8 input,
the same `[1,1,6144,2048]` LPBQ W4G32 weight, the same U8 output qparams, QAIRT
2.47.0.260601, SM8750/V79, and the same Conv2D expression.

- `omitted`: QNN Conv2d has activation and weight inputs.
- `explicit_u8_zero`: QNN Conv2d additionally receives a static `[2048]`
  `UFIXED_POINT_8` all-zero bias with zero-point 0 and output-matching scale.

The experiment-only option defaults to false, so existing model graphs are
unchanged.

## Acceptance protocol

1. Both contexts must finalize under QAIRT 2.47.
2. The manifest audit must prove that bias is the only logical contract delta.
3. Six deterministic fixtures (`encoded_zero`, qmin, qmax, alternating, ramp,
   and seeded random) run twice per variant. All four outputs per fixture must
   be byte-identical.
4. Profiling-off timing uses five fresh-process paired rounds with alternating
   order. Each process performs 20 warmups and 500 measured executions.
5. The candidate passes the speed gate only if its median of round medians is
   no more than 1% slower than omitted.
6. One Optrace capture per variant checks the physical HTP event signature and
   phase costs. Optrace cycles diagnose lowering but do not replace the
   profiling-off timing gate.

## Result

Correctness and structural gates passed; the speed gate failed.

| Metric | omitted | explicit U8 zero bias |
|---|---:|---:|
| Median of five round medians | 167 us | 170 us |
| Candidate / omitted | 1.000 | 1.018 |
| Speed change | — | -1.76% |
| Six-fixture output equality | byte exact | byte exact |
| Target Optrace work cycles | 1,101,781 | 1,103,444 |
| Target Optrace active union | 264,538 | 260,985 |

The physical event signatures are identical, including
`bias_to_vtcm`; explicit bias does not remove or change the physical phase
structure. Its one-capture bias-phase work was 39,405 cycles versus 34,882 for
omitted, while other overlapping phases varied in both directions. This is
consistent with the profiling-off result: QAIRT already materializes the
omitted bias into the same physical LPBQ Conv2D schedule, and making zero bias
explicit provides no speed benefit.

The candidate must not proceed to a full-model change. The original omitted
bias path remains the baseline.

## Reproduction and evidence

Source tools:

- `mllm-qwen3-lpbq-down-zero-bias-c`
- `mllm-qwen3-lpbq-down-zero-bias-runner`
- `scripts/audit_lpbq_down_zero_bias.py`
- `scripts/generate_lpbq_down_zero_bias_fixtures.py`
- `scripts/summarize_lpbq_down_zero_bias_gate.py`
- `scripts/summarize_lpbq_down_zero_bias_optrace.py`

Archived model/intermediate artifacts:

```text
D:\llm_exp\models\qwen3_sm8750_v79\g32\
  w4a8_rmsnorm_u8_lpbq_down_zero_bias\20260818_layer14_s32
```

Archived result:

```text
D:\llm_exp\results\
  qwen3_sm8750_v79_w4a8_rmsnorm_u8_lpbq_down_zero_bias_gate_20260818
```

The authoritative compact summaries are `manifest_audit.json`,
`gate_summary.json`, and `optrace_summary.json`. Raw paired timing CSVs and
decoded QNN Optrace outputs are retained under the same result directory.
