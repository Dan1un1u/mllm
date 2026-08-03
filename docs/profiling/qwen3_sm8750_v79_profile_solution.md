# Qwen3-1.7B × SM8750 V79 profiling solution

Canonical entry point:

```bash
cd /home/daniuniu/llm_exp/mllm
./run_qwen3_sm8750_v79_profile.sh
```

Each invocation creates a new directory:

```text
/home/daniuniu/llm_exp/results/qwen3_sm8750_v79_YYYYmmdd_HHMMSS/
```

The final report is:

```text
qwen3-sm8750-v79-e2e-critical-path.html
```

## What one run does

1. Builds and pushes the current Android QNN runner.
2. Verifies the native SM8750/V79 context SHA-256.
3. Runs three independent `profiling=off` benchmark processes and reports the
   median runner-phase prefill/decode throughput.
4. Runs a profiling-off, 100-case greedy short-answer accuracy sanity suite.
5. Starts one fresh process to capture the first `model.0.s32` execution.
6. Starts a second fresh process to capture the first `model.0.s1` execution.
7. Decodes both Optrace payloads with the matching V79 schematic.
8. Generates QNN operator CSVs, Qwen3 structure CSVs and the canonical HTML.

Optrace and throughput are intentionally collected in separate processes.
Optrace significantly perturbs the first execution and must not be used as the
measured token/s result.

## Report contents

- Full graph: mutually exclusive critical-path contribution in Qwen3 structure
  order, with all 28 layers aggregated by semantic stage.
- Layer 0: representative operator E2E envelopes filled by owned
  HMX/HVX/DMA critical contribution.
- Local timelines: Layer 0 and the lowered LM-head HMX/6×HVX/DMA pipeline.
- Resource pies: full graph, Layer 0 and LM head.
- Prefill and decode token/s from the independent runner-level benchmark.
- Lightweight aggregate accuracy score in the same HTML. Per-case outputs stay
  in the result CSV/JSON for debugging and are not embedded in the report.

The accuracy result is a low-cost regression sentinel, not MMLU/CMMLU. It is
intended to catch obvious model/context/runtime corruption or a large
quantization quality drop. The suite uses one loaded Runner, clears KV cache
in place between questions, and keeps QNN-bound graph I/O addresses stable.

The generator checks these invariants before writing the report:

- full HTP resource direct cycles equal mapped QNN-node direct cycles;
- stage, Layer 0 and LM-head resource splits close to their owner direct cycles;
- quantization/conversion critical cycles do not exceed HVX critical cycles;
- mapped QNN direct cycles do not exceed the official dominant path.

## Common overrides

```bash
BENCHMARK_RUNS=5 MAX_NEW_TOKENS=128 \
./run_qwen3_sm8750_v79_profile.sh
```

```bash
BUILD_ANDROID=0 PREPARE_DEVICE=0 \
./run_qwen3_sm8750_v79_profile.sh
```

Other supported variables include `ADB_SERIAL`, `RESULTS_BASE`,
`BENCHMARK_COOLDOWN_SEC`, `PROMPT`, `LOCAL_MODEL`, `LOCAL_TOKENIZER`,
`LOCAL_CONFIG`, `SCHEMATIC_DIR` and `CLEAN_REMOTE`.

The default run retains the two raw Chrome Traces and QHAS reports for
re-analysis. Plan for roughly 1.8–2.0 GiB of host storage per run.

## Timing boundary

`prefill_e2e` starts after tokenization and ends after first-token
detokenization/callback. `decode_e2e_after_first` covers the subsequent token
loop including cache/mask maintenance, logits transfer, sampling and token
callback. Neither includes process startup or QNN context loading.

If measured runner CSVs are unavailable, the summarizer can fall back to
traced QHAS `graphExecute` latency. The HTML labels that value as derived
because it excludes CPU/runtime work and may contain profiling perturbation.
