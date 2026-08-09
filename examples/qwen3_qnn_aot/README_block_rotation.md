# Qwen3 Layer 5 offline R1/R2 prototype

This prototype evaluates whether fixed offline R1/R2 rotations change QNN HTP block latency. It is intentionally scoped to Qwen3 Layer 5 and does not make an accuracy claim.

## Experiment contract

- A: original Layer 5 weights.
- B: identity R1/R2 with RMSNorm gamma folded into downstream matrices.
- C: normalized Sylvester Hadamard R1=H2048 and shared per-head R2=H128.
- C accepts and returns the residual stream in the R1 basis. Its value-cache boundary uses R2 per KV head; the key cache remains unrotated.
- s1 decode uses one current token and 1023 past KV tokens.
- s32 prefill/chunk uses 32 current tokens and 992 past KV tokens.
- Existing activation and KV QDQ parameters are reused for all variants.
- A pass is C median latency <=3% above A. A 3-5% increase requires a rerun; >5% fails.

See [`../../CONTEXT.md`](../../CONTEXT.md) for the precise terminology.

## Export and mathematical verification

Run from the repository root in the Python environment containing PyTorch and safetensors:

```bash
python scripts/qwen3_block_rotation.py \
  --source-model /path/to/Qwen3-origin \
  --base-quant-checkpoint /path/to/Qwen3-1.7B-G32-base/model.safetensors \
  --output-dir /path/to/qwen3_layer5_rotation

python scripts/verify_qwen3_block_rotation.py \
  --source-model /path/to/Qwen3-origin \
  --seq-lens 1 32 \
  --output /path/to/qwen3_layer5_rotation/math_verification.json
```

The exporter applies every basis change to canonical float OI matrices, then performs fresh LPBQ G32 quantization and the existing OI-to-HWIO conversion. No runtime rotation node is emitted.

## Build products

The CMake targets added by this prototype are:

- `mllm-qwen3-block-aot-sha-g32-c`: x86 QAIRT context compiler.
- `mllm-qwen3-block-aot-runner`: Android block runner and timing harness.

The compiler traces `model.0.s1` and `model.0.s32` into one context. Use `qnn_aot_cfg_block_1.7B_g32.json` and the matching `config_1.7B_g32.json`.

For WSL builds behind Windows Clash Verge, source:

```bash
source scripts/wsl_clash_proxy.sh
```

The helper discovers the Windows host through the WSL default route. Its default Clash mixed proxy port is 7897 and can be overridden with `CLASH_PROXY_PORT`.

## Profiling-disabled timing

Run the Android binary from `/data/local/tmp`, because the HTP DSP loader resolves `./libQnnHtpV79Skel.so` from that working directory on the tested device:

```sh
cd /data/local/tmp
export LD_LIBRARY_PATH=/data/local/tmp/qwen3_layer5_rotation:/data/local/tmp
export MLLM_QNN_PROFILE_LEVEL=off

/data/local/tmp/qwen3_layer5_rotation/mllm-qwen3-block-aot-runner \
  -m /data/local/tmp/qwen3_layer5_rotation/A_original.bin \
  --graph both --warmup 20 --iterations 200 --variant A_original \
  -o /data/local/tmp/qwen3_layer5_rotation/A_timing.json
```

Run A/B/C in one order and then in reverse order. Use the second, longer run as the canonical result after checking thermal and battery state. The timing boundary is host wall time immediately around `QnnGraph_execute`; context loading, allocation, input fill, and warmup are excluded.

Generate the latency and MIR graph-contract report with:

```bash
python scripts/qwen3_block_rotation_report.py \
  --timing-root /path/to/results/by_variant \
  --manifest-root /path/to/qwen3_layer5_rotation \
  --output-json /path/to/results/report.json \
  --output-md /path/to/results/report.md
```

## Internal HTP Optrace

Generate the contexts once with the following compiler environment. The resulting context bytes are unchanged; the extra host artifacts are graph-specific schematic and quantization manifest files.

```bash
export MLLM_QNN_AOT_OPTRACE=1
export MLLM_QNN_AOT_OPTRACE_DIR=optrace_artifacts
export MLLM_QNN_AOT_QUANT_MANIFEST_DIR=manifests
```

Capture each variant/workload in a fresh process. HTP Optrace must attach to the graph's first execution:

```sh
export MLLM_QNN_PROFILE_LEVEL=optrace
export MLLM_QNN_PROFILE_WARMUP=0
export MLLM_QNN_PROFILE_EVERY=1
export MLLM_QNN_PROFILE_MAX_CAPTURES=1
export MLLM_QNN_PROFILE_GRAPH=model.0.s1
export MLLM_QNN_PROFILE_SERIALIZE=1
export MLLM_QNN_PROFILE_DIR=/data/local/tmp/qwen3_layer5_rotation/optrace/A/s1

mllm-qwen3-block-aot-runner -m A_original.bin --graph s1 \
  --warmup 0 --iterations 1 --variant A_original -o "$MLLM_QNN_PROFILE_DIR/timing.json"
```

Decode `qnn-profiling-data.log` with QAIRT `qnn-profile-viewer`, `libQnnHtpOptraceProfilingReader.so`, and the exact matching `model.0.s1_schematic.bin` or `model.0.s32_schematic.bin`. Then run:

```bash
python scripts/qnn_optrace_qwen3_structure.py chrometrace.json \
  --htp-json chrometrace_htp.json \
  --qhas-json chrometrace_qnn_htp_analysis_summary.json \
  --output-prefix structure

python scripts/qwen3_block_optrace_report.py \
  --optrace-root /path/to/results/optrace \
  --timing-report /path/to/results/report.json \
  --output-prefix /path/to/results/optrace_comparison
```

Optrace serialization substantially inflates runner host wall time. Do not use that wall time for acceptance. The report uses QHAS graph execution and stage attribution only as diagnostics and keeps the profiling-disabled repeated median as the speed decision.
