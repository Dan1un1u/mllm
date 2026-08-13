#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results}"
work_root="${W4A8_WORK_ROOT:-${models_root}/w4a8g32}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
compiler="${W4A8_AOT_COMPILER:-${repo_root}/build-qnn-aot/bin/mllm-qwen3-aot-sha-g32-c}"
run_id="${1:?usage: build_qwen3_w4a8g32_context.sh RUN_ID [MODEL.mllm]}"
[[ "${run_id}" != "-h" && "${run_id}" != "--help" ]] || {
  echo "usage: $0 RUN_ID [MODEL.mllm]"
  exit 0
}
stage_dir="${W4A8_STAGE_DIR:-${work_root}/staging/${run_id}}"
model="${2:-${stage_dir}/qwen3_1.7b_w4a8g32.mllm}"
artifact_dir="${stage_dir}/qnn"
manifest_dir="${artifact_dir}/manifests"
optrace_dir="${artifact_dir}/schematics"
context="${artifact_dir}/qwen3-1.7B-w4a8g32-sha.bin"
log_dir="${W4A8_LOG_ROOT:-${results_root}/w4a8g32_build_logs}/${run_id}"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"

[[ -f "${model}" ]] || { echo "model missing: ${model}" >&2; exit 2; }
[[ -x "${compiler}" ]] || { echo "AOT compiler missing: ${compiler}" >&2; exit 2; }
[[ -f "${qnn_lib}/libQnnHtp.so" ]] || { echo "QAIRT HTP library missing: ${qnn_lib}" >&2; exit 2; }
[[ "${qairt_root##*/}" == 2.47.0.260601 ]] || { echo "unexpected QAIRT release: ${qairt_root}" >&2; exit 2; }
[[ ! -e "${context}" ]] || { echo "context already exists: ${context}" >&2; exit 2; }

mkdir -p "${artifact_dir}" "${manifest_dir}" "${optrace_dir}" "${log_dir}"
export LD_LIBRARY_PATH="${repo_root}/build-qnn-aot/bin:${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLLM_QNN_AOT_OPTRACE=1
export MLLM_QNN_AOT_QUANT_MANIFEST_DIR="${manifest_dir}"
export MLLM_QNN_AOT_OPTRACE_DIR="${optrace_dir}"

(
  cd "${artifact_dir}"
  "${compiler}" \
    -m "${model}" \
    -c "${repo_root}/examples/qwen3_qnn_aot/config_1.7B_g32.json" \
    -aot_cfg "${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32.json" \
    -qnn_env "${qnn_lib}/" \
    -o "${context}"
) 2>&1 | tee "${log_dir}/aot-context.log"

for graph in s1 s32; do
  [[ -s "${manifest_dir}/model.0.${graph}_quant_manifest.json" ]] \
    || { echo "manifest missing for ${graph}" >&2; exit 1; }
  [[ -s "${optrace_dir}/model.0.${graph}_schematic.bin" ]] \
    || { echo "schematic missing for ${graph}" >&2; exit 1; }
done
[[ -s "${context}" ]] || { echo "context was not generated" >&2; exit 1; }

sha256sum \
  "${context}" \
  "${manifest_dir}/model.0.s1_quant_manifest.json" \
  "${manifest_dir}/model.0.s32_quant_manifest.json" \
  "${optrace_dir}/model.0.s1_schematic.bin" \
  "${optrace_dir}/model.0.s32_schematic.bin" \
  > "${log_dir}/qnn-artifacts.sha256"

echo "W4A8G32 QNN context staging complete: ${context}"
