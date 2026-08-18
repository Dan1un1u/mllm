#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/w8a8_vs_lpbq_layer14_mlp/20260818}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
compiler="${COMPILER:-${repo_root}/build-qnn-aot/bin/mllm-qwen3-layer14-mlp-w8-lpbq-c}"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"

[[ "${qairt_root##*/}" == "2.47.0.260601" ]] || {
  echo "unexpected QAIRT release: ${qairt_root}" >&2
  exit 2
}
[[ -x "${compiler}" ]] || { echo "compiler missing: ${compiler}" >&2; exit 2; }
[[ -f "${qnn_lib}/libQnnHtp.so" ]] || { echo "QAIRT HTP library missing: ${qnn_lib}" >&2; exit 2; }

export LD_LIBRARY_PATH="${repo_root}/build-qnn-aot/bin:${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLLM_QNN_AOT_OPTRACE=1

for variant in lpbq w8a8; do
  if [[ "${variant}" == "lpbq" ]]; then
    model="${artifact_root}/qwen3-layer14-mlp-lpbq-w4a8.mllm"
    config="${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_layer14_mlp_lpbq.json"
  else
    model="${artifact_root}/qwen3-layer14-mlp-pertensor-w8a8.mllm"
    config="${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_layer14_mlp_w8a8.json"
  fi
  [[ -f "${model}" ]] || { echo "model missing: ${model}" >&2; exit 2; }

  for seq in 1 32; do
    case_dir="${artifact_root}/contexts/${variant}_s${seq}"
    context="${case_dir}/${variant}_s${seq}.bin"
    [[ ! -e "${case_dir}" ]] || { echo "refusing existing case directory: ${case_dir}" >&2; exit 2; }
    mkdir -p "${case_dir}/manifests" "${case_dir}/schematics"
    export MLLM_QNN_AOT_QUANT_MANIFEST_DIR="${case_dir}/manifests"
    export MLLM_QNN_AOT_OPTRACE_DIR="${case_dir}/schematics"

    (
      cd "${case_dir}"
      "${compiler}" \
        -m "${model}" \
        -aot_cfg "${config}" \
        -qnn_env "${qnn_lib}/" \
        -o "${context}" \
        --variant "${variant}" \
        --seq "${seq}"
    ) 2>&1 | tee "${case_dir}/compile.log"

    manifest="${case_dir}/manifests/model.0.s${seq}_quant_manifest.json"
    schematic="${case_dir}/schematics/model.0.s${seq}_schematic.bin"
    for output in "${context}" "${manifest}" "${schematic}"; do
      [[ -s "${output}" ]] || { echo "expected output missing: ${output}" >&2; exit 1; }
    done
    sha256sum "${context}" "${manifest}" "${schematic}" >"${case_dir}/artifacts.sha256"
  done
done

echo "Layer-14 LPBQ/W8A8 contexts complete: ${artifact_root}/contexts"
