#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_a16_vs_a8_projections/20260821}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
compiler="${COMPILER:-${repo_root}/build-qnn-aot/bin/mllm-qwen3-lpbq-a16-a8-projection-c}"
model="${artifact_root}/qwen3-lpbq-a16-a8-projections.mllm"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"

[[ "${qairt_root##*/}" == "2.47.0.260601" ]] || {
  echo "unexpected QAIRT release: ${qairt_root}" >&2
  exit 2
}
[[ -x "${compiler}" ]] || { echo "compiler missing: ${compiler}" >&2; exit 2; }
[[ -s "${model}" ]] || { echo "compact model missing: ${model}" >&2; exit 2; }
[[ -f "${qnn_lib}/libQnnHtp.so" ]] || { echo "QNN HTP library missing: ${qnn_lib}" >&2; exit 2; }

export LD_LIBRARY_PATH="${repo_root}/build-qnn-aot/bin:${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLLM_QNN_AOT_OPTRACE=1

for activation in a16 a8; do
  config="${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_lpbq_projection_${activation}.json"
  for projection in gate_proj up_proj down_proj lm_head; do
    for seq in 1 32; do
      tag="${activation}_${projection}_s${seq}"
      case_dir="${artifact_root}/contexts/${tag}"
      context="${case_dir}/${tag}.bin"
      manifest="${case_dir}/manifests/model.0.s${seq}_quant_manifest.json"
      schematic="${case_dir}/schematics/model.0.s${seq}_schematic.bin"
      if [[ -s "${context}" && -s "${manifest}" && -s "${schematic}" ]]; then
        echo "already complete: ${tag}"
        continue
      fi
      [[ ! -e "${case_dir}" ]] || {
        echo "refusing partial case directory: ${case_dir}" >&2
        exit 2
      }
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
          --activation "${activation}" \
          --projection "${projection}" \
          --seq "${seq}"
      ) >"${case_dir}/compile.log" 2>&1
      for output in "${context}" "${manifest}" "${schematic}"; do
        [[ -s "${output}" ]] || { echo "expected output missing: ${output}" >&2; exit 1; }
      done
      sha256sum "${context}" "${manifest}" "${schematic}" >"${case_dir}/artifacts.sha256"
      echo "completed: ${tag}"
    done
  done
done

echo "LPBQ A16/A8 projection contexts complete: ${artifact_root}/contexts"
