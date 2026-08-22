#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/kv_head_packing/20260822_split8}"
source_model="${SOURCE_MODEL:-${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/20260813_211529/qwen3-1.7B-w4a8g32-rmsnorm-u8.mllm}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
compiler="${COMPILER:-${repo_root}/build-qnn-aot/bin/mllm-qwen3-kv-head-packing-c}"
python="${PYTHON:-/home/daniuniu/mllm-quant-venv/bin/python3}"
config="${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_lpbq_projection_a8.json"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"
model="${artifact_root}/qwen3-kv-head-packing.mllm"

[[ "${qairt_root##*/}" == "2.47.0.260601" ]] || {
  echo "unexpected QAIRT release: ${qairt_root}" >&2
  exit 2
}
[[ -x "${compiler}" ]] || { echo "compiler missing: ${compiler}" >&2; exit 2; }
[[ -x "${python}" ]] || { echo "Python environment missing: ${python}" >&2; exit 2; }
[[ -s "${source_model}" ]] || { echo "source model missing: ${source_model}" >&2; exit 2; }
[[ -f "${qnn_lib}/libQnnHtp.so" ]] || { echo "QNN HTP library missing: ${qnn_lib}" >&2; exit 2; }

mkdir -p "${artifact_root}"
if [[ ! -s "${model}" || ! -s "${artifact_root}/artifact_report.json" ]]; then
  [[ ! -e "${model}" && ! -e "${artifact_root}/artifact_report.json" ]] || {
    echo "refusing partial compact artifact in ${artifact_root}" >&2
    exit 2
  }
  artifact_stage="$(mktemp -d "${repo_root}/tmp/kv_head_packing_artifact.XXXXXX")"
  trap 'rm -rf "${artifact_stage}"' ERR INT TERM
  "${python}" "${repo_root}/scripts/qnn_kv_head_packing_artifact.py" \
    "${source_model}" "${artifact_stage}" \
    --report "${artifact_stage}/artifact_report.json" \
    >"${artifact_stage}/artifact_generation.log"
  cp -f "${artifact_stage}/qwen3-kv-head-packing.mllm" \
    "${artifact_stage}/artifact_report.json" \
    "${artifact_stage}/artifact_generation.log" \
    "${artifact_stage}"/input_*_a8.raw "${artifact_root}/"
  rm -rf "${artifact_stage}"
  trap - ERR INT TERM
fi

export LD_LIBRARY_PATH="${repo_root}/build-qnn-aot/bin:${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLLM_QNN_AOT_OPTRACE=1
export MLLM_QNN_AOT_FINALIZE_P=19

for projection in k_proj v_proj; do
  for variant in per_head packed; do
    for seq in 1 32; do
      tag="a8_${projection}_${variant}_split8_s${seq}_p19"
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
      stage="$(mktemp -d "${repo_root}/tmp/kv_head_packing_compile.XXXXXX")"
      trap 'rm -rf "${stage}"' ERR INT TERM
      mkdir -p "${stage}/manifests" "${stage}/schematics"
      export MLLM_QNN_AOT_QUANT_MANIFEST_DIR="${stage}/manifests"
      export MLLM_QNN_AOT_OPTRACE_DIR="${stage}/schematics"
      (
        cd "${stage}"
        "${compiler}" \
          -m "${model}" \
          -aot_cfg "${config}" \
          -qnn_env "${qnn_lib}/" \
          -o "${stage}/${tag}.bin" \
          --projection "${projection}" \
          --variant "${variant}" \
          --seq "${seq}"
      ) >"${stage}/compile.log" 2>&1
      grep -q "Prepare: Graph model.0.s${seq} with init graph option: P = 19" \
        "${stage}/compile.log" || {
          echo "P19 compiler evidence missing: ${tag}" >&2
          exit 1
        }
      for output in \
        "${stage}/${tag}.bin" \
        "${stage}/manifests/model.0.s${seq}_quant_manifest.json" \
        "${stage}/schematics/model.0.s${seq}_schematic.bin"; do
        [[ -s "${output}" ]] || { echo "expected output missing: ${output}" >&2; exit 1; }
      done
      sha256sum \
        "${stage}/${tag}.bin" \
        "${stage}/manifests/model.0.s${seq}_quant_manifest.json" \
        "${stage}/schematics/model.0.s${seq}_schematic.bin" \
        >"${stage}/artifacts.sha256"
      publish="${case_dir}.tmp.$$"
      [[ ! -e "${publish}" ]] || { echo "publish stage exists: ${publish}" >&2; exit 2; }
      mkdir -p "${publish}"
      cp -a "${stage}/." "${publish}/"
      mkdir -p "$(dirname "${case_dir}")"
      mv "${publish}" "${case_dir}"
      rm -rf "${stage}"
      trap - ERR INT TERM
      echo "completed: ${tag}"
    done
  done
done

echo "K/V head-packing contexts complete: ${artifact_root}/contexts"
