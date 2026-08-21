#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
source_root="${SOURCE_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_a16_vs_a8_projections/20260821}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_p_point_search/20260821}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
compiler="${COMPILER:-${repo_root}/build-qnn-aot/bin/mllm-qwen3-lpbq-a16-a8-projection-c}"
model="${source_root}/qwen3-lpbq-a16-a8-projections.mllm"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"
read -r -a points <<<"${POINTS:-0 1 2 3 4 5 6 8 13 15 16 17 19 20 21 22 23}"
read -r -a projections <<<"${PROJECTIONS:-gate_proj up_proj}"

[[ "${qairt_root##*/}" == "2.47.0.260601" ]] || {
  echo "unexpected QAIRT release: ${qairt_root}" >&2
  exit 2
}
[[ -x "${compiler}" ]] || { echo "compiler missing: ${compiler}" >&2; exit 2; }
[[ -s "${model}" ]] || { echo "compact model missing: ${model}" >&2; exit 2; }
[[ -f "${qnn_lib}/libQnnHtp.so" ]] || { echo "QNN HTP library missing: ${qnn_lib}" >&2; exit 2; }
case "${artifact_root}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/lpbq_p_point_search/*) ;;
  *) echo "refusing unexpected ARTIFACT_ROOT: ${artifact_root}" >&2; exit 2 ;;
esac

mkdir -p "${artifact_root}/contexts" "${artifact_root}/failures"
export LD_LIBRARY_PATH="${repo_root}/build-qnn-aot/bin:${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLLM_QNN_AOT_OPTRACE=1

for projection in "${projections[@]}"; do
  case "${projection}" in
    gate_proj|up_proj|lm_head) ;;
    *) echo "unsupported projection for P-point search: ${projection}" >&2; exit 2 ;;
  esac
  for point in "${points[@]}"; do
    tag="p${point}_${projection}_s1"
    case_dir="${artifact_root}/contexts/${tag}"
    context="${case_dir}/${tag}.bin"
    manifest="${case_dir}/manifests/model.0.s1_quant_manifest.json"
    schematic="${case_dir}/schematics/model.0.s1_schematic.bin"
    if [[ -s "${context}" && -s "${manifest}" && -s "${schematic}" && -s "${case_dir}/compile.log" ]]; then
      echo "already complete: ${tag}"
      continue
    fi
    [[ ! -e "${case_dir}" ]] || {
      echo "refusing partial case directory: ${case_dir}" >&2
      exit 2
    }

    staging="${case_dir}.tmp.$$"
    case "${staging}" in
      "${artifact_root}/contexts/"*.tmp.*) ;;
      *) echo "refusing unexpected staging directory: ${staging}" >&2; exit 2 ;;
    esac
    mkdir -p "${staging}/manifests" "${staging}/schematics"
    export MLLM_QNN_AOT_FINALIZE_P="${point}"
    export MLLM_QNN_AOT_QUANT_MANIFEST_DIR="${staging}/manifests"
    export MLLM_QNN_AOT_OPTRACE_DIR="${staging}/schematics"
    if (
      cd "${staging}"
      "${compiler}" \
        -m "${model}" \
        -aot_cfg "${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_lpbq_projection_a8.json" \
        -qnn_env "${qnn_lib}/" \
        -o "${tag}.bin" \
        --activation a8 \
        --projection "${projection}" \
        --seq 1
    ) >"${staging}/compile.log" 2>&1; then
      for output in "${staging}/${tag}.bin" \
                    "${staging}/manifests/model.0.s1_quant_manifest.json" \
                    "${staging}/schematics/model.0.s1_schematic.bin"; do
        [[ -s "${output}" ]] || { echo "expected output missing: ${output}" >&2; exit 1; }
      done
      sha256sum "${staging}/${tag}.bin" \
                "${staging}/manifests/model.0.s1_quant_manifest.json" \
                "${staging}/schematics/model.0.s1_schematic.bin" \
        >"${staging}/artifacts.sha256"
      mv "${staging}" "${case_dir}"
      echo "completed: ${tag}"
    else
      mv "${staging}/compile.log" "${artifact_root}/failures/${tag}.log"
      rm -rf "${staging}"
      echo "prepare failed: ${tag}" >&2
    fi
  done
done

unset MLLM_QNN_AOT_FINALIZE_P
echo "LPBQ P-point contexts complete: ${artifact_root}/contexts"
