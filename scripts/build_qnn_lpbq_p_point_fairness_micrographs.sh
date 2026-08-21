#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
source_root="${SOURCE_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_a16_vs_a8_projections/20260821}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_p_point_fairness/20260822}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
compiler="${COMPILER:-${repo_root}/build-qnn-aot/bin/mllm-qwen3-lpbq-a16-a8-projection-c}"
model="${source_root}/qwen3-lpbq-a16-a8-projections.mllm"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"
read -r -a points <<<"${POINTS:-0 1 2 3 4 5 6 8 13 15 16 17 19 20 21 22 23}"
read -r -a activations <<<"${ACTIVATIONS:-a8 a16}"
read -r -a sequences <<<"${SEQUENCES:-1 32}"
read -r -a projections <<<"${PROJECTIONS:-gate_proj up_proj down_proj lm_head}"

[[ "${qairt_root##*/}" == "2.47.0.260601" ]] || {
  echo "unexpected QAIRT release: ${qairt_root}" >&2
  exit 2
}
[[ -x "${compiler}" ]] || { echo "compiler missing: ${compiler}" >&2; exit 2; }
[[ -s "${model}" ]] || { echo "compact model missing: ${model}" >&2; exit 2; }
[[ -f "${qnn_lib}/libQnnHtp.so" ]] || { echo "QNN HTP library missing: ${qnn_lib}" >&2; exit 2; }
case "${artifact_root}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/lpbq_p_point_fairness/*) ;;
  *) echo "refusing unexpected ARTIFACT_ROOT: ${artifact_root}" >&2; exit 2 ;;
esac

mkdir -p "${artifact_root}/contexts" "${artifact_root}/failures"
export LD_LIBRARY_PATH="${repo_root}/build-qnn-aot/bin:${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset MLLM_QNN_AOT_OPTRACE MLLM_QNN_AOT_QUANT_MANIFEST_DIR MLLM_QNN_AOT_OPTRACE_DIR

for activation in "${activations[@]}"; do
  case "${activation}" in a8|a16) ;; *) echo "unsupported activation: ${activation}" >&2; exit 2 ;; esac
  aot_cfg="${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_lpbq_projection_${activation}.json"
  [[ -s "${aot_cfg}" ]] || { echo "AOT config missing: ${aot_cfg}" >&2; exit 2; }
  for seq in "${sequences[@]}"; do
    case "${seq}" in 1|32) ;; *) echo "unsupported sequence: ${seq}" >&2; exit 2 ;; esac
    graph="s${seq}"
    for projection in "${projections[@]}"; do
      case "${projection}" in gate_proj|up_proj|down_proj|lm_head) ;;
        *) echo "unsupported projection: ${projection}" >&2; exit 2 ;;
      esac
      for point in "${points[@]}"; do
        tag="${activation}_${graph}_${projection}_p${point}"
        case_dir="${artifact_root}/contexts/${tag}"
        context="${case_dir}/context.bin"
        if [[ -s "${context}" && -s "${case_dir}/compile.log" && -s "${case_dir}/artifacts.sha256" ]]; then
          echo "already complete: ${tag}"
          continue
        fi
        [[ ! -e "${case_dir}" ]] || {
          echo "refusing partial case directory: ${case_dir}" >&2
          exit 2
        }

        staging="${case_dir}.tmp.$$"
        case "${staging}" in "${artifact_root}/contexts/"*.tmp.*) ;;
          *) echo "refusing unexpected staging directory: ${staging}" >&2; exit 2 ;;
        esac
        mkdir -p "${staging}"
        export MLLM_QNN_AOT_FINALIZE_P="${point}"
        if (
          cd "${staging}"
          "${compiler}" \
            -m "${model}" \
            -aot_cfg "${aot_cfg}" \
            -qnn_env "${qnn_lib}/" \
            -o context.bin \
            --activation "${activation}" \
            --projection "${projection}" \
            --seq "${seq}"
        ) >"${staging}/compile.log" 2>&1; then
          [[ -s "${staging}/context.bin" ]] || { echo "context missing: ${tag}" >&2; exit 1; }
          grep -q "Graph model.0.${graph} with init graph option: P = ${point}" \
            "${staging}/compile.log" || {
              echo "compile log does not prove P${point}: ${tag}" >&2
              exit 1
            }
          {
            printf 'activation=%s\nsequence=%s\ngraph=%s\nprojection=%s\nfinalize_O=3\nfinalize_P=%s\n' \
              "${activation}" "${seq}" "${graph}" "${projection}" "${point}"
            printf 'source_model_sha256=%s\n' "$(sha256sum "${model}" | awk '{print $1}')"
            printf 'git_commit=%s\n' "$(git -C "${repo_root}" rev-parse HEAD)"
          } >"${staging}/provenance.env"
          sha256sum "${staging}/context.bin" "${staging}/provenance.env" >"${staging}/artifacts.sha256"
          mv "${staging}" "${case_dir}"
          echo "completed: ${tag}"
        else
          mv "${staging}/compile.log" "${artifact_root}/failures/${tag}.log"
          rm -rf "${staging}"
          echo "prepare failed: ${tag}" >&2
        fi
      done
    done
  done
done

unset MLLM_QNN_AOT_FINALIZE_P
echo "LPBQ P-point fairness micrographs complete: ${artifact_root}/contexts"
