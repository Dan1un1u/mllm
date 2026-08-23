#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/kv_head_packing_qairt249/20260824}"
source_root="${artifact_root}/source"
source_model="${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/20260813_211529/qwen3-1.7B-w4a8g32-rmsnorm-u8.mllm"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.49.0.260730}"
compiler="${COMPILER:-${repo_root}/build-qnn-aot-qairt249/bin/mllm-qwen3-kv-head-packing-c}"
python_bin="${PYTHON_BIN:-/home/daniuniu/mllm-quant-venv/bin/python}"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"
projections=(k_proj v_proj)
modes=(split packed)
sequences=(32 64)

case "${artifact_root}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/kv_head_packing_qairt249/20260824) ;;
  *) echo "refusing unexpected ARTIFACT_ROOT: ${artifact_root}" >&2; exit 2 ;;
esac
[[ "${qairt_root##*/}" == "2.49.0.260730" ]] || {
  echo "unexpected QAIRT release" >&2
  exit 2
}
[[ -x "${compiler}" && -x "${python_bin}" && -s "${source_model}" ]] || {
  echo "compiler, Python, or accepted source model missing" >&2
  exit 2
}
[[ -f "${qnn_lib}/libQnnHtp.so" ]] || { echo "QNN HTP library missing" >&2; exit 2; }

mkdir -p "${artifact_root}/contexts" /home/daniuniu/llm_exp_work
if [[ ! -s "${source_root}/qwen3-layer14-kv-head-packing-w4g32-a8.mllm" ]]; then
  [[ ! -e "${source_root}" ]] || {
    echo "refusing partial source artifact: ${source_root}" >&2
    exit 2
  }
  source_stage="$(mktemp -d /home/daniuniu/llm_exp_work/kv_head_packing_source.XXXXXX)"
  source_publish="${source_root}.tmp.$$"
  case "${source_stage}" in /home/daniuniu/llm_exp_work/kv_head_packing_source.*) ;;
    *) echo "unexpected source staging: ${source_stage}" >&2; exit 2 ;;
  esac
  case "${source_publish}" in "${artifact_root}/source.tmp."*) ;;
    *) echo "unexpected source publish staging: ${source_publish}" >&2; exit 2 ;;
  esac
  trap 'rm -rf "${source_stage}" "${source_publish}"' ERR INT TERM
  "${python_bin}" "${repo_root}/scripts/qnn_kv_head_packing_artifact.py" \
    --source "${source_model}" --output-dir "${source_stage}/source" \
    >"${source_stage}/generator.stdout"
  (
    cd "${source_stage}/source"
    sha256sum ./* >artifacts.sha256
  )
  mkdir -p "${source_publish}"
  cp -a "${source_stage}/source/." "${source_publish}/"
  cp -f "${source_stage}/generator.stdout" "${source_publish}/"
  mv "${source_publish}" "${source_root}"
  rm -rf "${source_stage}"
  trap - ERR INT TERM
fi

compact_model="${source_root}/qwen3-layer14-kv-head-packing-w4g32-a8.mllm"
[[ -s "${compact_model}" && -s "${source_root}/artifact.json" ]] || {
  echo "compact source artifact is incomplete" >&2
  exit 2
}

for projection in "${projections[@]}"; do
  for mode in "${modes[@]}"; do
    for sequence in "${sequences[@]}"; do
      tag="${projection}_${mode}_s${sequence}_p19"
      case_dir="${artifact_root}/contexts/${tag}"
      manifest="${case_dir}/manifests/model.0.s${sequence}_quant_manifest.json"
      schematic="${case_dir}/schematics/model.0.s${sequence}_schematic.bin"
      if [[ -s "${case_dir}/context.bin" && -s "${manifest}" && -s "${schematic}" \
            && -s "${case_dir}/audit.json" ]]; then
        echo "already complete: ${tag}"
        continue
      fi
      [[ ! -e "${case_dir}" ]] || { echo "refusing partial case: ${case_dir}" >&2; exit 2; }

      stage="$(mktemp -d "/home/daniuniu/llm_exp_work/kv_head_packing_${tag}.XXXXXX")"
      publish="${case_dir}.tmp.$$"
      case "${stage}" in /home/daniuniu/llm_exp_work/kv_head_packing_${tag}.*) ;;
        *) echo "unexpected case staging: ${stage}" >&2; exit 2 ;;
      esac
      case "${publish}" in "${artifact_root}/contexts/${tag}.tmp."*) ;;
        *) echo "unexpected case publish staging: ${publish}" >&2; exit 2 ;;
      esac
      mkdir -p "${stage}/manifests" "${stage}/schematics"
      (
        cd "${stage}"
        env QAIRT_SDK_ROOT="${qairt_root}" \
          LD_LIBRARY_PATH="$(dirname "${compiler}"):${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
          MLLM_QNN_AOT_FINALIZE_P=19 \
          MLLM_QNN_AOT_OPTRACE=1 \
          MLLM_QNN_AOT_QUANT_MANIFEST_DIR="${stage}/manifests" \
          MLLM_QNN_AOT_OPTRACE_DIR="${stage}/schematics" \
          "${compiler}" -m "${compact_model}" \
            -aot_cfg "${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_lpbq_projection_a8.json" \
            -qnn_env "${qnn_lib}/" -o context.bin \
            --projection "${projection}" --mode "${mode}" --seq "${sequence}"
      ) >"${stage}/compile.log" 2>&1

      manifest="${stage}/manifests/model.0.s${sequence}_quant_manifest.json"
      schematic="${stage}/schematics/model.0.s${sequence}_schematic.bin"
      for output in "${stage}/context.bin" "${manifest}" "${schematic}"; do
        [[ -s "${output}" ]] || { echo "missing output for ${tag}: ${output}" >&2; exit 1; }
      done
      grep -q "Graph model.0.s${sequence} with init graph option: P = 19" \
        "${stage}/compile.log" || { echo "compile log does not prove P19: ${tag}" >&2; exit 1; }
      python3 "${repo_root}/scripts/qnn_kv_head_packing_audit.py" \
        --manifest "${manifest}" --projection "${projection}" --mode "${mode}" \
        --seq "${sequence}" --report "${stage}/audit.json" >"${stage}/audit.stdout"
      {
        printf 'projection=%s\nmode=%s\nsequence=%s\nfinalize_p=19\n' \
          "${projection}" "${mode}" "${sequence}"
        printf 'qairt_release=2.49.0.260730\nsource_artifact_sha256=%s\n' \
          "$(sha256sum "${compact_model}" | awk '{print $1}')"
        printf 'git_commit=%s\ngit_branch=%s\n' \
          "$(git -C "${repo_root}" rev-parse HEAD)" "$(git -C "${repo_root}" branch --show-current)"
      } >"${stage}/provenance.txt"
      (
        cd "${stage}"
        sha256sum context.bin manifests/* schematics/* audit.json provenance.txt >artifacts.sha256
      )
      mkdir -p "${publish}"
      cp -a "${stage}/." "${publish}/"
      mv "${publish}" "${case_dir}"
      rm -rf "${stage}"
      echo "completed: ${tag}"
    done
  done
done

inventory="${artifact_root}/inventory.tsv.tmp.$$"
printf 'projection\tmode\tsequence\tfinalize_p\tcontext\tmanifest\tschematic\n' >"${inventory}"
for projection in "${projections[@]}"; do
  for mode in "${modes[@]}"; do
    for sequence in "${sequences[@]}"; do
      tag="${projection}_${mode}_s${sequence}_p19"
      case_dir="${artifact_root}/contexts/${tag}"
      [[ -s "${case_dir}/context.bin" ]] || { echo "required context missing: ${tag}" >&2; exit 1; }
      printf '%s\t%s\t%s\t19\t%s\t%s\t%s\n' \
        "${projection}" "${mode}" "${sequence}" "${case_dir}/context.bin" \
        "${case_dir}/manifests/model.0.s${sequence}_quant_manifest.json" \
        "${case_dir}/schematics/model.0.s${sequence}_schematic.bin" >>"${inventory}"
    done
  done
done
mv "${inventory}" "${artifact_root}/inventory.tsv"
echo "K/V head-packing contexts complete: ${artifact_root}"
