#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/compact_prefill_attention_qairt249/20260823_boolmask}"
source_model="${SOURCE_MODEL:-${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/20260813_211529/qwen3-1.7B-w4a8g32-rmsnorm-u8.mllm}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.49.0.260730}"
compiler="${COMPILER:-${repo_root}/build-qnn-aot-qairt249/bin/mllm-qwen3-compact-prefill-attention-c}"
config="${AOT_CONFIG:-${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32.json}"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"

[[ "${qairt_root##*/}" == "2.49.0.260730" ]] || {
  echo "unexpected QAIRT release: ${qairt_root}" >&2
  exit 2
}
[[ -x "${compiler}" ]] || { echo "compiler missing: ${compiler}" >&2; exit 2; }
[[ -s "${source_model}" ]] || { echo "source model missing: ${source_model}" >&2; exit 2; }
[[ -f "${qnn_lib}/libQnnHtp.so" ]] || { echo "QNN HTP library missing: ${qnn_lib}" >&2; exit 2; }

mkdir -p "${repo_root}/tmp" "${artifact_root}/contexts"
index_tmp="${artifact_root}/contexts/index.tsv.tmp.$$"
printf 'variant\twidth\tcontext\tmanifest\tschematic\n' >"${index_tmp}"

for variant in full compact; do
  case "${variant}" in
    full) width=1024 ;;
    compact) width=32 ;;
  esac
  tag="${variant}_w${width}_s32_p19"
  case_dir="${artifact_root}/contexts/${tag}"
  context="${case_dir}/${tag}.bin"
  if [[ -s "${context}" && -s "${case_dir}/manifest.json" && -s "${case_dir}/schematic.bin" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "${variant}" "${width}" "${context}" \
      "${case_dir}/manifest.json" "${case_dir}/schematic.bin" >>"${index_tmp}"
    continue
  fi
  [[ ! -e "${case_dir}" ]] || { echo "refusing partial case: ${case_dir}" >&2; exit 2; }

  stage="$(mktemp -d "${repo_root}/tmp/compact_prefill_attention_compile.XXXXXX")"
  trap 'rm -rf "${stage}"' ERR INT TERM
  mkdir -p "${stage}/manifests" "${stage}/schematics"
  trap - ERR
  set +e
  (
    cd "${stage}"
    env \
      QAIRT_SDK_ROOT="${qairt_root}" \
      LD_LIBRARY_PATH="$(dirname "${compiler}"):${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      MLLM_QNN_AOT_OPTRACE=1 \
      MLLM_QNN_AOT_FINALIZE_P=19 \
      MLLM_QNN_AOT_QUANT_MANIFEST_DIR="${stage}/manifests" \
      MLLM_QNN_AOT_OPTRACE_DIR="${stage}/schematics" \
      "${compiler}" \
        -m "${source_model}" \
        -aot_cfg "${config}" \
        -qnn_env "${qnn_lib}/" \
        -o "${stage}/${tag}.bin" \
        --width "${width}"
  ) >"${stage}/compile.log" 2>&1
  status=$?
  set -e
  trap 'rm -rf "${stage}"' ERR INT TERM
  if (( status != 0 )); then
    rejected="${artifact_root}/rejected/${tag}"
    mkdir -p "${rejected}"
    cp -f "${stage}/compile.log" "${rejected}/"
    printf '%s\n' "${status}" >"${rejected}/exit_code.txt"
    rm -rf "${stage}"
    trap - ERR INT TERM
    echo "compile rejected: ${tag}" >&2
    exit "${status}"
  fi

  grep -q 'P = 19' "${stage}/compile.log" || {
    echo "forced P19 evidence missing: ${tag}" >&2
    exit 1
  }
  manifest="$(find "${stage}/manifests" -maxdepth 1 -type f -name '*_quant_manifest.json' -print -quit)"
  schematic="$(find "${stage}/schematics" -maxdepth 1 -type f -name '*_schematic.bin' -print -quit)"
  for output in "${stage}/${tag}.bin" "${manifest}" "${schematic}"; do
    [[ -s "${output}" ]] || { echo "expected output missing: ${output}" >&2; exit 1; }
  done
  cp -f "${manifest}" "${stage}/manifest.json"
  cp -f "${schematic}" "${stage}/schematic.bin"
  sha256sum "${stage}/${tag}.bin" "${stage}/manifest.json" \
    "${stage}/schematic.bin" >"${stage}/artifacts.sha256"
  publish="${case_dir}.tmp.$$"
  mkdir -p "${publish}"
  cp -a "${stage}/." "${publish}/"
  mv "${publish}" "${case_dir}"
  rm -rf "${stage}"
  trap - ERR INT TERM
  printf '%s\t%s\t%s\t%s\t%s\n' "${variant}" "${width}" "${context}" \
    "${case_dir}/manifest.json" "${case_dir}/schematic.bin" >>"${index_tmp}"
  echo "PASS ${tag}"
done

mv "${index_tmp}" "${artifact_root}/contexts/index.tsv"
echo "Compact-prefill attention contexts complete: ${artifact_root}/contexts"
