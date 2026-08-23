#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
variant="${1:?usage: build_qnn_ar64_fairness_context.sh a8_p19|a16_default}"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/ar64_fairness_qairt249/20260824}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.49.0.260730}"
compiler="${COMPILER:-${repo_root}/build-qnn-aot-qairt249/bin/mllm-qwen3-aot-sha-g32-c}"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"

case "${variant}" in
  a8_p19)
    source_model="${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/20260813_211529/qwen3-1.7B-w4a8g32-rmsnorm-u8.mllm"
    model_config="${repo_root}/examples/qwen3_qnn_aot/config_1.7B_g32.json"
    aot_config="${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32.json"
    finalize_p=19
    ;;
  a16_default)
    source_model="${models_root}/qwen3_sm8750_v79/g32/w4a16/source_g32_export/qwen3_1.7b_g32.mllm"
    model_config="${repo_root}/examples/qwen3_qnn_aot/config_1.7B_g32_a16.json"
    aot_config="${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32_a16.json"
    compiler="${A16_COMPILER:-${repo_root}/build-qnn-aot-qairt249/bin/mllm-qwen3-aot-sha-a16-g32-c}"
    finalize_p=default
    ;;
  *) echo "unsupported variant: ${variant}" >&2; exit 2 ;;
esac

case "${artifact_root}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/ar64_fairness_qairt249/20260824) ;;
  *) echo "refusing unexpected ARTIFACT_ROOT: ${artifact_root}" >&2; exit 2 ;;
esac
publish_root="${artifact_root}/${variant}"
[[ "${qairt_root##*/}" == "2.49.0.260730" ]] || { echo "unexpected QAIRT release" >&2; exit 2; }
[[ -x "${compiler}" && -s "${source_model}" ]] || { echo "compiler/model missing" >&2; exit 2; }
[[ ! -e "${publish_root}" ]] || { echo "refusing existing publish root: ${publish_root}" >&2; exit 2; }

mkdir -p /home/daniuniu/llm_exp_work "${artifact_root}"
stage="$(mktemp -d "/home/daniuniu/llm_exp_work/ar64_fairness_${variant}.XXXXXX")"
case "${stage}" in
  /home/daniuniu/llm_exp_work/ar64_fairness_${variant}.*) ;;
  *) echo "unexpected staging: ${stage}" >&2; exit 2 ;;
esac
trap 'rm -rf "${stage}"' ERR INT TERM
mkdir -p "${stage}/manifests" "${stage}/schematics"
context="${stage}/qwen3-1.7B-${variant}-ar64-qairt249.bin"

compile_env=(
  QAIRT_SDK_ROOT="${qairt_root}"
  LD_LIBRARY_PATH="$(dirname "${compiler}"):${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  MLLM_QNN_AOT_OPTRACE=1
  MLLM_QNN_AOT_QUANT_MANIFEST_DIR="${stage}/manifests"
  MLLM_QNN_AOT_OPTRACE_DIR="${stage}/schematics"
)
if [[ "${finalize_p}" != default ]]; then
  compile_env+=(MLLM_QNN_AOT_FINALIZE_P="${finalize_p}")
fi

trap - ERR
set +e
(
  cd "${stage}"
  env -u MLLM_QNN_AOT_FINALIZE_P -u MLLM_QNN_AOT_FINALIZE_P_S1 \
    -u MLLM_QNN_AOT_FINALIZE_P_S32 "${compile_env[@]}" \
    "${compiler}" \
      -m "${source_model}" \
      -c "${model_config}" \
      -aot_cfg "${aot_config}" \
      -qnn_env "${qnn_lib}/" \
      -o "${context}" \
      --prefill_seq 64
) >"${stage}/compile.log" 2>&1
status=$?
set -e
if (( status != 0 )); then
  rejected="${artifact_root}/rejected/${variant}"
  mkdir -p "${rejected}"
  cp -f "${stage}/compile.log" "${rejected}/"
  printf '%s\n' "${status}" >"${rejected}/exit_code.txt"
  rm -rf "${stage}"
  trap - ERR INT TERM
  echo "compile rejected: ${variant}" >&2
  exit "${status}"
fi

for graph in s1 s64; do
  [[ -s "${stage}/manifests/model.0.${graph}_quant_manifest.json" ]] || { echo "missing ${graph} manifest" >&2; exit 1; }
  [[ -s "${stage}/schematics/model.0.${graph}_schematic.bin" ]] || { echo "missing ${graph} schematic" >&2; exit 1; }
done
[[ -s "${context}" ]] || { echo "context missing" >&2; exit 1; }
if [[ "${finalize_p}" == default ]]; then
  if grep -q 'init graph option: P =' "${stage}/compile.log"; then
    echo "default candidate unexpectedly used explicit P" >&2
    exit 1
  fi
else
  [[ "$(grep -c "init graph option: P = ${finalize_p}" "${stage}/compile.log")" -eq 2 ]] || {
    echo "P${finalize_p} was not applied to both graphs" >&2
    exit 1
  }
fi

{
  printf 'variant=%s\nsource_model=%s\nsource_model_sha256=%s\n' \
    "${variant}" "${source_model}" "$(sha256sum "${source_model}" | awk '{print $1}')"
  printf 'qairt_release=2.49.0.260730\nprefill_seq=64\nfinalize_p=%s\n' "${finalize_p}"
  printf 'git_commit=%s\ngit_branch=%s\n' \
    "$(git -C "${repo_root}" rev-parse HEAD)" "$(git -C "${repo_root}" branch --show-current)"
  printf 'git_status_begin\n'
  git -C "${repo_root}" status --short
  printf 'git_status_end\n'
} >"${stage}/provenance.txt"
(
  cd "${stage}"
  sha256sum "$(basename "${context}")" manifests/* schematics/* >artifacts.sha256
)

publish="${publish_root}.tmp.$$"
case "${publish}" in
  "${artifact_root}"/*.tmp.*) ;;
  *) echo "unexpected publish staging: ${publish}" >&2; exit 2 ;;
esac
mkdir -p "${publish}/manifests" "${publish}/schematics"
cp -f "${context}" "${publish}/"
cp -f "${stage}/compile.log" "${stage}/provenance.txt" "${stage}/artifacts.sha256" "${publish}/"
cp -f "${stage}/manifests/"* "${publish}/manifests/"
cp -f "${stage}/schematics/"* "${publish}/schematics/"
mv "${publish}" "${publish_root}"
rm -rf "${stage}"
trap - ERR INT TERM
echo "AR64 fairness context complete: ${publish_root}"
