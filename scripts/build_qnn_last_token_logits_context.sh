#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/last_token_logits_qairt249/20260824_sha_compact_p19}"
source_model="${SOURCE_MODEL:-${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/20260813_211529/qwen3-1.7B-w4a8g32-rmsnorm-u8.mllm}"
control_root="${CONTROL_ROOT:-${models_root}/qwen3_sm8750_v79/g32/compact_initial_prefill_qairt249/20260823_sha_boolmask/contexts/compact_w32_s32_p19}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.49.0.260730}"
compiler="${COMPILER:-${repo_root}/build-qnn-aot-qairt249/bin/mllm-qwen3-compact-initial-prefill-c}"
model_config="${MODEL_CONFIG:-${repo_root}/examples/qwen3_qnn_aot/config_1.7B_g32.json}"
aot_config="${AOT_CONFIG:-${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32.json}"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"
tag="last_logits_w32_s32_p19"
case_dir="${artifact_root}/contexts/${tag}"

case "${artifact_root}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/last_token_logits_qairt249/20260824_sha_compact_p19) ;;
  *) echo "refusing unexpected ARTIFACT_ROOT: ${artifact_root}" >&2; exit 2 ;;
esac
[[ "${qairt_root##*/}" == "2.49.0.260730" ]] || { echo "unexpected QAIRT release" >&2; exit 2; }
[[ -x "${compiler}" && -s "${source_model}" ]] || { echo "compiler/model missing" >&2; exit 2; }
[[ -s "${control_root}/compact_w32_s32_p19.bin" && -s "${control_root}/manifest.json" ]] || {
  echo "accepted compact-prefill control missing: ${control_root}" >&2
  exit 2
}
[[ ! -e "${case_dir}" ]] || { echo "refusing existing candidate: ${case_dir}" >&2; exit 2; }

mkdir -p "${repo_root}/tmp" "${artifact_root}/contexts"
stage="$(mktemp -d "${repo_root}/tmp/last_token_logits_compile.XXXXXX")"
case "${stage}" in
  "${repo_root}"/tmp/last_token_logits_compile.*) ;;
  *) echo "unexpected staging: ${stage}" >&2; exit 2 ;;
esac
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
      -c "${model_config}" \
      -aot_cfg "${aot_config}" \
      -qnn_env "${qnn_lib}/" \
      -o "${stage}/${tag}.bin" \
      --width 32 \
      --logits last
) >"${stage}/compile.log" 2>&1
status=$?
set -e
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

grep -q 'P = 19' "${stage}/compile.log" || { echo "P19 evidence missing" >&2; exit 1; }
manifest="$(find "${stage}/manifests" -maxdepth 1 -type f -name '*_quant_manifest.json' -print -quit)"
schematic="$(find "${stage}/schematics" -maxdepth 1 -type f -name '*_schematic.bin' -print -quit)"
for output in "${stage}/${tag}.bin" "${manifest}" "${schematic}"; do
  [[ -s "${output}" ]] || { echo "output missing: ${output}" >&2; exit 1; }
done
cp -f "${manifest}" "${stage}/manifest.json"
cp -f "${schematic}" "${stage}/schematic.bin"
{
  printf 'control_context=%s\n' "${control_root}/compact_w32_s32_p19.bin"
  printf 'control_manifest=%s\n' "${control_root}/manifest.json"
  printf 'source_model=%s\nqairt_release=2.49.0.260730\nfinalize_p=19\n' "${source_model}"
  printf 'git_commit=%s\ngit_branch=%s\n' \
    "$(git -C "${repo_root}" rev-parse HEAD)" "$(git -C "${repo_root}" branch --show-current)"
} >"${stage}/provenance.txt"
sha256sum "${stage}/${tag}.bin" "${stage}/manifest.json" \
  "${stage}/schematic.bin" >"${stage}/artifacts.sha256"

publish="${case_dir}.tmp.$$"
case "${publish}" in
  "${artifact_root}"/contexts/*.tmp.*) ;;
  *) echo "unexpected publish staging: ${publish}" >&2; exit 2 ;;
esac
mkdir -p "${publish}"
cp -a "${stage}/." "${publish}/"
mv "${publish}" "${case_dir}"
rm -rf "${stage}"
trap - ERR INT TERM
echo "Last-token logits context complete: ${case_dir}"
