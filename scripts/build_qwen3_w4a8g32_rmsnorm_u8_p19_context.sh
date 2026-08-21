#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results}"
source_root="${SOURCE_ROOT:-${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/20260813_211529}"
publish_root="${PUBLISH_ROOT:-${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_p19/20260821}"
work_root="${WORK_ROOT:-/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/rmsnorm_u8_p19_20260821}"
model="${source_root}/qwen3-1.7B-w4a8g32-rmsnorm-u8.mllm"
run_id="rmsnorm_u8_p19_20260821"
stage="${work_root}/staging"
logs="${work_root}/logs"

case "${publish_root}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_p19/20260821) ;;
  *) echo "refusing unexpected PUBLISH_ROOT: ${publish_root}" >&2; exit 2 ;;
esac
case "${work_root}" in
  /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/rmsnorm_u8_p19_20260821) ;;
  *) echo "refusing unexpected WORK_ROOT: ${work_root}" >&2; exit 2 ;;
esac
[[ -s "${model}" ]] || { echo "accepted RMSNorm-A8 model missing: ${model}" >&2; exit 2; }
[[ ! -e "${publish_root}" ]] || { echo "published context already exists: ${publish_root}" >&2; exit 2; }
[[ ! -e "${work_root}" ]] || { echo "WSL staging already exists: ${work_root}" >&2; exit 2; }
mkdir -p "${stage}" "${logs}"

export W4A8_STAGE_DIR="${stage}"
export W4A8_LOG_ROOT="${logs}"
export MLLM_QNN_AOT_FINALIZE_P=19
"${repo_root}/scripts/build_qwen3_w4a8g32_context.sh" "${run_id}" "${model}"
unset MLLM_QNN_AOT_FINALIZE_P

qnn="${stage}/qnn"
context="${qnn}/qwen3-1.7B-w4a8g32-sha.bin"
for required in "${context}" \
  "${qnn}/manifests/model.0.s1_quant_manifest.json" \
  "${qnn}/manifests/model.0.s32_quant_manifest.json" \
  "${qnn}/schematics/model.0.s1_schematic.bin" \
  "${qnn}/schematics/model.0.s32_schematic.bin" \
  "${logs}/${run_id}/aot-context.log"; do
  [[ -s "${required}" ]] || { echo "required build output missing: ${required}" >&2; exit 1; }
done
grep -q "init graph option: P = 19" "${logs}/${run_id}/aot-context.log" || {
  echo "compile log does not prove P19 was applied" >&2
  exit 1
}
for graph in s1 s32; do
  cmp "${source_root}/manifests/model.0.${graph}_quant_manifest.json" \
      "${qnn}/manifests/model.0.${graph}_quant_manifest.json"
done

publish="${publish_root}.tmp.$$"
case "${publish}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_p19/20260821.tmp.*) ;;
  *) echo "refusing unexpected publish staging: ${publish}" >&2; exit 2 ;;
esac
mkdir -p "${publish}/manifests" "${publish}/schematics"
trap 'rm -rf "${publish}"' ERR INT TERM
cp "${context}" "${publish}/qwen3-1.7B-w4a8g32-rmsnorm-u8-p19.bin"
cp "${qnn}/manifests/"*.json "${publish}/manifests/"
cp "${qnn}/schematics/"*.bin "${publish}/schematics/"
cp "${logs}/${run_id}/aot-context.log" "${publish}/compile.log"
{
  printf 'source_model=%s\n' "${model}"
  printf 'source_context=%s\n' "${source_root}/qwen3-1.7B-w4a8g32-rmsnorm-u8.bin"
  printf 'git_commit=%s\n' "$(git -C "${repo_root}" rev-parse HEAD)"
  printf 'git_branch=%s\n' "$(git -C "${repo_root}" branch --show-current)"
  printf 'qairt_release=2.47.0.260601\n'
  printf 'finalize_O=3\nfinalize_P=19\n'
  printf 'logical_quant_manifests=byte-identical to accepted RMSNorm-A8 baseline\n'
} >"${publish}/provenance.txt"
(
  cd "${publish}"
  sha256sum qwen3-1.7B-w4a8g32-rmsnorm-u8-p19.bin \
    manifests/model.0.*.json schematics/model.0.*.bin
) >"${publish}/artifact.sha256"
mv "${publish}" "${publish_root}"
trap - ERR INT TERM

resolved_work="$(realpath -e "${work_root}")"
[[ "${resolved_work}" == "/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/rmsnorm_u8_p19_20260821" ]] || exit 2
rm -rf "${resolved_work}"
echo "Full RMSNorm-A8 P19 context published: ${publish_root}"
