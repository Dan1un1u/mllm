#!/usr/bin/env bash
set -Eeuo pipefail

# Recompile the accepted RMSNorm-U8 W4A8G32 model with an isolated QAIRT 2.49
# toolchain. The source .mllm and logical quantization contract stay fixed;
# only QNN graph finalization and the serialized context are regenerated.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
variant="${1:?usage: build_qwen3_w4a8g32_rmsnorm_u8_qairt249_context.sh default|p19}"
case "${variant}" in
  default) finalize_p=default ;;
  p19) finalize_p=19 ;;
  *) echo "unsupported variant: ${variant} (expected default or p19)" >&2; exit 2 ;;
esac

models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.49.0.260730}"
source_root="${SOURCE_ROOT:-${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/20260813_211529}"
model="${source_root}/qwen3-1.7B-w4a8g32-rmsnorm-u8.mllm"
compiler="${COMPILER:-${repo_root}/build-qnn-aot-qairt249/bin/mllm-qwen3-aot-sha-g32-c}"
work_root="${WORK_ROOT:-/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/rmsnorm_u8_qairt249_20260823/${variant}}"
publish_root="${PUBLISH_ROOT:-${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_qairt249/20260823/${variant}}"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"
manifest_dir="${work_root}/manifests"
schematic_dir="${work_root}/schematics"
context="${work_root}/qwen3-1.7B-w4a8g32-rmsnorm-u8-qairt249-${variant}.bin"
compile_log="${work_root}/compile.log"

case "${work_root}" in
  /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/rmsnorm_u8_qairt249_20260823/default|\
  /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/rmsnorm_u8_qairt249_20260823/p19) ;;
  *) echo "refusing unexpected WORK_ROOT: ${work_root}" >&2; exit 2 ;;
esac
case "${publish_root}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_qairt249/20260823/default|\
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_qairt249/20260823/p19) ;;
  *) echo "refusing unexpected PUBLISH_ROOT: ${publish_root}" >&2; exit 2 ;;
esac
[[ "${qairt_root##*/}" == 2.49.0.260730 ]] || {
  echo "unexpected QAIRT release: ${qairt_root}" >&2
  exit 2
}
[[ -s "${model}" ]] || { echo "accepted RMSNorm-U8 model missing: ${model}" >&2; exit 2; }
[[ -x "${compiler}" ]] || { echo "QAIRT 2.49 compiler missing: ${compiler}" >&2; exit 2; }
[[ -f "${qnn_lib}/libQnnHtp.so" ]] || { echo "QAIRT 2.49 HTP library missing" >&2; exit 2; }
[[ ! -e "${work_root}" ]] || { echo "work root already exists: ${work_root}" >&2; exit 2; }
[[ ! -e "${publish_root}" ]] || { echo "publish root already exists: ${publish_root}" >&2; exit 2; }
[[ -z "$(git -C "${repo_root}" status --porcelain=v1)" ]] || {
  echo "refusing to build a formal candidate from a dirty Git worktree" >&2
  exit 2
}

mkdir -p "${manifest_dir}" "${schematic_dir}"
export LD_LIBRARY_PATH="$(dirname "${compiler}"):${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLLM_QNN_AOT_QUANT_MANIFEST_DIR="${manifest_dir}"
export MLLM_QNN_AOT_OPTRACE=1
export MLLM_QNN_AOT_OPTRACE_DIR="${schematic_dir}"
unset MLLM_QNN_AOT_FINALIZE_P MLLM_QNN_AOT_FINALIZE_P_S1 MLLM_QNN_AOT_FINALIZE_P_S32
if [[ "${finalize_p}" != default ]]; then
  export MLLM_QNN_AOT_FINALIZE_P="${finalize_p}"
fi

(
  cd "${work_root}"
  "${compiler}" \
    -m "${model}" \
    -c "${repo_root}/examples/qwen3_qnn_aot/config_1.7B_g32.json" \
    -aot_cfg "${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32.json" \
    -qnn_env "${qnn_lib}/" \
    -o "${context}"
) >"${compile_log}" 2>&1

for graph in s1 s32; do
  [[ -s "${manifest_dir}/model.0.${graph}_quant_manifest.json" ]] || exit 1
  [[ -s "${schematic_dir}/model.0.${graph}_schematic.bin" ]] || exit 1
  python3 "${repo_root}/scripts/qnn_quant_manifest_equivalent.py" \
    "${source_root}/manifests/model.0.${graph}_quant_manifest.json" \
    "${manifest_dir}/model.0.${graph}_quant_manifest.json" \
    >"${manifest_dir}/model.0.${graph}_canonical.sha256"
done
[[ -s "${context}" ]] || exit 1
if [[ "${finalize_p}" == default ]]; then
  if grep -q "Graph model.0.* with init graph option: P =" "${compile_log}"; then
    echo "default candidate unexpectedly used an explicit P override" >&2
    exit 1
  fi
else
  [[ "$(grep -c "init graph option: P = ${finalize_p}" "${compile_log}")" -eq 2 ]] || {
    echo "compile log does not prove P${finalize_p} was applied to s1 and s32" >&2
    exit 1
  }
fi

publish="${publish_root}.tmp.$$"
case "${publish}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_qairt249/20260823/*.tmp.*) ;;
  *) echo "refusing unexpected publish staging: ${publish}" >&2; exit 2 ;;
esac
mkdir -p "${publish}/manifests" "${publish}/schematics"
trap 'rm -rf "${publish}"' ERR INT TERM
cp "${context}" "${publish}/$(basename "${context}")"
cp "${manifest_dir}"/*.json "${manifest_dir}"/*.sha256 "${publish}/manifests/"
cp "${schematic_dir}"/*.bin "${publish}/schematics/"
cp "${compile_log}" "${publish}/compile.log"
{
  printf 'source_model=%s\nsource_model_sha256=%s\n' \
    "${model}" "$(sha256sum "${model}" | awk '{print $1}')"
  printf 'source_quant_contract=%s\n' "${source_root}/manifests"
  printf 'git_commit=%s\ngit_branch=%s\n' \
    "$(git -C "${repo_root}" rev-parse HEAD)" "$(git -C "${repo_root}" branch --show-current)"
  printf 'qairt_release=2.49.0.260730\nqnn_api=2.38.0\n'
  printf 'finalize_O=3\nfinalize_P=%s\n' "${finalize_p}"
  printf 'logical_quant_manifests=canonical-equivalent to accepted QAIRT 2.47 RMSNorm-U8 baseline\n'
} >"${publish}/provenance.env"

context_name="$(basename "${context}")"
tokenizer="${models_root}/Qwen3-origin/qwen3-tokenizer.json"
config="${repo_root}/examples/qwen3_qnn_aot/config_1.7B_g32.json"
accuracy_suite="${repo_root}/scripts/qwen3_sm8750_v79_accuracy.tsv"
runner="${repo_root}/build-android-arm64-v8a-qnn-qairt249/bin/mllm-qwen3-aot-runner"
cat >"${publish}/profile-contract.env" <<EOF
BASELINE_ID="w4a8g32-rmsnorm-u8-qairt249-${variant}-20260823"
BASELINE_PROFILE_SCHEME="native_w4a8g32_rmsnorm_u8_qairt249_${variant}"
BASELINE_SOURCE_COMMIT="$(git -C "${repo_root}" rev-parse HEAD)"
BASELINE_REFERENCE_RESULT="qwen3_sm8750_v79_w4a8_rmsnorm_u8_p19_20260821_231318"
BASELINE_REFERENCE_COMMIT="2bd2de28387d7c506edfd1d749c4893b4c019e9b"
BASELINE_REFERENCE_CRITICAL_PATH="qwen3-sm8750-v79-g32-e2e-critical-path.html"
BASELINE_QAIRT_RELEASE="2.49.0.260730"
BASELINE_MODEL_REL="qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_qairt249/20260823/${variant}/${context_name}"
BASELINE_SCHEMATIC_REL="qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_qairt249/20260823/${variant}/schematics"
BASELINE_MANIFEST_REL="qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_qairt249/20260823/${variant}/manifests"
BASELINE_TOKENIZER_REL="Qwen3-origin/qwen3-tokenizer.json"
BASELINE_CONFIG_REL="examples/qwen3_qnn_aot/config_1.7B_g32.json"
BASELINE_ACCURACY_SUITE_REL="scripts/qwen3_sm8750_v79_accuracy.tsv"
BASELINE_RUNNER_REL="build-android-arm64-v8a-qnn-qairt249/bin/mllm-qwen3-aot-runner"
BASELINE_REMOTE_RUNNER="mllm-qwen3-aot-runner"
BASELINE_REMOTE_MODEL="${context_name}"
BASELINE_REMOTE_TOKENIZER="qwen3-tokenizer.json"
BASELINE_REMOTE_CONFIG="config_1.7B_g32.json"
BASELINE_CONTEXT_SHA256="$(sha256sum "${publish}/${context_name}" | awk '{print $1}')"
BASELINE_RUNNER_SHA256="$(sha256sum "${runner}" | awk '{print $1}')"
BASELINE_TOKENIZER_SHA256="$(sha256sum "${tokenizer}" | awk '{print $1}')"
BASELINE_CONFIG_SHA256="$(sha256sum "${config}" | awk '{print $1}')"
BASELINE_ACCURACY_SUITE_SHA256="$(sha256sum "${accuracy_suite}" | awk '{print $1}')"
BASELINE_S1_SCHEMATIC_SHA256="$(sha256sum "${publish}/schematics/model.0.s1_schematic.bin" | awk '{print $1}')"
BASELINE_S32_SCHEMATIC_SHA256="$(sha256sum "${publish}/schematics/model.0.s32_schematic.bin" | awk '{print $1}')"
BASELINE_S1_MANIFEST_SHA256="$(sha256sum "${publish}/manifests/model.0.s1_quant_manifest.json" | awk '{print $1}')"
BASELINE_S32_MANIFEST_SHA256="$(sha256sum "${publish}/manifests/model.0.s32_quant_manifest.json" | awk '{print $1}')"
EOF
(
  cd "${publish}"
  sha256sum "${context_name}" manifests/* schematics/* provenance.env profile-contract.env >artifacts.sha256
)
mv "${publish}" "${publish_root}"
trap - ERR INT TERM

resolved_work="$(realpath -e "${work_root}")"
case "${resolved_work}" in
  /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/rmsnorm_u8_qairt249_20260823/default|\
  /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/rmsnorm_u8_qairt249_20260823/p19) ;;
  *) echo "refusing to clean unexpected work root: ${resolved_work}" >&2; exit 2 ;;
esac
rm -rf "${resolved_work}"
echo "QAIRT 2.49 RMSNorm-U8 context published: ${publish_root}"
