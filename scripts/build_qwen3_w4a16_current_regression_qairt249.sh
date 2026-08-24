#!/usr/bin/env bash
set -Eeuo pipefail

# Rebuild the archived W4A16 graph contract from the current source tree.  The
# model, QAIRT release, graph shapes, default finalizer, and profiler inputs are
# pinned to the earlier QAIRT 2.49 W4A16 run; only the current shared mllm source
# is allowed to differ.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.49.0.260730}"
source_model="${models_root}/qwen3_sm8750_v79/g32/w4a16/source_g32_export/qwen3_1.7b_g32.mllm"
compiler="${COMPILER:-${repo_root}/build-qnn-aot-qairt249/bin/mllm-qwen3-aot-sha-a16-g32-c}"
runner="${repo_root}/build-android-arm64-v8a-qnn-qairt249/bin/mllm-qwen3-aot-runner"
config="${repo_root}/examples/qwen3_qnn_aot/config_1.7B_g32.json"
aot_config="${repo_root}/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_g32.json"
reference_result="qwen3_sm8750_v79_w4a16_qairt249_default_20260822_223744"
reference_root="/mnt/d/llm_exp/results/${reference_result}"
work_root="${WORK_ROOT:-/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/w4a16_current_regression_qairt249_20260824}"
publish_root="${PUBLISH_ROOT:-${models_root}/qwen3_sm8750_v79/g32/w4a16_current_regression_qairt249/20260824}"
qnn_lib="${qairt_root}/lib/x86_64-linux-clang"

case "${work_root}" in
  /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/w4a16_current_regression_qairt249_20260824) ;;
  *) echo "refusing unexpected WORK_ROOT: ${work_root}" >&2; exit 2 ;;
esac
case "${publish_root}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/w4a16_current_regression_qairt249/20260824) ;;
  *) echo "refusing unexpected PUBLISH_ROOT: ${publish_root}" >&2; exit 2 ;;
esac
[[ "${qairt_root##*/}" == 2.49.0.260730 ]] || {
  echo "unexpected QAIRT release: ${qairt_root}" >&2
  exit 2
}
for path in "${source_model}" "${compiler}" "${runner}" "${config}" "${aot_config}"; do
  [[ -s "${path}" ]] || { echo "required input missing: ${path}" >&2; exit 2; }
done
for graph in s1 s32; do
  [[ -s "${reference_root}/manifests/model.0.${graph}_quant_manifest.json" ]] || {
    echo "reference manifest missing for ${graph}" >&2
    exit 2
  }
done
[[ ! -e "${work_root}" ]] || { echo "work root already exists: ${work_root}" >&2; exit 2; }
[[ ! -e "${publish_root}" ]] || { echo "publish root already exists: ${publish_root}" >&2; exit 2; }
[[ -z "$(git -C "${repo_root}" status --porcelain=v1)" ]] || {
  echo "refusing a formal regression build from a dirty Git worktree" >&2
  exit 2
}

manifest_dir="${work_root}/manifests"
schematic_dir="${work_root}/schematics"
context="${work_root}/context.bin"
compile_log="${work_root}/compile.log"
mkdir -p "${manifest_dir}" "${schematic_dir}"

export LD_LIBRARY_PATH="$(dirname "${compiler}"):${qnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLLM_QNN_AOT_QUANT_MANIFEST_DIR="${manifest_dir}"
export MLLM_QNN_AOT_OPTRACE=1
export MLLM_QNN_AOT_OPTRACE_DIR="${schematic_dir}"
unset MLLM_QNN_AOT_FINALIZE_P MLLM_QNN_AOT_FINALIZE_P_S1 MLLM_QNN_AOT_FINALIZE_P_S32

(
  cd "${work_root}"
  "${compiler}" \
    -m "${source_model}" \
    -c "${config}" \
    -aot_cfg "${aot_config}" \
    -qnn_env "${qnn_lib}/" \
    --prefill_seq 32 \
    -o "${context}"
) >"${compile_log}" 2>&1

[[ -s "${context}" ]] || { echo "context was not generated" >&2; exit 1; }
if grep -q "with init graph option: P =" "${compile_log}"; then
  echo "W4A16 control unexpectedly used an explicit P override" >&2
  exit 1
fi
for graph in s1 s32; do
  manifest="${manifest_dir}/model.0.${graph}_quant_manifest.json"
  schematic="${schematic_dir}/model.0.${graph}_schematic.bin"
  [[ -s "${manifest}" && -s "${schematic}" ]] || {
    echo "missing ${graph} manifest or schematic" >&2
    exit 1
  }
  python3 "${repo_root}/scripts/qnn_quant_manifest_equivalent.py" \
    "${reference_root}/manifests/model.0.${graph}_quant_manifest.json" \
    "${manifest}" >"${manifest_dir}/model.0.${graph}_reference-equivalence.txt"
done

publish="${publish_root}.tmp.$$"
case "${publish}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/w4a16_current_regression_qairt249/20260824.tmp.*) ;;
  *) echo "refusing unexpected publish staging path: ${publish}" >&2; exit 2 ;;
esac
trap 'rm -rf "${publish}"' ERR INT TERM
mkdir -p "${publish}/manifests" "${publish}/schematics"
cp "${context}" "${publish}/context.bin"
cp "${manifest_dir}"/* "${publish}/manifests/"
cp "${schematic_dir}"/* "${publish}/schematics/"
cp "${compile_log}" "${publish}/compile.log"

git_commit="$(git -C "${repo_root}" rev-parse HEAD)"
tokenizer="${models_root}/Qwen3-origin/qwen3-tokenizer.json"
accuracy_suite="${repo_root}/scripts/qwen3_sm8750_v79_accuracy.tsv"
cat >"${publish}/provenance.env" <<EOF
git_commit=${git_commit}
git_branch=$(git -C "${repo_root}" branch --show-current)
source_model=${source_model}
source_model_sha256=$(sha256sum "${source_model}" | awk '{print $1}')
qairt_release=2.49.0.260730
qnn_api=2.38.0
finalize_O=3
finalize_P=default
graph_contract=legacy full-width s1/s32 W4A16G32
reference_result=${reference_result}
reference_manifests=canonical-equivalent
EOF

cat >"${publish}/profile-contract.env" <<EOF
BASELINE_ID="w4a16g32-current-source-regression-qairt249-20260824"
BASELINE_PROFILE_SCHEME="native_w4a16g32_current_source_regression_qairt249"
BASELINE_SOURCE_COMMIT="${git_commit}"
BASELINE_REFERENCE_RESULT="${reference_result}"
BASELINE_REFERENCE_COMMIT="a49d078f"
BASELINE_REFERENCE_CRITICAL_PATH="qwen3-sm8750-v79-g32-e2e-critical-path.html"
BASELINE_QAIRT_RELEASE="2.49.0.260730"
BASELINE_MODEL_REL="qwen3_sm8750_v79/g32/w4a16_current_regression_qairt249/20260824/context.bin"
BASELINE_SCHEMATIC_REL="qwen3_sm8750_v79/g32/w4a16_current_regression_qairt249/20260824/schematics"
BASELINE_MANIFEST_REL="qwen3_sm8750_v79/g32/w4a16_current_regression_qairt249/20260824/manifests"
BASELINE_TOKENIZER_REL="Qwen3-origin/qwen3-tokenizer.json"
BASELINE_CONFIG_REL="examples/qwen3_qnn_aot/config_1.7B_g32.json"
BASELINE_ACCURACY_SUITE_REL="scripts/qwen3_sm8750_v79_accuracy.tsv"
BASELINE_RUNNER_REL="build-android-arm64-v8a-qnn-qairt249/bin/mllm-qwen3-aot-runner"
BASELINE_REMOTE_RUNNER="mllm-qwen3-aot-runner"
BASELINE_REMOTE_MODEL="context.bin"
BASELINE_REMOTE_TOKENIZER="qwen3-tokenizer.json"
BASELINE_REMOTE_CONFIG="config_1.7B_g32.json"
BASELINE_CONTEXT_SHA256="$(sha256sum "${publish}/context.bin" | awk '{print $1}')"
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
  sha256sum context.bin manifests/* schematics/* compile.log provenance.env profile-contract.env \
    >artifacts.sha256
)
mv "${publish}" "${publish_root}"
trap - ERR INT TERM

resolved_work="$(realpath -e "${work_root}")"
[[ "${resolved_work}" == "/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/w4a16_current_regression_qairt249_20260824" ]] || {
  echo "refusing to clean unexpected work root: ${resolved_work}" >&2
  exit 2
}
rm -rf "${resolved_work}"
echo "W4A16 current-source regression context published: ${publish_root}"
