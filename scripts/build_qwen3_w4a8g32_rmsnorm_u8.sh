#!/usr/bin/env bash
set -Eeuo pipefail

# Build the isolated native-U8 RMSNorm experiment without using any prior
# W4A8 artifact. Python/model staging and QNN AOT compilation run on WSL ext4;
# only final .mllm/.bin/QNN evidence is copied to D:.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="${1:-$(date +%Y%m%d_%H%M%S)}"
if [[ "${run_id}" == "-h" || "${run_id}" == "--help" ]]; then
    echo "usage: $0 [RUN_ID]"
    echo "Build native-U8 RMSNorm from Qwen3-origin in the WSL workspace and publish final artifacts."
    exit 0
fi
[[ "${run_id}" =~ ^[0-9]{8}_[0-9]{6}$ ]] || {
    echo "RUN_ID must match YYYYMMDD_HHMMSS (got ${run_id})" >&2
    exit 2
}
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results}"
work_root="${W4A8_WORK_ROOT:-/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/rmsnorm_u8}"
stage_dir="${W4A8_STAGE_DIR:-${work_root}/staging/${run_id}}"
reuse_stage="${W4A8_REUSE_STAGE:-0}"
build_log_root="${W4A8_LOG_ROOT:-${work_root}/logs}"
publish_models="${PUBLISH_MODELS_ROOT:-${models_root}}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/${run_id}"
publish_results="${PUBLISH_RESULTS_ROOT:-${results_root}}/qwen3_sm8750_v79_w4a8_rmsnorm_u8_${run_id}"

[[ -d "${repo_root}/.git" ]] || { echo "not an mllm source checkout: ${repo_root}" >&2; exit 2; }
if [[ "${reuse_stage}" != "1" && -e "${stage_dir}" ]]; then
    echo "staging run already exists: ${stage_dir}" >&2
    exit 2
fi
[[ ! -e "${publish_models}" ]] || { echo "published model run already exists: ${publish_models}" >&2; exit 2; }
[[ ! -e "${publish_results}" ]] || { echo "published result run already exists: ${publish_results}" >&2; exit 2; }

mkdir -p "${work_root}" "${build_log_root}" "${results_root}"
export MODELS_ROOT="${models_root}" RESULTS_ROOT="${results_root}"
export W4A8_WORK_ROOT="${work_root}" W4A8_STAGE_DIR="${stage_dir}"
export W4A8_LOG_ROOT="${build_log_root}"

{
    echo "run_id=${run_id}"
    echo "repo_root=${repo_root}"
    echo "work_root=${work_root}"
    echo "stage_dir=${stage_dir}"
    echo "publish_models=${publish_models}"
    echo "publish_results=${publish_results}"
    echo "scope=native U8 RMSNorm only; all Qwen3 RMSNorm gamma and synthetic zero bias"
    echo "historical_w4a8_inputs=forbidden"
    echo "source_model=${SOURCE_MODEL:-${models_root}/Qwen3-origin}"
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short -- . \
        ':(exclude)third_party/half/include/half/half.hpp'
} | tee "${build_log_root}/${run_id}.provenance.txt"

model_path="${stage_dir}/qwen3_1.7b_w4a8g32.mllm"
if [[ "${reuse_stage}" == "1" && -s "${model_path}" ]]; then
    echo "Reusing completed model staging: ${model_path}"
else
    "${repo_root}/scripts/build_qwen3_w4a8g32_model.sh" "${run_id}" \
        2>&1 | tee "${build_log_root}/${run_id}.model.log"
    grep -q "W4A8G32 model staging complete" "${build_log_root}/${run_id}.model.log" || {
        echo "model staging did not complete" >&2
        exit 1
    }
fi

artifact_dir="${stage_dir}/qnn"
context_path="${artifact_dir}/qwen3-1.7B-w4a8g32-sha.bin"
if [[ "${reuse_stage}" == "1" && -s "${context_path}" \
    && -s "${artifact_dir}/manifests/model.0.s1_quant_manifest.json" \
    && -s "${artifact_dir}/manifests/model.0.s32_quant_manifest.json" ]]; then
    echo "Reusing completed QNN context staging: ${context_path}"
else
    "${repo_root}/scripts/build_qwen3_w4a8g32_context.sh" "${run_id}" "${model_path}" \
        2>&1 | tee "${build_log_root}/${run_id}.context.log"
fi

manifest_dir="${artifact_dir}/manifests"
schematic_dir="${artifact_dir}/schematics"
for path in "${model_path}" "${context_path}" \
    "${manifest_dir}/model.0.s1_quant_manifest.json" \
    "${manifest_dir}/model.0.s32_quant_manifest.json" \
    "${schematic_dir}/model.0.s1_schematic.bin" \
    "${schematic_dir}/model.0.s32_schematic.bin"; do
    [[ -s "${path}" ]] || { echo "required build artifact missing: ${path}" >&2; exit 1; }
done

# Publish atomically on the Windows-backed volume. A partially copied run is
# never exposed under the final namespace.
publish_models_tmp="${publish_models}.tmp.$$"
publish_results_tmp="${publish_results}.tmp.$$"
rm -rf "${publish_models_tmp}" "${publish_results_tmp}"
mkdir -p "${publish_models_tmp}/manifests" "${publish_models_tmp}/schematics" "${publish_results_tmp}"
cp --reflink=auto "${model_path}" "${publish_models_tmp}/qwen3-1.7B-w4a8g32-rmsnorm-u8.mllm"
cp --reflink=auto "${context_path}" "${publish_models_tmp}/qwen3-1.7B-w4a8g32-rmsnorm-u8.bin"
cp --reflink=auto "${manifest_dir}"/model.0.*_quant_manifest.json "${publish_models_tmp}/manifests/"
cp --reflink=auto "${schematic_dir}"/model.0.*_schematic.bin "${publish_models_tmp}/schematics/"
(
    cd "${publish_models_tmp}"
    sha256sum qwen3-1.7B-w4a8g32-rmsnorm-u8.mllm \
        qwen3-1.7B-w4a8g32-rmsnorm-u8.bin \
        manifests/model.0.*_quant_manifest.json \
        schematics/model.0.*_schematic.bin
) > "${publish_models_tmp}/artifact.sha256"
cp --reflink=auto "${build_log_root}/${run_id}.provenance.txt" "${publish_results_tmp}/provenance.txt"
cp --reflink=auto "${build_log_root}/${run_id}.model.log" "${publish_results_tmp}/model-build.log"
if [[ -f "${build_log_root}/${run_id}.context.log" ]]; then
    cp --reflink=auto "${build_log_root}/${run_id}.context.log" "${publish_results_tmp}/context-build.log"
else
    printf '%s\n' "context build was resumed from completed WSL staging; see staging/qnn artifacts" \
        > "${publish_results_tmp}/context-build.log"
fi
cp --reflink=auto "${publish_models_tmp}/artifact.sha256" "${publish_results_tmp}/artifact.sha256"
model_rel="qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/${run_id}/qwen3-1.7B-w4a8g32-rmsnorm-u8.bin"
schematic_rel="qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/${run_id}/schematics"
manifest_rel="qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/${run_id}/manifests"
tokenizer_path="${models_root}/Qwen3-origin/qwen3-tokenizer.json"
config_path="${repo_root}/examples/qwen3_qnn_aot/config_1.7B_g32.json"
accuracy_path="${repo_root}/scripts/qwen3_sm8750_v79_accuracy.tsv"
runner_path="${repo_root}/build-android-arm64-v8a-qnn/bin/mllm-qwen3-aot-runner"
runner_sha="candidate-runner-not-pinned"
if [[ -f "${runner_path}" ]]; then
    runner_sha="$(sha256sum "${runner_path}" | awk '{print $1}')"
fi
cat > "${publish_results_tmp}/profile-contract.env" <<EOF
# Generated candidate contract for the native-U8 RMSNorm experiment.
BASELINE_ID="w4a8g32-rmsnorm-u8-${run_id}"
BASELINE_PROFILE_SCHEME="native_w4a8g32_rmsnorm_u8"
BASELINE_SOURCE_COMMIT="$(git -C "${repo_root}" rev-parse HEAD)"
BASELINE_REFERENCE_RESULT="qwen3_sm8750_v79_g32_20260807_230410"
BASELINE_REFERENCE_COMMIT="fad71e9f3e0348d1afa1b1238e83cbfc40a4e57b"
BASELINE_REFERENCE_CRITICAL_PATH="qwen3-sm8750-v79-g32-e2e-critical-path.html"
BASELINE_QAIRT_RELEASE="2.47.0.260601"
BASELINE_MODEL_REL="${model_rel}"
BASELINE_SCHEMATIC_REL="${schematic_rel}"
BASELINE_MANIFEST_REL="${manifest_rel}"
BASELINE_TOKENIZER_REL="Qwen3-origin/qwen3-tokenizer.json"
BASELINE_CONFIG_REL="examples/qwen3_qnn_aot/config_1.7B_g32.json"
BASELINE_ACCURACY_SUITE_REL="scripts/qwen3_sm8750_v79_accuracy.tsv"
BASELINE_RUNNER_REL="build-android-arm64-v8a-qnn/bin/mllm-qwen3-aot-runner"
BASELINE_REMOTE_RUNNER="mllm-qwen3-aot-runner"
BASELINE_REMOTE_MODEL="qwen3-1.7B-w4a8g32-rmsnorm-u8.bin"
BASELINE_REMOTE_TOKENIZER="qwen3-tokenizer.json"
BASELINE_REMOTE_CONFIG="config_1.7B_g32.json"
BASELINE_CONTEXT_SHA256="$(sha256sum "${publish_models_tmp}/qwen3-1.7B-w4a8g32-rmsnorm-u8.bin" | awk '{print $1}')"
BASELINE_RUNNER_SHA256="${runner_sha}"
BASELINE_TOKENIZER_SHA256="$(sha256sum "${tokenizer_path}" | awk '{print $1}')"
BASELINE_CONFIG_SHA256="$(sha256sum "${config_path}" | awk '{print $1}')"
BASELINE_ACCURACY_SUITE_SHA256="$(sha256sum "${accuracy_path}" | awk '{print $1}')"
BASELINE_S1_SCHEMATIC_SHA256="$(sha256sum "${publish_models_tmp}/schematics/model.0.s1_schematic.bin" | awk '{print $1}')"
BASELINE_S32_SCHEMATIC_SHA256="$(sha256sum "${publish_models_tmp}/schematics/model.0.s32_schematic.bin" | awk '{print $1}')"
BASELINE_S1_MANIFEST_SHA256="$(sha256sum "${publish_models_tmp}/manifests/model.0.s1_quant_manifest.json" | awk '{print $1}')"
BASELINE_S32_MANIFEST_SHA256="$(sha256sum "${publish_models_tmp}/manifests/model.0.s32_quant_manifest.json" | awk '{print $1}')"
EOF
printf '%s\n' \
    "model=${publish_models}/qwen3-1.7B-w4a8g32-rmsnorm-u8.mllm" \
    "context=${publish_models}/qwen3-1.7B-w4a8g32-rmsnorm-u8.bin" \
    "manifests=${publish_models}/manifests" \
    "schematics=${publish_models}/schematics" \
    > "${publish_results_tmp}/published-paths.txt"
mv "${publish_models_tmp}" "${publish_models}"
mv "${publish_results_tmp}" "${publish_results}"

echo "Native-U8 RMSNorm build and atomic publish complete."
echo "Model artifacts: ${publish_models}"
echo "Build result:     ${publish_results}"
