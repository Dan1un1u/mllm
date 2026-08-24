#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
artifact_root="${ARTIFACT_ROOT:-/mnt/d/llm_exp}"
candidate_root="${artifact_root}/models/qwen3_sm8750_v79/g32/w4a16_current_regression_qairt249/20260824"
reference_root="${artifact_root}/results/qwen3_sm8750_v79_w4a16_qairt249_default_20260822_223744"
compact_result="${artifact_root}/results/qwen3_sm8750_v79_compact_initial_prefill_sha_qairt249_boolmask_20260823"
result_prefix="qwen3_sm8750_v79_w4a16_current_regression_qairt249"
viewer_root="/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/w4a16_current_regression_qairt249_20260824_viewer"
remote_dir="/data/local/tmp/mllm_w4a16_current_regression_qairt249"

[[ -r "${candidate_root}/profile-contract.env" ]] || {
  echo "candidate contract missing; run scripts/build_qwen3_w4a16_current_regression_qairt249.sh first" >&2
  exit 2
}
[[ -d "${reference_root}" ]] || { echo "reference result missing" >&2; exit 2; }
[[ ! -e "${viewer_root}" ]] || { echo "viewer work root already exists: ${viewer_root}" >&2; exit 2; }
case "${remote_dir}" in /data/local/tmp/mllm_w4a16_current_regression_qairt249) ;; *) exit 2 ;; esac

export ARTIFACT_ROOT="${artifact_root}"
export MODEL_ROOT="${artifact_root}/models"
export RESULTS_BASE="${artifact_root}/results"
export CONTRACT_FILE="${candidate_root}/profile-contract.env"
export RESULT_PREFIX="${result_prefix}"
export RMSNORM_U8_CONTRACT=0
export BUILD_ANDROID=0
export PREPARE_DEVICE=1
export BENCHMARK_RUNS=5
export MAX_NEW_TOKENS=64
export ACCURACY_MAX_NEW_TOKENS=64
export AR_LEN=32
export QAIRT_SDK_ROOT="${artifact_root}/models/qualcomm-sdk/qairt/2.49.0.260730"
export LOCAL_BUILD_BIN="${script_dir}/build-android-arm64-v8a-qnn-qairt249/bin"
export LOCAL_RUNNER="${script_dir}/build-android-arm64-v8a-qnn-qairt249/bin/mllm-qwen3-aot-runner"
export LLAMA_PACKAGE_BUILD="/home/daniuniu/llm_exp_work/qualcomm-sdk/qairt/2.49.0.260730/LLaMAPackage"
export ANDROID_NDK_PATH="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
export ADB_BIN="${ADB_BIN:-${script_dir}/scripts/adb_wsl_path_wrapper.sh}"
export ADB_SERIAL="${ADB_SERIAL:-3B15C8007Z300000}"
export REMOTE_DIR="${remote_dir}"
export PROFILE_WORK_ROOT="${viewer_root}"
export PROFILE_WRAPPER="${BASH_SOURCE[0]}"
export CLEAN_REMOTE=1

before="$(find "${RESULTS_BASE}" -maxdepth 1 -type d -name "${result_prefix}_*" -printf '%p\n' | sort)"
"${script_dir}/run_qwen3_sm8750_v79_g32_profile.sh"
after="$(find "${RESULTS_BASE}" -maxdepth 1 -type d -name "${result_prefix}_*" -printf '%p\n' | sort)"
result_root="$(comm -13 <(printf '%s\n' "${before}" | sed '/^$/d') <(printf '%s\n' "${after}" | sed '/^$/d'))"
[[ -n "${result_root}" && "${result_root}" != *$'\n'* ]] || {
  echo "could not identify exactly one new result directory" >&2
  exit 1
}

cp "${candidate_root}/profile-contract.env" "${result_root}/profile-contract.env"
cp "${candidate_root}/provenance.env" "${result_root}/context-provenance.env"
python3 "${script_dir}/scripts/qnn_w4a16_current_regression_result.py" \
  --candidate "${result_root}" \
  --reference "${reference_root}" \
  --compact-full-output "${compact_result}/optrace/full/output.raw" \
  --compact-cropped-output "${compact_result}/optrace/compact/output.raw" \
  --output "${result_root}/w4a16-current-regression.json" \
  --summary "${result_root}/SUMMARY.md" \
  | tee "${result_root}/w4a16-current-regression.log"

case "${viewer_root}" in
  /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/w4a16_current_regression_qairt249_20260824_viewer) rm -rf "${viewer_root}" ;;
  *) exit 2 ;;
esac
"${ADB_BIN}" -s "${ADB_SERIAL}" shell "rm -rf '${remote_dir}'"
echo "W4A16 current-source canonical regression result: ${result_root}"
