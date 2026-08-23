#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
artifact_root="${ARTIFACT_ROOT:-/mnt/d/llm_exp}"
variant="${QAIRT249_VARIANT:-${1:-p19}}"
case "${variant}" in default|p19) ;; *) echo "expected default or p19" >&2; exit 2 ;; esac

export ARTIFACT_ROOT="${artifact_root}"
export CONTRACT_FILE="${artifact_root}/models/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_qairt249/20260823/${variant}/profile-contract.env"
export RESULT_PREFIX="qwen3_sm8750_v79_w4a8_rmsnorm_u8_qairt249_${variant}"
export RMSNORM_U8_CONTRACT=1
export BUILD_ANDROID=0
export QAIRT_SDK_ROOT="${artifact_root}/models/qualcomm-sdk/qairt/2.49.0.260730"
export LOCAL_BUILD_BIN="${script_dir}/build-android-arm64-v8a-qnn-qairt249/bin"
export LLAMA_PACKAGE_BUILD="/home/daniuniu/llm_exp_work/qualcomm-sdk/qairt/2.49.0.260730/LLaMAPackage"
export ANDROID_NDK_PATH="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
export ADB_BIN="${ADB_BIN:-/mnt/c/adb/adb.exe}"
export ADB_SERIAL="${ADB_SERIAL:-3B15C8007Z300000}"
export REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/mllm_w4a8_rmsnorm_u8_qairt249}"
export PROFILE_WRAPPER="${BASH_SOURCE[0]}"

[[ -r "${CONTRACT_FILE}" ]] || {
  echo "candidate contract missing: ${CONTRACT_FILE}" >&2
  echo "build ${variant} first with scripts/build_qwen3_w4a8g32_rmsnorm_u8_qairt249_context.sh" >&2
  exit 2
}
exec "${script_dir}/run_qwen3_sm8750_v79_g32_profile.sh"
