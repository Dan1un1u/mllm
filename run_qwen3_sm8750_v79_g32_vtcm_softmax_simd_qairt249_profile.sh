#!/usr/bin/env bash
set -Eeuo pipefail

# Profile the isolated SIMD/VTCM masked-Softmax experiment against the
# accepted QAIRT 2.49 RMSNorm-U8 baseline.  The context contract pins all
# source artifacts; runtime op-package registration is enabled only here.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
artifact_root="${ARTIFACT_ROOT:-/mnt/d/llm_exp}"

export ARTIFACT_ROOT="${artifact_root}"
export CONTRACT_FILE="${artifact_root}/models/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_vtcm_softmax_simd_qairt249/20260823/p19/profile-contract.env"
export RESULT_PREFIX="qwen3_sm8750_v79_w4a8_rmsnorm_u8_vtcm_softmax_simd_qairt249_p19"
export RMSNORM_U8_CONTRACT=1
export BUILD_ANDROID=0
export QAIRT_SDK_ROOT="${artifact_root}/models/qualcomm-sdk/qairt/2.49.0.260730"
export LOCAL_BUILD_BIN="${script_dir}/build-android-arm64-v8a-qnn-qairt249/bin"
export LLAMA_PACKAGE_BUILD="/home/daniuniu/llm_exp_work/qualcomm-sdk/qairt/2.49.0.260730/LLaMAPackage"
export ANDROID_NDK_PATH="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
export ADB_BIN="${ADB_BIN:-/mnt/c/adb/adb.exe}"
export ADB_SERIAL="${ADB_SERIAL:-3B15C8007Z300000}"
export REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/mllm_w4a8_vtcm_softmax_simd_qairt249}"
export REMOTE_OP_PACKAGE_PATH="${REMOTE_DIR}/libQnnLLaMAPackage_HTP.so"
export REMOTE_OP_PACKAGE_PROVIDER="LLaMAPackageInterfaceProvider"
export REMOTE_OP_PACKAGE_TARGET="HTP"
export PROFILE_WRAPPER="${BASH_SOURCE[0]}"

[[ -r "${CONTRACT_FILE}" ]] || {
  echo "candidate contract missing: ${CONTRACT_FILE}" >&2
  exit 2
}

exec "${script_dir}/run_qwen3_sm8750_v79_g32_profile.sh"
