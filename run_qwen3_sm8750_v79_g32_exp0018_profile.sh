#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
artifact_root="${ARTIFACT_ROOT:-/mnt/d/llm_exp}"

export ARTIFACT_ROOT="${artifact_root}"
export CONTRACT_FILE="${script_dir}/profiles/qwen3_sm8750_v79_g32/exp0018.env"
export RESULT_PREFIX="exp0018_progressive_a8_qairt249_p19"
export RMSNORM_U8_CONTRACT=1
export BUILD_ANDROID=0
export QAIRT_SDK_ROOT="${artifact_root}/models/qualcomm-sdk/qairt/2.49.0.260730"
export LOCAL_BUILD_BIN="${script_dir}/build-android-arm64-v8a-qnn-qairt249/bin"
export LLAMA_PACKAGE_BUILD="/home/daniuniu/llm_exp_work/qualcomm-sdk/qairt/2.49.0.260730/LLaMAPackage"
export ANDROID_NDK_PATH="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
export ADB_BIN="${ADB_BIN:-/mnt/c/adb/adb.exe}"
export ADB_SERIAL="${ADB_SERIAL:-3B15C8007Z300000}"
export REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/mllm_exp0018_progressive_a8}"
export PROFILE_WRAPPER="${BASH_SOURCE[0]}"
# qnn-profile-viewer expands the trace into hundreds of megabytes of small
# JSON writes.  Keep that work on WSL ext4 and let the canonical profiler copy
# only completed artifacts to the contracted D: result directory.
export PROFILE_WORK_ROOT="${PROFILE_WORK_ROOT:-/home/daniuniu/llm_exp_work/profiling/exp0018_progressive_a8_qairt249_p19}"

exec "${script_dir}/run_qwen3_sm8750_v79_g32_profile.sh"
