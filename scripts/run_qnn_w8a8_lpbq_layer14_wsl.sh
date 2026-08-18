#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/w8a8_vs_lpbq_layer14_mlp/20260818}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_layer14_mlp_w8a8_vs_lpbq_20260818}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
runner="${RUNNER:-${repo_root}/build-android-arm64-v8a-qnn/bin/mllm-qwen3-layer14-mlp-runner}"
build_bin="$(dirname "${runner}")"
ndk_root="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
libomp="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
adb="${ADB_WRAPPER:-${repo_root}/scripts/adb_wsl_path_wrapper.sh}"
serial="${ADB_SERIAL:-3B15C8007Z300000}"
remote="${REMOTE_DIR:-/data/local/tmp/mllm_w8a8_lpbq_layer14}"

case "${remote}" in
  /data/local/tmp/mllm_w8a8_lpbq_layer14) ;;
  *) echo "refusing unexpected REMOTE_DIR: ${remote}" >&2; exit 2 ;;
esac
case "${results_root}" in
  /mnt/d/llm_exp/results/qwen3_sm8750_v79_layer14_mlp_w8a8_vs_lpbq_20260818) ;;
  *) echo "refusing unexpected RESULTS_ROOT: ${results_root}" >&2; exit 2 ;;
esac
if [[ -e "${results_root}" ]]; then
  echo "refusing to overwrite existing result: ${results_root}" >&2
  exit 2
fi
staging="${results_root}.tmp.$$"
mkdir -p "${staging}"
trap 'rm -rf "${staging}"' ERR INT TERM

adb_cmd() { "${adb}" -s "${serial}" "$@"; }

push_file() {
  local source="$1" destination="$2"
  [[ -f "${source}" ]] || { echo "missing source: ${source}" >&2; exit 2; }
  adb_cmd push "${source}" "${destination}" >/dev/null
}

context_path() {
  local variant="$1" seq="$2"
  printf '%s/contexts/%s_s%s/%s_s%s.bin\n' "${artifact_root}" "${variant}" "${seq}" "${variant}" "${seq}"
}

prepare_device() {
  adb_cmd get-state >/dev/null
  adb_cmd shell "rm -rf '${remote}' && mkdir -p '${remote}/timing' '${remote}/outputs' '${remote}/optrace'"
  for library in libMllmCPUBackend.so libMllmQNNBackend.so libMllmRT.so; do
    push_file "${build_bin}/${library}" "${remote}/${library}"
  done
  push_file "${runner}" "${remote}/mllm-qwen3-layer14-mlp-runner"
  push_file "${libomp}" "${remote}/libomp.so"
  for library in libQnnHtp.so libQnnSystem.so libQnnHtpV79Stub.so; do
    push_file "${qairt_root}/lib/aarch64-android/${library}" "${remote}/${library}"
  done
  push_file "${qairt_root}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" \
    "${remote}/libQnnHtpV79Skel.so"
  for variant in lpbq w8a8; do
    for seq in 1 32; do
      push_file "$(context_path "${variant}" "${seq}")" "${remote}/${variant}_s${seq}.bin"
    done
  done
  for seq in 1 32; do
    push_file "${artifact_root}/input_s${seq}.raw" "${remote}/input_s${seq}.raw"
  done
  adb_cmd shell "chmod 755 '${remote}/mllm-qwen3-layer14-mlp-runner'"
}

execute_case() {
  local variant="$1" seq="$2" tag="$3" warmup="$4" iterations="$5" profile_level="$6"
  local host_dir="${staging}/${tag}/${variant}_s${seq}"
  local remote_dir="${remote}/${tag}/${variant}_s${seq}"
  mkdir -p "${host_dir}"
  adb_cmd shell "mkdir -p '${remote_dir}' && cd '${remote}' && env \
    LD_LIBRARY_PATH='${remote}' ADSP_LIBRARY_PATH='${remote}' \
    MLLM_QNN_PROFILE_WARMUP=0 MLLM_QNN_PROFILE_EVERY=1 \
    MLLM_QNN_PROFILE_MAX_CAPTURES=1 MLLM_QNN_PROFILE_SERIALIZE=1 \
    ./mllm-qwen3-layer14-mlp-runner \
    --context '${remote}/${variant}_s${seq}.bin' --graph 'model.0.s${seq}' \
    --input '${remote}/input_s${seq}.raw' --output '${remote_dir}/output.raw' \
    --timing_csv '${remote_dir}/timing.csv' --seq '${seq}' \
    --warmup '${warmup}' --iterations '${iterations}' --profile_level '${profile_level}'" \
    >"${host_dir}/run.log" 2>&1
  adb_cmd pull "${remote_dir}/output.raw" "${host_dir}/output.raw" >/dev/null
  adb_cmd pull "${remote_dir}/timing.csv" "${host_dir}/timing.csv" >/dev/null
  if [[ "${profile_level}" == "optrace" ]]; then
    adb_cmd pull "${remote_dir}/." "${host_dir}/" >/dev/null
  fi
}

run_correctness() {
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-correctness-before.txt"
  for variant in lpbq w8a8; do
    for seq in 1 32; do
      execute_case "${variant}" "${seq}" "correctness/first" 0 1 off
      execute_case "${variant}" "${seq}" "correctness/repeat" 0 1 off
    done
  done
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-correctness-after.txt"
}

run_paired_speed() {
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-speed-before.txt"
  for round in 1 2 3 4 5; do
    if (( round % 2 )); then
      order=(lpbq w8a8)
    else
      order=(w8a8 lpbq)
    fi
    for seq in 1 32; do
      for variant in "${order[@]}"; do
        execute_case "${variant}" "${seq}" "speed/round${round}" 20 500 off
      done
    done
  done
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-speed-after.txt"
}

run_optrace() {
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-optrace-before.txt"
  for variant in lpbq w8a8; do
    for seq in 1 32; do
      execute_case "${variant}" "${seq}" "optrace" 0 1 optrace
    done
  done
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-optrace-after.txt"
}

record_provenance() {
  {
    printf 'git_commit=%s\n' "$(git -C "${repo_root}" rev-parse HEAD)"
    printf 'git_status_begin\n'
    git -C "${repo_root}" status --short
    printf 'git_status_end\n'
    printf 'device_serial=%s\n' "${serial}"
    printf 'qairt_release=2.47.0.260601\n'
    printf 'runner_sha256=%s\n' "$(sha256sum "${runner}" | awk '{print $1}')"
    for variant in lpbq w8a8; do
      for seq in 1 32; do
        printf '%s_s%s_context_sha256=%s\n' "${variant}" "${seq}" \
          "$(sha256sum "$(context_path "${variant}" "${seq}")" | awk '{print $1}')"
      done
    done
  } >"${staging}/provenance.txt"
  adb_cmd shell getprop >"${staging}/device-getprop.txt"
}

prepare_device
record_provenance
run_correctness
run_paired_speed
run_optrace
mv "${staging}" "${results_root}"
trap - ERR INT TERM
echo "W8A8 versus LPBQ layer-14 MLP run complete: ${results_root}"
