#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_a16_vs_a8_projections/20260821}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_lpbq_a16_vs_a8_projections_20260821}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
runner="${RUNNER:-${repo_root}/build-android-arm64-v8a-qnn/bin/mllm-qwen3-lpbq-a16-a8-projection-runner}"
build_bin="$(dirname "${runner}")"
ndk_root="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
libomp="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
adb="${ADB_WRAPPER:-${repo_root}/scripts/adb_wsl_path_wrapper.sh}"
serial="${ADB_SERIAL:-3B15C8007Z300000}"
remote="${REMOTE_DIR:-/data/local/tmp/mllm_lpbq_a16_a8_projection}"

case "${remote}" in
  /data/local/tmp/mllm_lpbq_a16_a8_projection) ;;
  *) echo "refusing unexpected REMOTE_DIR: ${remote}" >&2; exit 2 ;;
esac
case "${results_root}" in
  /mnt/d/llm_exp/results/qwen3_sm8750_v79_lpbq_a16_vs_a8_projections_20260821) ;;
  *) echo "refusing unexpected RESULTS_ROOT: ${results_root}" >&2; exit 2 ;;
esac
[[ "${qairt_root##*/}" == "2.47.0.260601" ]] || {
  echo "unexpected QAIRT release: ${qairt_root}" >&2
  exit 2
}
[[ ! -e "${results_root}" ]] || { echo "refusing existing result: ${results_root}" >&2; exit 2; }

staging="${results_root}.tmp.$$"
mkdir -p "${staging}"
trap 'rm -rf "${staging}"' ERR INT TERM

adb_cmd() { "${adb}" -s "${serial}" "$@"; }

push_file() {
  local source="$1" destination="$2"
  [[ -f "${source}" ]] || { echo "missing source: ${source}" >&2; exit 2; }
  adb_cmd push "${source}" "${destination}" >/dev/null
}

shape() {
  case "$1" in
    gate_proj|up_proj) printf '2048 6144\n' ;;
    down_proj) printf '6144 2048\n' ;;
    lm_head) printf '2048 151936\n' ;;
    *) return 2 ;;
  esac
}

tag() { printf '%s_%s_s%s\n' "$1" "$2" "$3"; }

context_path() {
  local case_tag
  case_tag="$(tag "$1" "$2" "$3")"
  printf '%s/contexts/%s/%s.bin\n' "${artifact_root}" "${case_tag}" "${case_tag}"
}

input_path() {
  printf '%s/input_%s_s%s_%s.raw\n' "${artifact_root}" "$2" "$3" "$1"
}

prepare_device() {
  adb_cmd get-state >/dev/null
  adb_cmd shell "rm -rf '${remote}' && mkdir -p '${remote}'"
  for library in libMllmCPUBackend.so libMllmQNNBackend.so libMllmRT.so; do
    push_file "${build_bin}/${library}" "${remote}/${library}"
  done
  push_file "${runner}" "${remote}/projection-runner"
  push_file "${libomp}" "${remote}/libomp.so"
  for library in libQnnHtp.so libQnnSystem.so libQnnHtpV79Stub.so; do
    push_file "${qairt_root}/lib/aarch64-android/${library}" "${remote}/${library}"
  done
  push_file "${qairt_root}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" \
    "${remote}/libQnnHtpV79Skel.so"
  for activation in a16 a8; do
    for projection in gate_proj up_proj down_proj lm_head; do
      for seq in 1 32; do
        local case_tag
        case_tag="$(tag "${activation}" "${projection}" "${seq}")"
        push_file "$(context_path "${activation}" "${projection}" "${seq}")" \
          "${remote}/${case_tag}.bin"
        push_file "$(input_path "${activation}" "${projection}" "${seq}")" \
          "${remote}/input_${case_tag}.raw"
      done
    done
  done
  adb_cmd shell "chmod 755 '${remote}/projection-runner'"
}

execute_case() {
  local activation="$1" projection="$2" seq="$3" run_tag="$4" warmup="$5" iterations="$6" profile="$7"
  local case_tag remote_dir host_dir in_channels out_channels
  case_tag="$(tag "${activation}" "${projection}" "${seq}")"
  read -r in_channels out_channels <<<"$(shape "${projection}")"
  host_dir="${staging}/${run_tag}/${case_tag}"
  remote_dir="${remote}/${run_tag}/${case_tag}"
  mkdir -p "${host_dir}"
  adb_cmd shell "mkdir -p '${remote_dir}' && cd '${remote}' && env \
    LD_LIBRARY_PATH='${remote}' ADSP_LIBRARY_PATH='${remote}' \
    MLLM_QNN_PROFILE_WARMUP=0 MLLM_QNN_PROFILE_EVERY=1 \
    MLLM_QNN_PROFILE_MAX_CAPTURES=1 MLLM_QNN_PROFILE_SERIALIZE=1 \
    ./projection-runner \
    --context '${remote}/${case_tag}.bin' --graph 'model.0.s${seq}' \
    --input '${remote}/input_${case_tag}.raw' --output '${remote_dir}/output.raw' \
    --timing_csv '${remote_dir}/timing.csv' --activation '${activation}' \
    --in_channels '${in_channels}' --out_channels '${out_channels}' --seq '${seq}' \
    --warmup '${warmup}' --iterations '${iterations}' --profile_level '${profile}'" \
    >"${host_dir}/run.log" 2>&1
  adb_cmd pull "${remote_dir}/output.raw" "${host_dir}/output.raw" >/dev/null
  adb_cmd pull "${remote_dir}/timing.csv" "${host_dir}/timing.csv" >/dev/null
  if [[ "${profile}" == "optrace" ]]; then
    adb_cmd pull "${remote_dir}/." "${host_dir}/" >/dev/null
  fi
}

run_correctness() {
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-correctness-before.txt"
  for activation in a16 a8; do
    for projection in gate_proj up_proj down_proj lm_head; do
      for seq in 1 32; do
        execute_case "${activation}" "${projection}" "${seq}" correctness/first 0 1 off
        execute_case "${activation}" "${projection}" "${seq}" correctness/repeat 0 1 off
      done
    done
  done
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-correctness-after.txt"
}

run_speed() {
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-speed-before.txt"
  for round in 1 2 3 4 5; do
    if (( round % 2 )); then order=(a16 a8); else order=(a8 a16); fi
    for projection in gate_proj up_proj down_proj lm_head; do
      for seq in 1 32; do
        for activation in "${order[@]}"; do
          execute_case "${activation}" "${projection}" "${seq}" "speed/round${round}" 20 500 off
        done
      done
    done
  done
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-speed-after.txt"
}

run_optrace() {
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-optrace-before.txt"
  for activation in a16 a8; do
    for projection in gate_proj up_proj down_proj lm_head; do
      for seq in 1 32; do
        execute_case "${activation}" "${projection}" "${seq}" optrace 0 1 optrace
      done
    done
  done
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-optrace-after.txt"
}

record_provenance() {
  {
    printf 'git_commit=%s\n' "$(git -C "${repo_root}" rev-parse HEAD)"
    printf 'git_branch=%s\n' "$(git -C "${repo_root}" branch --show-current)"
    printf 'git_status_begin\n'
    git -C "${repo_root}" status --short
    printf 'git_status_end\n'
    printf 'device_serial=%s\n' "${serial}"
    printf 'qairt_release=2.47.0.260601\n'
    printf 'runner_sha256=%s\n' "$(sha256sum "${runner}" | awk '{print $1}')"
    printf 'compact_artifact_sha256=%s\n' \
      "$(sha256sum "${artifact_root}/qwen3-lpbq-a16-a8-projections.mllm" | awk '{print $1}')"
  } >"${staging}/provenance.txt"
  adb_cmd shell getprop >"${staging}/device-getprop.txt"
}

prepare_device
record_provenance
run_correctness
run_speed
run_optrace
mv "${staging}" "${results_root}"
trap - ERR INT TERM
echo "LPBQ A16/A8 projection run complete: ${results_root}"
