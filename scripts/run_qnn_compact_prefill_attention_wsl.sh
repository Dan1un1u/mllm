#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/compact_prefill_attention_qairt249/20260823_boolmask}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_compact_prefill_attention_qairt249_boolmask_20260823}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.49.0.260730}"
runner="${RUNNER:-${repo_root}/build-android-arm64-v8a-qnn-qairt249/bin/mllm-qwen3-compact-prefill-attention-runner}"
build_bin="$(dirname "${runner}")"
ndk_root="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
libomp="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
adb="${ADB_WRAPPER:-${repo_root}/scripts/adb_wsl_path_wrapper.sh}"
serial="${ADB_SERIAL:-3B15C8007Z300000}"
remote="${REMOTE_DIR:-/data/local/tmp/mllm_compact_prefill_attention_qairt249}"

case "${remote}" in
  /data/local/tmp/mllm_compact_prefill_attention_qairt249) ;;
  *) echo "refusing unexpected REMOTE_DIR: ${remote}" >&2; exit 2 ;;
esac
case "${results_root}" in
  /mnt/d/llm_exp/results/qwen3_sm8750_v79_compact_prefill_attention_qairt249_boolmask_20260823) ;;
  *) echo "refusing unexpected RESULTS_ROOT: ${results_root}" >&2; exit 2 ;;
esac
[[ "${qairt_root##*/}" == "2.49.0.260730" ]] || {
  echo "unexpected QAIRT release: ${qairt_root}" >&2
  exit 2
}
[[ -x "${runner}" ]] || { echo "runner missing: ${runner}" >&2; exit 2; }
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

width_for() {
  case "$1" in
    full) printf '1024\n' ;;
    compact) printf '32\n' ;;
    *) return 2 ;;
  esac
}

context_for() {
  case "$1" in
    full) printf '%s/contexts/full_w1024_s32_p19/full_w1024_s32_p19.bin\n' "${artifact_root}" ;;
    compact) printf '%s/contexts/compact_w32_s32_p19/compact_w32_s32_p19.bin\n' "${artifact_root}" ;;
    *) return 2 ;;
  esac
}

fixture_for() {
  local variant="$1" kind="$2"
  case "${variant}:${kind}" in
    full:key) printf '%s/fixtures/key_full_w1024_u8.raw\n' "${artifact_root}" ;;
    full:value) printf '%s/fixtures/value_full_w1024_u8.raw\n' "${artifact_root}" ;;
    full:mask) printf '%s/fixtures/mask_full_w1024_u8.raw\n' "${artifact_root}" ;;
    compact:key) printf '%s/fixtures/key_compact_w32_u8.raw\n' "${artifact_root}" ;;
    compact:value) printf '%s/fixtures/value_compact_w32_u8.raw\n' "${artifact_root}" ;;
    compact:mask) printf '%s/fixtures/mask_compact_w32_u8.raw\n' "${artifact_root}" ;;
    *) return 2 ;;
  esac
}

prepare_device() {
  adb_cmd get-state >/dev/null
  adb_cmd shell "rm -rf '${remote}' && mkdir -p '${remote}'"
  for library in libMllmCPUBackend.so libMllmQNNBackend.so libMllmRT.so; do
    push_file "${build_bin}/${library}" "${remote}/${library}"
  done
  push_file "${runner}" "${remote}/attention-runner"
  push_file "${libomp}" "${remote}/libomp.so"
  for library in libQnnHtp.so libQnnSystem.so libQnnHtpV79Stub.so; do
    push_file "${qairt_root}/lib/aarch64-android/${library}" "${remote}/${library}"
  done
  push_file "${qairt_root}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" \
    "${remote}/libQnnHtpV79Skel.so"
  push_file "${artifact_root}/fixtures/query_s32_u8.raw" "${remote}/query.raw"
  push_file "${artifact_root}/fixtures/host_expected_output_u8.raw" "${remote}/host_expected.raw"
  for variant in full compact; do
    push_file "$(context_for "${variant}")" "${remote}/${variant}.bin"
    push_file "$(fixture_for "${variant}" key)" "${remote}/${variant}_key.raw"
    push_file "$(fixture_for "${variant}" value)" "${remote}/${variant}_value.raw"
    push_file "$(fixture_for "${variant}" mask)" "${remote}/${variant}_mask.raw"
  done
  adb_cmd shell "chmod 755 '${remote}/attention-runner'"
}

execute_case() {
  local variant="$1" run_tag="$2" warmup="$3" iterations="$4" profile="$5"
  local width host_dir remote_dir
  width="$(width_for "${variant}")"
  host_dir="${staging}/${run_tag}/${variant}"
  remote_dir="${remote}/${run_tag}/${variant}"
  mkdir -p "${host_dir}"
  adb_cmd shell "mkdir -p '${remote_dir}' && cd '${remote}' && env \
    LD_LIBRARY_PATH='${remote}' ADSP_LIBRARY_PATH='${remote}' \
    MLLM_QNN_PROFILE_WARMUP=0 MLLM_QNN_PROFILE_EVERY=1 \
    MLLM_QNN_PROFILE_MAX_CAPTURES=1 MLLM_QNN_PROFILE_SERIALIZE=1 \
    ./attention-runner \
      --context '${remote}/${variant}.bin' --graph 'model.0.s1' \
      --query '${remote}/query.raw' --key '${remote}/${variant}_key.raw' \
      --value '${remote}/${variant}_value.raw' --mask '${remote}/${variant}_mask.raw' \
      --output '${remote_dir}/output.raw' --timing_csv '${remote_dir}/timing.csv' \
      --width '${width}' --warmup '${warmup}' --iterations '${iterations}' \
      --profile_level '${profile}'" >"${host_dir}/run.log" 2>&1
  adb_cmd pull "${remote_dir}/output.raw" "${host_dir}/output.raw" >/dev/null
  adb_cmd pull "${remote_dir}/timing.csv" "${host_dir}/timing.csv" >/dev/null
  if [[ "${profile}" == "optrace" ]]; then
    adb_cmd pull "${remote_dir}/." "${host_dir}/" >/dev/null
  fi
}

run_correctness() {
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-correctness-before.txt"
  for variant in full compact; do
    execute_case "${variant}" correctness/first 0 1 off
    execute_case "${variant}" correctness/repeat 0 1 off
  done
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-correctness-after.txt"
}

run_speed() {
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-speed-before.txt"
  for round in 1 2 3 4 5 6 7 8 9 10; do
    if (( round % 2 )); then order=(full compact); else order=(compact full); fi
    for variant in "${order[@]}"; do
      execute_case "${variant}" "speed/round${round}" 50 1000 off
    done
  done
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-speed-after.txt"
}

run_optrace() {
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-optrace-before.txt"
  for variant in full compact; do
    execute_case "${variant}" optrace 0 1 optrace
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
    printf 'qairt_release=2.49.0.260730\n'
    printf 'finalize_p=19\n'
    printf 'runner_sha256=%s\n' "$(sha256sum "${runner}" | awk '{print $1}')"
    printf 'fixture_report_sha256=%s\n' \
      "$(sha256sum "${artifact_root}/fixture_report.json" | awk '{print $1}')"
  } >"${staging}/provenance.txt"
  adb_cmd shell getprop >"${staging}/device-getprop.txt"
}

prepare_device
record_provenance
run_correctness
run_speed
run_optrace
cp -f "${artifact_root}/fixture_report.json" "${staging}/"
mv "${staging}" "${results_root}"
trap - ERR INT TERM
echo "Compact-prefill attention run complete: ${results_root}"
