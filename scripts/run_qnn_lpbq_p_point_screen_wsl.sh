#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
source_root="${SOURCE_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_a16_vs_a8_projections/20260821}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_p_point_search/20260821}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_lpbq_p_point_search_20260821}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
runner="${RUNNER:-${repo_root}/build-android-arm64-v8a-qnn/bin/mllm-qwen3-lpbq-a16-a8-projection-runner}"
build_bin="$(dirname "${runner}")"
ndk_root="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
libomp="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
adb="${ADB_WRAPPER:-${repo_root}/scripts/adb_wsl_path_wrapper.sh}"
serial="${ADB_SERIAL:-3B15C8007Z300000}"
remote="${REMOTE_DIR:-/data/local/tmp/mllm_lpbq_p_point_search}"
points=(default 0 1 2 3 4 5 6 8 13 15 16 17 19 20 21 22 23)
projections=(gate_proj up_proj)

case "${remote}" in
  /data/local/tmp/mllm_lpbq_p_point_search) ;;
  *) echo "refusing unexpected REMOTE_DIR: ${remote}" >&2; exit 2 ;;
esac
case "${results_root}" in
  /mnt/d/llm_exp/results/qwen3_sm8750_v79_lpbq_p_point_search_20260821) ;;
  *) echo "refusing unexpected RESULTS_ROOT: ${results_root}" >&2; exit 2 ;;
esac
[[ "${qairt_root##*/}" == "2.47.0.260601" ]] || {
  echo "unexpected QAIRT release: ${qairt_root}" >&2
  exit 2
}
[[ -x "${runner}" ]] || { echo "runner missing: ${runner}" >&2; exit 2; }
[[ ! -e "${results_root}" ]] || { echo "refusing existing result: ${results_root}" >&2; exit 2; }

staging="${STAGING_ROOT:-${repo_root}/build-lpbq-p-point-screen-staging}"
case "${staging}" in
  "${repo_root}/build-lpbq-p-point-screen-staging") ;;
  *) echo "refusing unexpected staging directory: ${staging}" >&2; exit 2 ;;
esac
mkdir -p "${staging}"

adb_cmd() {
  local attempt
  for attempt in 1 2 3; do
    if "${adb}" -s "${serial}" "$@"; then return 0; fi
    sleep 1
  done
  return 1
}

push_file() {
  local source="$1" destination="$2"
  [[ -f "${source}" ]] || { echo "missing source: ${source}" >&2; exit 2; }
  adb_cmd push "${source}" "${destination}" >/dev/null
}

context_path() {
  local point="$1" projection="$2"
  if [[ "${point}" == default ]]; then
    printf '%s/contexts/a8_%s_s1/a8_%s_s1.bin\n' "${source_root}" "${projection}" "${projection}"
  else
    printf '%s/contexts/p%s_%s_s1/p%s_%s_s1.bin\n' \
      "${artifact_root}" "${point}" "${projection}" "${point}" "${projection}"
  fi
}

prepare_device() {
  adb_cmd get-state >/dev/null
  if adb_cmd shell "test -f '${remote}/.prepared'"; then
    echo "device staging already prepared"
    return
  fi
  adb_cmd shell "rm -rf '${remote}' && mkdir -p '${remote}/contexts' '${remote}/runs'"
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
  for projection in "${projections[@]}"; do
    push_file "${source_root}/input_${projection}_s1_a8.raw" \
      "${remote}/input_${projection}.raw"
    for point in "${points[@]}"; do
      push_file "$(context_path "${point}" "${projection}")" \
        "${remote}/contexts/${point}_${projection}.bin"
    done
  done
  adb_cmd shell "chmod 755 '${remote}/projection-runner'"
  adb_cmd shell "touch '${remote}/.prepared'"
}

execute_case() {
  local point="$1" projection="$2" run_tag="$3" warmup="$4" iterations="$5"
  local host_dir remote_dir host_tmp
  host_dir="${staging}/${run_tag}/${point}_${projection}"
  remote_dir="${remote}/runs/${run_tag}/${point}_${projection}"
  if [[ -s "${host_dir}/output.raw" && -s "${host_dir}/timing.csv" ]]; then
    echo "already complete: ${run_tag}/${point}_${projection}"
    return
  fi
  host_tmp="${host_dir}.tmp"
  case "${host_tmp}" in
    "${staging}/"*.tmp) ;;
    *) echo "refusing unexpected case staging: ${host_tmp}" >&2; exit 2 ;;
  esac
  rm -rf "${host_tmp}"
  mkdir -p "${host_tmp}"
  adb_cmd shell "mkdir -p '${remote_dir}' && cd '${remote}' && env \
    LD_LIBRARY_PATH='${remote}' ADSP_LIBRARY_PATH='${remote}' \
    ./projection-runner \
    --context '${remote}/contexts/${point}_${projection}.bin' --graph 'model.0.s1' \
    --input '${remote}/input_${projection}.raw' --output '${remote_dir}/output.raw' \
    --timing_csv '${remote_dir}/timing.csv' --activation a8 \
    --in_channels 2048 --out_channels 6144 --seq 1 \
    --warmup '${warmup}' --iterations '${iterations}' --profile_level off" \
    >"${host_tmp}/run.log" 2>&1
  adb_cmd pull "${remote_dir}/output.raw" "${host_tmp}/output.raw" >/dev/null
  adb_cmd pull "${remote_dir}/timing.csv" "${host_tmp}/timing.csv" >/dev/null
  [[ -s "${host_tmp}/output.raw" && -s "${host_tmp}/timing.csv" ]]
  [[ ! -e "${host_dir}" ]] || { echo "refusing partial case directory: ${host_dir}" >&2; exit 2; }
  mv "${host_tmp}" "${host_dir}"
}

run_correctness() {
  for projection in "${projections[@]}"; do
    for point in "${points[@]}"; do
      execute_case "${point}" "${projection}" correctness 0 1
    done
    for point in "${points[@]}"; do
      cmp "${staging}/correctness/default_${projection}/output.raw" \
          "${staging}/correctness/${point}_${projection}/output.raw"
    done
  done
  find "${staging}/correctness" -name output.raw -print0 | sort -z | xargs -0 sha256sum \
    >"${staging}/correctness.sha256"
}

run_speed() {
  for round in 1 2 3 4 5; do
    if (( round % 2 )); then
      order=("${points[@]}")
    else
      order=()
      for ((index=${#points[@]} - 1; index >= 0; --index)); do order+=("${points[index]}"); done
    fi
    for projection in "${projections[@]}"; do
      for point in "${order[@]}"; do
        execute_case "${point}" "${projection}" "speed/round${round}" 20 500
      done
    done
  done
}

record_provenance() {
  [[ ! -e "${staging}/provenance.txt" ]] || return
  {
    printf 'git_commit=%s\n' "$(git -C "${repo_root}" rev-parse HEAD)"
    printf 'git_branch=%s\n' "$(git -C "${repo_root}" branch --show-current)"
    printf 'git_status_begin\n'
    git -C "${repo_root}" status --short
    printf 'git_status_end\n'
    printf 'device_serial=%s\n' "${serial}"
    printf 'qairt_release=2.47.0.260601\n'
    printf 'runner_sha256=%s\n' "$(sha256sum "${runner}" | awk '{print $1}')"
  } >"${staging}/provenance.txt"
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-before.txt"
}

prepare_device
record_provenance
run_correctness
run_speed
adb_cmd shell dumpsys thermalservice >"${staging}/thermal-after.txt"
adb_cmd shell "rm -rf '${remote}'"
publish="${results_root}.tmp.$$"
case "${publish}" in
  /mnt/d/llm_exp/results/qwen3_sm8750_v79_lpbq_p_point_search_20260821.tmp.*) ;;
  *) echo "refusing unexpected publish directory: ${publish}" >&2; exit 2 ;;
esac
[[ ! -e "${publish}" ]] || { echo "refusing existing publish directory: ${publish}" >&2; exit 2; }
mkdir -p "${publish}"
trap 'rm -rf "${publish}"' ERR INT TERM
cp -a "${staging}/." "${publish}/"
mv "${publish}" "${results_root}"
rm -rf "${staging}"
trap - ERR INT TERM
echo "LPBQ P-point screen complete: ${results_root}"
