#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
source_root="${SOURCE_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_a16_vs_a8_projections/20260821}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_p_point_search/20260821}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_lpbq_p19_lm_head_20260821}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
runner="${RUNNER:-${repo_root}/build-android-arm64-v8a-qnn/bin/mllm-qwen3-lpbq-a16-a8-projection-runner}"
build_bin="$(dirname "${runner}")"
ndk_root="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
libomp="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
adb="${ADB_WRAPPER:-${repo_root}/scripts/adb_wsl_path_wrapper.sh}"
serial="${ADB_SERIAL:-3B15C8007Z300000}"
remote="/data/local/tmp/mllm_lpbq_p19_lm_head"
points=(default 19)

case "${results_root}" in
  /mnt/d/llm_exp/results/qwen3_sm8750_v79_lpbq_p19_lm_head_20260821) ;;
  *) echo "refusing unexpected RESULTS_ROOT: ${results_root}" >&2; exit 2 ;;
esac
[[ "${qairt_root##*/}" == "2.47.0.260601" ]] || exit 2
[[ -x "${runner}" && ! -e "${results_root}" ]] || exit 2
staging="${repo_root}/build-lpbq-p19-lm-head-staging"
mkdir -p "${staging}"

adb_cmd() {
  local attempt
  for attempt in 1 2 3; do
    if "${adb}" -s "${serial}" "$@"; then return 0; fi
    sleep 1
  done
  return 1
}
push_file() { [[ -f "$1" ]] && adb_cmd push "$1" "$2" >/dev/null; }

context_path() {
  if [[ "$1" == default ]]; then
    printf '%s/contexts/a8_lm_head_s1/a8_lm_head_s1.bin\n' "${source_root}"
  else
    printf '%s/contexts/p19_lm_head_s1/p19_lm_head_s1.bin\n' "${artifact_root}"
  fi
}

prepare_device() {
  adb_cmd get-state >/dev/null
  if adb_cmd shell "test -f '${remote}/.prepared'"; then return; fi
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
  push_file "${source_root}/input_lm_head_s1_a8.raw" "${remote}/input.raw"
  for point in "${points[@]}"; do
    push_file "$(context_path "${point}")" "${remote}/contexts/${point}.bin"
  done
  adb_cmd shell "chmod 755 '${remote}/projection-runner' && touch '${remote}/.prepared'"
}

execute_case() {
  local point="$1" run_tag="$2" warmup="$3" iterations="$4" profile="$5"
  local host_dir="${staging}/${run_tag}/${point}" host_tmp remote_dir
  if [[ -s "${host_dir}/output.raw" && -s "${host_dir}/timing.csv" ]]; then return; fi
  host_tmp="${host_dir}.tmp"
  case "${host_tmp}" in "${staging}/"*.tmp) ;; *) exit 2 ;; esac
  rm -rf "${host_tmp}"
  mkdir -p "${host_tmp}"
  remote_dir="${remote}/runs/${run_tag}/${point}"
  adb_cmd shell "mkdir -p '${remote_dir}' && cd '${remote}' && env \
    LD_LIBRARY_PATH='${remote}' ADSP_LIBRARY_PATH='${remote}' \
    MLLM_QNN_PROFILE_WARMUP=0 MLLM_QNN_PROFILE_EVERY=1 \
    MLLM_QNN_PROFILE_MAX_CAPTURES=1 MLLM_QNN_PROFILE_SERIALIZE=1 \
    ./projection-runner --context '${remote}/contexts/${point}.bin' --graph model.0.s1 \
    --input '${remote}/input.raw' --output '${remote_dir}/output.raw' \
    --timing_csv '${remote_dir}/timing.csv' --activation a8 \
    --in_channels 2048 --out_channels 151936 --seq 1 \
    --warmup '${warmup}' --iterations '${iterations}' --profile_level '${profile}'" \
    >"${host_tmp}/run.log" 2>&1
  adb_cmd pull "${remote_dir}/output.raw" "${host_tmp}/output.raw" >/dev/null
  adb_cmd pull "${remote_dir}/timing.csv" "${host_tmp}/timing.csv" >/dev/null
  if [[ "${profile}" == optrace ]]; then adb_cmd pull "${remote_dir}/." "${host_tmp}/" >/dev/null; fi
  [[ ! -e "${host_dir}" ]] || exit 2
  mv "${host_tmp}" "${host_dir}"
}

prepare_device
{
  printf 'git_commit=%s\n' "$(git -C "${repo_root}" rev-parse HEAD)"
  printf 'git_branch=%s\n' "$(git -C "${repo_root}" branch --show-current)"
  printf 'qairt_release=2.47.0.260601\n'
} >"${staging}/provenance.txt"
execute_case default correctness 0 1 off
execute_case 19 correctness 0 1 off
cmp "${staging}/correctness/default/output.raw" "${staging}/correctness/19/output.raw"
for round in $(seq 1 10); do
  if (( round % 2 )); then order=(default 19); else order=(19 default); fi
  for point in "${order[@]}"; do execute_case "${point}" "speed/round${round}" 20 500 off; done
done
for point in "${points[@]}"; do execute_case "${point}" optrace 0 1 optrace; done
adb_cmd shell "rm -rf '${remote}'"

publish="${results_root}.tmp.$$"
case "${publish}" in
  /mnt/d/llm_exp/results/qwen3_sm8750_v79_lpbq_p19_lm_head_20260821.tmp.*) ;;
  *) exit 2 ;;
esac
mkdir -p "${publish}"
trap 'rm -rf "${publish}"' ERR INT TERM
cp -a "${staging}/." "${publish}/"
mv "${publish}" "${results_root}"
rm -rf "${staging}"
trap - ERR INT TERM
echo "LPBQ P19 lm_head run complete: ${results_root}"
