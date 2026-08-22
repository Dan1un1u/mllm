#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/native_u8_masked_softmax_qairt249/20260822}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_native_u8_masked_softmax_qairt249_20260822}"
qairt247="${models_root}/qualcomm-sdk/qairt/2.47.0.260601"
qairt249="${models_root}/qualcomm-sdk/qairt/2.49.0.260730"
build247="${repo_root}/build-android-arm64-v8a-qnn/bin"
build249="${repo_root}/build-android-arm64-v8a-qnn-qairt249/bin"
ndk_root="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
libomp="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
adb="${ADB_WRAPPER:-${repo_root}/scripts/adb_wsl_path_wrapper.sh}"
serial="${ADB_SERIAL:-3B15C8007Z300000}"
remote="${REMOTE_DIR:-/data/local/tmp/mllm_native_u8_softmax_qairt249}"

case "${remote}" in
  /data/local/tmp/mllm_native_u8_softmax_qairt249) ;;
  *) echo "refusing unexpected REMOTE_DIR: ${remote}" >&2; exit 2 ;;
esac
[[ ! -e "${results_root}" ]] || { echo "refusing existing result: ${results_root}" >&2; exit 2; }
[[ -s "${artifact_root}/contexts/index.tsv" ]] || { echo "context index missing" >&2; exit 2; }

mkdir -p "${repo_root}/tmp"
staging="$(mktemp -d "${repo_root}/tmp/native_u8_softmax_results.XXXXXX")"
trap 'rm -rf "${staging}"' ERR INT TERM

adb_cmd() { "${adb}" -s "${serial}" "$@" </dev/null; }
push_file() {
  local source="$1" destination="$2"
  [[ -f "${source}" ]] || { echo "missing source: ${source}" >&2; exit 2; }
  adb_cmd push "${source}" "${destination}" >/dev/null
}
sdk_root() { [[ "$1" == 247 ]] && printf '%s\n' "${qairt247}" || printf '%s\n' "${qairt249}"; }
build_root() { [[ "$1" == 247 ]] && printf '%s\n' "${build247}" || printf '%s\n' "${build249}"; }

prepare_device() {
  adb_cmd get-state >/dev/null
  adb_cmd shell "rm -rf '${remote}' && mkdir -p '${remote}/contexts' '${remote}/fixtures'"
  for sdk in 247 249; do
    local sdk_dir qairt build
    sdk_dir="${remote}/sdk${sdk}"
    qairt="$(sdk_root "${sdk}")"
    build="$(build_root "${sdk}")"
    adb_cmd shell "mkdir -p '${sdk_dir}'"
    for library in libMllmCPUBackend.so libMllmQNNBackend.so libMllmRT.so; do
      push_file "${build}/${library}" "${sdk_dir}/${library}"
    done
    push_file "${build}/mllm-qwen3-native-u8-masked-softmax-runner" "${sdk_dir}/runner"
    push_file "${libomp}" "${sdk_dir}/libomp.so"
    for library in libQnnHtp.so libQnnSystem.so libQnnHtpV79Stub.so; do
      push_file "${qairt}/lib/aarch64-android/${library}" "${sdk_dir}/${library}"
    done
    push_file "${qairt}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" \
      "${sdk_dir}/libQnnHtpV79Skel.so"
    adb_cmd shell "chmod 755 '${sdk_dir}/runner'"
  done
  for seq in 1 32; do
    push_file "${artifact_root}/attn_s${seq}_u8.raw" "${remote}/fixtures/attn_s${seq}_u8.raw"
    push_file "${artifact_root}/causal_mask_s${seq}_u8.raw" "${remote}/fixtures/causal_mask_s${seq}_u8.raw"
  done
  while IFS=$'\t' read -r sdk seq point tag context _manifest _schematic; do
    [[ "${sdk}" != sdk ]] || continue
    push_file "${context}" "${remote}/contexts/${tag}.bin"
  done <"${artifact_root}/contexts/index.tsv"
}

execute_case() {
  local tag="$1" run_group="$2" warmup="$3" iterations="$4" profile="$5"
  local sdk seq point remote_dir host_dir
  IFS=_ read -r sdk seq point <<<"${tag}"
  sdk="${sdk#qairt}"
  seq="${seq#s}"
  host_dir="${staging}/${run_group}/${tag}"
  remote_dir="${remote}/runs/${run_group}/${tag}"
  mkdir -p "${host_dir}"
  adb_cmd shell "mkdir -p '${remote_dir}' && cd '${remote}/sdk${sdk}' && env \
    LD_LIBRARY_PATH='${remote}/sdk${sdk}' ADSP_LIBRARY_PATH='${remote}/sdk${sdk}' \
    MLLM_QNN_PROFILE_WARMUP=0 MLLM_QNN_PROFILE_EVERY=1 \
    MLLM_QNN_PROFILE_MAX_CAPTURES=1 MLLM_QNN_PROFILE_SERIALIZE=1 \
    ./runner --context '${remote}/contexts/${tag}.bin' --graph 'model.0.s1' \
    --attn '${remote}/fixtures/attn_s${seq}_u8.raw' \
    --mask '${remote}/fixtures/causal_mask_s${seq}_u8.raw' \
    --output '${remote_dir}/output.raw' --timing_csv '${remote_dir}/timing.csv' \
    --seq '${seq}' --warmup '${warmup}' --iterations '${iterations}' \
    --profile_level '${profile}'" >"${host_dir}/run.log" 2>&1
  adb_cmd pull "${remote_dir}/output.raw" "${host_dir}/output.raw" >/dev/null
  adb_cmd pull "${remote_dir}/timing.csv" "${host_dir}/timing.csv" >/dev/null
  if [[ "${profile}" == optrace ]]; then
    adb_cmd pull "${remote_dir}/." "${host_dir}/" >/dev/null
  fi
}

record_provenance() {
  {
    printf 'git_commit=%s\n' "$(git -C "${repo_root}" rev-parse HEAD)"
    printf 'git_branch=%s\n' "$(git -C "${repo_root}" branch --show-current)"
    printf 'device_serial=%s\n' "${serial}"
    printf 'qairt247=%s\nqairt249=%s\n' "${qairt247}" "${qairt249}"
    printf 'sdk247_runner_sha256=%s\n' "$(sha256sum "${build247}/mllm-qwen3-native-u8-masked-softmax-runner" | awk '{print $1}')"
    printf 'sdk249_runner_sha256=%s\n' "$(sha256sum "${build249}/mllm-qwen3-native-u8-masked-softmax-runner" | awk '{print $1}')"
    printf 'git_status_begin\n'; git -C "${repo_root}" status --short; printf 'git_status_end\n'
  } >"${staging}/provenance.txt"
  adb_cmd shell getprop >"${staging}/device-getprop.txt"
}

run_stage1() {
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-stage1-before.txt"
  while IFS=$'\t' read -r sdk seq point tag _context _manifest _schematic; do
    [[ "${sdk}" != sdk ]] || continue
    execute_case "${tag}" stage1 20 300 off
  done <"${artifact_root}/contexts/index.tsv"
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-stage1-after.txt"
  python3 "${repo_root}/scripts/qnn_native_u8_masked_softmax_summary.py" rank \
    --results-root "${staging}" --output "${staging}/stage1_rank.json" \
    --top-file "${staging}/top_cases.txt" --top 3
}

run_stage2() {
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-stage2-before.txt"
  for round in 1 2 3 4 5 6 7; do
    mapfile -t cases <"${staging}/top_cases.txt"
    if (( round % 2 == 0 )); then mapfile -t cases < <(printf '%s\n' "${cases[@]}" | tac); fi
    for tag in "${cases[@]}"; do execute_case "${tag}" "stage2/round${round}" 50 1000 off; done
  done
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-stage2-after.txt"
  python3 "${repo_root}/scripts/qnn_native_u8_masked_softmax_summary.py" select \
    --results-root "${staging}" --output "${staging}/stage2_rank.json" \
    --winner-file "${staging}/winner_cases.txt"
}

run_correctness_and_optrace() {
  while read -r tag; do
    execute_case "${tag}" correctness/first 0 1 off
    execute_case "${tag}" correctness/repeat 0 1 off
    execute_case "${tag}" optrace 0 1 optrace
  done <"${staging}/winner_cases.txt"
  python3 "${repo_root}/scripts/qnn_native_u8_masked_softmax_summary.py" final \
    --results-root "${staging}" --artifact-root "${artifact_root}" \
    --output "${staging}/summary.json"
}

prepare_device
record_provenance
run_stage1
run_stage2
run_correctness_and_optrace

publish="${results_root}.tmp.$$"
[[ ! -e "${publish}" ]] || { echo "publish stage exists: ${publish}" >&2; exit 2; }
mkdir -p "${publish}"
cp -a "${staging}/." "${publish}/"
mv "${publish}" "${results_root}"
rm -rf "${staging}"
trap - ERR INT TERM
echo "Native-U8 masked-Softmax run complete: ${results_root}"
