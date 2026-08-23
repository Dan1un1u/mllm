#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/ar64_fairness_qairt249/20260824}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_ar64_fairness_qairt249_20260824}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.49.0.260730}"
runner="${RUNNER:-${repo_root}/build-android-arm64-v8a-qnn-qairt249/bin/mllm-qwen3-aot-runner}"
build_bin="$(dirname "${runner}")"
tokenizer="${models_root}/Qwen3-origin/qwen3-tokenizer.json"
ndk_root="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
libomp="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
adb="${ADB_WRAPPER:-${repo_root}/scripts/adb_wsl_path_wrapper.sh}"
serial="${ADB_SERIAL:-3B15C8007Z300000}"
remote="${REMOTE_DIR:-/data/local/tmp/mllm_ar64_fairness_qairt249}"
recover_existing_remote="${RECOVER_EXISTING_REMOTE:-0}"
work_root="${WORK_ROOT:-/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/ar64_fairness_qairt249_20260824}"

case "${remote}" in
  /data/local/tmp/mllm_ar64_fairness_qairt249) ;;
  *) echo "refusing unexpected REMOTE_DIR: ${remote}" >&2; exit 2 ;;
esac
case "${results_root}" in
  /mnt/d/llm_exp/results/qwen3_sm8750_v79_ar64_fairness_qairt249_20260824) ;;
  *) echo "refusing unexpected RESULTS_ROOT: ${results_root}" >&2; exit 2 ;;
esac
[[ "${qairt_root##*/}" == "2.49.0.260730" ]] || { echo "unexpected QAIRT release" >&2; exit 2; }
[[ -x "${runner}" && -s "${tokenizer}" ]] || { echo "runner/tokenizer missing" >&2; exit 2; }
[[ ! -e "${results_root}" ]] || { echo "refusing existing result: ${results_root}" >&2; exit 2; }
[[ "${recover_existing_remote}" == 0 || "${recover_existing_remote}" == 1 ]] || {
  echo "RECOVER_EXISTING_REMOTE must be 0 or 1" >&2
  exit 2
}
case "${work_root}" in
  /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/ar64_fairness_qairt249_*) ;;
  *) echo "unexpected WORK_ROOT: ${work_root}" >&2; exit 2 ;;
esac

context_for() {
  case "$1" in
    a8) printf '%s/a8_p19/qwen3-1.7B-a8_p19-ar64-qairt249.bin\n' "${artifact_root}" ;;
    a16) printf '%s/a16_default/qwen3-1.7B-a16_default-ar64-qairt249.bin\n' "${artifact_root}" ;;
    *) return 2 ;;
  esac
}
config_for() {
  case "$1" in
    a8) printf '%s/examples/qwen3_qnn_aot/config_1.7B_g32.json\n' "${repo_root}" ;;
    a16) printf '%s/examples/qwen3_qnn_aot/config_1.7B_g32_a16.json\n' "${repo_root}" ;;
    *) return 2 ;;
  esac
}
schematic_for() {
  case "$1" in
    a8) printf '%s/a8_p19/schematics/model.0.s64_schematic.bin\n' "${artifact_root}" ;;
    a16) printf '%s/a16_default/schematics/model.0.s64_schematic.bin\n' "${artifact_root}" ;;
    *) return 2 ;;
  esac
}
for variant in a8 a16; do
  [[ -s "$(context_for "${variant}")" && -s "$(config_for "${variant}")" && -s "$(schematic_for "${variant}")" ]] || {
    echo "missing ${variant} artifact" >&2
    exit 2
  }
done

staging="${work_root}.tmp.$$"
publish="${results_root}.tmp.$$"
case "${staging}" in
  /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/ar64_fairness_qairt249_*.tmp.*) ;;
  *) echo "unexpected WSL staging: ${staging}" >&2; exit 2 ;;
esac
case "${publish}" in
  /mnt/d/llm_exp/results/qwen3_sm8750_v79_ar64_fairness_qairt249_20260824.tmp.*) ;;
  *) echo "unexpected result publish staging: ${publish}" >&2; exit 2 ;;
esac
mkdir -p "${staging}"
trap 'rm -rf "${staging}" "${publish}"' ERR INT TERM

prompt_unit='Explain how quantized Transformer inference maps matrix multiplication, attention, KV cache, MLP, normalization, and data movement onto a mobile NPU, with enough concrete detail to analyze hardware bottlenecks. '
prompt=''
for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do prompt+="${prompt_unit}"; done
printf '%s\n' "${prompt}" >"${staging}/prompt.txt"

adb_cmd() { "${adb}" -s "${serial}" "$@"; }
push_file() {
  local source="$1" destination="$2"
  [[ -f "${source}" ]] || { echo "missing source: ${source}" >&2; exit 2; }
  adb_cmd push "${source}" "${destination}" >/dev/null
}

adb_cmd get-state >/dev/null
if [[ "${recover_existing_remote}" == 0 ]]; then
  adb_cmd shell "rm -rf '${remote}' && mkdir -p '${remote}'"
  for library in libMllmCPUBackend.so libMllmQNNBackend.so libMllmRT.so; do
    push_file "${build_bin}/${library}" "${remote}/${library}"
  done
  push_file "${runner}" "${remote}/runner"
  push_file "${libomp}" "${remote}/libomp.so"
  for library in libQnnHtp.so libQnnSystem.so libQnnHtpV79Stub.so libQnnHtpProfilingReader.so \
    libQnnHtpOptraceProfilingReader.so libQnnHtpPrepare.so; do
    push_file "${qairt_root}/lib/aarch64-android/${library}" "${remote}/${library}"
  done
  push_file "${qairt_root}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" "${remote}/libQnnHtpV79Skel.so"
  push_file "${tokenizer}" "${remote}/tokenizer.json"
  push_file "${staging}/prompt.txt" "${remote}/prompt.txt"
  for variant in a8 a16; do
    push_file "$(context_for "${variant}")" "${remote}/${variant}.bin"
    push_file "$(config_for "${variant}")" "${remote}/${variant}.json"
  done
  adb_cmd shell "chmod 755 '${remote}/runner'"
else
  adb_cmd shell "test -d '${remote}/benchmark' && test -d '${remote}/optrace/a16' && test -d '${remote}/optrace/a8'"
fi

execute_benchmark_once() {
  local variant="$1" round="$2" host_dir remote_dir
  host_dir="${staging}/benchmark/round${round}/${variant}"
  remote_dir="${remote}/benchmark/round${round}/${variant}"
  mkdir -p "${host_dir}"
  adb_cmd shell "mkdir -p '${remote_dir}' && cd '${remote}' && env \
    LD_LIBRARY_PATH='${remote}' ADSP_LIBRARY_PATH='${remote}' \
    MLLM_QNN_PROFILE_LEVEL=off MLLM_QNN_PROFILE_DIR='${remote_dir}' \
    ./runner -m '${variant}.bin' -t tokenizer.json -c '${variant}.json' \
      --ar_len 64 --max_new_tokens 1 --perf < prompt.txt" >"${host_dir}/runner.log" 2>&1
  adb_cmd pull "${remote_dir}/qnn_runner_e2e.csv" "${host_dir}/qnn_runner_e2e.csv" >/dev/null
  if adb_cmd shell "test -f '${remote_dir}/qnn_e2e_profile.csv'"; then
    adb_cmd pull "${remote_dir}/qnn_e2e_profile.csv" "${host_dir}/qnn_e2e_profile.csv" >/dev/null
  fi
}

execute_benchmark() {
  local variant="$1" round="$2" attempt
  for attempt in 1 2 3; do
    if execute_benchmark_once "${variant}" "${round}"; then return 0; fi
    mv "${staging}/benchmark/round${round}/${variant}/runner.log" \
      "${staging}/benchmark/round${round}/${variant}/runner.attempt${attempt}.log" 2>/dev/null || true
  done
  return 1
}

if [[ "${recover_existing_remote}" == 0 ]]; then
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-benchmark-before.txt"
  for round in 1 2 3 4 5 6 7 8 9 10; do
    if (( round % 2 )); then order=(a16 a8); else order=(a8 a16); fi
    for variant in "${order[@]}"; do execute_benchmark "${variant}" "${round}"; done
  done
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-benchmark-after.txt"

  for variant in a16 a8; do
    host_dir="${staging}/optrace/${variant}"
    remote_dir="${remote}/optrace/${variant}"
    mkdir -p "${host_dir}"
    adb_cmd shell "mkdir -p '${remote_dir}' && cd '${remote}' && env \
      LD_LIBRARY_PATH='${remote}' ADSP_LIBRARY_PATH='${remote}' \
      MLLM_QNN_PROFILE_LEVEL=optrace MLLM_QNN_PROFILE_WARMUP=0 \
      MLLM_QNN_PROFILE_EVERY=1 MLLM_QNN_PROFILE_MAX_CAPTURES=1 \
      MLLM_QNN_PROFILE_GRAPH='model.0.s64' MLLM_QNN_PROFILE_SERIALIZE=1 \
      MLLM_QNN_PROFILE_DIR='${remote_dir}' \
      ./runner -m '${variant}.bin' -t tokenizer.json -c '${variant}.json' \
        --ar_len 64 --max_new_tokens 1 < prompt.txt" >"${host_dir}/runner.log" 2>&1
    adb_cmd pull "${remote_dir}/." "${host_dir}/" >/dev/null
  done
else
  mkdir -p "${staging}/benchmark" "${staging}/optrace"
  adb_cmd pull "${remote}/benchmark/." "${staging}/benchmark/" >/dev/null
  adb_cmd pull "${remote}/optrace/." "${staging}/optrace/" >/dev/null
  adb_cmd shell dumpsys thermalservice >"${staging}/thermal-recovery.txt"
fi

python3 "${repo_root}/scripts/qnn_ar64_fairness_audit.py" \
  --a8-s1 "${artifact_root}/a8_p19/manifests/model.0.s1_quant_manifest.json" \
  --a8-s64 "${artifact_root}/a8_p19/manifests/model.0.s64_quant_manifest.json" \
  --a16-s1 "${artifact_root}/a16_default/manifests/model.0.s1_quant_manifest.json" \
  --a16-s64 "${artifact_root}/a16_default/manifests/model.0.s64_quant_manifest.json" \
  --report "${staging}/manifest_audit.json" >/dev/null

viewer="${qairt_root}/bin/x86_64-linux-clang/qnn-profile-viewer"
reader="${qairt_root}/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so"
for variant in a16 a8; do
  "${viewer}" --reader "${reader}" \
    --input_log "${staging}/optrace/${variant}/qnn-profiling-data.log" \
    --schematic "$(schematic_for "${variant}")" \
    --output "${staging}/optrace/${variant}/chrometrace.json" \
    >"${staging}/optrace/${variant}/profile-viewer.log" 2>&1
done

{
  printf 'git_commit=%s\ngit_branch=%s\n' \
    "$(git -C "${repo_root}" rev-parse HEAD)" "$(git -C "${repo_root}" branch --show-current)"
  printf 'device_serial=%s\nqairt_release=2.49.0.260730\nar_len=64\n' "${serial}"
  printf 'runner_sha256=%s\ntokenizer_sha256=%s\n' \
    "$(sha256sum "${runner}" | awk '{print $1}')" "$(sha256sum "${tokenizer}" | awk '{print $1}')"
  for variant in a16 a8; do
    printf '%s_context_sha256=%s\n' "${variant}" "$(sha256sum "$(context_for "${variant}")" | awk '{print $1}')"
  done
  printf 'git_status_begin\n'
  git -C "${repo_root}" status --short
  printf 'git_status_end\n'
} >"${staging}/provenance.txt"
adb_cmd shell getprop >"${staging}/device-getprop.txt"

python3 "${repo_root}/scripts/qnn_ar64_fairness_result.py" \
  --result-root "${staging}" --report "${staging}/summary.json" \
  --markdown "${staging}/SUMMARY.md"
mkdir -p "${publish}"
cp -a "${staging}/." "${publish}/"
mv "${publish}" "${results_root}"
rm -rf "${staging}"
trap - ERR INT TERM
adb_cmd shell "rm -rf '${remote}'"
echo "AR64 fairness run complete: ${results_root}"
