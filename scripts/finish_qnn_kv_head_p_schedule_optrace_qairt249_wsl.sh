#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/kv_head_p_schedule_qairt249/20260824}"
source_root="${artifact_root}/source"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_kv_head_p_schedule_qairt249_20260824}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.49.0.260730}"
runner="${RUNNER:-${repo_root}/build-android-arm64-v8a-qnn-qairt249/bin/mllm-qwen3-lpbq-a16-a8-projection-runner}"
build_bin="$(dirname "${runner}")"
ndk_root="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
libomp="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
adb="${ADB_WRAPPER:-${repo_root}/scripts/adb_wsl_path_wrapper.sh}"
serial="${ADB_SERIAL:-3B15C8007Z300000}"
remote="${REMOTE_DIR:-/data/local/tmp/mllm_kv_head_p_schedule_optrace_qairt249}"
work_root="${WORK_ROOT:-/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/kv_head_p_schedule_optrace_qairt249_20260824}"

case "${artifact_root}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/kv_head_p_schedule_qairt249/20260824) ;;
  *) echo "refusing unexpected ARTIFACT_ROOT: ${artifact_root}" >&2; exit 2 ;;
esac
case "${results_root}" in
  /mnt/d/llm_exp/results/qwen3_sm8750_v79_kv_head_p_schedule_qairt249_20260824) ;;
  *) echo "refusing unexpected RESULTS_ROOT: ${results_root}" >&2; exit 2 ;;
esac
case "${remote}" in /data/local/tmp/mllm_kv_head_p_schedule_optrace_qairt249) ;;
  *) echo "refusing unexpected REMOTE_DIR: ${remote}" >&2; exit 2 ;;
esac
case "${work_root}" in
  /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/kv_head_p_schedule_optrace_qairt249_*) ;;
  *) echo "refusing unexpected WORK_ROOT: ${work_root}" >&2; exit 2 ;;
esac
[[ "${qairt_root##*/}" == "2.49.0.260730" ]] || { echo "unexpected QAIRT release" >&2; exit 2; }
[[ -x "${runner}" && -x "${adb}" && -s "${libomp}" ]] || { echo "runtime missing" >&2; exit 2; }
[[ -s "${results_root}/winners.tsv" && -s "${results_root}/summary.screen.json" ]] || {
  echo "published screen result is incomplete" >&2
  exit 2
}

staging="${work_root}.tmp.$$"
publish="${results_root}.optrace.tmp.$$"
backup="${results_root}.pre_optrace.$$"
case "${staging}" in
  /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/kv_head_p_schedule_optrace_qairt249_*.tmp.*) ;;
  *) echo "unexpected WSL staging: ${staging}" >&2; exit 2 ;;
esac
case "${publish}" in "${results_root}.optrace.tmp."*) ;;
  *) echo "unexpected publish staging: ${publish}" >&2; exit 2 ;;
esac
case "${backup}" in "${results_root}.pre_optrace."*) ;;
  *) echo "unexpected result backup: ${backup}" >&2; exit 2 ;;
esac
[[ ! -e "${staging}" && ! -e "${publish}" && ! -e "${backup}" ]] || {
  echo "refusing existing completion staging" >&2
  exit 2
}
mkdir -p "${staging}"
cp -a "${results_root}/." "${staging}/"
trap 'echo "Optrace completion failed; retained WSL staging: ${staging}" >&2' ERR INT TERM

adb_cmd() {
  local attempt
  for attempt in 1 2 3; do
    if "${adb}" -s "${serial}" "$@" </dev/null; then return 0; fi
    sleep 1
  done
  return 1
}

push_file() {
  [[ -f "$1" ]] || { echo "missing source: $1" >&2; exit 2; }
  adb_cmd push "$1" "$2" >/dev/null
}

case_tag() { printf '%s_s%s_%s\n' "$1" "$2" "$3"; }

schematic_path() {
  printf '%s/contexts/%s/schematics/model.0.s%s_schematic.bin\n' \
    "${artifact_root}" "$(case_tag "$1" "$2" "$3")" "$2"
}

adb_cmd get-state >/dev/null
adb_cmd shell "rm -rf '${remote}' && mkdir -p '${remote}/runs'"
for library in libMllmCPUBackend.so libMllmQNNBackend.so libMllmRT.so; do
  push_file "${build_bin}/${library}" "${remote}/${library}"
done
push_file "${runner}" "${remote}/projection-runner"
push_file "${libomp}" "${remote}/libomp.so"
for library in libQnnHtp.so libQnnSystem.so libQnnHtpV79Stub.so \
  libQnnHtpProfilingReader.so libQnnHtpOptraceProfilingReader.so; do
  push_file "${qairt_root}/lib/aarch64-android/${library}" "${remote}/${library}"
done
push_file "${qairt_root}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" \
  "${remote}/libQnnHtpV79Skel.so"
adb_cmd shell "chmod 755 '${remote}/projection-runner'"

viewer="${qairt_root}/bin/x86_64-linux-clang/qnn-profile-viewer"
reader="${qairt_root}/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so"
exec 3<"${staging}/winners.tsv"
while IFS=$'\t' read -r projection sequence winner <&3; do
  profile_points=(19)
  if [[ "${winner}" != 19 ]]; then profile_points+=("${winner}"); fi
  for point in "${profile_points[@]}"; do
    tag="$(case_tag "${projection}" "${sequence}" "${point}")"
    profile_dir="${staging}/optrace/${tag}"
    if [[ -s "${profile_dir}/qnn_detail_profile.txt" \
          && -s "${profile_dir}/chrometrace.json" ]]; then
      echo "already complete: ${tag}"
      continue
    fi
    context="${artifact_root}/contexts/${tag}/context.bin"
    input="${source_root}/input_s${sequence}_a8.raw"
    remote_dir="${remote}/runs/${tag}"
    mkdir -p "${profile_dir}"
    push_file "${context}" "${remote}/context.bin"
    push_file "${input}" "${remote}/input.raw"
    adb_cmd shell "rm -rf '${remote_dir}' && mkdir -p '${remote_dir}' && cd '${remote}' && env \
      LD_LIBRARY_PATH='${remote}' ADSP_LIBRARY_PATH='${remote}' \
      MLLM_QNN_PROFILE_WARMUP=0 MLLM_QNN_PROFILE_EVERY=1 \
      MLLM_QNN_PROFILE_MAX_CAPTURES=1 MLLM_QNN_PROFILE_SERIALIZE=1 \
      ./projection-runner --context '${remote}/context.bin' --graph 'model.0.s${sequence}' \
        --input '${remote}/input.raw' --output '${remote_dir}/output.raw' \
        --timing_csv '${remote_dir}/timing.csv' --activation a8 \
        --in_channels 2048 --out_channels 128 --seq '${sequence}' \
        --warmup 0 --iterations 1 --profile_level optrace" >"${profile_dir}/run.log" 2>&1
    adb_cmd pull "${remote_dir}/." "${profile_dir}/" >/dev/null
    cmp "${staging}/correctness/${tag}/output.raw" "${profile_dir}/output.raw"
    "${viewer}" --reader "${reader}" \
      --input_log "${profile_dir}/qnn-profiling-data.log" \
      --schematic "$(schematic_path "${projection}" "${sequence}" "${point}")" \
      --output "${profile_dir}/chrometrace.json" \
      >"${profile_dir}/profile-viewer.log" 2>&1
    echo "completed: ${tag}"
  done
done
exec 3<&-

python3 "${repo_root}/scripts/qnn_kv_head_p_schedule_result.py" \
  --result-root "${staging}" --source-root "${source_root}" \
  --report "${staging}/summary.json" --csv "${staging}/ranking.csv" \
  --markdown "${staging}/SUMMARY.md" >"${staging}/summary.stdout"
printf 'completed_at=%s\nreason=finish all winner-versus-P19 Optrace after stdin isolation fix\n' \
  "$(date --iso-8601=seconds)" >"${staging}/optrace_completion.txt"

mkdir -p "${publish}"
cp -a "${staging}/." "${publish}/"
mv "${results_root}" "${backup}"
if mv "${publish}" "${results_root}"; then
  rm -rf "${backup}" "${staging}"
else
  mv "${backup}" "${results_root}"
  exit 1
fi
adb_cmd shell "rm -rf '${remote}'"
trap - ERR INT TERM
echo "K/V-head Optrace completion finished: ${results_root}"
