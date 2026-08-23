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
remote="${REMOTE_DIR:-/data/local/tmp/mllm_kv_head_p_schedule_qairt249}"
work_root="${WORK_ROOT:-/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/kv_head_p_schedule_qairt249_20260824}"
rounds="${ROUNDS:-5}"
warmup="${WARMUP:-100}"
iterations="${ITERATIONS:-1000}"
points=(default 0 1 2 3 4 5 6 8 13 15 16 17 19 20 21 22 23)
projections=(k_proj v_proj)
sequences=(1 32 64)

case "${artifact_root}" in
  /mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/kv_head_p_schedule_qairt249/20260824) ;;
  *) echo "refusing unexpected ARTIFACT_ROOT: ${artifact_root}" >&2; exit 2 ;;
esac
case "${results_root}" in
  /mnt/d/llm_exp/results/qwen3_sm8750_v79_kv_head_p_schedule_qairt249_20260824) ;;
  *) echo "refusing unexpected RESULTS_ROOT: ${results_root}" >&2; exit 2 ;;
esac
case "${remote}" in /data/local/tmp/mllm_kv_head_p_schedule_qairt249) ;;
  *) echo "refusing unexpected REMOTE_DIR: ${remote}" >&2; exit 2 ;;
esac
case "${work_root}" in
  /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/kv_head_p_schedule_qairt249_*) ;;
  *) echo "refusing unexpected WORK_ROOT: ${work_root}" >&2; exit 2 ;;
esac
[[ "${qairt_root##*/}" == "2.49.0.260730" ]] || { echo "unexpected QAIRT release" >&2; exit 2; }
[[ -x "${runner}" && -x "${adb}" && -s "${libomp}" ]] || {
  echo "runner, ADB wrapper, or libomp missing" >&2
  exit 2
}
[[ -s "${artifact_root}/inventory.tsv" && -s "${source_root}/artifact.json" ]] || {
  echo "K/V schedule artifacts are incomplete" >&2
  exit 2
}
[[ ! -e "${results_root}" ]] || { echo "refusing existing result: ${results_root}" >&2; exit 2; }
[[ "${rounds}" =~ ^[1-9][0-9]*$ && "${warmup}" =~ ^[0-9]+$ \
    && "${iterations}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid timing controls" >&2; exit 2; }

staging="${work_root}.tmp.$$"
publish="${results_root}.tmp.$$"
case "${staging}" in
  /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/kv_head_p_schedule_qairt249_*.tmp.*) ;;
  *) echo "unexpected WSL staging: ${staging}" >&2; exit 2 ;;
esac
case "${publish}" in
  /mnt/d/llm_exp/results/qwen3_sm8750_v79_kv_head_p_schedule_qairt249_20260824.tmp.*) ;;
  *) echo "unexpected publish staging: ${publish}" >&2; exit 2 ;;
esac
mkdir -p "${staging}"
trap 'echo "K/V schedule run failed; retained WSL staging: ${staging}" >&2' ERR INT TERM

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

context_path() {
  printf '%s/contexts/%s/context.bin\n' "${artifact_root}" "$(case_tag "$1" "$2" "$3")"
}

schematic_path() {
  printf '%s/contexts/%s/schematics/model.0.s%s_schematic.bin\n' \
    "${artifact_root}" "$(case_tag "$1" "$2" "$3")" "$2"
}

run_case() {
  local projection="$1" sequence="$2" point="$3" run_root="$4"
  local case_warmup="$5" case_iterations="$6" profile="$7" pull_output="$8"
  local tag host_dir remote_dir
  tag="$(case_tag "${projection}" "${sequence}" "${point}")"
  host_dir="${staging}/${run_root}/${tag}"
  remote_dir="${remote}/runs/${run_root}/${tag}"
  mkdir -p "${host_dir}"
  adb_cmd shell "rm -rf '${remote_dir}' && mkdir -p '${remote_dir}'"
  adb_cmd shell "cd '${remote}' && env \
    LD_LIBRARY_PATH='${remote}' ADSP_LIBRARY_PATH='${remote}' \
    MLLM_QNN_PROFILE_WARMUP=0 MLLM_QNN_PROFILE_EVERY=1 \
    MLLM_QNN_PROFILE_MAX_CAPTURES=1 MLLM_QNN_PROFILE_SERIALIZE=1 \
    ./projection-runner --context '${remote}/contexts/${tag}.bin' \
      --graph 'model.0.s${sequence}' --input '${remote}/input_s${sequence}.raw' \
      --output '${remote_dir}/output.raw' --timing_csv '${remote_dir}/timing.csv' \
      --activation a8 --in_channels 2048 --out_channels 128 --seq '${sequence}' \
      --warmup '${case_warmup}' --iterations '${case_iterations}' \
      --profile_level '${profile}'" >"${host_dir}/run.log" 2>&1
  adb_cmd pull "${remote_dir}/timing.csv" "${host_dir}/timing.csv" >/dev/null
  if [[ "${pull_output}" == 1 ]]; then
    adb_cmd pull "${remote_dir}/output.raw" "${host_dir}/output.raw" >/dev/null
  fi
  if [[ "${profile}" == optrace ]]; then
    adb_cmd pull "${remote_dir}/." "${host_dir}/" >/dev/null
  fi
}

adb_cmd get-state >/dev/null
adb_cmd shell "rm -rf '${remote}' && mkdir -p '${remote}/contexts' '${remote}/runs'"
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
for sequence in "${sequences[@]}"; do
  push_file "${source_root}/input_s${sequence}_a8.raw" "${remote}/input_s${sequence}.raw"
done
for projection in "${projections[@]}"; do
  for sequence in "${sequences[@]}"; do
    for point in "${points[@]}"; do
      tag="$(case_tag "${projection}" "${sequence}" "${point}")"
      push_file "$(context_path "${projection}" "${sequence}" "${point}")" \
        "${remote}/contexts/${tag}.bin"
    done
  done
done
adb_cmd shell "chmod 755 '${remote}/projection-runner'"

correctness_points=(19 default 0 1 2 3 4 5 6 8 13 15 16 17 20 21 22 23)
for projection in "${projections[@]}"; do
  for sequence in "${sequences[@]}"; do
    control="${staging}/correctness/$(case_tag "${projection}" "${sequence}" 19)/output.raw"
    for point in "${correctness_points[@]}"; do
      run_case "${projection}" "${sequence}" "${point}" correctness 0 1 off 1
      candidate="${staging}/correctness/$(case_tag "${projection}" "${sequence}" "${point}")/output.raw"
      if [[ "${point}" != 19 ]]; then cmp "${control}" "${candidate}"; fi
    done
  done
done
find "${staging}/correctness" -type f -name output.raw -print0 | sort -z | xargs -0 sha256sum \
  >"${staging}/correctness.sha256"

adb_cmd shell dumpsys thermalservice >"${staging}/thermal-speed-before.txt" || true
for round in $(seq 1 "${rounds}"); do
  for projection in "${projections[@]}"; do
    for sequence in "${sequences[@]}"; do
      if (( round % 2 )); then
        for point in "${points[@]}"; do
          run_case "${projection}" "${sequence}" "${point}" "speed/round${round}" \
            "${warmup}" "${iterations}" off 0
        done
      else
        for ((index=${#points[@]} - 1; index >= 0; --index)); do
          point="${points[index]}"
          run_case "${projection}" "${sequence}" "${point}" "speed/round${round}" \
            "${warmup}" "${iterations}" off 0
        done
      fi
    done
  done
done
adb_cmd shell dumpsys thermalservice >"${staging}/thermal-speed-after.txt" || true

python3 "${repo_root}/scripts/qnn_kv_head_p_schedule_result.py" \
  --result-root "${staging}" --source-root "${source_root}" \
  --report "${staging}/summary.screen.json" --csv "${staging}/ranking.screen.csv" \
  --markdown "${staging}/SUMMARY.screen.md" >"${staging}/summary.screen.stdout"
python3 - "${staging}/summary.screen.json" >"${staging}/winners.tsv" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for projection in ("k_proj", "v_proj"):
    for sequence in (1, 32, 64):
        key = f"{projection}_s{sequence}"
        print(projection, sequence, payload["rankings"][key][0]["point"], sep="\t")
PY

viewer="${qairt_root}/bin/x86_64-linux-clang/qnn-profile-viewer"
reader="${qairt_root}/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so"
while IFS=$'\t' read -r projection sequence winner; do
  profile_points=(19)
  if [[ "${winner}" != 19 ]]; then profile_points+=("${winner}"); fi
  for point in "${profile_points[@]}"; do
    run_case "${projection}" "${sequence}" "${point}" optrace 0 1 optrace 1
    tag="$(case_tag "${projection}" "${sequence}" "${point}")"
    profile_dir="${staging}/optrace/${tag}"
    cmp "${staging}/correctness/${tag}/output.raw" "${profile_dir}/output.raw"
    "${viewer}" --reader "${reader}" \
      --input_log "${profile_dir}/qnn-profiling-data.log" \
      --schematic "$(schematic_path "${projection}" "${sequence}" "${point}")" \
      --output "${profile_dir}/chrometrace.json" \
      >"${profile_dir}/profile-viewer.log" 2>&1
  done
done <"${staging}/winners.tsv"

python3 "${repo_root}/scripts/qnn_kv_head_p_schedule_result.py" \
  --result-root "${staging}" --source-root "${source_root}" \
  --report "${staging}/summary.json" --csv "${staging}/ranking.csv" \
  --markdown "${staging}/SUMMARY.md" >"${staging}/summary.stdout"
{
  printf 'git_commit=%s\ngit_branch=%s\n' \
    "$(git -C "${repo_root}" rev-parse HEAD)" "$(git -C "${repo_root}" branch --show-current)"
  printf 'device_serial=%s\nqairt_release=2.49.0.260730\n' "${serial}"
  printf 'rounds=%s\nwarmup=%s\niterations=%s\n' "${rounds}" "${warmup}" "${iterations}"
  printf 'artifact_sha256=%s\nrunner_sha256=%s\n' \
    "$(sha256sum "${source_root}/qwen3-layer14-kv-head0-w4g32-a8.mllm" | awk '{print $1}')" \
    "$(sha256sum "${runner}" | awk '{print $1}')"
  printf 'git_status_begin\n'
  git -C "${repo_root}" status --short
  printf 'git_status_end\n'
} >"${staging}/provenance.txt"
adb_cmd shell getprop >"${staging}/device-getprop.txt"

mkdir -p "${publish}"
cp -a "${staging}/." "${publish}/"
mv "${publish}" "${results_root}"
rm -rf "${staging}"
adb_cmd shell "rm -rf '${remote}'"
trap - ERR INT TERM
echo "K/V-head P-schedule run complete: ${results_root}"
