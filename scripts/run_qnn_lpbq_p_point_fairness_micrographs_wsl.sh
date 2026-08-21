#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
source_root="${SOURCE_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_a16_vs_a8_projections/20260821}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_p_point_fairness/20260822}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_lpbq_p_point_fairness_micrographs_20260822}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
runner="${RUNNER:-${repo_root}/build-android-arm64-v8a-qnn/bin/mllm-qwen3-lpbq-a16-a8-projection-runner}"
build_bin="$(dirname "${runner}")"
ndk_root="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
libomp="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
adb="${ADB_WRAPPER:-${repo_root}/scripts/adb_wsl_path_wrapper.sh}"
serial="${ADB_SERIAL:-3B15C8007Z300000}"
remote="${REMOTE_DIR:-/data/local/tmp/mllm_lpbq_p_point_fairness}"
read -r -a points <<<"${POINTS:-default 0 1 2 3 4 5 6 8 13 15 16 17 19 20 21 22 23}"
read -r -a activations <<<"${ACTIVATIONS:-a8 a16}"
read -r -a sequences <<<"${SEQUENCES:-1 32}"
read -r -a projections <<<"${PROJECTIONS:-gate_proj up_proj down_proj lm_head}"
rounds="${ROUNDS:-3}"
warmup="${WARMUP:-20}"
iterations="${ITERATIONS:-300}"
staging="${STAGING_ROOT:-/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/lpbq_p_point_fairness_micrographs_20260822}"

case "${remote}" in /data/local/tmp/mllm_lpbq_p_point_fairness) ;;
  *) echo "refusing unexpected REMOTE_DIR: ${remote}" >&2; exit 2 ;;
esac
case "${results_root}" in /mnt/d/llm_exp/results/qwen3_sm8750_v79_lpbq_p_point_fairness_micrographs_*) ;;
  *) echo "refusing unexpected RESULTS_ROOT: ${results_root}" >&2; exit 2 ;;
esac
case "${staging}" in /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/lpbq_p_point_fairness_micrographs_*) ;;
  *) echo "refusing unexpected STAGING_ROOT: ${staging}" >&2; exit 2 ;;
esac
[[ "${qairt_root##*/}" == "2.47.0.260601" ]] || exit 2
[[ -x "${runner}" ]] || { echo "runner missing: ${runner}" >&2; exit 2; }
[[ ! -e "${results_root}" ]] || { echo "result already exists: ${results_root}" >&2; exit 2; }
[[ "${rounds}" =~ ^[1-9][0-9]*$ && "${warmup}" =~ ^[0-9]+$ && "${iterations}" =~ ^[1-9][0-9]*$ ]] || exit 2
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
  [[ -f "$1" ]] || { echo "missing source: $1" >&2; exit 2; }
  adb_cmd push "$1" "$2" >/dev/null
}

channels() {
  case "$1" in
    gate_proj|up_proj) printf '2048 6144\n' ;;
    down_proj) printf '6144 2048\n' ;;
    lm_head) printf '2048 151936\n' ;;
    *) return 2 ;;
  esac
}

context_path() {
  local activation="$1" seq="$2" projection="$3" point="$4"
  if [[ "${point}" == default ]]; then
    printf '%s/contexts/%s_%s_s%s/%s_%s_s%s.bin\n' \
      "${source_root}" "${activation}" "${projection}" "${seq}" \
      "${activation}" "${projection}" "${seq}"
  else
    printf '%s/contexts/%s_s%s_%s_p%s/context.bin\n' \
      "${artifact_root}" "${activation}" "${seq}" "${projection}" "${point}"
  fi
}

prepare_device() {
  adb_cmd get-state >/dev/null
  adb_cmd shell "rm -rf '${remote}' && mkdir -p '${remote}/runs'"
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
  adb_cmd shell "chmod 755 '${remote}/projection-runner'"
}

execute_case() {
  local activation="$1" seq="$2" projection="$3" point="$4" run_tag="$5" warm="$6" iters="$7"
  local in_channels out_channels case_tag host_dir host_tmp remote_run need_output=0
  read -r in_channels out_channels < <(channels "${projection}")
  case_tag="${activation}_s${seq}_${projection}_${point}"
  host_dir="${staging}/${run_tag}/${case_tag}"
  [[ "${run_tag}" == correctness ]] && need_output=1
  if [[ -s "${host_dir}/timing.csv" ]] \
      && (( ! need_output || $(stat -c %s "${host_dir}/output.raw" 2>/dev/null || echo 0) > 0 )); then
    return
  fi
  host_tmp="${host_dir}.tmp.$$"
  case "${host_tmp}" in "${staging}/"*.tmp.*) ;; *) exit 2 ;; esac
  rm -rf "${host_tmp}"
  mkdir -p "${host_tmp}"
  remote_run="${remote}/runs/${run_tag}/${case_tag}"
  adb_cmd shell "rm -rf '${remote_run}' && mkdir -p '${remote_run}'"
  adb_cmd shell "cd '${remote}' && env LD_LIBRARY_PATH='${remote}' ADSP_LIBRARY_PATH='${remote}' \
    ./projection-runner --context '${remote}/context.bin' --graph 'model.0.s${seq}' \
    --input '${remote}/input.raw' --output '${remote_run}/output.raw' \
    --timing_csv '${remote_run}/timing.csv' --activation '${activation}' \
    --in_channels '${in_channels}' --out_channels '${out_channels}' --seq '${seq}' \
    --warmup '${warm}' --iterations '${iters}' --profile_level off" \
    >"${host_tmp}/run.log" 2>&1
  if (( need_output )); then
    adb_cmd pull "${remote_run}/output.raw" "${host_tmp}/output.raw" >/dev/null
  fi
  adb_cmd pull "${remote_run}/timing.csv" "${host_tmp}/timing.csv" >/dev/null
  [[ -s "${host_tmp}/timing.csv" ]]
  (( ! need_output )) || [[ -s "${host_tmp}/output.raw" ]]
  [[ ! -e "${host_dir}" ]] || exit 2
  mv "${host_tmp}" "${host_dir}"
}

prepare_device
{
  printf 'git_commit=%s\n' "$(git -C "${repo_root}" rev-parse HEAD)"
  printf 'git_branch=%s\nqairt_release=2.47.0.260601\ndevice_serial=%s\n' \
    "$(git -C "${repo_root}" branch --show-current)" "${serial}"
  printf 'rounds=%s\nwarmup=%s\niterations=%s\n' "${rounds}" "${warmup}" "${iterations}"
} >"${staging}/provenance.env"
adb_cmd shell dumpsys thermalservice >"${staging}/thermal-before.txt" || true

for activation in "${activations[@]}"; do
  for seq in "${sequences[@]}"; do
    for projection in "${projections[@]}"; do
      input="${source_root}/input_${projection}_s${seq}_${activation}.raw"
      push_file "${input}" "${remote}/input.raw"
      reference="${staging}/correctness/${activation}_s${seq}_${projection}_default/output.raw"
      for point in "${points[@]}"; do
        context="$(context_path "${activation}" "${seq}" "${projection}" "${point}")"
        push_file "${context}" "${remote}/context.bin"
        execute_case "${activation}" "${seq}" "${projection}" "${point}" correctness 0 1
        if [[ "${point}" == default ]]; then
          [[ -s "${reference}" ]] || exit 1
        else
          cmp "${reference}" "${staging}/correctness/${activation}_s${seq}_${projection}_${point}/output.raw"
        fi
        for round in $(seq 1 "${rounds}"); do
          execute_case "${activation}" "${seq}" "${projection}" "${point}" \
            "speed/round${round}" "${warmup}" "${iterations}"
        done
        adb_cmd shell "rm -f '${remote}/context.bin'"
      done
    done
  done
done

find "${staging}/correctness" -name output.raw -print0 | sort -z | xargs -0 sha256sum \
  >"${staging}/correctness.sha256"
# Hashes plus the retained default outputs are sufficient to audit the
# byte-exact comparison; discard the duplicate candidate payloads before
# publishing, especially the 9.7 MB lm_head/s32 tensors.
find "${staging}/correctness" -mindepth 2 -maxdepth 2 -type f -name output.raw \
  ! -path '*_default/output.raw' -delete
adb_cmd shell dumpsys thermalservice >"${staging}/thermal-after.txt" || true
adb_cmd shell "rm -rf '${remote}'"
python3 "${repo_root}/scripts/qnn_lpbq_p_point_fairness_summary.py" \
  "${staging}" --output-json "${staging}/summary.json" --output-csv "${staging}/summary.csv"

publish="${results_root}.tmp.$$"
case "${publish}" in /mnt/d/llm_exp/results/qwen3_sm8750_v79_lpbq_p_point_fairness_micrographs_*.tmp.*) ;;
  *) exit 2 ;;
esac
mkdir -p "${publish}"
trap 'rm -rf "${publish}"' ERR INT TERM
cp -a "${staging}/." "${publish}/"
mv "${publish}" "${results_root}"
resolved_staging="$(realpath -e "${staging}")"
case "${resolved_staging}" in /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/lpbq_p_point_fairness_micrographs_*) ;;
  *) exit 2 ;;
esac
rm -rf "${resolved_staging}"
trap - ERR INT TERM
echo "LPBQ P-point fairness micrograph screen complete: ${results_root}"
