#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_w4a8_rmsnorm_u8_lpbq_mlp_gate_20260818}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_lpbq_mlp/20260818_micrograph_layer14}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
runner="${RUNNER:-${repo_root}/build-android-arm64-v8a-qnn/bin/mllm-qnn-lpbq-mlp-runner}"
build_bin="$(dirname "${runner}")"
ndk_root="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
libomp="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
adb="${ADB_WRAPPER:-${repo_root}/scripts/adb_wsl_path_wrapper.sh}"
serial="${ADB_SERIAL:-3B15C8007Z300000}"
remote="${REMOTE_DIR:-/data/local/tmp/mllm_w4a8_lpbq_mlp}"
mode="${1:-correctness}"

case "${remote}" in
  /data/local/tmp/mllm_w4a8_lpbq_mlp) ;;
  *) echo "refusing unexpected REMOTE_DIR: ${remote}" >&2; exit 2 ;;
esac

projections=(gate_proj up_proj down_proj)
sequences=(1 32)
fixtures=(zero qmin qmax alternating random calibration_qparam_replay)

adb_cmd() { "${adb}" -s "${serial}" "$@"; }

shape_for() {
  case "$1" in
    gate_proj|up_proj) printf '2048 6144\n' ;;
    down_proj) printf '6144 2048\n' ;;
    *) return 2 ;;
  esac
}

remote_sha() {
  adb_cmd shell "sha256sum '${1}' 2>/dev/null" | awk '{print $1}' | tr -d '\r' || true
}

push_if_changed() {
  local source="$1" destination="$2" expected actual
  expected="$(sha256sum "${source}" | awk '{print $1}')"
  actual="$(remote_sha "${destination}")"
  if [[ "${expected}" != "${actual}" ]]; then
    adb_cmd push "${source}" "${destination}" >/dev/null
  fi
}

prepare_device() {
  adb_cmd get-state >/dev/null
  adb_cmd shell "mkdir -p '${remote}/timing' '${remote}/outputs'"
  for library in libMllmCPUBackend.so libMllmQNNBackend.so libMllmRT.so; do
    push_if_changed "${build_bin}/${library}" "${remote}/${library}"
  done
  push_if_changed "${runner}" "${remote}/mllm-qnn-lpbq-mlp-runner"
  push_if_changed "${libomp}" "${remote}/libomp.so"
  for library in libQnnHtp.so libQnnSystem.so libQnnHtpV79Stub.so; do
    push_if_changed "${qairt_root}/lib/aarch64-android/${library}" "${remote}/${library}"
  done
  push_if_changed "${qairt_root}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" \
    "${remote}/libQnnHtpV79Skel.so"
  adb_cmd shell "chmod 755 '${remote}/mllm-qnn-lpbq-mlp-runner'"
}

prepare_contexts_and_fixtures() {
  local layout projection seq case_name context fixture
  for layout in conv matmul; do
    for projection in "${projections[@]}"; do
      for seq in "${sequences[@]}"; do
        case_name="${layout}_${projection}_s${seq}"
        context="${artifact_root}/contexts/${case_name}/${case_name}.bin"
        push_if_changed "${context}" "${remote}/${case_name}.bin"
      done
    done
  done
  for projection in "${projections[@]}"; do
    for seq in "${sequences[@]}"; do
      for fixture in "${fixtures[@]}"; do
        push_if_changed "${artifact_root}/fixtures/${projection}_s${seq}/${fixture}.raw" \
          "${remote}/${projection}_s${seq}_${fixture}.raw"
      done
    done
  done
}

execute_case() {
  local layout="$1" projection="$2" seq="$3" fixture="$4" tag="$5" warmup="$6" iterations="$7"
  local in_channels out_channels case_name host_dir remote_timing remote_output remote_tag
  read -r in_channels out_channels < <(shape_for "${projection}")
  case_name="${layout}_${projection}_s${seq}"
  host_dir="${results_root}/${tag}/${case_name}"
  mkdir -p "${host_dir}"
  remote_tag="${tag//\//_}"
  remote_timing="${remote}/timing/${remote_tag}_${case_name}_${fixture}.csv"
  remote_output="${remote}/outputs/${remote_tag}_${case_name}_${fixture}.raw"
  adb_cmd shell "cd '${remote}' && env LD_LIBRARY_PATH='${remote}' ADSP_LIBRARY_PATH='${remote}' \
    ./mllm-qnn-lpbq-mlp-runner \
    --context '${remote}/${case_name}.bin' --graph 'model.0.s${seq}' \
    --input '${remote}/${projection}_s${seq}_${fixture}.raw' \
    --output '${remote_output}' --timing_csv '${remote_timing}' \
    --in_channels '${in_channels}' --out_channels '${out_channels}' --seq '${seq}' \
    --warmup '${warmup}' --iterations '${iterations}'" \
    >"${host_dir}/${fixture}.log" 2>&1
  adb_cmd pull "${remote_output}" "${host_dir}/${fixture}.raw" >/dev/null
  adb_cmd pull "${remote_timing}" "${host_dir}/${fixture}.timing.csv" >/dev/null
}

run_correctness() {
  local layout projection seq fixture original
  mkdir -p "${results_root}/correctness"
  adb_cmd shell dumpsys thermalservice >"${results_root}/correctness/thermal-before.txt"
  for layout in conv matmul; do
    for projection in "${projections[@]}"; do
      for seq in "${sequences[@]}"; do
        for fixture in "${fixtures[@]}"; do
          execute_case "${layout}" "${projection}" "${seq}" "${fixture}" correctness 0 1
          original="${results_root}/correctness/${layout}_${projection}_s${seq}/${fixture}.raw"
          mv "${original}" "${original%.raw}.first.raw"
          execute_case "${layout}" "${projection}" "${seq}" "${fixture}" correctness 0 1
          mv "${results_root}/correctness/${layout}_${projection}_s${seq}/${fixture}.raw" \
            "${results_root}/correctness/${layout}_${projection}_s${seq}/${fixture}.repeat.raw"
          mv "${original%.raw}.first.raw" "${original}"
        done
      done
    done
  done
  adb_cmd shell dumpsys thermalservice >"${results_root}/correctness/thermal-after.txt"
}

run_warm() {
  local round order layout projection seq fixture=calibration_qparam_replay
  mkdir -p "${results_root}/warm"
  adb_cmd shell dumpsys thermalservice >"${results_root}/warm/thermal-before.txt"
  for round in 1 2 3 4 5; do
    if (( round % 2 )); then order=(conv matmul); else order=(matmul conv); fi
    for projection in "${projections[@]}"; do
      for seq in "${sequences[@]}"; do
        for layout in "${order[@]}"; do
          execute_case "${layout}" "${projection}" "${seq}" "${fixture}" "warm/round${round}" 10 100
        done
      done
    done
  done
  adb_cmd shell dumpsys thermalservice >"${results_root}/warm/thermal-after.txt"
}

prepare_device
prepare_contexts_and_fixtures
case "${mode}" in
  correctness) run_correctness ;;
  warm) run_warm ;;
  all) run_correctness; run_warm ;;
  *) echo "usage: $0 [correctness|warm|all]" >&2; exit 2 ;;
esac
