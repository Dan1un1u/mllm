#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_p_point_fairness_full_screen_20260822}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
runner="${RUNNER:-${repo_root}/build-android-arm64-v8a-qnn/bin/mllm-qwen3-aot-runner}"
build_bin="$(dirname "${runner}")"
ndk_root="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
libomp="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
tokenizer="${models_root}/Qwen3-origin/qwen3-tokenizer.json"
config_a8="${repo_root}/examples/qwen3_qnn_aot/config_1.7B_g32.json"
config_a16="${repo_root}/examples/qwen3_qnn_aot/config_1.7B_g32_a16.json"
llama_package_build="${repo_root}/mllm/backends/qnn/custom-op-package/LLaMAPackage/build"
adb="${ADB_WRAPPER:-${repo_root}/scripts/adb_wsl_path_wrapper.sh}"
serial="${ADB_SERIAL:-3B15C8007Z300000}"
remote="${REMOTE_DIR:-/data/local/tmp/mllm_p_point_fairness_full}"
rounds="${ROUNDS:-5}"
max_new_tokens="${MAX_NEW_TOKENS:-64}"
cooldown="${COOLDOWN_SEC:-2}"
prompt="${PROMPT:-Explain how quantized Transformer inference maps matrix, vector, and data movement work onto a mobile NPU. Discuss attention, KV cache, MLP, and the cost of quantization conversions in enough detail to continue for at least sixty-four generated tokens.}"
staging="${STAGING_ROOT:-/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/p_point_fairness_full_screen_20260822}"
screen_scope="${SCREEN_SCOPE:-full}"

case "${remote}" in /data/local/tmp/mllm_p_point_fairness_full) ;; *) exit 2 ;; esac
case "${results_root}" in /mnt/d/llm_exp/results/qwen3_sm8750_v79_p_point_fairness_*_20260822) ;; *) exit 2 ;; esac
case "${staging}" in /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/p_point_fairness_*_20260822) ;; *) exit 2 ;; esac
[[ ! -e "${results_root}" && ! -e "${staging}" ]] || { echo "screen output already exists" >&2; exit 2; }
[[ "${qairt_root##*/}" == 2.47.0.260601 && "${rounds}" =~ ^[1-9][0-9]*$ ]] || exit 2
mkdir -p "${staging}/benchmark"

case "${screen_scope}" in
  full)
    labels=(
      a8_default a8_global_p19 a8_p19_default a8_p19_p17 a8_p19_p16
      a16_default a16_p6_default a16_p6_p0 a16_p6_p20
    )
    paths=(
      "${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/20260813_211529/qwen3-1.7B-w4a8g32-rmsnorm-u8.bin"
      "${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_p19/20260821/qwen3-1.7B-w4a8g32-rmsnorm-u8-p19.bin"
      "${models_root}/qwen3_sm8750_v79/g32/p_point_fairness_full/20260822/a8_p19_default/context.bin"
      "${models_root}/qwen3_sm8750_v79/g32/p_point_fairness_full/20260822/a8_p19_p17/context.bin"
      "${models_root}/qwen3_sm8750_v79/g32/p_point_fairness_full/20260822/a8_p19_p16/context.bin"
      "${models_root}/qwen3_sm8750_v79/g32/w4a16/qwen3-1.7B-lpbq-sha-g32.bin"
      "${models_root}/qwen3_sm8750_v79/g32/p_point_fairness_full/20260822/a16_p6_default/context.bin"
      "${models_root}/qwen3_sm8750_v79/g32/p_point_fairness_full/20260822/a16_p6_p0/context.bin"
      "${models_root}/qwen3_sm8750_v79/g32/p_point_fairness_full/20260822/a16_p6_p20/context.bin"
    )
    ;;
  finalists)
    labels=(a8_default a8_global_p19 a8_p19_p17 a16_default a16_p6_default a16_p6_p0)
    paths=(
      "${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/20260813_211529/qwen3-1.7B-w4a8g32-rmsnorm-u8.bin"
      "${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_p19/20260821/qwen3-1.7B-w4a8g32-rmsnorm-u8-p19.bin"
      "${models_root}/qwen3_sm8750_v79/g32/p_point_fairness_full/20260822/a8_p19_p17/context.bin"
      "${models_root}/qwen3_sm8750_v79/g32/w4a16/qwen3-1.7B-lpbq-sha-g32.bin"
      "${models_root}/qwen3_sm8750_v79/g32/p_point_fairness_full/20260822/a16_p6_default/context.bin"
      "${models_root}/qwen3_sm8750_v79/g32/p_point_fairness_full/20260822/a16_p6_p0/context.bin"
    )
    ;;
  best)
    labels=(a8_default a8_global_p19 a16_default a16_p6_default)
    paths=(
      "${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8/20260813_211529/qwen3-1.7B-w4a8g32-rmsnorm-u8.bin"
      "${models_root}/qwen3_sm8750_v79/g32/w4a8_rmsnorm_u8_p19/20260821/qwen3-1.7B-w4a8g32-rmsnorm-u8-p19.bin"
      "${models_root}/qwen3_sm8750_v79/g32/w4a16/qwen3-1.7B-lpbq-sha-g32.bin"
      "${models_root}/qwen3_sm8750_v79/g32/p_point_fairness_full/20260822/a16_p6_default/context.bin"
    )
    ;;
  *) echo "unknown SCREEN_SCOPE: ${screen_scope}" >&2; exit 2 ;;
esac

adb_cmd() {
  local attempt
  for attempt in 1 2 3; do
    if "${adb}" -s "${serial}" "$@"; then return 0; fi
    sleep 1
  done
  return 1
}
push_file() { [[ -f "$1" ]] || { echo "missing source: $1" >&2; exit 2; }; adb_cmd push "$1" "$2" >/dev/null; }

adb_cmd get-state >/dev/null
adb_cmd shell "rm -rf '${remote}' && mkdir -p '${remote}/contexts' '${remote}/runs'"
for library in libMllmCPUBackend.so libMllmQNNBackend.so libMllmRT.so; do
  push_file "${build_bin}/${library}" "${remote}/${library}"
done
push_file "${runner}" "${remote}/runner"
push_file "${libomp}" "${remote}/libomp.so"
for library in libQnnHtp.so libQnnSystem.so libQnnHtpV79Stub.so; do
  push_file "${qairt_root}/lib/aarch64-android/${library}" "${remote}/${library}"
done
push_file "${qairt_root}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" "${remote}/libQnnHtpV79Skel.so"
push_file "${llama_package_build}/libQnnLLaMAPackage_CPU.so" "${remote}/libQnnLLaMAPackage_CPU.so"
push_file "${llama_package_build}/libQnnLLaMAPackage_HTP.so" "${remote}/libQnnLLaMAPackage_HTP.so"
push_file "${tokenizer}" "${remote}/tokenizer.json"
push_file "${config_a8}" "${remote}/config_a8.json"
push_file "${config_a16}" "${remote}/config_a16.json"
adb_cmd shell "chmod 755 '${remote}/runner'"

for index in "${!labels[@]}"; do
  label="${labels[index]}"
  path="${paths[index]}"
  push_file "${path}" "${remote}/contexts/${label}.bin"
  printf '%s %s %s\n' "${label}" "$(sha256sum "${path}" | awk '{print $1}')" "${path}" \
    >>"${staging}/contexts.sha256"
done
{
  printf 'git_commit=%s\ngit_branch=%s\nqairt_release=2.47.0.260601\ndevice_serial=%s\n' \
    "$(git -C "${repo_root}" rev-parse HEAD)" "$(git -C "${repo_root}" branch --show-current)" "${serial}"
  printf 'screen_scope=%s\nrounds=%s\nmax_new_tokens=%s\nprompt=%s\n' \
    "${screen_scope}" "${rounds}" "${max_new_tokens}" "${prompt}"
  printf 'config_a8_sha256=%s\nconfig_a16_sha256=%s\n' \
    "$(sha256sum "${config_a8}" | awk '{print $1}')" \
    "$(sha256sum "${config_a16}" | awk '{print $1}')"
} >"${staging}/provenance.env"
adb_cmd shell dumpsys thermalservice >"${staging}/thermal-before.txt" || true

for round in $(seq 1 "${rounds}"); do
  if (( round % 2 )); then
    order=("${labels[@]}")
  else
    order=()
    for ((index=${#labels[@]} - 1; index >= 0; --index)); do order+=("${labels[index]}"); done
  fi
  for label in "${order[@]}"; do
    case "${label}" in
      a8_*) runtime_config=config_a8.json ;;
      a16_*) runtime_config=config_a16.json ;;
      *) echo "unknown activation contract for ${label}" >&2; exit 2 ;;
    esac
    host_dir="${staging}/benchmark/round${round}/${label}"
    remote_run="${remote}/runs/round${round}/${label}"
    mkdir -p "${host_dir}"
    adb_cmd shell "mkdir -p '${remote_run}'"
    adb_cmd shell dumpsys thermalservice >"${host_dir}/thermal-before.txt" || true
    printf '%s\n' "${prompt}" | adb_cmd shell "cd '${remote}' && env \
      LD_LIBRARY_PATH='${remote}' ADSP_LIBRARY_PATH='${remote}' \
      MLLM_QNN_PROFILE_LEVEL=off MLLM_QNN_PROFILE_DIR='${remote_run}' \
      ./runner -m 'contexts/${label}.bin' -t tokenizer.json -c '${runtime_config}' \
      --ar_len 32 --max_new_tokens '${max_new_tokens}' --perf" \
      >"${host_dir}/runner.log" 2>&1
    adb_cmd pull "${remote_run}/qnn_runner_e2e.csv" "${host_dir}/qnn_runner_e2e.csv" >/dev/null
    [[ -s "${host_dir}/qnn_runner_e2e.csv" ]] || exit 1
  done
  (( round == rounds || cooldown == 0 )) || sleep "${cooldown}"
done
adb_cmd shell dumpsys thermalservice >"${staging}/thermal-after.txt" || true
python3 "${repo_root}/scripts/qnn_p_point_full_screen_summary.py" "${staging}" \
  --output-json "${staging}/summary.json" --output-csv "${staging}/summary.csv"
adb_cmd shell "rm -rf '${remote}'"

publish="${results_root}.tmp.$$"
case "${publish}" in /mnt/d/llm_exp/results/qwen3_sm8750_v79_p_point_fairness_*_20260822.tmp.*) ;; *) exit 2 ;; esac
mkdir -p "${publish}"
trap 'rm -rf "${publish}"' ERR INT TERM
cp -a "${staging}/." "${publish}/"
mv "${publish}" "${results_root}"
resolved_staging="$(realpath -e "${staging}")"
case "${resolved_staging}" in /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/p_point_fairness_*_20260822) ;; *) exit 2 ;; esac
rm -rf "${resolved_staging}"
trap - ERR INT TERM
echo "full P-point candidate screen complete: ${results_root}"
