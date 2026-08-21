#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_a16_p_point_accuracy_20260822}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
runner="${RUNNER:-${repo_root}/build-android-arm64-v8a-qnn/bin/mllm-qwen3-aot-runner}"
build_bin="$(dirname "${runner}")"
ndk_root="${ANDROID_NDK_PATH:-/home/daniuniu/toolchains/android-ndk-r26c}"
libomp="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
tokenizer="${models_root}/Qwen3-origin/qwen3-tokenizer.json"
config="${repo_root}/examples/qwen3_qnn_aot/config_1.7B_g32_a16.json"
suite="${repo_root}/scripts/qwen3_sm8750_v79_accuracy.tsv"
llama_package_build="${repo_root}/mllm/backends/qnn/custom-op-package/LLaMAPackage/build"
adb="${ADB_WRAPPER:-${repo_root}/scripts/adb_wsl_path_wrapper.sh}"
serial="${ADB_SERIAL:-3B15C8007Z300000}"
remote="${REMOTE_DIR:-/data/local/tmp/mllm_a16_p_point_accuracy}"
max_new_tokens="${ACCURACY_MAX_NEW_TOKENS:-64}"
staging="${STAGING_ROOT:-/home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/a16_p_point_accuracy_20260822}"
archived_csv="${ARCHIVED_ACCURACY_CSV:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_g32_20260807_230410/accuracy/qnn_accuracy_eval.csv}"

case "${remote}" in /data/local/tmp/mllm_a16_p_point_accuracy) ;; *) exit 2 ;; esac
case "${results_root}" in /mnt/d/llm_exp/results/qwen3_sm8750_v79_a16_p_point_accuracy_20260822) ;; *) exit 2 ;; esac
case "${staging}" in /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/a16_p_point_accuracy_20260822) ;; *) exit 2 ;; esac
[[ ! -e "${results_root}" && ! -e "${staging}" ]] || { echo "accuracy output already exists" >&2; exit 2; }
[[ "${qairt_root##*/}" == 2.47.0.260601 && "${max_new_tokens}" =~ ^[1-9][0-9]*$ ]] || exit 2

labels=(a16_default a16_p6_default)
paths=(
  "${models_root}/qwen3_sm8750_v79/g32/w4a16/qwen3-1.7B-lpbq-sha-g32.bin"
  "${models_root}/qwen3_sm8750_v79/g32/p_point_fairness_full/20260822/a16_p6_default/context.bin"
)

adb_cmd() {
  local attempt
  for attempt in 1 2 3; do
    if "${adb}" -s "${serial}" "$@"; then return 0; fi
    sleep 1
  done
  return 1
}
push_file() { [[ -f "$1" ]] || { echo "missing source: $1" >&2; exit 2; }; adb_cmd push "$1" "$2" >/dev/null; }

mkdir -p "${staging}"
adb_cmd get-state >/dev/null
adb_cmd shell "rm -rf '${remote}' && mkdir -p '${remote}/contexts' '${remote}/accuracy'"
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
push_file "${config}" "${remote}/config.json"
push_file "${suite}" "${remote}/accuracy_suite.tsv"
adb_cmd shell "chmod 755 '${remote}/runner'"

for index in "${!labels[@]}"; do
  label="${labels[index]}"
  path="${paths[index]}"
  push_file "${path}" "${remote}/contexts/${label}.bin"
  printf '%s %s %s\n' "${label}" "$(sha256sum "${path}" | awk '{print $1}')" "${path}" \
    >>"${staging}/contexts.sha256"
done

for label in "${labels[@]}"; do
  mkdir -p "${staging}/${label}"
  adb_cmd shell "mkdir -p '${remote}/accuracy/${label}'"
  adb_cmd shell "cd '${remote}' && env \
    LD_LIBRARY_PATH='${remote}' ADSP_LIBRARY_PATH='${remote}' \
    MLLM_QNN_PROFILE_LEVEL=off MLLM_QNN_PROFILE_DIR='${remote}/accuracy/${label}' \
    ./runner -m 'contexts/${label}.bin' -t tokenizer.json -c config.json \
    --ar_len 32 --max_new_tokens '${max_new_tokens}' --eval_file accuracy_suite.tsv" \
    >"${staging}/${label}/runner.log" 2>&1
  adb_cmd pull "${remote}/accuracy/${label}/qnn_accuracy_eval.csv" \
    "${staging}/${label}/qnn_accuracy_eval.csv" >/dev/null
  python3 "${repo_root}/scripts/qnn_accuracy_summary.py" \
    "${staging}/${label}/qnn_accuracy_eval.csv" --max-new-tokens "${max_new_tokens}" \
    --output "${staging}/${label}/summary.json" \
    >"${staging}/${label}/summary.log"
done

if cmp "${staging}/a16_default/qnn_accuracy_eval.csv" \
       "${staging}/a16_p6_default/qnn_accuracy_eval.csv" >/dev/null; then
  identical=true
else
  identical=false
  diff -u "${staging}/a16_default/qnn_accuracy_eval.csv" \
          "${staging}/a16_p6_default/qnn_accuracy_eval.csv" \
          >"${staging}/default_vs_p6.diff" || true
fi
archived_identical=false
if [[ -f "${archived_csv}" ]] && cmp "${archived_csv}" \
    "${staging}/a16_default/qnn_accuracy_eval.csv" >/dev/null; then
  archived_identical=true
fi
{
  printf 'git_commit=%s\ngit_branch=%s\nqairt_release=2.47.0.260601\ndevice_serial=%s\n' \
    "$(git -C "${repo_root}" rev-parse HEAD)" "$(git -C "${repo_root}" branch --show-current)" "${serial}"
  printf 'runner_sha256=%s\nconfig_sha256=%s\nsuite_sha256=%s\n' \
    "$(sha256sum "${runner}" | awk '{print $1}')" \
    "$(sha256sum "${config}" | awk '{print $1}')" \
    "$(sha256sum "${suite}" | awk '{print $1}')"
  printf 'default_vs_p6_csv_byte_identical=%s\narchived_vs_current_default_csv_byte_identical=%s\n' \
    "${identical}" "${archived_identical}"
} >"${staging}/provenance.env"

adb_cmd shell "rm -rf '${remote}'"
publish="${results_root}.tmp.$$"
case "${publish}" in /mnt/d/llm_exp/results/qwen3_sm8750_v79_a16_p_point_accuracy_20260822.tmp.*) ;; *) exit 2 ;; esac
mkdir -p "${publish}"
trap 'rm -rf "${publish}"' ERR INT TERM
cp -a "${staging}/." "${publish}/"
mv "${publish}" "${results_root}"
test "$(realpath -e "${staging}")" = /home/daniuniu/llm_exp_work/qwen3_sm8750_v79/g32/a16_p_point_accuracy_20260822
rm -rf "${staging}"
trap - ERR INT TERM
echo "A16 P-point accuracy comparison complete: ${results_root}"
