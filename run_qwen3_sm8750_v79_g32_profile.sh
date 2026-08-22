#!/usr/bin/env bash

# Standalone Qwen3-1.7B W4A8G32 profiling solution for SM8750/V79.
#
# This file intentionally has its own G32 defaults.  It must not silently
# inherit the native G16 context/configuration from the original script.
#
# The run is split into four independent workloads:
#   1. profiling-off runner E2E benchmark (median of multiple fresh processes);
#   2. profiling-off lightweight accuracy sanity suite;
#   3. fresh-process model.0.s32 Optrace capture;
#   4. fresh-process model.0.s1 Optrace capture.
#
# Every invocation creates a new timestamped directory under RESULTS_BASE.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
CONTRACT_FILE="${CONTRACT_FILE:-${REPO_ROOT}/profiles/qwen3_sm8750_v79_g32/baseline.env}"
[[ -r "${CONTRACT_FILE}" ]] || {
    echo "ERROR: baseline contract not found: ${CONTRACT_FILE}" >&2
    exit 1
}
# shellcheck disable=SC1090
source "${CONTRACT_FILE}"
# Formal models, intermediates, and results live outside Git by contract.
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/d/llm_exp}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

RESULTS_BASE="${RESULTS_BASE:-${ARTIFACT_ROOT}/results}"
RESUME_RESULT_ROOT="${RESUME_RESULT_ROOT:-}"
RESULT_PREFIX="${RESULT_PREFIX:-qwen3_sm8750_v79_w4a8g32}"
if [[ -n "${RESUME_RESULT_ROOT}" ]]; then
    RESULT_ROOT="${RESUME_RESULT_ROOT}"
    TIMESTAMP="${RESULT_ROOT##*_}"
else
    RESULT_ROOT="${RESULTS_BASE}/${RESULT_PREFIX}_${TIMESTAMP}"
fi
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/mllm_w4a8g32}"
REMOTE_ROOT="${REMOTE_ROOT:-${REMOTE_DIR}/qwen3_sm8750_v79_g32_profile_${TIMESTAMP}}"
ADB_SERIAL="${ADB_SERIAL:-}"
MODEL_ROOT="${MODEL_ROOT:-${ARTIFACT_ROOT}/models}"
ADB_BIN="${ADB_BIN:-adb}"

# Keep the candidate baseline identity explicit in every result.  This script
# is intentionally not a generic scheme dispatcher.
PROFILE_SCHEME="${BASELINE_PROFILE_SCHEME}"
PROFILE_WRAPPER="${PROFILE_WRAPPER:-${BASH_SOURCE[0]}}"

BUILD_ANDROID="${BUILD_ANDROID:-1}"
PREPARE_DEVICE="${PREPARE_DEVICE:-1}"
BENCHMARK_RUNS="${BENCHMARK_RUNS:-3}"
BENCHMARK_COOLDOWN_SEC="${BENCHMARK_COOLDOWN_SEC:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
ACCURACY_MAX_NEW_TOKENS="${ACCURACY_MAX_NEW_TOKENS:-64}"
AR_LEN="${AR_LEN:-32}"
CLEAN_REMOTE="${CLEAN_REMOTE:-1}"
RMSNORM_U8_CONTRACT="${RMSNORM_U8_CONTRACT:-0}"
REMOTE_OP_PACKAGE_PATH="${REMOTE_OP_PACKAGE_PATH:-}"
REMOTE_OP_PACKAGE_TARGET="${REMOTE_OP_PACKAGE_TARGET:-}"
PROMPT="${PROMPT:-Explain how quantized Transformer inference maps matrix, vector, and data movement work onto a mobile NPU. Discuss attention, KV cache, MLP, and the cost of quantization conversions in enough detail to continue for at least sixty-four generated tokens.}"

REMOTE_RUNNER="${BASELINE_REMOTE_RUNNER}"
REMOTE_MODEL="${BASELINE_REMOTE_MODEL}"
REMOTE_TOKENIZER="${BASELINE_REMOTE_TOKENIZER}"
REMOTE_CONFIG="${BASELINE_REMOTE_CONFIG}"

REMOTE_OP_PACKAGE_ENV=""
if [[ -n "${REMOTE_OP_PACKAGE_PATH}" ]]; then
    REMOTE_OP_PACKAGE_ENV="export MLLM_QNN_OP_PACKAGE_PATH='${REMOTE_OP_PACKAGE_PATH}' &&"
fi
if [[ -n "${REMOTE_OP_PACKAGE_TARGET}" ]]; then
    REMOTE_OP_PACKAGE_ENV+=" export MLLM_QNN_OP_PACKAGE_TARGET='${REMOTE_OP_PACKAGE_TARGET}' &&"
fi

LOCAL_BUILD_BIN="${LOCAL_BUILD_BIN:-${REPO_ROOT}/build-android-arm64-v8a-qnn/bin}"
ANDROID_NDK_PATH="${ANDROID_NDK_PATH:-${ANDROID_NDK_ROOT:-}}"
LOCAL_LIBOMP="${LOCAL_LIBOMP:-${ANDROID_NDK_PATH:+${ANDROID_NDK_PATH}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so}}"
QAIRT_ANDROID_LIB="${QAIRT_ANDROID_LIB:-${QAIRT_SDK_ROOT}/lib/aarch64-android}"
QAIRT_V79_LIB="${QAIRT_V79_LIB:-${QAIRT_SDK_ROOT}/lib/hexagon-v79/unsigned}"
LLAMA_PACKAGE_BUILD="${LLAMA_PACKAGE_BUILD:-${REPO_ROOT}/mllm/backends/qnn/custom-op-package/LLaMAPackage/build}"
LOCAL_RUNNER="${LOCAL_RUNNER:-${REPO_ROOT}/${BASELINE_RUNNER_REL}}"
LOCAL_MODEL="${LOCAL_MODEL:-${MODEL_ROOT}/${BASELINE_MODEL_REL}}"
LOCAL_TOKENIZER="${LOCAL_TOKENIZER:-${MODEL_ROOT}/${BASELINE_TOKENIZER_REL}}"
LOCAL_CONFIG="${LOCAL_CONFIG:-${REPO_ROOT}/${BASELINE_CONFIG_REL}}"
ACCURACY_SUITE="${ACCURACY_SUITE:-${REPO_ROOT}/${BASELINE_ACCURACY_SUITE_REL}}"
SCHEMATIC_DIR="${SCHEMATIC_DIR:-${MODEL_ROOT}/${BASELINE_SCHEMATIC_REL}}"
MANIFEST_DIR="${MANIFEST_DIR:-${MODEL_ROOT}/${BASELINE_MANIFEST_REL}}"

# These values are fixed by the candidate contract.  Changing a pinned data
# artifact creates a new baseline; it must not silently reuse this result
# name.  The runner digest is retained as reference provenance only: this
# WSL build may differ because the ELF embeds source/debug paths and commit
# data, so the runner is presence-checked but deliberately not SHA-pinned.
EXPECTED_CONTEXT_SHA="${BASELINE_CONTEXT_SHA256}"
REFERENCE_RUNNER_SHA="${BASELINE_RUNNER_SHA256}"
EXPECTED_TOKENIZER_SHA="${BASELINE_TOKENIZER_SHA256}"
EXPECTED_CONFIG_SHA="${BASELINE_CONFIG_SHA256}"
EXPECTED_ACCURACY_SUITE_SHA="${BASELINE_ACCURACY_SUITE_SHA256}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

check_sha() {
    local label="$1"
    local path="$2"
    local expected="$3"
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] \
        || die "${label} SHA mismatch: expected ${expected}, got ${actual} (${path})"
}

[[ "${BENCHMARK_RUNS}" =~ ^[1-9][0-9]*$ ]] || die "BENCHMARK_RUNS must be a positive integer"
[[ "${MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_NEW_TOKENS must be a positive integer"
[[ "${ACCURACY_MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]] \
    || die "ACCURACY_MAX_NEW_TOKENS must be a positive integer"
[[ "${AR_LEN}" == "32" ]] || die "This V79 context contains s1 and s32; AR_LEN must be 32"
[[ -n "${QAIRT_SDK_ROOT:-}" ]] || die "QAIRT_SDK_ROOT is not set"
[[ "${QAIRT_SDK_ROOT}" == */"${BASELINE_QAIRT_RELEASE}" ]] \
    || die "QAIRT SDK release mismatch: expected ${BASELINE_QAIRT_RELEASE}, got ${QAIRT_SDK_ROOT}"
if (( ACCURACY_MAX_NEW_TOKENS < 32 )); then
    echo "WARNING: ACCURACY_MAX_NEW_TOKENS=${ACCURACY_MAX_NEW_TOKENS} is likely to truncate answers;"
    echo "         use 64 or more for a meaningful accuracy sanity score."
fi
command -v "${ADB_BIN}" >/dev/null || die "${ADB_BIN} is not in PATH"
command -v python3 >/dev/null || die "python3 is not in PATH"
if [[ -n "${RESUME_RESULT_ROOT}" ]]; then
    [[ -d "${RESULT_ROOT}" ]] || die "resume result directory does not exist: ${RESULT_ROOT}"
else
    [[ ! -e "${RESULT_ROOT}" ]] || die "timestamped result directory already exists: ${RESULT_ROOT}"
fi

PROFILE_VIEWER="${QAIRT_SDK_ROOT}/bin/x86_64-linux-clang/qnn-profile-viewer"
OPTRACE_READER="${QAIRT_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so"
PROFILE_WORK_ROOT="${PROFILE_WORK_ROOT:-}"
[[ -x "${PROFILE_VIEWER}" ]] || die "qnn-profile-viewer not found: ${PROFILE_VIEWER}"
[[ -f "${OPTRACE_READER}" ]] || die "Optrace reader not found: ${OPTRACE_READER}"
for graph in s1 s32; do
    [[ -f "${SCHEMATIC_DIR}/model.0.${graph}_schematic.bin" ]] \
        || die "schematic missing: ${SCHEMATIC_DIR}/model.0.${graph}_schematic.bin"
done
for graph in s1 s32; do
    [[ -f "${MANIFEST_DIR}/model.0.${graph}_quant_manifest.json" ]] \
        || die "quantization manifest missing: ${MANIFEST_DIR}/model.0.${graph}_quant_manifest.json"
done
check_sha "s1 quant manifest" "${MANIFEST_DIR}/model.0.s1_quant_manifest.json" \
    "${BASELINE_S1_MANIFEST_SHA256}"
check_sha "s32 quant manifest" "${MANIFEST_DIR}/model.0.s32_quant_manifest.json" \
    "${BASELINE_S32_MANIFEST_SHA256}"

ADB=("${ADB_BIN}")
if [[ -n "${ADB_SERIAL}" ]]; then
    ADB+=(-s "${ADB_SERIAL}")
fi
ADB_STATE="$("${ADB[@]}" get-state 2>/dev/null | tr -d '\r' || true)"
[[ "${ADB_STATE}" == "device" ]] \
    || die "no online Android device is available (adb state: ${ADB_STATE:-none})"

# Windows adb.exe runs outside WSL and cannot reliably resolve Linux mount
# paths such as /mnt/d/... on its host-side push/pull arguments. Keep paths in
# native WSL form for validation, hashing, and the profile viewer, translating
# only at the adb.exe boundary.
ADB_IS_WINDOWS=0
case "$(basename "${ADB_BIN}")" in
    *.exe|*.EXE) ADB_IS_WINDOWS=1 ;;
esac
adb_host_path() {
    local path="$1"
    if (( ADB_IS_WINDOWS )); then
        wslpath -w -- "${path}"
    else
        printf '%s\n' "${path}"
    fi
}
adb_push() {
    local source="$1"
    local destination="$2"
    "${ADB[@]}" push "$(adb_host_path "${source}")" "${destination}"
}
adb_pull() {
    local source="$1"
    local destination="$2"
    "${ADB[@]}" pull "${source}" "$(adb_host_path "${destination}")"
}

if [[ "${BUILD_ANDROID}" == "1" ]]; then
    echo "===== Build Android QNN runner ====="
    (cd "${REPO_ROOT}" && python3 task.py tasks/build_android_qnn.yaml)
fi

for path in "${LOCAL_RUNNER}" "${LOCAL_MODEL}" "${LOCAL_TOKENIZER}" "${LOCAL_CONFIG}"; do
    [[ -f "${path}" ]] || die "local artifact missing: ${path}"
done
[[ -n "${LOCAL_LIBOMP}" && -f "${LOCAL_LIBOMP}" ]] \
    || die "Android OpenMP runtime missing; set ANDROID_NDK_PATH or LOCAL_LIBOMP"
for qnn_lib in libQnnHtp.so libQnnSystem.so libQnnHtpV79Stub.so \
    libQnnHtpProfilingReader.so libQnnHtpOptraceProfilingReader.so libQnnHtpPrepare.so; do
    [[ -f "${QAIRT_ANDROID_LIB}/${qnn_lib}" ]] || die "QAIRT Android library missing: ${qnn_lib}"
done
[[ -f "${QAIRT_V79_LIB}/libQnnHtpV79Skel.so" ]] \
    || die "QAIRT V79 skel missing: ${QAIRT_V79_LIB}/libQnnHtpV79Skel.so"
[[ -f "${LLAMA_PACKAGE_BUILD}/aarch64-android/libQnnLLaMAPackage.so" ]] \
    || die "QNN LLaMA CPU package missing"
[[ -f "${LLAMA_PACKAGE_BUILD}/hexagon-v79/libQnnLLaMAPackage.so" ]] \
    || die "QNN LLaMA V79 package missing"
[[ -f "${ACCURACY_SUITE}" ]] || die "accuracy suite missing: ${ACCURACY_SUITE}"
LOCAL_RUNNER_SHA="$(sha256sum "${LOCAL_RUNNER}" | awk '{print $1}')"
check_sha "tokenizer" "${LOCAL_TOKENIZER}" "${EXPECTED_TOKENIZER_SHA}"
check_sha "config" "${LOCAL_CONFIG}" "${EXPECTED_CONFIG_SHA}"
check_sha "accuracy suite" "${ACCURACY_SUITE}" "${EXPECTED_ACCURACY_SUITE_SHA}"
LOCAL_CONTEXT_SHA="$(sha256sum "${LOCAL_MODEL}" | awk '{print $1}')"
[[ "${LOCAL_CONTEXT_SHA}" == "${EXPECTED_CONTEXT_SHA}" ]] \
    || die "G32 V79 context SHA mismatch: expected ${EXPECTED_CONTEXT_SHA}, got ${LOCAL_CONTEXT_SHA}"
check_sha "s1 schematic" "${SCHEMATIC_DIR}/model.0.s1_schematic.bin" \
    "${BASELINE_S1_SCHEMATIC_SHA256}"
check_sha "s32 schematic" "${SCHEMATIC_DIR}/model.0.s32_schematic.bin" \
    "${BASELINE_S32_SCHEMATIC_SHA256}"

mkdir -p "${RESULT_ROOT}/benchmark" "${RESULT_ROOT}/accuracy"
if [[ -z "${RESUME_RESULT_ROOT}" ]]; then
    cp "${CONTRACT_FILE}" "${RESULT_ROOT}/profile-contract.env"
    cp "${PROFILE_WRAPPER}" "${RESULT_ROOT}/experiment_script.sh"
    cp "${BASH_SOURCE[0]}" "${RESULT_ROOT}/base_profile_script.sh"
    cp "${ACCURACY_SUITE}" "${RESULT_ROOT}/accuracy/accuracy_suite.tsv"
    mkdir -p "${RESULT_ROOT}/manifests"
    cp "${MANIFEST_DIR}"/*_quant_manifest.json "${RESULT_ROOT}/manifests/"
    git -C "${REPO_ROOT}" rev-parse HEAD >"${RESULT_ROOT}/git_commit.txt"
    # The half.hpp modification predates this baseline and belongs to the user.
    # Preserve it without allowing it to contaminate this run's source-status proof.
    git -C "${REPO_ROOT}" status --short -- . \
        ':(exclude)third_party/half/include/half/half.hpp' >"${RESULT_ROOT}/git_status.txt"
    sha256sum "${LOCAL_RUNNER}" "${LOCAL_MODEL}" "${LOCAL_TOKENIZER}" "${LOCAL_CONFIG}" \
        "${ACCURACY_SUITE}" \
        "${SCHEMATIC_DIR}/model.0.s1_schematic.bin" "${SCHEMATIC_DIR}/model.0.s32_schematic.bin" \
        "${MANIFEST_DIR}/model.0.s1_quant_manifest.json" \
        "${MANIFEST_DIR}/model.0.s32_quant_manifest.json" \
        >"${RESULT_ROOT}/artifact_sha256.txt"
    "${ADB[@]}" shell getprop >"${RESULT_ROOT}/device_getprop.txt"
fi

{
    echo "timestamp=${TIMESTAMP}"
    echo "result_root=${RESULT_ROOT}"
    echo "capture_context=SM8750 native V79 G32"
    echo "baseline_id=${BASELINE_ID}"
    echo "baseline_contract=${CONTRACT_FILE}"
    echo "baseline_source_commit=${BASELINE_SOURCE_COMMIT}"
    echo "baseline_reference_result=${BASELINE_REFERENCE_RESULT}"
    echo "baseline_reference_commit=${BASELINE_REFERENCE_COMMIT}"
    echo "profile_scheme=${PROFILE_SCHEME}"
    echo "profile_wrapper=${PROFILE_WRAPPER}"
    echo "context_sha256=${EXPECTED_CONTEXT_SHA}"
    echo "runner_sha256=${LOCAL_RUNNER_SHA}"
    echo "runner_reference_sha256=${REFERENCE_RUNNER_SHA}"
    echo "tokenizer_sha256=${EXPECTED_TOKENIZER_SHA}"
    echo "config_sha256=${EXPECTED_CONFIG_SHA}"
    echo "accuracy_suite_sha256=${EXPECTED_ACCURACY_SUITE_SHA}"
    echo "s1_schematic_sha256=${BASELINE_S1_SCHEMATIC_SHA256}"
    echo "s32_schematic_sha256=${BASELINE_S32_SCHEMATIC_SHA256}"
    echo "qairt_sdk_root=${QAIRT_SDK_ROOT}"
    echo "benchmark_runs=${BENCHMARK_RUNS}"
    echo "max_new_tokens=${MAX_NEW_TOKENS}"
    echo "accuracy_suite=${ACCURACY_SUITE}"
    echo "accuracy_max_new_tokens=${ACCURACY_MAX_NEW_TOKENS}"
    echo "ar_len=${AR_LEN}"
    echo "remote_op_package_path=${REMOTE_OP_PACKAGE_PATH:-none}"
    echo "remote_op_package_target=${REMOTE_OP_PACKAGE_TARGET:-none}"
    echo "prompt=${PROMPT}"
    echo "benchmark_semantics=profiling off; fresh process per round; runner-level prefill/decode E2E"
    echo "accuracy_semantics=profiling off; one Runner reused with KV reset; greedy short-answer sanity suite"
    echo "accuracy_acceptance=informational only; no accuracy threshold"
    echo "speed_acceptance=informational only; runner E2E CSV required; no speed threshold"
    echo "optrace_semantics=fresh process per graph; first selected graph execution; one payload"
    echo "reference_result=${BASELINE_REFERENCE_RESULT}"
    echo "reference_critical_path=${BASELINE_REFERENCE_CRITICAL_PATH}"
} >"${RESULT_ROOT}/experiment_metadata.txt"
if [[ -n "${RESUME_RESULT_ROOT}" ]]; then
    printf 'resumed_at=%s\n' "$(date --iso-8601=seconds)" >>"${RESULT_ROOT}/experiment_metadata.txt"
fi

if [[ "${PREPARE_DEVICE}" == "1" ]]; then
    echo "===== Prepare device ====="
    "${ADB[@]}" shell "mkdir -p '${REMOTE_DIR}'"
    for local_so in "${LOCAL_BUILD_BIN}"/*.so; do
        [[ -f "${local_so}" ]] || continue
        adb_push "${local_so}" "${REMOTE_DIR}/" >/dev/null
    done
    adb_push "${LOCAL_LIBOMP}" "${REMOTE_DIR}/libomp.so" >/dev/null
    for qnn_lib in libQnnHtp.so libQnnSystem.so libQnnHtpV79Stub.so \
        libQnnHtpProfilingReader.so libQnnHtpOptraceProfilingReader.so libQnnHtpPrepare.so; do
        adb_push "${QAIRT_ANDROID_LIB}/${qnn_lib}" "${REMOTE_DIR}/${qnn_lib}" >/dev/null
    done
    adb_push "${QAIRT_V79_LIB}/libQnnHtpV79Skel.so" \
        "${REMOTE_DIR}/libQnnHtpV79Skel.so" >/dev/null
    adb_push "${LLAMA_PACKAGE_BUILD}/aarch64-android/libQnnLLaMAPackage.so" \
        "${REMOTE_DIR}/libQnnLLaMAPackage_CPU.so" >/dev/null
    adb_push "${LLAMA_PACKAGE_BUILD}/hexagon-v79/libQnnLLaMAPackage.so" \
        "${REMOTE_DIR}/libQnnLLaMAPackage_HTP.so" >/dev/null
    adb_push "${LOCAL_RUNNER}" "${REMOTE_DIR}/${REMOTE_RUNNER}" >/dev/null
    # Windows ADB may not preserve the executable bit when pushing from a
    # mounted WSL path.  The runner is the only pushed artifact that must be
    # executable; shared libraries are loaded by the runner and need no mode
    # change.
    "${ADB[@]}" shell "chmod 755 '${REMOTE_DIR}/${REMOTE_RUNNER}'"
    adb_push "${LOCAL_TOKENIZER}" "${REMOTE_DIR}/${REMOTE_TOKENIZER}" >/dev/null
    adb_push "${LOCAL_CONFIG}" "${REMOTE_DIR}/${REMOTE_CONFIG}" >/dev/null
    REMOTE_CONTEXT_SHA="$("${ADB[@]}" shell "sha256sum '${REMOTE_DIR}/${REMOTE_MODEL}' 2>/dev/null" \
        | awk '{print $1}' | tr -d '\r' || true)"
    if [[ "${REMOTE_CONTEXT_SHA}" != "${EXPECTED_CONTEXT_SHA}" ]]; then
        echo "Pushing 1.6 GiB G32 V79 context ..."
        adb_push "${LOCAL_MODEL}" "${REMOTE_DIR}/${REMOTE_MODEL}" >/dev/null
    fi
fi

for remote_file in "${REMOTE_RUNNER}" "${REMOTE_MODEL}" "${REMOTE_TOKENIZER}" "${REMOTE_CONFIG}"; do
    "${ADB[@]}" shell "test -r '${REMOTE_DIR}/${remote_file}'" \
        || die "device artifact missing: ${REMOTE_DIR}/${remote_file}"
done
REMOTE_CONTEXT_SHA="$("${ADB[@]}" shell "sha256sum '${REMOTE_DIR}/${REMOTE_MODEL}'" \
    | awk '{print $1}' | tr -d '\r')"
[[ "${REMOTE_CONTEXT_SHA}" == "${EXPECTED_CONTEXT_SHA}" ]] \
    || die "device context SHA mismatch: expected ${EXPECTED_CONTEXT_SHA}, got ${REMOTE_CONTEXT_SHA}"

check_remote_sha() {
    local label="$1"
    local remote_path="$2"
    local expected="$3"
    local actual
    actual="$("${ADB[@]}" shell "sha256sum '${remote_path}'" \
        | awk '{print $1}' | tr -d '\r')"
    [[ "${actual}" == "${expected}" ]] \
        || die "device ${label} SHA mismatch: expected ${expected}, got ${actual} (${remote_path})"
}

check_remote_sha "tokenizer" "${REMOTE_DIR}/${REMOTE_TOKENIZER}" "${EXPECTED_TOKENIZER_SHA}"
check_remote_sha "config" "${REMOTE_DIR}/${REMOTE_CONFIG}" "${EXPECTED_CONFIG_SHA}"
"${ADB[@]}" shell "mkdir -p '${REMOTE_ROOT}'"
adb_push "${ACCURACY_SUITE}" "${REMOTE_ROOT}/accuracy_suite.tsv" >/dev/null

echo "===== Profiling-off E2E throughput (${BENCHMARK_RUNS} rounds) ====="
for ((round = 1; round <= BENCHMARK_RUNS; ++round)); do
    run_name="$(printf 'run_%02d' "${round}")"
    host_dir="${RESULT_ROOT}/benchmark/${run_name}"
    remote_run="${REMOTE_ROOT}/benchmark/${run_name}"
    if [[ -s "${host_dir}/qnn_runner_e2e.csv" ]]; then
        echo "Resume: keeping completed ${run_name}"
        continue
    fi
    mkdir -p "${host_dir}"
    "${ADB[@]}" shell "mkdir -p '${remote_run}'"
    "${ADB[@]}" shell dumpsys thermalservice >"${host_dir}/thermal_before.txt" || true
    set +e
    printf '%s\n' "${PROMPT}" | "${ADB[@]}" shell "
        cd '${REMOTE_DIR}' &&
        export LD_LIBRARY_PATH=.:${REMOTE_DIR} &&
        export ADSP_LIBRARY_PATH='${REMOTE_DIR}' &&
        ${REMOTE_OP_PACKAGE_ENV}
        export MLLM_QNN_PROFILE_LEVEL=off &&
        export MLLM_QNN_PROFILE_DIR='${remote_run}' &&
        './${REMOTE_RUNNER}' \
            -m '${REMOTE_MODEL}' \
            -t '${REMOTE_TOKENIZER}' \
            -c '${REMOTE_CONFIG}' \
            --ar_len '${AR_LEN}' \
            --max_new_tokens '${MAX_NEW_TOKENS}' \
            --perf
    " 2>&1 | tee "${host_dir}/runner.log"
    run_status="${PIPESTATUS[1]}"
    set -e
    [[ "${run_status}" == "0" ]] || die "benchmark ${run_name} failed with status ${run_status}"
    adb_pull "${remote_run}/qnn_runner_e2e.csv" "${host_dir}/qnn_runner_e2e.csv" >/dev/null
    if "${ADB[@]}" shell "test -f '${remote_run}/qnn_e2e_profile.csv'"; then
        adb_pull "${remote_run}/qnn_e2e_profile.csv" "${host_dir}/qnn_graph_module_e2e.csv" >/dev/null
    fi
    "${ADB[@]}" shell dumpsys thermalservice >"${host_dir}/thermal_after.txt" || true
    if ((round < BENCHMARK_RUNS && BENCHMARK_COOLDOWN_SEC > 0)); then
        sleep "${BENCHMARK_COOLDOWN_SEC}"
    fi
done

echo "===== Profiling-off lightweight accuracy sanity ====="
remote_accuracy="${REMOTE_ROOT}/accuracy"
"${ADB[@]}" shell "mkdir -p '${remote_accuracy}'"
if [[ -s "${RESULT_ROOT}/qwen3-sm8750-v79-g32-accuracy.json" ]]; then
    echo "Resume: keeping completed accuracy sanity"
else
set +e
"${ADB[@]}" shell "
    cd '${REMOTE_DIR}' &&
    export LD_LIBRARY_PATH=.:${REMOTE_DIR} &&
    export ADSP_LIBRARY_PATH='${REMOTE_DIR}' &&
    ${REMOTE_OP_PACKAGE_ENV}
    export MLLM_QNN_PROFILE_LEVEL=off &&
    export MLLM_QNN_PROFILE_DIR='${remote_accuracy}' &&
    './${REMOTE_RUNNER}' \
        -m '${REMOTE_MODEL}' \
        -t '${REMOTE_TOKENIZER}' \
        -c '${REMOTE_CONFIG}' \
        --ar_len '${AR_LEN}' \
        --max_new_tokens '${ACCURACY_MAX_NEW_TOKENS}' \
        --eval_file '${REMOTE_ROOT}/accuracy_suite.tsv'
" 2>&1 | tee "${RESULT_ROOT}/accuracy/runner.log"
accuracy_status="${PIPESTATUS[0]}"
set -e
[[ "${accuracy_status}" == "0" ]] || die "accuracy sanity runner failed with status ${accuracy_status}"
adb_pull "${remote_accuracy}/qnn_accuracy_eval.csv" \
    "${RESULT_ROOT}/accuracy/qnn_accuracy_eval.csv" >/dev/null
python3 "${REPO_ROOT}/scripts/qnn_accuracy_summary.py" \
    "${RESULT_ROOT}/accuracy/qnn_accuracy_eval.csv" \
    --max-new-tokens "${ACCURACY_MAX_NEW_TOKENS}" \
    --output "${RESULT_ROOT}/qwen3-sm8750-v79-g32-accuracy.json" \
    | tee "${RESULT_ROOT}/accuracy-summary.log"
fi

echo "===== Fresh-process Optrace captures ====="
for graph in s32 s1; do
    host_prefix="${RESULT_ROOT}/qwen3-sm8750-v79-g32-${graph}"
    remote_capture="${REMOTE_ROOT}/optrace_${graph}"
    graph_name="model.0.${graph}"
    schematic="${SCHEMATIC_DIR}/${graph_name}_schematic.bin"
    chrome_trace="${host_prefix}-chrometrace.json"
    htp_json="${host_prefix}-chrometrace_htp.json"
    qhas_json="${host_prefix}-chrometrace_qnn_htp_analysis_summary.json"
    if [[ -s "${chrome_trace}" && -s "${htp_json}" && -s "${qhas_json}" ]]; then
        echo "Resume: keeping completed ${graph} Optrace decode"
    else
    "${ADB[@]}" shell "mkdir -p '${remote_capture}'"
    "${ADB[@]}" shell dumpsys thermalservice >"${host_prefix}-thermal-before.txt" || true

    set +e
    printf '%s\n' "${PROMPT}" | "${ADB[@]}" shell "
        cd '${REMOTE_DIR}' &&
        export LD_LIBRARY_PATH=.:${REMOTE_DIR} &&
        export ADSP_LIBRARY_PATH='${REMOTE_DIR}' &&
        ${REMOTE_OP_PACKAGE_ENV}
        export MLLM_QNN_PROFILE_LEVEL=optrace &&
        export MLLM_QNN_PROFILE_WARMUP=0 &&
        export MLLM_QNN_PROFILE_EVERY=1 &&
        export MLLM_QNN_PROFILE_MAX_CAPTURES=1 &&
        export MLLM_QNN_PROFILE_GRAPH='${graph_name}' &&
        export MLLM_QNN_PROFILE_SERIALIZE=1 &&
        export MLLM_QNN_PROFILE_DIR='${remote_capture}' &&
        './${REMOTE_RUNNER}' \
            -m '${REMOTE_MODEL}' \
            -t '${REMOTE_TOKENIZER}' \
            -c '${REMOTE_CONFIG}' \
            --ar_len '${AR_LEN}' \
            --max_new_tokens 8
    " 2>&1 | tee "${host_prefix}-runner.log"
    run_status="${PIPESTATUS[1]}"
    set -e
    [[ "${run_status}" == "0" ]] || die "${graph} Optrace runner failed with status ${run_status}"

    adb_pull "${remote_capture}/qnn-profiling-data.log" "${host_prefix}-optrace.log" >/dev/null
    for profile_file in qnn_detail_profile.txt qnn_macro_profile.csv qnn_e2e_profile.csv; do
        if "${ADB[@]}" shell "test -f '${remote_capture}/${profile_file}'"; then
            suffix="${profile_file#qnn_}"
            adb_pull "${remote_capture}/${profile_file}" "${host_prefix}-${suffix}" >/dev/null
        fi
    done
    "${ADB[@]}" shell dumpsys thermalservice >"${host_prefix}-thermal-after.txt" || true

    if [[ -n "${PROFILE_WORK_ROOT}" ]]; then
        viewer_work="${PROFILE_WORK_ROOT}/${graph}"
        mkdir -p "${viewer_work}"
        viewer_prefix="${viewer_work}/$(basename "${host_prefix}")"
        cp "${host_prefix}-optrace.log" "${viewer_prefix}-optrace.log"
        cp "${schematic}" "${viewer_work}/$(basename "${schematic}")"
        "${PROFILE_VIEWER}" \
            --reader "${OPTRACE_READER}" \
            --input_log "${viewer_prefix}-optrace.log" \
            --schematic "${viewer_work}/$(basename "${schematic}")" \
            --output "${viewer_prefix}-chrometrace.json" \
            2>&1 | tee "${host_prefix}-profile-viewer.log"
        for viewer_artifact in "${viewer_prefix}"*; do
            [[ -f "${viewer_artifact}" ]] || continue
            cp "${viewer_artifact}" "${RESULT_ROOT}/$(basename "${viewer_artifact}")"
        done
    else
        "${PROFILE_VIEWER}" \
            --reader "${OPTRACE_READER}" \
            --input_log "${host_prefix}-optrace.log" \
            --schematic "${schematic}" \
            --output "${chrome_trace}" \
            2>&1 | tee "${host_prefix}-profile-viewer.log"
    fi
    [[ -s "${chrome_trace}" && -s "${htp_json}" && -s "${qhas_json}" ]] \
        || die "${graph}: viewer did not generate all required artifacts"
    fi

    python3 "${REPO_ROOT}/scripts/qnn_optrace_summary.py" \
        "${chrome_trace}" \
        --htp-json "${htp_json}" \
        --output "${host_prefix}-operators.csv" \
        --type-summary-output "${host_prefix}-logical-types.csv" \
        >"${host_prefix}-summary-generation.log"
    python3 "${REPO_ROOT}/scripts/qnn_optrace_qwen3_structure.py" \
        "${chrome_trace}" \
        --htp-json "${htp_json}" \
        --qhas-json "${qhas_json}" \
        --output-prefix "${host_prefix}" \
        >"${host_prefix}-structure-generation.log"
    python3 "${REPO_ROOT}/scripts/qnn_optrace_quantization.py" \
        "${chrome_trace}" \
        --qhas-json "${qhas_json}" \
        --quant-manifest "${MANIFEST_DIR}/${graph_name}_quant_manifest.json" \
        --output-prefix "${host_prefix}" \
        >"${host_prefix}-quantization-generation.log"
done

acceptance_args=()
if [[ "${RMSNORM_U8_CONTRACT}" == "1" ]]; then
    acceptance_args+=(--rmsnorm-u8)
fi
python3 "${REPO_ROOT}/scripts/qnn_w4a8_acceptance.py" \
    --s1-manifest "${MANIFEST_DIR}/model.0.s1_quant_manifest.json" \
    --s1-trace "${RESULT_ROOT}/qwen3-sm8750-v79-g32-s1-chrometrace.json" \
    --s1-qhas "${RESULT_ROOT}/qwen3-sm8750-v79-g32-s1-chrometrace_qnn_htp_analysis_summary.json" \
    --s32-manifest "${MANIFEST_DIR}/model.0.s32_quant_manifest.json" \
    --s32-trace "${RESULT_ROOT}/qwen3-sm8750-v79-g32-s32-chrometrace.json" \
    --s32-qhas "${RESULT_ROOT}/qwen3-sm8750-v79-g32-s32-chrometrace_qnn_htp_analysis_summary.json" \
    --output "${RESULT_ROOT}/qwen3-sm8750-v79-g32-w4a8-acceptance.json" \
    "${acceptance_args[@]}" \
    | tee "${RESULT_ROOT}/w4a8-acceptance.log"

echo "===== Throughput summary and canonical report ====="
benchmark_csv=("${RESULT_ROOT}"/benchmark/run_*/qnn_runner_e2e.csv)
[[ "${#benchmark_csv[@]}" == "${BENCHMARK_RUNS}" ]] \
    || die "runner E2E CSV count mismatch: expected ${BENCHMARK_RUNS}, got ${#benchmark_csv[@]}"
for csv_path in "${benchmark_csv[@]}"; do
    [[ -s "${csv_path}" ]] || die "formal comparison forbids QHAS fallback; missing runner CSV: ${csv_path}"
done
python3 "${REPO_ROOT}/scripts/qnn_profile_speed_summary.py" \
    --forbid-fallback \
    "${benchmark_csv[@]}" \
    --s1-qhas "${RESULT_ROOT}/qwen3-sm8750-v79-g32-s1-chrometrace_qnn_htp_analysis_summary.json" \
    --s32-qhas "${RESULT_ROOT}/qwen3-sm8750-v79-g32-s32-chrometrace_qnn_htp_analysis_summary.json" \
    --output "${RESULT_ROOT}/qwen3-sm8750-v79-g32-speed.json" \
    | tee "${RESULT_ROOT}/speed-summary.log"
python3 "${REPO_ROOT}/scripts/qnn_speed_comparison.py" \
    --candidate "${RESULT_ROOT}/qwen3-sm8750-v79-g32-speed.json" \
    --reference "${RESULTS_BASE}/${BASELINE_REFERENCE_RESULT}/qwen3-sm8750-v79-g32-speed.json" \
    --reference-id "${BASELINE_REFERENCE_RESULT}" \
    --output "${RESULT_ROOT}/qwen3-sm8750-v79-g32-speed-comparison.json" \
    | tee "${RESULT_ROOT}/speed-comparison.log"

# The shared report generator expects the historical qwen3-sm8750-v79-
# {s1,s32} basename. Keep the G32 artifacts authoritative, while adding
# result-local aliases only for that reader.
for graph in s1 s32; do
    g32_prefix="${RESULT_ROOT}/qwen3-sm8750-v79-g32-${graph}-"
    legacy_prefix="${RESULT_ROOT}/qwen3-sm8750-v79-${graph}-"
    for source in "${g32_prefix}"*; do
        [[ -e "${source}" ]] || continue
        suffix="${source#${g32_prefix}}"
        ln -sfn "$(basename "${source}")" "${legacy_prefix}${suffix}"
    done
done

FINAL_REPORT="${RESULT_ROOT}/qwen3-sm8750-v79-g32-e2e-critical-path.html"
python3 "${REPO_ROOT}/scripts/qnn_e2e_critical_path_report.py" \
    --results-dir "${RESULT_ROOT}" \
    --speed-json "${RESULT_ROOT}/qwen3-sm8750-v79-g32-speed.json" \
    --accuracy-json "${RESULT_ROOT}/qwen3-sm8750-v79-g32-accuracy.json" \
    --speed-comparison-json "${RESULT_ROOT}/qwen3-sm8750-v79-g32-speed-comparison.json" \
    --output "${FINAL_REPORT}" \
    | tee "${RESULT_ROOT}/report-generation.log"
[[ -s "${FINAL_REPORT}" ]] || die "final HTML report was not generated"

if [[ "${CLEAN_REMOTE}" == "1" ]]; then
    "${ADB[@]}" shell "rm -rf '${REMOTE_ROOT}'"
fi

echo
echo "Finished."
echo "Results: ${RESULT_ROOT}"
echo "Report:  ${FINAL_REPORT}"
