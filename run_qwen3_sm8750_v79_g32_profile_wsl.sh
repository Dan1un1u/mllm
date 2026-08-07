#!/usr/bin/env bash

# Thin WSL/host entry point.  All capture, pulling, QNN viewer conversion,
# structure summaries, and final HTML generation remain in the canonical
# repository script.  Do not duplicate the device protocol in a PowerShell
# wrapper; that is how host runs previously lost the post-processing stage.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-${ARTIFACT_ROOT}/models}"
RESULTS_BASE="${RESULTS_BASE:-${ARTIFACT_ROOT}/results}"
QAIRT_SDK_ROOT="${QAIRT_SDK_ROOT:-/opt/qcom/aistack/qairt/2.47.0.260601}"
ADB_BIN="${ADB_BIN:-adb}"

CANONICAL="${REPO_ROOT}/run_qwen3_sm8750_v79_g32_profile.sh"
[[ -f "${CANONICAL}" ]] || {
    echo "ERROR: canonical G32 profiling script not found: ${CANONICAL}" >&2
    exit 1
}

# Baseline defaults.  Every value remains overridable for a deliberate
# experiment, but the wrapper never substitutes a different model or suite.
export REPO_ROOT ARTIFACT_ROOT MODEL_ROOT RESULTS_BASE QAIRT_SDK_ROOT ADB_BIN
export BUILD_ANDROID="${BUILD_ANDROID:-0}"
export PREPARE_DEVICE="${PREPARE_DEVICE:-1}"
export BENCHMARK_RUNS="${BENCHMARK_RUNS:-3}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export ACCURACY_MAX_NEW_TOKENS="${ACCURACY_MAX_NEW_TOKENS:-64}"
export AR_LEN="${AR_LEN:-32}"
export REMOTE_MODEL="${REMOTE_MODEL:-qwen3-1.7B-lpbq-sha-g32.bin}"
export LOCAL_MODEL="${LOCAL_MODEL:-${MODEL_ROOT}/qwen3_sm8750_v79/g32/w4a16/qwen3-1.7B-lpbq-sha-g32.bin}"
export LOCAL_TOKENIZER="${LOCAL_TOKENIZER:-${MODEL_ROOT}/Qwen3-origin/qwen3-tokenizer.json}"
export LOCAL_CONFIG="${LOCAL_CONFIG:-${REPO_ROOT}/examples/qwen3_qnn_aot/config_1.7B_g32.json}"
export ACCURACY_SUITE="${ACCURACY_SUITE:-${REPO_ROOT}/scripts/qwen3_sm8750_v79_accuracy.tsv}"
export SCHEMATIC_DIR="${SCHEMATIC_DIR:-${MODEL_ROOT}/qwen3_sm8750_v79/g32/w4a16/schematics}"
export EXPECTED_CONTEXT_SHA="${EXPECTED_CONTEXT_SHA:-f637b4ddbd63478205679f40642fd24801093bb98ddf0808f13d99e3fb155d5d}"

# A baseline run must use the same answer budget as the VM reference.  Set
# STRICT_BASELINE=0 only for an intentionally shortened smoke test.
if [[ "${STRICT_BASELINE:-1}" == "1" && "${ACCURACY_MAX_NEW_TOKENS}" != "64" ]]; then
    echo "ERROR: baseline accuracy requires ACCURACY_MAX_NEW_TOKENS=64; got ${ACCURACY_MAX_NEW_TOKENS}" >&2
    exit 2
fi

if [[ "${STRICT_BASELINE:-1}" == "1" ]]; then
    # These are the VM G32 baseline identities recorded in the reference
    # result.  A host build with another runner/suite/config is allowed only
    # when STRICT_BASELINE=0 and must not be compared as the same baseline.
    export EXPECTED_RUNNER_SHA="${EXPECTED_RUNNER_SHA:-f8a00d53b001e017405b6d3061b57195580bfc2877bb14eb8e335f21c7f6452d}"
    export EXPECTED_TOKENIZER_SHA="${EXPECTED_TOKENIZER_SHA:-aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4}"
    export EXPECTED_CONFIG_SHA="${EXPECTED_CONFIG_SHA:-1cf89d946a8138be13050bb2125b1302d9157a308d77e391d3ce7eceb2f22db0}"
    export EXPECTED_ACCURACY_SUITE_SHA="${EXPECTED_ACCURACY_SUITE_SHA:-5bbcd2f39d176511d216c897191887c9e4577dc6a49d85c01c4dd7a4c6ad931c}"
fi

exec "${CANONICAL}"
