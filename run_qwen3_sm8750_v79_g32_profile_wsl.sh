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

# WSL/ADB defaults.  Artifact paths and hashes come only from the canonical
# contract; this wrapper never creates a model/config/suite variant.
export REPO_ROOT ARTIFACT_ROOT MODEL_ROOT RESULTS_BASE QAIRT_SDK_ROOT ADB_BIN
export CONTRACT_FILE="${CONTRACT_FILE:-${REPO_ROOT}/profiles/qwen3_sm8750_v79_g32/baseline.env}"
export BUILD_ANDROID="${BUILD_ANDROID:-0}"
export PREPARE_DEVICE="${PREPARE_DEVICE:-1}"
export BENCHMARK_RUNS="${BENCHMARK_RUNS:-3}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export ACCURACY_MAX_NEW_TOKENS="${ACCURACY_MAX_NEW_TOKENS:-64}"
export AR_LEN="${AR_LEN:-32}"
# The canonical script reads all model/config/hash values from CONTRACT_FILE;
# the wrapper only selects the WSL/ADB defaults and never creates a variant.

exec "${CANONICAL}"
