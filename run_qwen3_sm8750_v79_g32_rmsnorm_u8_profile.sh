#!/usr/bin/env bash
set -Eeuo pipefail

# Candidate-only wrapper. The build script publishes a run-specific contract;
# this wrapper selects it and turns on the all-729 native-U8 RMSNorm audit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/d/llm_exp}"
RUN_ID="${1:?usage: run_qwen3_sm8750_v79_g32_rmsnorm_u8_profile.sh RUN_ID}"
CONTRACT_FILE="${CONTRACT_FILE:-${ARTIFACT_ROOT}/results/qwen3_sm8750_v79_w4a8_rmsnorm_u8_${RUN_ID}/profile-contract.env}"

[[ -r "${CONTRACT_FILE}" ]] || {
    echo "ERROR: candidate profile contract not found: ${CONTRACT_FILE}" >&2
    echo "Run scripts/build_qwen3_w4a8g32_rmsnorm_u8.sh ${RUN_ID} first." >&2
    exit 2
}

export ARTIFACT_ROOT
export CONTRACT_FILE
export RESULT_PREFIX="qwen3_sm8750_v79_w4a8_rmsnorm_u8"
export RMSNORM_U8_CONTRACT=1
export REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/mllm_w4a8_rmsnorm_u8}"
exec "${SCRIPT_DIR}/run_qwen3_sm8750_v79_g32_profile.sh" "$@"
