#!/usr/bin/env bash

# Host-only preflight for the experimental W4A8G32 baseline.  It deliberately
# does not require adb or QAIRT, so it can be run immediately after moving the
# repository/models tree to another VM before attempting a device run.
#
# The runner is intentionally presence-checked but not SHA-pinned.  Its ELF
# contains build-path/debug and source-commit data, so a functionally
# equivalent WSL build can legitimately have a different digest from the VM
# reference binary.  The reference digest remains in baseline.env as
# provenance; model/config/schematic artifacts remain SHA-pinned below.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONTRACT_FILE="${CONTRACT_FILE:-${REPO_ROOT}/profiles/qwen3_sm8750_v79_g32/baseline.env}"
[[ -r "${CONTRACT_FILE}" ]] || {
    echo "ERROR: baseline contract not found: ${CONTRACT_FILE}" >&2
    exit 1
}
# shellcheck disable=SC1090
source "${CONTRACT_FILE}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/d/llm_exp}"
MODEL_ROOT="${MODEL_ROOT:-${ARTIFACT_ROOT}/models}"
LOCAL_RUNNER="${LOCAL_RUNNER:-${REPO_ROOT}/${BASELINE_RUNNER_REL}}"
LOCAL_MODEL="${LOCAL_MODEL:-${MODEL_ROOT}/${BASELINE_MODEL_REL}}"
LOCAL_TOKENIZER="${LOCAL_TOKENIZER:-${MODEL_ROOT}/${BASELINE_TOKENIZER_REL}}"
LOCAL_CONFIG="${LOCAL_CONFIG:-${REPO_ROOT}/${BASELINE_CONFIG_REL}}"
ACCURACY_SUITE="${ACCURACY_SUITE:-${REPO_ROOT}/${BASELINE_ACCURACY_SUITE_REL}}"
SCHEMATIC_DIR="${SCHEMATIC_DIR:-${MODEL_ROOT}/${BASELINE_SCHEMATIC_REL}}"
MANIFEST_DIR="${MANIFEST_DIR:-${MODEL_ROOT}/${BASELINE_MANIFEST_REL}}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

check_sha() {
    local label="$1"
    local path="$2"
    local expected="$3"
    [[ -f "${path}" ]] || die "${label} missing: ${path}"
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] \
        || die "${label} SHA mismatch: expected ${expected}, got ${actual} (${path})"
    printf 'OK  %-16s %s\n' "${label}" "${actual}"
}

[[ -f "${LOCAL_RUNNER}" ]] || die "runner missing: ${LOCAL_RUNNER}"
printf 'INFO runner         SHA check disabled; actual %s\n' \
    "$(sha256sum "${LOCAL_RUNNER}" | awk '{print $1}')"
check_sha context "${LOCAL_MODEL}" "${BASELINE_CONTEXT_SHA256}"
check_sha tokenizer "${LOCAL_TOKENIZER}" "${BASELINE_TOKENIZER_SHA256}"
check_sha config "${LOCAL_CONFIG}" "${BASELINE_CONFIG_SHA256}"
check_sha accuracy_suite "${ACCURACY_SUITE}" "${BASELINE_ACCURACY_SUITE_SHA256}"
check_sha schematic_s1 "${SCHEMATIC_DIR}/model.0.s1_schematic.bin" \
    "${BASELINE_S1_SCHEMATIC_SHA256}"
check_sha schematic_s32 "${SCHEMATIC_DIR}/model.0.s32_schematic.bin" \
    "${BASELINE_S32_SCHEMATIC_SHA256}"
check_sha manifest_s1 "${MANIFEST_DIR}/model.0.s1_quant_manifest.json" \
    "${BASELINE_S1_MANIFEST_SHA256}"
check_sha manifest_s32 "${MANIFEST_DIR}/model.0.s32_quant_manifest.json" \
    "${BASELINE_S32_MANIFEST_SHA256}"

echo
echo "Baseline contract: ${BASELINE_ID}"
echo "Source commit:     ${BASELINE_SOURCE_COMMIT}"
echo "Reference result:  ${BASELINE_REFERENCE_RESULT}"
echo "Reference commit:  ${BASELINE_REFERENCE_COMMIT}"
echo "QAIRT release:     ${BASELINE_QAIRT_RELEASE}"
echo "Preflight: PASS"
