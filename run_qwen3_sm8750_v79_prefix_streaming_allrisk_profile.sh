#!/usr/bin/env bash

# One-click SM8750/V79 QNN profiling for the prefix-streaming all-risk policy.
#
# This wrapper deliberately requires a native context compiled from the
# streaming scales.  It must not fall back to the archived G32 baseline or to
# the older actaware/selective contexts that are still present on the device.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
SCHEME="prefix_streaming_allrisk"
SCALE_DIR="${STREAMING_SCALE_DIR:-${REPO_ROOT}/artifacts/p1/streaming-full-allrisk-mapzp}"
STREAMING_MANIFEST="${STREAMING_MANIFEST:-${SCALE_DIR}/streaming-train.json}"
PRECISION_MAP="${PRECISION_MAP:-${REPO_ROOT}/artifacts/p0/static_a8/mixed-precision-map-all-risk-full.json}"
RESULTS_BASE="${RESULTS_BASE:-/mnt/d/llm_exp/results/${SCHEME}}"
QAIRT_SDK_ROOT="${QAIRT_SDK_ROOT:-/mnt/d/llm_exp/models/qualcomm-sdk/qairt/2.47.0.260601}"
LOCAL_MODEL="${LOCAL_MODEL:-/tmp/qwen3-1.7B-prefix-streaming-allrisk-sm8750-v79.bin}"
REMOTE_MODEL="${REMOTE_MODEL:-qwen3-1.7B-prefix-streaming-allrisk-sm8750-v79.bin}"
ADB_BIN="${ADB_BIN:-adb.exe}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "${SCALE_DIR}" ]] || die "streaming scale directory missing: ${SCALE_DIR}"
[[ -f "${STREAMING_MANIFEST}" ]] || die "streaming manifest missing: ${STREAMING_MANIFEST}"
[[ -f "${PRECISION_MAP}" ]] || die "precision map missing: ${PRECISION_MAP}"
scale_count="$(find "${SCALE_DIR}" -maxdepth 1 -type f -name 'layer*-lpbq-scales.safetensors' | wc -l)"
[[ "${scale_count}" == "28" ]] || die "expected 28 layer scale files in ${SCALE_DIR}, found ${scale_count}"
[[ -f "${LOCAL_MODEL}" ]] || die "native streaming all-risk context missing: ${LOCAL_MODEL}; compile the G32 V79 context first"

LOCAL_CONTEXT_SHA="$(sha256sum "${LOCAL_MODEL}" | awk '{print $1}')"
EXPECTED_CONTEXT_SHA="${EXPECTED_CONTEXT_SHA:-${LOCAL_CONTEXT_SHA}}"
[[ "${LOCAL_CONTEXT_SHA}" == "${EXPECTED_CONTEXT_SHA}" ]] \
    || die "native context SHA mismatch: expected ${EXPECTED_CONTEXT_SHA}, got ${LOCAL_CONTEXT_SHA}"

export REPO_ROOT RESULTS_BASE QAIRT_SDK_ROOT LOCAL_MODEL REMOTE_MODEL ADB_BIN
export PROFILE_SCHEME="${SCHEME}"
export PROFILE_WRAPPER="${BASH_SOURCE[0]}"
export STREAMING_SCALE_DIR="${SCALE_DIR}"
export STREAMING_MANIFEST PRECISION_MAP
export NATIVE_CONTEXT_PROVENANCE="prefix-streaming all-risk native G32 V79 context; SHA verified"
export OFFLINE_LOGITS_COSINE="0.889310"
export OFFLINE_TOP1_AGREEMENT="1.000000"
export OFFLINE_LOGITS_NMSE="0.221470"

exec bash "${REPO_ROOT}/run_qwen3_sm8750_v79_g32_profile.sh" "$@"
