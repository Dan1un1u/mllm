#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALIB_DIR="${CALIB_DIR:-/home/daniuniu/llm_exp/calibration/qwen3-p0-all-layers-seed17-s96}"
MODEL_DIR="${MODEL_DIR:-/home/daniuniu/llm_exp/models/Qwen3-origin}"
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/artifacts/p0/static_a8/sensitivity-map.json}"

python "${REPO_ROOT}/scripts/qwen3_p0_sensitivity_map.py" \
  --calibration-dir "${CALIB_DIR}" \
  --model "${MODEL_DIR}" \
  --learnable-steps "${LEARNABLE_STEPS:-50}" \
  --learnable-lr "${LEARNABLE_LR:-0.03}" \
  --a8-nmse-threshold "${A8_NMSE_THRESHOLD:-0.02}" \
  --device "${DEVICE:-cuda}" \
  --output-json "${OUTPUT_JSON}"

