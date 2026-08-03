#!/usr/bin/env bash
set -euo pipefail

# Run the first P0 matrix on the already collected 0/13/27 BF16 shards.
# The fit/eval split is selected by the Python runner from shard filenames;
# no model graph is kept alive between rows.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALIB_DIR="${CALIB_DIR:-/home/daniuniu/llm_exp/calibration/qwen3-p1-layers-0-13-27-seed17-s96}"
MODEL_DIR="${MODEL_DIR:-/home/daniuniu/llm_exp/models/Qwen3-origin}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/p0/static_a8}"
MAX_INPUT_VALUES="${MAX_INPUT_VALUES:-10000}"
LEARNABLE_STEPS="${LEARNABLE_STEPS:-100}"
LEARNABLE_LR="${LEARNABLE_LR:-0.03}"

mkdir -p "${OUTPUT_DIR}"
for layer in 0 13 27; do
  for projection in o_proj down_proj; do
    if [[ "${projection}" == "o_proj" ]]; then
      input_key="layer_$(printf '%02d' "${layer}").o_proj_input"
      weight_key="model.layers.${layer}.self_attn.o_proj.weight"
    else
      input_key="layer_$(printf '%02d' "${layer}").down_proj_input"
      weight_key="model.layers.${layer}.mlp.down_proj.weight"
    fi
    output_json="${OUTPUT_DIR}/layer$(printf '%02d' "${layer}")-${projection}.json"
    echo "[P0] layer=${layer} projection=${projection}"
    python "${REPO_ROOT}/scripts/qwen3_p0_static_a8.py" \
      --inputs "${CALIB_DIR}" \
      --input-key "${input_key}" \
      --weight "${MODEL_DIR}" \
      --weight-key "${weight_key}" \
      --max-input-values "${MAX_INPUT_VALUES}" \
      --device cuda \
      --learnable-steps "${LEARNABLE_STEPS}" \
      --learnable-lr "${LEARNABLE_LR}" \
      --output-json "${output_json}" \
      > "${output_json%.json}.log"
  done
done

echo "P0 matrix complete: ${OUTPUT_DIR}"
