#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root=/mnt/d/llm_exp/models
results_root=/mnt/d/llm_exp/results
source_model="${SOURCE_MODEL:-${models_root}/Qwen3-origin}"
venv="${W4A8_VENV:-${models_root}/w4a8g32/intermediate/python/venv}"
run_id="${1:-$(date +%Y%m%d_%H%M%S)}"
stage_dir="${models_root}/w4a8g32/staging/${run_id}"
calibration_dir="${models_root}/w4a8g32/calibration"
corpus="${calibration_dir}/qwen3_wikitext_103_v1_train_128x512.jsonl"
log_dir="${results_root}/w4a8g32_build_logs/${run_id}"
output_model="${stage_dir}/qwen3_1.7b_w4a8g32.mllm"

[[ -d "${repo_root}/.git" ]] || { echo "not an mllm source checkout: ${repo_root}" >&2; exit 2; }
[[ -d "${source_model}" ]] || { echo "source model missing: ${source_model}" >&2; exit 2; }
[[ -x "${venv}/bin/python" ]] || { echo "Python environment missing: ${venv}" >&2; exit 2; }
[[ ! -e "${stage_dir}" ]] || { echo "staging run already exists: ${stage_dir}" >&2; exit 2; }

mkdir -p "${stage_dir}/intermediate" "${calibration_dir}" "${log_dir}"

capture_args=()
if [[ ! -f "${corpus}" ]]; then
  capture_args+=(--capture_calibration_corpus)
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

{
  echo "run_id=${run_id}"
  echo "source_model=${source_model}"
  echo "output_model=${output_model}"
  echo "calibration_corpus=${corpus}"
  "${venv}/bin/python" -m pymllm.mobile.backends.qualcomm.transformers.qwen3.train \
    --model_path "${source_model}" \
    --max_length 512 \
    --num_samples 128 \
    --calibration_corpus "${corpus}" \
    "${capture_args[@]}" \
    --activation_bits 8 \
    --linear_block_size 32 \
    --infer_max_new_tokens 8 \
    --output_dir "${stage_dir}/intermediate" \
    --output_mllm "${output_model}"

  python3 "${repo_root}/scripts/verify_mllm_v2.py" "${output_model}" \
    --output-json "${log_dir}/mllm-v2-audit.json"
  sha256sum "${output_model}" > "${log_dir}/model.sha256"
  sha256sum "${corpus}" > "${log_dir}/calibration-corpus.sha256"
  git -C "${repo_root}" rev-parse HEAD > "${log_dir}/source-head.txt"
  git -C "${repo_root}" diff --binary -- . ':!third_party/half/include/half/half.hpp' \
    > "${log_dir}/source-working-tree.patch"
  "${venv}/bin/python" -m pip freeze > "${log_dir}/python-freeze.txt"
  printf '%s\n' "${output_model}" > "${log_dir}/output-model.txt"
} 2>&1 | tee "${log_dir}/build.log"

echo "W4A8G32 model staging complete: ${stage_dir}"
