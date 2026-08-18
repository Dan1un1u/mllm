#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/w8a8_vs_lpbq_layer14_mlp/20260818}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_layer14_mlp_w8a8_vs_lpbq_20260818}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
profile_viewer="${qairt_root}/bin/x86_64-linux-clang/qnn-profile-viewer"
optrace_reader="${qairt_root}/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so"

[[ "${qairt_root##*/}" == "2.47.0.260601" ]] || {
  echo "unexpected QAIRT release: ${qairt_root}" >&2
  exit 2
}
[[ -x "${profile_viewer}" ]] || { echo "profile viewer missing: ${profile_viewer}" >&2; exit 2; }
[[ -f "${optrace_reader}" ]] || { echo "Optrace reader missing: ${optrace_reader}" >&2; exit 2; }
export LD_LIBRARY_PATH="${qairt_root}/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

for variant in lpbq w8a8; do
  for seq in 1 32; do
    directory="${results_root}/optrace/${variant}_s${seq}"
    schematic="${artifact_root}/contexts/${variant}_s${seq}/schematics/model.0.s${seq}_schematic.bin"
    base="${directory}/layer14-${variant}-s${seq}"
    if [[ -s "${base}-operators.csv" && -s "${base}-chrometrace_htp.json" ]]; then
      echo "already decoded: ${variant} s${seq}"
      continue
    fi
    "${profile_viewer}" \
      --reader "${optrace_reader}" \
      --input_log "${directory}/qnn-profiling-data.log" \
      --schematic "${schematic}" \
      --output "${base}-chrometrace.json" \
      >"${directory}/profile_viewer.log" 2>&1
    python3 "${repo_root}/scripts/qnn_optrace_summary.py" \
      "${base}-chrometrace.json" \
      --htp-json "${base}-chrometrace_htp.json" \
      --output "${base}-operators.csv" \
      --type-summary-output "${base}-logical-types.csv" \
      >"${directory}/summary_generation.log"
  done
done

echo "Decoded layer-14 LPBQ/W8A8 Optrace: ${results_root}/optrace"
