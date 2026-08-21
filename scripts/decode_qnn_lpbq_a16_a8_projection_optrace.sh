#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_a16_vs_a8_projections/20260821}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_lpbq_a16_vs_a8_projections_20260821}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
profile_viewer="${qairt_root}/bin/x86_64-linux-clang/qnn-profile-viewer"
optrace_reader="${qairt_root}/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so"
decode_root="${DECODE_WORK_ROOT:-${repo_root}/build-optrace-decode/lpbq_a16_a8_20260821}"

[[ "${qairt_root##*/}" == "2.47.0.260601" ]] || {
  echo "unexpected QAIRT release: ${qairt_root}" >&2
  exit 2
}
[[ -x "${profile_viewer}" ]] || { echo "profile viewer missing: ${profile_viewer}" >&2; exit 2; }
[[ -f "${optrace_reader}" ]] || { echo "Optrace reader missing: ${optrace_reader}" >&2; exit 2; }
export LD_LIBRARY_PATH="${qairt_root}/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
case "${decode_root}" in
  "${repo_root}"/build-optrace-decode/lpbq_a16_a8_20260821) ;;
  *) echo "refusing unexpected DECODE_WORK_ROOT: ${decode_root}" >&2; exit 2 ;;
esac
mkdir -p "${decode_root}"

for activation in a16 a8; do
  for projection in gate_proj up_proj down_proj lm_head; do
    for seq in 1 32; do
      tag="${activation}_${projection}_s${seq}"
      directory="${results_root}/optrace/${tag}"
      schematic="${artifact_root}/contexts/${tag}/schematics/model.0.s${seq}_schematic.bin"
      base="${directory}/${tag}"
      if [[ -s "${base}-operators.csv" && -s "${base}-chrometrace_htp.json" ]]; then
        echo "already decoded: ${tag}"
        continue
      fi
      stage="${decode_root}/${tag}"
      [[ ! -e "${stage}" ]] || { echo "refusing existing decode stage: ${stage}" >&2; exit 2; }
      mkdir -p "${stage}"
      stage_base="${stage}/${tag}"
      "${profile_viewer}" \
        --reader "${optrace_reader}" \
        --input_log "${directory}/qnn-profiling-data.log" \
        --schematic "${schematic}" \
        --output "${stage_base}-chrometrace.json" \
        >"${stage}/profile_viewer.log" 2>&1
      python3 "${repo_root}/scripts/qnn_optrace_summary.py" \
        "${stage_base}-chrometrace.json" \
        --htp-json "${stage_base}-chrometrace_htp.json" \
        --output "${stage_base}-operators.csv" \
        --type-summary-output "${stage_base}-logical-types.csv" \
        >"${stage}/summary_generation.log"
      cp -f "${stage}"/* "${directory}/"
      rm -rf "${stage}"
      echo "decoded: ${tag}"
    done
  done
done
rmdir "${decode_root}" 2>/dev/null || true

echo "Decoded LPBQ A16/A8 projection Optrace: ${results_root}/optrace"
