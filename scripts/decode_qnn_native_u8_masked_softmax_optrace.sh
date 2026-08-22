#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/native_u8_masked_softmax_qairt249/20260822}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_native_u8_masked_softmax_qairt249_20260822}"
decode_root="${DECODE_WORK_ROOT:-${repo_root}/build-optrace-decode/native_u8_masked_softmax_qairt249_20260822}"

case "${decode_root}" in
  "${repo_root}"/build-optrace-decode/native_u8_masked_softmax_qairt249_20260822) ;;
  *) echo "refusing unexpected DECODE_WORK_ROOT: ${decode_root}" >&2; exit 2 ;;
esac
mkdir -p "${decode_root}"

while read -r tag; do
  sdk="${tag#qairt}"; sdk="${sdk%%_*}"
  if [[ "${sdk}" == 247 ]]; then release=2.47.0.260601; else release=2.49.0.260730; fi
  qairt="${models_root}/qualcomm-sdk/qairt/${release}"
  profile_viewer="${qairt}/bin/x86_64-linux-clang/qnn-profile-viewer"
  reader="${qairt}/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so"
  directory="${results_root}/optrace/${tag}"
  schematic="${artifact_root}/contexts/${tag}/schematics/model.0.s1_schematic.bin"
  stage="${decode_root}/${tag}"
  [[ ! -e "${stage}" ]] || { echo "refusing existing decode stage: ${stage}" >&2; exit 2; }
  [[ -s "${directory}/qnn-profiling-data.log" && -s "${schematic}" ]] || {
    echo "missing Optrace input for ${tag}" >&2; exit 2;
  }
  mkdir -p "${stage}"
  base="${stage}/${tag}"
  env LD_LIBRARY_PATH="${qairt}/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${profile_viewer}" --reader "${reader}" \
    --input_log "${directory}/qnn-profiling-data.log" --schematic "${schematic}" \
    --output "${base}-chrometrace.json" >"${stage}/profile_viewer.log" 2>&1
  python3 "${repo_root}/scripts/qnn_optrace_summary.py" \
    "${base}-chrometrace.json" --htp-json "${base}-chrometrace_htp.json" \
    --output "${base}-operators.csv" --type-summary-output "${base}-logical-types.csv" \
    >"${stage}/summary_generation.log"
  cp -f "${stage}"/* "${directory}/"
  rm -rf "${stage}"
  echo "decoded ${tag} with QAIRT ${release}"
done <"${results_root}/winner_cases.txt"

rmdir "${decode_root}" 2>/dev/null || true
echo "Native-U8 masked-Softmax Optrace decoding complete"
