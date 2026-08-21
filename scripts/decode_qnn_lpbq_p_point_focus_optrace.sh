#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${MODELS_ROOT:-/mnt/d/llm_exp/models}"
source_root="${SOURCE_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_a16_vs_a8_projections/20260821}"
artifact_root="${ARTIFACT_ROOT:-${models_root}/qwen3_sm8750_v79/g32/lpbq_p_point_search/20260821}"
results_root="${RESULTS_ROOT:-/mnt/d/llm_exp/results/qwen3_sm8750_v79_lpbq_p_point_focus_20260821}"
qairt_root="${QAIRT_SDK_ROOT:-${models_root}/qualcomm-sdk/qairt/2.47.0.260601}"
profile_viewer="${qairt_root}/bin/x86_64-linux-clang/qnn-profile-viewer"
optrace_reader="${qairt_root}/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so"
decode_root="${repo_root}/build-optrace-decode/lpbq_p_point_focus_20260821"
read -r -a points <<<"${POINTS:-default 2 15 19}"
read -r -a projections <<<"${PROJECTIONS:-gate_proj up_proj}"

[[ "${qairt_root##*/}" == "2.47.0.260601" ]] || exit 2
[[ -x "${profile_viewer}" && -f "${optrace_reader}" ]] || exit 2
case "${decode_root}" in
  "${repo_root}/build-optrace-decode/lpbq_p_point_focus_20260821") ;;
  *) exit 2 ;;
esac
mkdir -p "${decode_root}"
export LD_LIBRARY_PATH="${qairt_root}/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

schematic_path() {
  if [[ "$1" == default ]]; then
    printf '%s/contexts/a8_%s_s1/schematics/model.0.s1_schematic.bin\n' "${source_root}" "$2"
  else
    printf '%s/contexts/p%s_%s_s1/schematics/model.0.s1_schematic.bin\n' \
      "${artifact_root}" "$1" "$2"
  fi
}

for projection in "${projections[@]}"; do
  for point in "${points[@]}"; do
    tag="${point}_${projection}"
    result_tag="${tag}"
    if [[ "${SHORT_RESULT_TAGS:-0}" == 1 ]]; then result_tag="${point}"; fi
    directory="${results_root}/optrace/${result_tag}"
    base="${directory}/${tag}"
    if [[ -s "${base}-operators.csv" && -s "${base}-chrometrace_htp.json" ]]; then
      echo "already decoded: ${tag}"
      continue
    fi
    schematic="$(schematic_path "${point}" "${projection}")"
    [[ -s "${schematic}" && -s "${directory}/qnn-profiling-data.log" ]] || exit 2
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
rmdir "${decode_root}" 2>/dev/null || true
echo "Decoded LPBQ P-point focused Optrace: ${results_root}/optrace"
