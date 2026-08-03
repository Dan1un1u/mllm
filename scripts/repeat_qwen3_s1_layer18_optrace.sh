#!/usr/bin/env bash

# Repeat the exact capture semantics used by the native SM8750/V79 s1 report:
# one fresh runner process -> first model.0.s1 execution -> one Optrace payload.
#
# By default this script prepares the device from the known local V79 artifacts,
# captures, pulls, decodes, regenerates the Qwen3 structure report, and then
# compares Layer 18 both across runs and against the other 27 layers.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ ! -f "${REPO_ROOT}/scripts/qnn_optrace_qwen3_structure.py" ]]; then
    REPO_ROOT="/home/daniuniu/llm_exp/mllm"
fi
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

RUNS="${RUNS:-5}"
LAYER="${LAYER:-18}"
COOLDOWN_SEC="${COOLDOWN_SEC:-5}"
PROMPT="${PROMPT:-hello!}"
AR_LEN="${AR_LEN:-32}"

ADB_SERIAL="${ADB_SERIAL:-}"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp}"
REMOTE_RUNNER="${REMOTE_RUNNER:-mllm-qwen3-aot-runner}"
REMOTE_MODEL="${REMOTE_MODEL:-qwen3-1.7B-lpbq-sha-sm8750-v79.bin}"
REMOTE_TOKENIZER="${REMOTE_TOKENIZER:-qwen3-tokenizer.json}"
REMOTE_CONFIG="${REMOTE_CONFIG:-config_1.7B.json}"
REMOTE_EXPERIMENT_ROOT="${REMOTE_EXPERIMENT_ROOT:-${REMOTE_DIR}/qwen3_s1_repeat_${TIMESTAMP}}"

LOCAL_BUILD_BIN="${LOCAL_BUILD_BIN:-${REPO_ROOT}/build-android-arm64-v8a-qnn/bin}"
LOCAL_RUNNER="${LOCAL_RUNNER:-${LOCAL_BUILD_BIN}/mllm-qwen3-aot-runner}"
LOCAL_MODEL="${LOCAL_MODEL:-/home/daniuniu/llm_exp/models/output/qwen3-1.7B-lpbq-sha-sm8750-v79.bin}"
LOCAL_TOKENIZER="${LOCAL_TOKENIZER:-/home/daniuniu/llm_exp/models/Qwen3-origin/qwen3-tokenizer.json}"
LOCAL_CONFIG="${LOCAL_CONFIG:-/home/daniuniu/llm_exp/models/Qwen3-origin/config_1.7B.json}"
PREPARE_DEVICE="${PREPARE_DEVICE:-1}"
BUILD_ANDROID="${BUILD_ANDROID:-0}"

ARCHIVE_DIR="${ARCHIVE_DIR:-/home/daniuniu/llm_exp/results/qwen3_sm8750_v79_20260713}"
SCHEMATIC="${SCHEMATIC:-${ARCHIVE_DIR}/schematics/model.0.s1_schematic.bin}"
BASELINE_LAYER_CSV="${BASELINE_LAYER_CSV:-${ARCHIVE_DIR}/qwen3-sm8750-v79-s1-qwen3-layer-stage.csv}"
RESULT_ROOT="${RESULT_ROOT:-/home/daniuniu/llm_exp/results/qwen3_sm8750_v79_s1_layer18_repeat_${TIMESTAMP}}"

# This hash is the native SM8750/V79 context used by the archived report.
VERIFY_CONTEXT_SHA="${VERIFY_CONTEXT_SHA:-1}"
EXPECTED_CONTEXT_SHA="${EXPECTED_CONTEXT_SHA:-727fe97725abb0d6ff4efb5fa06564746bae89339b9079c88f8d6f865acb173d}"
CLEAN_REMOTE="${CLEAN_REMOTE:-1}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ "${RUNS}" =~ ^[1-9][0-9]*$ ]] || die "RUNS must be a positive integer"
[[ "${LAYER}" =~ ^[0-9]+$ ]] || die "LAYER must be a non-negative integer"
[[ "${COOLDOWN_SEC}" =~ ^[0-9]+$ ]] || die "COOLDOWN_SEC must be a non-negative integer"
[[ -n "${QAIRT_SDK_ROOT:-}" ]] || die "QAIRT_SDK_ROOT is not set"

PROFILE_VIEWER="${QAIRT_SDK_ROOT}/bin/x86_64-linux-clang/qnn-profile-viewer"
OPTRACE_READER="${QAIRT_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so"
[[ -x "${PROFILE_VIEWER}" ]] || die "qnn-profile-viewer not found: ${PROFILE_VIEWER}"
[[ -f "${OPTRACE_READER}" ]] || die "Optrace reader not found: ${OPTRACE_READER}"
[[ -f "${SCHEMATIC}" ]] || die "s1 schematic not found: ${SCHEMATIC}"
[[ -f "${REPO_ROOT}/scripts/qnn_optrace_summary.py" ]] || die "repository profiling scripts are missing"
[[ -f "${REPO_ROOT}/scripts/qnn_optrace_qwen3_structure.py" ]] || die "repository profiling scripts are missing"
command -v adb >/dev/null || die "adb is not in PATH"
command -v python3 >/dev/null || die "python3 is not in PATH"

ADB=(adb)
if [[ -n "${ADB_SERIAL}" ]]; then
    ADB+=(-s "${ADB_SERIAL}")
fi

"${ADB[@]}" get-state >/dev/null

if [[ "${BUILD_ANDROID}" == "1" ]]; then
    echo "Building Android QNN target ..."
    (cd "${REPO_ROOT}" && python3 task.py tasks/build_android_qnn.yaml)
fi

if [[ "${PREPARE_DEVICE}" == "1" ]]; then
    [[ -x "${LOCAL_RUNNER}" ]] || die "local runner not found: ${LOCAL_RUNNER}"
    [[ -f "${LOCAL_MODEL}" ]] || die "local native V79 context not found: ${LOCAL_MODEL}"
    [[ -f "${LOCAL_TOKENIZER}" ]] || die "local tokenizer not found: ${LOCAL_TOKENIZER}"
    [[ -f "${LOCAL_CONFIG}" ]] || die "local config not found: ${LOCAL_CONFIG}"

    LOCAL_CONTEXT_SHA="$(sha256sum "${LOCAL_MODEL}" | awk '{print $1}')"
    [[ "${LOCAL_CONTEXT_SHA}" == "${EXPECTED_CONTEXT_SHA}" ]] \
        || die "local V79 context SHA mismatch: expected ${EXPECTED_CONTEXT_SHA}, got ${LOCAL_CONTEXT_SHA}"

    echo "Preparing runner and mllm libraries on the device ..."
    "${ADB[@]}" push "${LOCAL_BUILD_BIN}"/*.so "${REMOTE_DIR}/" >/dev/null
    "${ADB[@]}" push "${LOCAL_RUNNER}" "${REMOTE_DIR}/${REMOTE_RUNNER}" >/dev/null
    "${ADB[@]}" push "${LOCAL_TOKENIZER}" "${REMOTE_DIR}/${REMOTE_TOKENIZER}" >/dev/null
    "${ADB[@]}" push "${LOCAL_CONFIG}" "${REMOTE_DIR}/${REMOTE_CONFIG}" >/dev/null

    remote_v79_sha="$("${ADB[@]}" shell "sha256sum '${REMOTE_DIR}/${REMOTE_MODEL}' 2>/dev/null" \
        | awk '{print $1}' | tr -d '\r' || true)"
    if [[ "${remote_v79_sha}" != "${EXPECTED_CONTEXT_SHA}" ]]; then
        echo "Pushing the 1.5 GiB native SM8750/V79 context ..."
        "${ADB[@]}" push "${LOCAL_MODEL}" "${REMOTE_DIR}/${REMOTE_MODEL}" >/dev/null
    else
        echo "Native SM8750/V79 context is already present; skipping model push."
    fi
fi

for remote_file in "${REMOTE_RUNNER}" "${REMOTE_MODEL}" "${REMOTE_TOKENIZER}" "${REMOTE_CONFIG}"; do
    "${ADB[@]}" shell "test -r '${REMOTE_DIR}/${remote_file}'" \
        || die "device file is missing: ${REMOTE_DIR}/${remote_file}"
done

if [[ "${VERIFY_CONTEXT_SHA}" == "1" ]]; then
    echo "Checking native V79 context SHA-256 once ..."
    ACTUAL_CONTEXT_SHA="$("${ADB[@]}" shell "sha256sum '${REMOTE_DIR}/${REMOTE_MODEL}'" | awk '{print $1}' | tr -d '\r')"
    [[ "${ACTUAL_CONTEXT_SHA}" == "${EXPECTED_CONTEXT_SHA}" ]] \
        || die "context SHA mismatch: expected ${EXPECTED_CONTEXT_SHA}, got ${ACTUAL_CONTEXT_SHA}"
fi

mkdir -p "${RESULT_ROOT}"
cp "${BASH_SOURCE[0]}" "${RESULT_ROOT}/experiment_script.sh"
cp "${SCHEMATIC}" "${RESULT_ROOT}/model.0.s1_schematic.bin"
if [[ -f "${LOCAL_RUNNER}" && -f "${LOCAL_MODEL}" && -f "${LOCAL_TOKENIZER}" && -f "${LOCAL_CONFIG}" ]]; then
    sha256sum "${LOCAL_RUNNER}" "${LOCAL_MODEL}" "${LOCAL_TOKENIZER}" "${LOCAL_CONFIG}" \
        >"${RESULT_ROOT}/local_artifact_sha256.txt"
fi

cat >"${RESULT_ROOT}/experiment_metadata.txt" <<EOF
timestamp=${TIMESTAMP}
runs=${RUNS}
layer=${LAYER}
prompt=${PROMPT}
ar_len=${AR_LEN}
device_serial=${ADB_SERIAL:-default}
remote_runner=${REMOTE_DIR}/${REMOTE_RUNNER}
remote_model=${REMOTE_DIR}/${REMOTE_MODEL}
local_runner=${LOCAL_RUNNER}
local_model=${LOCAL_MODEL}
expected_context_sha256=${EXPECTED_CONTEXT_SHA}
schematic=${SCHEMATIC}
baseline_layer_csv=${BASELINE_LAYER_CSV}
capture_semantics=fresh process; first model.0.s1 execution; one Optrace payload
cooldown_seconds=${COOLDOWN_SEC}
prepare_device=${PREPARE_DEVICE}
build_android=${BUILD_ANDROID}
EOF

git -C "${REPO_ROOT}" rev-parse HEAD >"${RESULT_ROOT}/git_commit.txt"
git -C "${REPO_ROOT}" status --short >"${RESULT_ROOT}/git_status.txt"

"${ADB[@]}" shell getprop >"${RESULT_ROOT}/device_getprop.txt"
"${ADB[@]}" shell "mkdir -p '${REMOTE_EXPERIMENT_ROOT}'"

echo "Results: ${RESULT_ROOT}"
echo "Estimated storage: roughly 0.8-0.9 GiB per run if all viewer artifacts are retained."

for ((run = 1; run <= RUNS; ++run)); do
    run_name="$(printf 'run_%02d' "${run}")"
    host_run_dir="${RESULT_ROOT}/${run_name}"
    remote_capture_dir="${REMOTE_EXPERIMENT_ROOT}/${run_name}"
    mkdir -p "${host_run_dir}"
    "${ADB[@]}" shell "mkdir -p '${remote_capture_dir}'"

    echo
    echo "===== ${run_name}/${RUNS}: thermal state before capture ====="
    "${ADB[@]}" shell dumpsys battery >"${host_run_dir}/battery_before.txt" || true
    "${ADB[@]}" shell dumpsys thermalservice >"${host_run_dir}/thermal_before.txt" || true

    echo "===== ${run_name}/${RUNS}: capture first model.0.s1 execution ====="
    set +e
    printf '%s\n' "${PROMPT}" | "${ADB[@]}" shell "
        cd '${REMOTE_DIR}' &&
        export LD_LIBRARY_PATH=. &&
        export MLLM_QNN_PROFILE_LEVEL=optrace &&
        export MLLM_QNN_PROFILE_WARMUP=0 &&
        export MLLM_QNN_PROFILE_EVERY=1 &&
        export MLLM_QNN_PROFILE_MAX_CAPTURES=1 &&
        export MLLM_QNN_PROFILE_GRAPH=model.0.s1 &&
        export MLLM_QNN_PROFILE_SERIALIZE=1 &&
        export MLLM_QNN_PROFILE_DIR='${remote_capture_dir}' &&
        './${REMOTE_RUNNER}' \
            -m '${REMOTE_MODEL}' \
            -t '${REMOTE_TOKENIZER}' \
            -c '${REMOTE_CONFIG}' \
            --ar_len '${AR_LEN}'
    " 2>&1 | tee "${host_run_dir}/runner.log"
    runner_status="${PIPESTATUS[1]}"
    set -e
    [[ "${runner_status}" == "0" ]] || die "${run_name}: runner failed with status ${runner_status}"

    echo "===== ${run_name}/${RUNS}: pull raw profile ====="
    "${ADB[@]}" pull "${remote_capture_dir}/qnn-profiling-data.log" "${host_run_dir}/qnn-profiling-data.log" >/dev/null
    for profile_file in qnn_detail_profile.txt qnn_macro_profile.csv qnn_e2e_profile.csv; do
        if "${ADB[@]}" shell "test -f '${remote_capture_dir}/${profile_file}'"; then
            "${ADB[@]}" pull "${remote_capture_dir}/${profile_file}" "${host_run_dir}/${profile_file}" >/dev/null
        fi
    done
    "${ADB[@]}" shell dumpsys battery >"${host_run_dir}/battery_after.txt" || true
    "${ADB[@]}" shell dumpsys thermalservice >"${host_run_dir}/thermal_after.txt" || true

    echo "===== ${run_name}/${RUNS}: decode Optrace and regenerate structure report ====="
    chrome_trace="${host_run_dir}/qwen3-s1-chrometrace.json"
    "${PROFILE_VIEWER}" \
        --reader "${OPTRACE_READER}" \
        --input_log "${host_run_dir}/qnn-profiling-data.log" \
        --schematic "${SCHEMATIC}" \
        --output "${chrome_trace}" \
        2>&1 | tee "${host_run_dir}/profile_viewer.log"

    htp_json="${host_run_dir}/qwen3-s1-chrometrace_htp.json"
    qhas_json="${host_run_dir}/qwen3-s1-chrometrace_qnn_htp_analysis_summary.json"
    [[ -s "${chrome_trace}" ]] || die "${run_name}: Chrome Trace was not generated"
    [[ -s "${htp_json}" ]] || die "${run_name}: HTP JSON was not generated"
    [[ -s "${qhas_json}" ]] || die "${run_name}: QHAS JSON was not generated"

    python3 "${REPO_ROOT}/scripts/qnn_optrace_summary.py" \
        "${chrome_trace}" \
        --htp-json "${htp_json}" \
        --output "${host_run_dir}/qwen3-s1-operators.csv" \
        --type-summary-output "${host_run_dir}/qwen3-s1-logical-types.csv" \
        >"${host_run_dir}/summary_generation.log"

    python3 "${REPO_ROOT}/scripts/qnn_optrace_qwen3_structure.py" \
        "${chrome_trace}" \
        --htp-json "${htp_json}" \
        --qhas-json "${qhas_json}" \
        --output-prefix "${host_run_dir}/qwen3-s1" \
        >"${host_run_dir}/structure_generation.log"

    [[ -s "${host_run_dir}/qwen3-s1-qwen3-layer-stage.csv" ]] \
        || die "${run_name}: layer-stage CSV was not generated"

    if ((run < RUNS && COOLDOWN_SEC > 0)); then
        echo "Cooling down for ${COOLDOWN_SEC}s ..."
        sleep "${COOLDOWN_SEC}"
    fi
done

echo
echo "===== Build cross-run Layer ${LAYER} comparison ====="
export RESULT_ROOT LAYER BASELINE_LAYER_CSV
python3 - <<'PY'
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["RESULT_ROOT"])
target_layer = int(os.environ["LAYER"])
baseline = Path(os.environ["BASELINE_LAYER_CSV"])

samples = []
if baseline.is_file():
    samples.append(("baseline_20260713", "baseline", baseline))
for run_dir in sorted(root.glob("run_[0-9][0-9]")):
    path = run_dir / "qwen3-s1-qwen3-layer-stage.csv"
    if path.is_file():
        samples.append((run_dir.name, "repeat", path))
if not samples:
    raise SystemExit("No layer-stage CSV files found")

overview_rows = []
stage_rows = []
all_layer_rows = []
repeat_totals = []

for sample, kind, path in samples:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_layer = defaultdict(float)
    by_stage = defaultdict(dict)
    target_rows = []
    for row in rows:
        layer = int(row["layer"])
        stage = row["stage"]
        value = float(row["critical_path_us_estimate"])
        by_layer[layer] += value
        by_stage[stage][layer] = value
        if layer == target_layer:
            target_rows.append(row)
    if target_layer not in by_layer:
        raise RuntimeError(f"{sample}: Layer {target_layer} is absent")

    layer_values = list(by_layer.values())
    layer_median = statistics.median(layer_values)
    target_total = by_layer[target_layer]
    target_rank = 1 + sum(value > target_total for value in layer_values)

    graph_execute_candidates = []
    for row in rows:
        percent = float(row["critical_path_percent"])
        if percent > 0:
            graph_execute_candidates.append(float(row["critical_path_us_estimate"]) * 100.0 / percent)
    graph_execute_us = statistics.median(graph_execute_candidates)

    overview_rows.append({
        "sample": sample,
        "sample_kind": kind,
        "graph_execute_us": graph_execute_us,
        "layer": target_layer,
        "layer_total_attributed_us": target_total,
        "layer_percent_graph_execute": 100.0 * target_total / graph_execute_us,
        "rank_slowest_of_layers": target_rank,
        "num_layers": len(by_layer),
        "all_layer_median_us": layer_median,
        "ratio_to_all_layer_median": target_total / layer_median if layer_median else 0.0,
    })
    if kind == "repeat":
        repeat_totals.append(target_total)

    for layer, value in sorted(by_layer.items()):
        all_layer_rows.append({
            "sample": sample,
            "sample_kind": kind,
            "layer": layer,
            "total_attributed_us": value,
            "ratio_to_sample_layer_median": value / layer_median if layer_median else 0.0,
        })

    for row in sorted(target_rows, key=lambda item: int(item["stage_order"])):
        stage = row["stage"]
        values = list(by_stage[stage].values())
        stage_median = statistics.median(values)
        target_value = float(row["critical_path_us_estimate"])
        stage_rows.append({
            "sample": sample,
            "sample_kind": kind,
            "layer": target_layer,
            "stage_order": int(row["stage_order"]),
            "stage": stage,
            "layer_stage_attributed_us": target_value,
            "same_stage_all_layer_median_us": stage_median,
            "ratio_to_same_stage_median": target_value / stage_median if stage_median else 0.0,
            "rank_slowest_of_stage": 1 + sum(value > target_value for value in values),
            "num_layers_with_stage": len(values),
            "dominant_path_cycles": int(row["num_dominant_path_cycles_htp_0"]),
            "work_cycles": int(row["cycles"]),
            "active_union_cycles": int(row["active_union_cycles"]),
            "wall_span_cycles": int(row["wall_span_cycles"]),
            "max_parallelism": int(row["max_parallelism"]),
        })

def write_csv(name, rows):
    if not rows:
        return
    with (root / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

write_csv(f"layer{target_layer}_by_run.csv", overview_rows)
write_csv(f"layer{target_layer}_stage_by_run.csv", stage_rows)
write_csv("all_layer_totals_by_run.csv", all_layer_rows)

lines = [
    f"Layer {target_layer} repeated Optrace summary",
    "",
    "Important: attributed_us is QHAS dominant-path attribution scaled to graphExecute;",
    "it is not a sum of serialized operator latency.",
    "",
]
for row in overview_rows:
    lines.append(
        f"{row['sample']:>18s}: total={row['layer_total_attributed_us']:.3f} us, "
        f"rank={row['rank_slowest_of_layers']}/{row['num_layers']}, "
        f"ratio_to_layer_median={row['ratio_to_all_layer_median']:.3f}, "
        f"graphExecute={row['graph_execute_us']:.3f} us"
    )

if repeat_totals:
    mean = statistics.mean(repeat_totals)
    median = statistics.median(repeat_totals)
    stdev = statistics.stdev(repeat_totals) if len(repeat_totals) > 1 else 0.0
    lines.extend([
        "",
        "repeat-only Layer total statistics:",
        f"count={len(repeat_totals)}",
        f"mean_us={mean:.3f}",
        f"median_us={median:.3f}",
        f"stdev_us={stdev:.3f}",
        f"cv_percent={(100.0 * stdev / mean if mean else 0.0):.3f}",
        f"min_us={min(repeat_totals):.3f}",
        f"max_us={max(repeat_totals):.3f}",
    ])

lines.extend([
    "",
    "Interpretation:",
    "- If Layer 18 remains rank 1 and the same stage ratio stays high in every run,",
    "  the anomaly is repeatable in this first-s1 scheduling/lowering path.",
    "- If only one run is high while repeat CV and stage ratios vary strongly,",
    "  it is more likely a transient scheduling, frequency, thermal, or Optrace perturbation.",
])
(root / f"layer{target_layer}_summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

if [[ "${CLEAN_REMOTE}" == "1" ]]; then
    "${ADB[@]}" shell "rm -rf '${REMOTE_EXPERIMENT_ROOT}'"
fi

echo
echo "Finished. Open per-run qwen3-s1-qwen3-structure.html files and compare:"
echo "  ${RESULT_ROOT}/layer${LAYER}_by_run.csv"
echo "  ${RESULT_ROOT}/layer${LAYER}_stage_by_run.csv"
echo "  ${RESULT_ROOT}/layer${LAYER}_summary.txt"
