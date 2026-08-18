#!/usr/bin/env python3
"""Summarize correctness, profiling-off, and QHAS evidence for the OC split gate."""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PROJECTIONS = ("gate_proj", "up_proj")
LAYOUTS = ("full", "split2")


def number(row, key):
    return int(float(row.get(key, 0) or 0))


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def active_cycles(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection_cycles(left, right):
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def resource(row):
    if row.get("hmx"):
        return "hmx"
    if row.get("hvx"):
        return "hvx"
    if row.get("dma_wait"):
        return "dma_wait"
    if row.get("dma"):
        return "dma_transfer"
    if row.get("dma_set") or row.get("sync"):
        return "sync"
    return "other"


def category(row):
    name = row.get("htp_op", "").lower()
    if "expand_block_quant" in name:
        return "weight_expand"
    if "weights_to_vtcm" in name:
        return "weight_wait" if row.get("dma_wait") else "weight_transfer"
    if row.get("hmx"):
        return "hmx_mac"
    if row.get("dma_set") or row.get("sync") or "checkpoint" in name or "sync" in name:
        return "checkpoint_sync"
    if "concat" in name:
        return "concat"
    return "other"


def projection_qnn_names(rows, projection):
    marker = f".mlp.{projection}"
    return sorted(row["qnn_op"] for row in rows if marker in row["qnn_op"])


def qhas_case(path, projection):
    document = json.loads(path.read_text(encoding="utf-8"))
    data = document["data"]
    qnn_rows = data["qnn_op_instances_nodes"]["data"]
    names = projection_qnn_names(qnn_rows, projection)
    expected_count = 1 if path.parent.name.startswith("full_") else 2
    if len(names) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} projection nodes, got {names}")
    rows = [row for row in data["htp_op_instances"]["data"] if row["qnn_op"] in names]
    selected_qnn = [row for row in qnn_rows if row["qnn_op"] in names]

    categories = defaultdict(lambda: Counter({
        "instances": 0, "work_cycles": 0, "direct_cycles": 0,
        "dram_read": 0, "dram_write": 0, "vtcm_read": 0, "vtcm_write": 0,
    }))
    resources = defaultdict(list)
    op_envelopes = defaultdict(list)
    for row in rows:
        start = number(row, "start_cycle")
        end = start + number(row, "cycles")
        res = resource(row)
        cat = category(row)
        resources[res].append((start, end))
        op_envelopes[row["qnn_op"]].append((start, end))
        group = categories[cat]
        group["instances"] += 1
        group["work_cycles"] += number(row, "cycles")
        group["direct_cycles"] += number(row, "num_dominant_path_cycles")
        for key in ("dram_read", "dram_write", "vtcm_read", "vtcm_write"):
            group[key] += number(row, key)

    all_intervals = [interval for values in resources.values() for interval in values]
    start = min(item[0] for item in all_intervals)
    end = max(item[1] for item in all_intervals)
    hmx = resources["hmx"]
    hmx_start = min(item[0] for item in hmx)
    hmx_end = max(item[1] for item in hmx)
    hmx_active = active_cycles(hmx)
    qnn_totals = Counter()
    for row in selected_qnn:
        for key in ("num_dominant_path_cycles_htp_0", "num_htp_ops", "dram_read", "dram_write", "vtcm_read", "vtcm_write"):
            qnn_totals[key] += number(row, key)

    envelopes = {}
    for name, intervals in op_envelopes.items():
        envelopes[name] = [min(a for a, _ in intervals), max(b for _, b in intervals)]
    oc_overlap = 0
    if len(envelopes) == 2:
        left, right = envelopes.values()
        oc_overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))

    return {
        "qnn_nodes": names,
        "projection_envelope_cycles": end - start,
        "qnn_totals": dict(qnn_totals),
        "categories": {key: dict(value) for key, value in sorted(categories.items())},
        "resource_active_cycles": {key: active_cycles(value) for key, value in resources.items()},
        "overlaps": {
            "hmx_hvx": intersection_cycles(resources["hmx"], resources["hvx"]),
            "hmx_dma": intersection_cycles(resources["hmx"], resources["dma_transfer"]),
            "hvx_dma": intersection_cycles(resources["hvx"], resources["dma_transfer"]),
            "wait_compute": intersection_cycles(resources["dma_wait"], resources["hmx"] + resources["hvx"]),
            "split_qnn_node_envelope": oc_overlap,
        },
        "hmx": {
            "active_cycles": hmx_active,
            "first_to_last_span_cycles": hmx_end - hmx_start,
            "idle_gap_cycles": hmx_end - hmx_start - hmx_active,
        },
        "kernel_counts": dict(Counter(row["htp_op"] for row in rows)),
    }


def timing_values(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return [int(row["graph_execute_us"]) for row in csv.DictReader(stream) if row["phase"] == "measured"]


def perf_summary(root, projection):
    rounds = []
    for index in range(1, 6):
        directory = root / "profiling_off" / projection / f"round_{index}"
        full = statistics.median(timing_values(directory / "full_timing.csv"))
        split = statistics.median(timing_values(directory / "split2_timing.csv"))
        rounds.append({"round": index, "full_us": full, "split2_us": split, "ratio": split / full})
    full = statistics.median(row["full_us"] for row in rounds)
    split = statistics.median(row["split2_us"] for row in rounds)
    return {
        "rounds": rounds,
        "median_of_round_medians_us": {"full": full, "split2": split},
        "split2_over_full": split / full,
        "paired_ratio_median": statistics.median(row["ratio"] for row in rounds),
        "split2_slower_rounds": sum(row["split2_us"] > row["full_us"] for row in rounds),
        "gate_threshold": "split2 <= full * 1.01",
        "pass": split <= full * 1.01,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    correctness = json.loads((args.results_root / "correctness_summary.json").read_text(encoding="utf-8-sig"))
    correctness_pass = len(correctness) == 12 and all(row["byte_exact"] for row in correctness)
    performance = {projection: perf_summary(args.results_root, projection) for projection in PROJECTIONS}
    optrace = {}
    for projection in PROJECTIONS:
        for layout in LAYOUTS:
            case = f"{layout}_{projection}"
            qhas = args.results_root / "optrace" / case / "trace_qnn_htp_analysis_summary.json"
            optrace[case] = qhas_case(qhas, projection)

    result = {
        "status": "pass" if correctness_pass and all(item["pass"] for item in performance.values()) else "fail",
        "correctness": {"pass": correctness_pass, "byte_exact_cases": sum(row["byte_exact"] for row in correctness), "total_cases": len(correctness)},
        "performance": performance,
        "optrace": optrace,
        "decision": "stop_before_full_model" if not all(item["pass"] for item in performance.values()) else "eligible_for_full_model",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    rows = []
    for projection, item in performance.items():
        rows.append({
            "projection": projection,
            "full_us": item["median_of_round_medians_us"]["full"],
            "split2_us": item["median_of_round_medians_us"]["split2"],
            "ratio": item["split2_over_full"],
            "paired_ratio_median": item["paired_ratio_median"],
            "split2_slower_rounds": item["split2_slower_rounds"],
            "pass": item["pass"],
        })
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"gate={result['status']} correctness={correctness_pass} decision={result['decision']}")
    for row in rows:
        print(f"{row['projection']}: full={row['full_us']} us split2={row['split2_us']} us ratio={row['ratio']:.4f} pass={row['pass']}")


if __name__ == "__main__":
    main()
