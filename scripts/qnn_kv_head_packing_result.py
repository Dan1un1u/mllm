#!/usr/bin/env python3
"""Summarize the fixed-P19 K/V split-vs-packed LPBQ experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


PROJECTIONS = ("k_proj", "v_proj")
MODES = ("split", "packed")
SEQUENCES = (32, 64)


def measured_median(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        values = [
            float(row["graph_execute_us"])
            for row in csv.DictReader(stream)
            if row["phase"] == "measured"
        ]
    if not values:
        raise ValueError(f"no measured rows: {path}")
    return statistics.median(values)


def parse_case(name: str) -> tuple[str, str, int]:
    match = re.fullmatch(r"(k_proj|v_proj)_(split|packed)_s(32|64)_p19", name)
    if not match:
        raise ValueError(f"unrecognized case: {name}")
    return match.group(1), match.group(2), int(match.group(3))


def delta_percent(candidate: float, control: float) -> float:
    return (candidate / control - 1.0) * 100.0


def reference_error(output: Path, reference: Path) -> dict[str, float | int]:
    actual = output.read_bytes()
    expected = reference.read_bytes()
    if len(actual) != len(expected):
        raise AssertionError(f"reference size mismatch: {output} vs {reference}")
    differences = [abs(left - right) for left, right in zip(actual, expected)]
    return {
        "elements": len(actual),
        "exact_matches": sum(value == 0 for value in differences),
        "max_abs_code_error": max(differences, default=0),
        "mean_abs_code_error": statistics.fmean(differences) if differences else 0.0,
    }


def root_cycles(path: Path) -> int:
    pattern = re.compile(r"value=(\d+).*identifier=Accelerator \(execute\) time \(cycles\)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    raise AssertionError(f"accelerator cycles missing: {path}")


def qhas_metrics(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))["data"]
    overall = document["htp_overall_summary"]["data"][0]
    selected: dict[str, object] = {}
    keywords = ("ConvLayer", "weights_to_vtcm", "expand_block", "Slice", "Format")
    for item in document["htp_op_types"]["data"]:
        name = item["op"]
        if not any(keyword in name for keyword in keywords):
            continue
        selected[name] = {
            key: item[key]
            for key in (
                "cycles",
                "num_dominant_path_cycles_htp_0",
                "instances",
                "dram_read",
                "dram_write",
                "vtcm_read",
                "vtcm_write",
            )
        }
    return {
        "timeline_cycles": overall["timeline_cycles"],
        "graph_execute_us": overall["graph_execute_us"],
        "total_dram_read": overall["total_dram_read"],
        "total_dram_write": overall["total_dram_write"],
        "total_vtcm_read": overall["total_vtcm_read"],
        "total_vtcm_write": overall["total_vtcm_write"],
        "peak_vtcm_alloc": overall["peak_vtcm_alloc"],
        "selected_physical_kernels": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    round_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for timing in sorted(args.result_root.glob("speed/round*/*/timing.csv")):
        projection, mode, sequence = parse_case(timing.parent.name)
        round_values[(projection, mode, sequence)].append(measured_median(timing))

    comparisons: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for projection in PROJECTIONS:
        for sequence in SEQUENCES:
            key = f"{projection}_s{sequence}"
            split = round_values[(projection, "split", sequence)]
            packed = round_values[(projection, "packed", sequence)]
            if not split or len(split) != len(packed):
                raise AssertionError(f"incomplete paired timing rounds: {key}")
            split_median = statistics.median(split)
            packed_median = statistics.median(packed)
            split_output = (
                args.result_root / "correctness"
                / f"{projection}_split_s{sequence}_p19" / "output.raw"
            )
            packed_output = (
                args.result_root / "correctness"
                / f"{projection}_packed_s{sequence}_p19" / "output.raw"
            )
            exact = split_output.read_bytes() == packed_output.read_bytes()
            reference = args.source_root / f"reference_{projection}_s{sequence}_a8.raw"
            ref_error = reference_error(split_output, reference)
            gate = exact and packed_median <= split_median
            entry = {
                "split_median_us": split_median,
                "packed_median_us": packed_median,
                "packed_delta_percent": delta_percent(packed_median, split_median),
                "paired_packed_wins": sum(c < r for c, r in zip(packed, split)),
                "paired_ties": sum(c == r for c, r in zip(packed, split)),
                "round_split_medians_us": split,
                "round_packed_medians_us": packed,
                "split_vs_packed_byte_exact": exact,
                "split_vs_independent_host_lpbq_reference": ref_error,
                "speed_and_math_gate_passed": gate,
            }
            comparisons[key] = entry
            rows.append({"projection": projection, "sequence": sequence, **entry})

    profiles: dict[str, object] = {}
    for directory in sorted((args.result_root / "optrace").glob("*")):
        if not directory.is_dir():
            continue
        projection, mode, sequence = parse_case(directory.name)
        detail = directory / "qnn_detail_profile.txt"
        qhas = directory / "chrometrace_qnn_htp_analysis_summary.json"
        if detail.is_file() and qhas.is_file():
            profiles[directory.name] = {
                "projection": projection,
                "mode": mode,
                "sequence": sequence,
                "accelerator_cycles": root_cycles(detail),
                "htp_analysis": qhas_metrics(qhas),
            }

    trace_comparisons: dict[str, object] = {}
    for projection in PROJECTIONS:
        for sequence in SEQUENCES:
            key = f"{projection}_s{sequence}"
            split = profiles[f"{projection}_split_s{sequence}_p19"]
            packed = profiles[f"{projection}_packed_s{sequence}_p19"]
            split_htp = split["htp_analysis"]
            packed_htp = packed["htp_analysis"]
            trace_comparisons[key] = {
                "accelerator_work_delta_percent": delta_percent(
                    packed["accelerator_cycles"], split["accelerator_cycles"]
                ),
                "timeline_delta_percent": delta_percent(
                    packed_htp["timeline_cycles"], split_htp["timeline_cycles"]
                ),
                "dram_read_delta_bytes": (
                    packed_htp["total_dram_read"] - split_htp["total_dram_read"]
                ),
                "dram_write_delta_bytes": (
                    packed_htp["total_dram_write"] - split_htp["total_dram_write"]
                ),
                "peak_vtcm_delta_bytes": (
                    packed_htp["peak_vtcm_alloc"] - split_htp["peak_vtcm_alloc"]
                ),
            }

    overall_gate = all(
        item["speed_and_math_gate_passed"] for item in comparisons.values()
    )
    report = {
        "contract": {
            "qairt_release": "2.49.0.260730",
            "finalize_schedule": "P19 for both variants",
            "activation": "asymmetric U8 input/output",
            "weight": "signed W4 LPBQ G32",
            "control": "eight independent 2048x128 Conv2d projections",
            "candidate": "one 2048x1024 Conv2d plus eight output slices",
            "sequences": list(SEQUENCES),
            "gate": "byte-exact split/packed and packed median <= split median for every K/V s32/s64 case",
        },
        "comparisons": comparisons,
        "optrace": profiles,
        "optrace_comparisons": trace_comparisons,
        "overall_gate_passed": overall_gate,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "projection", "sequence", "split_median_us", "packed_median_us",
            "packed_delta_percent", "paired_packed_wins", "paired_ties",
            "split_vs_packed_byte_exact", "host_reference_max_abs_code_error",
            "speed_and_math_gate_passed",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "projection": row["projection"],
                    "sequence": row["sequence"],
                    "split_median_us": f"{row['split_median_us']:.3f}",
                    "packed_median_us": f"{row['packed_median_us']:.3f}",
                    "packed_delta_percent": f"{row['packed_delta_percent']:.6f}",
                    "paired_packed_wins": row["paired_packed_wins"],
                    "paired_ties": row["paired_ties"],
                    "split_vs_packed_byte_exact": row["split_vs_packed_byte_exact"],
                    "host_reference_max_abs_code_error": row[
                        "split_vs_independent_host_lpbq_reference"
                    ]["max_abs_code_error"],
                    "speed_and_math_gate_passed": row["speed_and_math_gate_passed"],
                }
            )

    markdown = "# QAIRT 2.49 K/V head-packing gate\n\n"
    markdown += f"Overall gate: **{'PASS' if overall_gate else 'FAIL'}**\n\n"
    markdown += "| Case | Split us | Packed us | Delta | Exact | Gate |\n"
    markdown += "|---|---:|---:|---:|:---:|:---:|\n"
    for key, item in comparisons.items():
        markdown += (
            f"| {key} | {item['split_median_us']:.3f} | "
            f"{item['packed_median_us']:.3f} | {item['packed_delta_percent']:+.2f}% | "
            f"{'yes' if item['split_vs_packed_byte_exact'] else 'no'} | "
            f"{'PASS' if item['speed_and_math_gate_passed'] else 'FAIL'} |\n"
        )
    args.markdown.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
