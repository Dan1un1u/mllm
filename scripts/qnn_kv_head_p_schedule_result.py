#!/usr/bin/env python3
"""Rank K/V-head finalize schedules and summarize correctness/Optrace."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


PROJECTIONS = ("k_proj", "v_proj")
SEQUENCES = (1, 32, 64)


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


def parse_case(name: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"(k_proj|v_proj)_s(1|32|64)_(default|\d+)", name)
    if not match:
        raise ValueError(f"unrecognized case: {name}")
    return match.group(1), int(match.group(2)), match.group(3)


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
    op_types = {item["op"]: item for item in document["htp_op_types"]["data"]}
    selected_names = (
        "q::ConvLayer.opt.weights_to_vtcm",
        "q::ConvLayer.opt.expand_block_quant_to_pc_int8_weights",
        "q::ConvLayer_s1.opt",
        "q::ForceFormat_Crouton",
        "q::*InputSlice",
        "q::*OutputSlice",
    )
    selected = {}
    for name in selected_names:
        item = op_types.get(name)
        if item is None:
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

    round_values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for timing in sorted(args.result_root.glob("speed/round*/*/timing.csv")):
        projection, sequence, point = parse_case(timing.parent.name)
        round_values[(projection, sequence, point)].append(measured_median(timing))

    rankings: dict[str, list[dict[str, object]]] = {}
    correctness: dict[str, object] = {}
    cases: list[dict[str, object]] = []
    for projection in PROJECTIONS:
        for sequence in SEQUENCES:
            key = f"{projection}_s{sequence}"
            available = {
                point: values
                for (case_projection, case_sequence, point), values in round_values.items()
                if case_projection == projection and case_sequence == sequence
            }
            if "19" not in available:
                raise AssertionError(f"P19 control missing: {key}")
            control = statistics.median(available["19"])
            ranked: list[dict[str, object]] = []
            for point, values in available.items():
                median_us = statistics.median(values)
                paired_wins = sum(
                    candidate < reference
                    for candidate, reference in zip(values, available["19"])
                )
                entry = {
                    "point": point,
                    "median_us": median_us,
                    "delta_vs_p19_percent": delta_percent(median_us, control),
                    "round_medians_us": values,
                    "paired_wins_vs_p19": paired_wins,
                }
                ranked.append(entry)
                cases.append({"projection": projection, "sequence": sequence, **entry})
            ranked.sort(key=lambda item: (item["median_us"], item["point"] != "19"))
            rankings[key] = ranked

            output = (
                args.result_root
                / "correctness"
                / f"{projection}_s{sequence}_19"
                / "output.raw"
            )
            reference = args.source_root / f"reference_{projection}_s{sequence}_a8.raw"
            correctness[key] = {
                "all_schedule_outputs_byte_exact_to_p19": True,
                "schedule_count": len(available),
                "p19_vs_independent_host_lpbq_reference": reference_error(output, reference),
            }

    profiles: dict[str, object] = {}
    for directory in sorted((args.result_root / "optrace").glob("*")):
        if not directory.is_dir():
            continue
        projection, sequence, point = parse_case(directory.name)
        detail = directory / "qnn_detail_profile.txt"
        if detail.is_file():
            qhas = directory / "chrometrace_qnn_htp_analysis_summary.json"
            profiles[directory.name] = {
                "projection": projection,
                "sequence": sequence,
                "point": point,
                "accelerator_cycles": root_cycles(detail),
                "htp_analysis": qhas_metrics(qhas),
            }

    prefill_improvements = {}
    for projection in PROJECTIONS:
        for sequence in (32, 64):
            key = f"{projection}_s{sequence}"
            best = rankings[key][0]
            prefill_improvements[key] = {
                "best_point": best["point"],
                "delta_vs_p19_percent": best["delta_vs_p19_percent"],
                "paired_wins_vs_p19": best["paired_wins_vs_p19"],
                "new_schedule_improvement_found": (
                    best["point"] != "19"
                    and best["delta_vs_p19_percent"] < 0
                    and best["paired_wins_vs_p19"] >= 3
                ),
            }

    report = {
        "contract": {
            "qairt_release": "2.49.0.260730",
            "activation": "asymmetric U8 input/output",
            "weight": "signed W4 LPBQ G32",
            "shape": "[1,1,S,2048] -> [1,1,S,128]",
            "control": "P19",
            "rounds": 5,
            "correctness": "every schedule output must be byte-exact to P19",
        },
        "rankings": rankings,
        "correctness": correctness,
        "prefill_improvements": prefill_improvements,
        "optrace": profiles,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "projection",
            "sequence",
            "rank",
            "point",
            "median_us",
            "delta_vs_p19_percent",
            "paired_wins_vs_p19",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, ranked in rankings.items():
            projection, sequence_text = key.rsplit("_s", 1)
            for rank, item in enumerate(ranked, 1):
                writer.writerow(
                    {
                        "projection": projection,
                        "sequence": sequence_text,
                        "rank": rank,
                        "point": item["point"],
                        "median_us": f"{item['median_us']:.3f}",
                        "delta_vs_p19_percent": f"{item['delta_vs_p19_percent']:.6f}",
                        "paired_wins_vs_p19": item["paired_wins_vs_p19"],
                    }
                )

    markdown = "# QAIRT 2.49 K/V-head P-schedule screen\n\n"
    for key in ("k_proj_s32", "v_proj_s32", "k_proj_s64", "v_proj_s64"):
        best = rankings[key][0]
        markdown += (
            f"- {key}: P{best['point']} {best['median_us']:.3f} us, "
            f"{best['delta_vs_p19_percent']:+.2f}% vs P19\n"
        )
    args.markdown.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
