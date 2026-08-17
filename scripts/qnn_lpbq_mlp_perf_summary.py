#!/usr/bin/env python3
"""Summarize profiling-off Conv/MatMul LPBQ micrograph timing sets."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
SEQUENCES = (1, 32)


def measured_median(path: Path) -> tuple[float, float, int]:
    with path.open(newline="") as stream:
        values = [
            float(row["graph_execute_us"])
            for row in csv.DictReader(stream)
            if row["phase"] == "measured"
        ]
    if not values:
        raise ValueError(f"no measured rows: {path}")
    median = statistics.median(values)
    p99 = sorted(values)[min(len(values) - 1, int(len(values) * 0.99))]
    return median, p99, len(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1.01)
    parser.add_argument("--dispersion-threshold", type=float, default=0.01)
    args = parser.parse_args()

    rows = []
    failures = []
    for projection in PROJECTIONS:
        for seq in SEQUENCES:
            layouts = {}
            for layout in ("conv", "matmul"):
                process_rows = []
                for round_index in range(1, args.rounds + 1):
                    path = (
                        args.root
                        / f"round{round_index}"
                        / f"{layout}_{projection}_s{seq}"
                        / "calibration_qparam_replay.timing.csv"
                    )
                    median, p99, samples = measured_median(path)
                    process_rows.append(
                        {"round": round_index, "median_us": median, "p99_us": p99, "samples": samples}
                    )
                process_medians = [value["median_us"] for value in process_rows]
                overall = statistics.median(process_medians)
                dispersion = (max(process_medians) - min(process_medians)) / overall
                layouts[layout] = {
                    "median_us": overall,
                    "process_dispersion": dispersion,
                    "processes": process_rows,
                }
            ratio = layouts["matmul"]["median_us"] / layouts["conv"]["median_us"]
            paired_ratios = [
                layouts["matmul"]["processes"][index]["median_us"]
                / layouts["conv"]["processes"][index]["median_us"]
                for index in range(args.rounds)
            ]
            direction_consistent = all(value <= 1.0 for value in paired_ratios) or all(
                value >= 1.0 for value in paired_ratios
            )
            valid = (
                layouts["conv"]["process_dispersion"] <= args.dispersion_threshold
                and layouts["matmul"]["process_dispersion"] <= args.dispersion_threshold
                and direction_consistent
            )
            passed = valid and ratio <= args.threshold
            case = {
                "projection": projection,
                "seq": seq,
                "conv": layouts["conv"],
                "matmul": layouts["matmul"],
                "ratio": ratio,
                "paired_ratios": paired_ratios,
                "direction_consistent": direction_consistent,
                "valid": valid,
                "pass": passed,
            }
            rows.append(case)
            if not valid:
                failures.append(f"{projection}_s{seq}: unstable warm set")
            elif not passed:
                failures.append(f"{projection}_s{seq}: MatMul/Conv={ratio:.6f} > {args.threshold:.6f}")

    report = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "measurement": "profiling off; QnnGraph_execute only; 10 warmup + 100 measured; five fresh processes",
        "threshold": args.threshold,
        "dispersion_threshold": args.dispersion_threshold,
        "cases": rows,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for row in rows:
        print(
            f"{row['projection']}_s{row['seq']}: "
            f"conv={row['conv']['median_us']:.3f} us "
            f"matmul={row['matmul']['median_us']:.3f} us "
            f"ratio={row['ratio']:.6f} valid={row['valid']} pass={row['pass']}"
        )
    if failures:
        print("failures:")
        for failure in failures:
            print(f"- {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
