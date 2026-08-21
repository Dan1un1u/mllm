#!/usr/bin/env python3
"""Rank QAIRT P points from repeated LPBQ projection micrographs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


WEIGHTS = {"gate_proj": 28, "up_proj": 28, "down_proj": 28, "lm_head": 1}


def measured_median(path: Path) -> float:
    with path.open(newline="") as handle:
        values = [float(row["graph_execute_us"]) for row in csv.DictReader(handle) if row["phase"] == "measured"]
    if not values:
        raise ValueError(f"no measured rows: {path}")
    return statistics.median(values)


def parse_case(name: str) -> tuple[str, str, str, str]:
    activation, graph, remainder = name.split("_", 2)
    for projection in ("gate_proj", "up_proj", "down_proj", "lm_head"):
        prefix = projection + "_"
        if remainder.startswith(prefix):
            return activation, graph, projection, remainder[len(prefix) :]
    raise ValueError(f"unrecognized case name: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    per_case: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for timing in sorted(args.result_root.glob("speed/round*/*/timing.csv")):
        activation, graph, projection, point = parse_case(timing.parent.name)
        per_case[(activation, graph, projection, point)].append(measured_median(timing))

    rows = []
    grouped: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (activation, graph, projection, point), round_medians in sorted(per_case.items()):
        median_us = statistics.median(round_medians)
        grouped[(activation, graph)][point][projection] = median_us
        rows.append(
            {
                "activation": activation,
                "graph": graph,
                "projection": projection,
                "point": point,
                "median_us": median_us,
                "round_medians_us": round_medians,
            }
        )

    rankings = {}
    for (activation, graph), points in sorted(grouped.items()):
        default = points["default"]
        selected_weights = {name: WEIGHTS[name] for name in WEIGHTS if name in default}
        if not selected_weights:
            raise ValueError(f"{activation}/{graph} has no recognized projections")
        ranked = []
        for point, projections in points.items():
            missing = set(selected_weights) - set(projections)
            if missing:
                raise ValueError(f"{activation}/{graph}/{point} missing {sorted(missing)}")
            weighted_us = sum(selected_weights[name] * projections[name] for name in selected_weights)
            default_weighted_us = sum(selected_weights[name] * default[name] for name in selected_weights)
            ranked.append(
                {
                    "point": point,
                    "weighted_projection_us": weighted_us,
                    "delta_vs_default_percent": 100.0 * (weighted_us / default_weighted_us - 1.0),
                    "projection_median_us": projections,
                }
            )
        ranked.sort(key=lambda item: item["weighted_projection_us"])
        rankings[f"{activation}_{graph}"] = ranked

    payload = {
        "method": "median within each process, median across rounds; 28x gate/up/down plus 1x lm_head heuristic",
        "weights": WEIGHTS,
        "rankings": rankings,
        "cases": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        fieldnames = ["activation", "graph", "rank", "point", "weighted_projection_us", "delta_vs_default_percent"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, ranked in rankings.items():
            activation, graph = key.split("_", 1)
            for rank, item in enumerate(ranked, 1):
                writer.writerow(
                    {
                        "activation": activation,
                        "graph": graph,
                        "rank": rank,
                        "point": item["point"],
                        "weighted_projection_us": f"{item['weighted_projection_us']:.3f}",
                        "delta_vs_default_percent": f"{item['delta_vs_default_percent']:.6f}",
                    }
                )
    for key, ranked in rankings.items():
        top = ", ".join(f"{item['point']} ({item['delta_vs_default_percent']:+.2f}%)" for item in ranked[:5])
        print(f"{key}: {top}")


if __name__ == "__main__":
    main()
