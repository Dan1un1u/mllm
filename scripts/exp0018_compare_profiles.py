#!/usr/bin/env python3
"""Compare EXP-0018 profiling evidence with A8 and W4A16 references."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PREFIX = "qwen3-sm8750-v79-g32"
STAGE_METRICS = (
    "critical_path_us_estimate",
    "num_dominant_path_cycles_htp_0",
    "trace_work_cycles",
    "active_union_cycles",
    "dram_read",
    "dram_write",
)
QUANT_METRICS = (
    "work_cycles",
    "hardware_active_cycles",
    "dominant_path_cycles",
    "active_union_cycles",
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _delta(candidate: float, reference: float) -> dict[str, float | None]:
    return {
        "candidate": candidate,
        "reference": reference,
        "delta": candidate - reference,
        "percent_change": (
            (candidate / reference - 1.0) * 100.0 if reference != 0.0 else None
        ),
    }


def _stage_comparison(candidate: Path, reference: Path, graph: str) -> list[dict[str, Any]]:
    suffix = f"{PREFIX}-{graph}-qwen3-stage-summary.csv"
    cand = {row["stage"]: row for row in _csv_rows(candidate / suffix)}
    ref = {row["stage"]: row for row in _csv_rows(reference / suffix)}
    result: list[dict[str, Any]] = []
    for stage in sorted(cand.keys() & ref.keys(), key=lambda name: int(cand[name]["stage_order"])):
        item: dict[str, Any] = {
            "stage": stage,
            "stage_label": cand[stage]["stage_label"],
        }
        for metric in STAGE_METRICS:
            item[metric] = _delta(_number(cand[stage][metric]), _number(ref[stage][metric]))
        result.append(item)
    return result


def _aggregate_quant(path: Path, graph: str) -> dict[str, dict[str, float]]:
    suffix = f"{PREFIX}-{graph}-quant-stage.csv"
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in _csv_rows(path / suffix):
        category = row["category"]
        for metric in QUANT_METRICS:
            totals[category][metric] += _number(row[metric])
    return {category: dict(values) for category, values in totals.items()}


def _quant_comparison(candidate: Path, reference: Path, graph: str) -> list[dict[str, Any]]:
    cand = _aggregate_quant(candidate, graph)
    ref = _aggregate_quant(reference, graph)
    result: list[dict[str, Any]] = []
    for category in sorted(cand.keys() | ref.keys()):
        item: dict[str, Any] = {"category": category}
        for metric in QUANT_METRICS:
            item[metric] = _delta(cand.get(category, {}).get(metric, 0.0), ref.get(category, {}).get(metric, 0.0))
        result.append(item)
    return result


def _speed(path: Path) -> dict[str, Any]:
    with (path / f"{PREFIX}-speed.json").open(encoding="utf-8") as handle:
        return json.load(handle)["phases"]


def _speed_comparison(candidate: Path, reference: Path) -> dict[str, Any]:
    cand = _speed(candidate)
    ref = _speed(reference)
    return {
        phase: _delta(
            float(cand[phase]["tokens_per_second_median"]),
            float(ref[phase]["tokens_per_second_median"]),
        )
        for phase in sorted(cand.keys() & ref.keys())
    }


def _top_stage_deltas(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: abs(float(row["critical_path_us_estimate"]["delta"])),
        reverse=True,
    )[:limit]


def _top_quant_deltas(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: abs(float(row["dominant_path_cycles"]["delta"])),
        reverse=True,
    )[:limit]


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# EXP-0018 profile comparison", ""]
    for label, comparison in report["comparisons"].items():
        lines.extend((f"## Candidate vs {label}", "", "### E2E speed", ""))
        for phase, values in comparison["speed"].items():
            lines.append(
                f"- {phase}: {values['candidate']:.3f} vs {values['reference']:.3f} "
                f"tok/s ({values['percent_change']:+.2f}%)"
            )
        for graph, graph_data in comparison["graphs"].items():
            lines.extend(("", f"### {graph} largest stage critical-path deltas", ""))
            for row in graph_data["top_stage_critical_path_deltas"]:
                values = row["critical_path_us_estimate"]
                lines.append(
                    f"- {row['stage']}: {values['candidate']:.3f} vs "
                    f"{values['reference']:.3f} us ({values['delta']:+.3f} us)"
                )
            lines.extend(("", f"### {graph} largest quant-kernel dominant-path deltas", ""))
            for row in graph_data["top_quant_dominant_path_deltas"]:
                values = row["dominant_path_cycles"]
                work = row["work_cycles"]
                active = row["hardware_active_cycles"]
                lines.append(
                    f"- {row['category']}: {values['candidate']:.0f} vs "
                    f"{values['reference']:.0f} dominant cycles ({values['delta']:+.0f}); "
                    f"work {work['candidate']:.0f} vs {work['reference']:.0f}; "
                    f"active {active['candidate']:.0f} vs {active['reference']:.0f}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--a8-reference", type=Path, required=True)
    parser.add_argument("--a16-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    comparisons: dict[str, Any] = {}
    for label, reference in (
        ("a8_reference", args.a8_reference),
        ("a16_reference", args.a16_reference),
    ):
        graph_results: dict[str, Any] = {}
        for graph in ("s1", "s32"):
            stages = _stage_comparison(args.candidate, reference, graph)
            quant = _quant_comparison(args.candidate, reference, graph)
            graph_results[graph] = {
                "top_stage_critical_path_deltas": _top_stage_deltas(stages),
                "top_quant_dominant_path_deltas": _top_quant_deltas(quant),
                "all_stage_deltas": stages,
                "all_quant_category_deltas": quant,
            }
        comparisons[label] = {
            "root": str(reference),
            "speed": _speed_comparison(args.candidate, reference),
            "graphs": graph_results,
        }

    report = {
        "experiment": "EXP-0018",
        "candidate_root": str(args.candidate),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(_markdown(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
