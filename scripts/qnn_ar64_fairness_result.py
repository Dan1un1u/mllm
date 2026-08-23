#!/usr/bin/env python3
"""Summarize runner-level and Optrace AR64 A8/A16 fairness results."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


VARIANTS = ("a16", "a8")


def _prefill_row(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row["phase"] == "prefill_e2e"]
    if len(matches) != 1:
        raise AssertionError(f"expected one prefill row in {path}")
    row = matches[0]
    return {
        "tokens": int(row["tokens"]),
        "duration_us": int(row["duration_us"]),
        "tokens_per_second": float(row["tokens_per_second"]),
    }


def _root_cycles(path: Path) -> int:
    pattern = re.compile(r"value=(\d+).*identifier=Accelerator \(execute\) time \(cycles\)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    raise AssertionError(f"accelerator cycles missing in {path}")


def _node_work(path: Path) -> dict[str, int]:
    event = re.compile(r"depth=1\|type=node.*unit=cycles.*value=(\d+).*identifier=(.*) \(cycles\)$")
    categories = {
        "softmax": 0,
        "rmsnorm": 0,
        "lm_head": 0,
        "lpbq_projection": 0,
        "attention_matmul": 0,
        "all_nodes": 0,
    }
    projection = re.compile(r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)(?:\.|:)")
    norm = re.compile(r"(?:input_layernorm|post_attention_layernorm|q_norm|k_norm|model\.norm)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = event.search(line)
        if not match:
            continue
        cycles = int(match.group(1))
        name = match.group(2)
        categories["all_nodes"] += cycles
        if "Softmax" in name:
            categories["softmax"] += cycles
        if norm.search(name):
            categories["rmsnorm"] += cycles
        if name.startswith("lm_head:"):
            categories["lm_head"] += cycles
        if projection.search(name):
            categories["lpbq_projection"] += cycles
        if ".MatMul." in name:
            categories["attention_matmul"] += cycles
    return categories


def _delta(candidate: float, control: float) -> float:
    return (candidate / control - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root

    rounds = {
        variant: [
            _prefill_row(root / "benchmark" / f"round{index}" / variant / "qnn_runner_e2e.csv")
            for index in range(1, 11)
        ]
        for variant in VARIANTS
    }
    token_counts = {row["tokens"] for values in rounds.values() for row in values}
    if len(token_counts) != 1:
        raise AssertionError(f"prompt token counts differ: {token_counts}")
    metrics = {}
    for variant in VARIANTS:
        durations = [item["duration_us"] for item in rounds[variant]]
        throughputs = [item["tokens_per_second"] for item in rounds[variant]]
        metrics[variant] = {
            "durations_us": durations,
            "throughputs_tokens_per_second": throughputs,
            "median_duration_us": statistics.median(durations),
            "median_tokens_per_second": statistics.median(throughputs),
        }
    paired_wins = sum(
        a8["duration_us"] < a16["duration_us"]
        for a16, a8 in zip(rounds["a16"], rounds["a8"])
    )

    profiles = {}
    for variant in VARIANTS:
        path = root / "optrace" / variant / "qnn_detail_profile.txt"
        profiles[variant] = {
            "accelerator_cycles": _root_cycles(path),
            "node_work_cycles": _node_work(path),
        }
    category_deltas = {
        key: _delta(profiles["a8"]["node_work_cycles"][key], profiles["a16"]["node_work_cycles"][key])
        for key in profiles["a16"]["node_work_cycles"]
        if profiles["a16"]["node_work_cycles"][key] > 0
    }
    report = {
        "contract": {
            "ar_len": 64,
            "prompt_tokens": next(iter(token_counts)),
            "rounds": 10,
            "fresh_process_per_variant_per_round": True,
            "alternating_order": True,
            "decode_steps": 0,
            "qairt_release": "2.49.0.260730",
            "a8_finalize": "P19",
            "a16_finalize": "default",
        },
        "runner_prefill_e2e": {
            "a16": metrics["a16"],
            "a8": metrics["a8"],
            "a8_duration_delta_percent": _delta(
                metrics["a8"]["median_duration_us"], metrics["a16"]["median_duration_us"]
            ),
            "a8_throughput_delta_percent": _delta(
                metrics["a8"]["median_tokens_per_second"], metrics["a16"]["median_tokens_per_second"]
            ),
            "a8_paired_wins_of_10": paired_wins,
        },
        "optrace_s64": {
            "a16": profiles["a16"],
            "a8": profiles["a8"],
            "a8_accelerator_cycles_delta_percent": _delta(
                profiles["a8"]["accelerator_cycles"], profiles["a16"]["accelerator_cycles"]
            ),
            "a8_node_work_delta_percent": category_deltas,
        },
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    e2e = report["runner_prefill_e2e"]
    optrace = report["optrace_s64"]
    markdown = "# AR64 W4A8 versus W4A16\n\n"
    markdown += f"- Prompt tokens: {report['contract']['prompt_tokens']}\n"
    markdown += f"- A16 prefill: {metrics['a16']['median_tokens_per_second']:.3f} tok/s\n"
    markdown += f"- A8 prefill: {metrics['a8']['median_tokens_per_second']:.3f} tok/s ({e2e['a8_throughput_delta_percent']:.2f}%)\n"
    markdown += f"- A8 paired wins: {paired_wins}/10\n"
    markdown += f"- s64 accelerator cycles: {profiles['a16']['accelerator_cycles']} → {profiles['a8']['accelerator_cycles']} ({optrace['a8_accelerator_cycles_delta_percent']:.2f}%)\n"
    args.markdown.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
