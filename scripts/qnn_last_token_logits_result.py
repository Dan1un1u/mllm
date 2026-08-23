#!/usr/bin/env python3
"""Validate and summarize the last-token-only lm_head experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


SEQUENCE = 32
VOCABULARY = 151936
LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
CONTROL_LOGITS_BYTES = SEQUENCE * VOCABULARY
CANDIDATE_LOGITS_BYTES = VOCABULARY
KV_BYTES = 2 * LAYERS * KV_HEADS * HEAD_DIM * SEQUENCE


def _read_measured(path: Path) -> list[int]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        values = [int(row["graph_execute_us"]) for row in rows if row["phase"] == "measured"]
    if not values:
        raise AssertionError(f"no measured timings in {path}")
    return values


def _round_medians(root: Path, variant: str) -> list[float]:
    return [
        statistics.median(_read_measured(root / "speed" / f"round{round_index}" / variant / "timing.csv"))
        for round_index in range(1, 11)
    ]


def _profile_value(path: Path, identifier: str) -> int:
    pattern = re.compile(rf"value=(\d+).*identifier={re.escape(identifier)}")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    raise AssertionError(f"profile event not found: {identifier} in {path}")


def _lm_head_cycles(path: Path) -> int:
    pattern = re.compile(r"value=(\d+).*identifier=lm_head:.* \(cycles\)")
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            values.append(int(match.group(1)))
    if len(values) != 1:
        raise AssertionError(f"expected one lm_head event in {path}, got {len(values)}")
    return values[0]


def _delta_percent(candidate: float, control: float) -> float:
    return (candidate / control - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root

    control_first = (root / "correctness/first/control/output.raw").read_bytes()
    control_repeat = (root / "correctness/repeat/control/output.raw").read_bytes()
    candidate_first = (root / "correctness/first/candidate/output.raw").read_bytes()
    candidate_repeat = (root / "correctness/repeat/candidate/output.raw").read_bytes()
    expected_control = CONTROL_LOGITS_BYTES + KV_BYTES
    expected_candidate = CANDIDATE_LOGITS_BYTES + KV_BYTES
    if len(control_first) != expected_control or len(candidate_first) != expected_candidate:
        raise AssertionError(
            f"unexpected output sizes: control={len(control_first)}, candidate={len(candidate_first)}"
        )

    repeat_exact = {
        "control": control_first == control_repeat,
        "candidate": candidate_first == candidate_repeat,
    }
    control_last_logits = control_first[
        CONTROL_LOGITS_BYTES - VOCABULARY : CONTROL_LOGITS_BYTES
    ]
    candidate_logits = candidate_first[:CANDIDATE_LOGITS_BYTES]
    logits_exact = control_last_logits == candidate_logits
    kv_exact = control_first[CONTROL_LOGITS_BYTES:] == candidate_first[CANDIDATE_LOGITS_BYTES:]

    control_rounds = _round_medians(root, "control")
    candidate_rounds = _round_medians(root, "candidate")
    control_median = statistics.median(control_rounds)
    candidate_median = statistics.median(candidate_rounds)
    paired_wins = sum(c < b for b, c in zip(control_rounds, candidate_rounds))

    control_profile = root / "optrace/control/qnn_detail_profile.txt"
    candidate_profile = root / "optrace/candidate/qnn_detail_profile.txt"
    accelerator_identifier = "Accelerator (execute) time (cycles)"
    control_cycles = _profile_value(control_profile, accelerator_identifier)
    candidate_cycles = _profile_value(candidate_profile, accelerator_identifier)
    control_lm_cycles = _lm_head_cycles(control_profile)
    candidate_lm_cycles = _lm_head_cycles(candidate_profile)

    correctness = all(repeat_exact.values()) and logits_exact and kv_exact
    speed_non_regression = candidate_median <= control_median
    report = {
        "gate": {
            "correctness_exact": correctness,
            "profiling_off_speed_non_regression": speed_non_regression,
            "passed": correctness and speed_non_regression,
        },
        "correctness": {
            "control_output_bytes": len(control_first),
            "candidate_output_bytes": len(candidate_first),
            "control_repeat_exact": repeat_exact["control"],
            "candidate_repeat_exact": repeat_exact["candidate"],
            "last_logits_exact": logits_exact,
            "all_kv_outputs_exact": kv_exact,
        },
        "profiling_off": {
            "control_round_medians_us": control_rounds,
            "candidate_round_medians_us": candidate_rounds,
            "control_median_us": control_median,
            "candidate_median_us": candidate_median,
            "candidate_delta_percent": _delta_percent(candidate_median, control_median),
            "candidate_paired_wins_of_10": paired_wins,
        },
        "optrace": {
            "control_accelerator_cycles": control_cycles,
            "candidate_accelerator_cycles": candidate_cycles,
            "accelerator_delta_percent": _delta_percent(candidate_cycles, control_cycles),
            "control_lm_head_cycles": control_lm_cycles,
            "candidate_lm_head_cycles": candidate_lm_cycles,
            "lm_head_delta_percent": _delta_percent(candidate_lm_cycles, control_lm_cycles),
        },
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown = "# Last-token-only lm_head experiment\n\n"
    markdown += f"Gate: **{'PASS' if report['gate']['passed'] else 'FAIL'}**\n\n"
    markdown += "- Last-row logits exact: " + str(logits_exact) + "\n"
    markdown += "- All 56 KV outputs exact: " + str(kv_exact) + "\n"
    markdown += f"- Profiling-off median: {control_median:.3f} → {candidate_median:.3f} us ({report['profiling_off']['candidate_delta_percent']:.2f}%)\n"
    markdown += f"- Accelerator cycles: {control_cycles} → {candidate_cycles} ({report['optrace']['accelerator_delta_percent']:.2f}%)\n"
    markdown += f"- lm_head cycles: {control_lm_cycles} → {candidate_lm_cycles} ({report['optrace']['lm_head_delta_percent']:.2f}%)\n"
    args.markdown.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
