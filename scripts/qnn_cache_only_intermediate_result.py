#!/usr/bin/env python3
"""Validate and summarize the cache-only intermediate-prefill experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


SEQUENCE = 32
VOCABULARY = 151936
HIDDEN = 2048
LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
CONTROL_PREFIX_BYTES = SEQUENCE * VOCABULARY
CANDIDATE_PREFIX_BYTES = SEQUENCE * HIDDEN
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
        statistics.median(_read_measured(root / "speed" / f"round{index}" / variant / "timing.csv"))
        for index in range(1, 11)
    ]


def _event_value(path: Path, identifier: str) -> int:
    pattern = re.compile(rf"value=(\d+).*identifier={re.escape(identifier)}")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    raise AssertionError(f"missing profile event {identifier} in {path}")


def _lm_head_cycles(path: Path) -> list[int]:
    pattern = re.compile(r"value=(\d+).*identifier=lm_head:.* \(cycles\)")
    return [
        int(match.group(1))
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if (match := pattern.search(line))
    ]


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
    if len(control_first) != CONTROL_PREFIX_BYTES + KV_BYTES:
        raise AssertionError(f"unexpected control output bytes: {len(control_first)}")
    if len(candidate_first) != CANDIDATE_PREFIX_BYTES + KV_BYTES:
        raise AssertionError(f"unexpected candidate output bytes: {len(candidate_first)}")

    control_repeat_exact = control_first == control_repeat
    candidate_repeat_exact = candidate_first == candidate_repeat
    kv_exact = control_first[CONTROL_PREFIX_BYTES:] == candidate_first[CANDIDATE_PREFIX_BYTES:]

    control_rounds = _round_medians(root, "control")
    candidate_rounds = _round_medians(root, "candidate")
    control_median = statistics.median(control_rounds)
    candidate_median = statistics.median(candidate_rounds)
    paired_wins = sum(candidate < control for control, candidate in zip(control_rounds, candidate_rounds))

    control_profile = root / "optrace/control/qnn_detail_profile.txt"
    candidate_profile = root / "optrace/candidate/qnn_detail_profile.txt"
    identifier = "Accelerator (execute) time (cycles)"
    control_cycles = _event_value(control_profile, identifier)
    candidate_cycles = _event_value(candidate_profile, identifier)
    control_lm = _lm_head_cycles(control_profile)
    candidate_lm = _lm_head_cycles(candidate_profile)
    if len(control_lm) != 1 or candidate_lm:
        raise AssertionError(f"unexpected lm_head profile events: control={control_lm}, candidate={candidate_lm}")

    correctness = control_repeat_exact and candidate_repeat_exact and kv_exact
    speed_non_regression = candidate_median <= control_median
    report = {
        "gate": {
            "kv_correctness_exact": correctness,
            "profiling_off_speed_non_regression": speed_non_regression,
            "passed": correctness and speed_non_regression,
        },
        "correctness": {
            "control_output_bytes": len(control_first),
            "candidate_output_bytes": len(candidate_first),
            "control_repeat_exact": control_repeat_exact,
            "candidate_repeat_exact": candidate_repeat_exact,
            "all_56_kv_outputs_exact": kv_exact,
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
            "control_lm_head_cycles": control_lm[0],
            "candidate_lm_head_events": 0,
        },
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown = "# Cache-only intermediate-prefill experiment\n\n"
    markdown += f"Gate: **{'PASS' if report['gate']['passed'] else 'FAIL'}**\n\n"
    markdown += f"- All 56 KV outputs exact: {kv_exact}\n"
    markdown += f"- Profiling-off median: {control_median:.3f} → {candidate_median:.3f} us ({report['profiling_off']['candidate_delta_percent']:.2f}%)\n"
    markdown += f"- Accelerator cycles: {control_cycles} → {candidate_cycles} ({report['optrace']['accelerator_delta_percent']:.2f}%)\n"
    markdown += f"- Removed lm_head event: {control_lm[0]} cycles → none\n"
    args.markdown.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
