#!/usr/bin/env python3
"""Summarize paired timing and byte-exact correctness for the bias gate."""

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


TIMING_RE = re.compile(r"round_(\d+)_(omitted|explicit_u8_zero)$")
FIXTURES = ("encoded_zero", "qmin", "qmax", "alternating", "ramp", "seeded_random")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    device_out = args.result_root / "device_out"

    rounds: dict[int, dict[str, dict]] = {}
    for directory in sorted(device_out.glob("round_*")):
        match = TIMING_RE.fullmatch(directory.name)
        if not match:
            continue
        round_index = int(match.group(1))
        variant = match.group(2)
        with (directory / "timing.csv").open(newline="", encoding="utf-8") as stream:
            values = [
                int(row["graph_execute_us"])
                for row in csv.DictReader(stream)
                if row["phase"] == "measured"
            ]
        assert len(values) == 500, (directory, len(values))
        rounds.setdefault(round_index, {})[variant] = {
            "samples": len(values),
            "median_us": statistics.median(values),
            "mean_us": statistics.fmean(values),
            "min_us": min(values),
            "p05_us": percentile(values, 0.05),
            "p95_us": percentile(values, 0.95),
            "max_us": max(values),
        }

    assert sorted(rounds) == [1, 2, 3, 4, 5]
    for variants in rounds.values():
        assert set(variants) == {"omitted", "explicit_u8_zero"}
        variants["candidate_over_omitted"] = (
            variants["explicit_u8_zero"]["median_us"] / variants["omitted"]["median_us"]
        )

    omitted_medians = [rounds[index]["omitted"]["median_us"] for index in sorted(rounds)]
    explicit_medians = [rounds[index]["explicit_u8_zero"]["median_us"] for index in sorted(rounds)]
    paired_ratios = [rounds[index]["candidate_over_omitted"] for index in sorted(rounds)]

    correctness = {}
    for fixture in FIXTURES:
        paths = {
            "omitted_1": device_out / f"omitted_{fixture}_1.bin",
            "omitted_2": device_out / f"omitted_{fixture}_2.bin",
            "explicit_1": device_out / f"explicit_{fixture}_1.bin",
            "explicit_2": device_out / f"explicit_{fixture}_2.bin",
        }
        hashes = {name: digest(path) for name, path in paths.items()}
        assert len(set(hashes.values())) == 1, (fixture, hashes)
        correctness[fixture] = {"byte_exact": True, "sha256": next(iter(hashes.values()))}

    overall_omitted = statistics.median(omitted_medians)
    overall_explicit = statistics.median(explicit_medians)
    report = {
        "status": "pass",
        "correctness": correctness,
        "timing_protocol": {
            "fresh_process_rounds": 5,
            "alternating_order": True,
            "warmup_per_process": 20,
            "measured_per_process": 500,
            "profiling": "off",
        },
        "rounds": rounds,
        "aggregate": {
            "omitted_median_of_round_medians_us": overall_omitted,
            "explicit_u8_zero_median_of_round_medians_us": overall_explicit,
            "candidate_over_omitted_ratio": overall_explicit / overall_omitted,
            "paired_ratio_median": statistics.median(paired_ratios),
            "candidate_speed_change_percent": (overall_omitted / overall_explicit - 1.0) * 100.0,
            "one_percent_no_regression_gate": overall_explicit <= overall_omitted * 1.01,
        },
    }
    output = args.result_root / "gate_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
