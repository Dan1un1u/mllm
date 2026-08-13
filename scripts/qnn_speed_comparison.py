#!/usr/bin/env python3
"""Compare two measured runner-E2E speed summaries without a speed gate."""

import argparse
import json
import math
from pathlib import Path


PHASES = ("prefill_e2e", "decode_e2e_after_first")


def measured_rate(document, phase, label):
    row = document.get("phases", {}).get(phase, {})
    if row.get("source") != "runner_e2e_measured" or row.get("rounds", 0) < 1:
        raise ValueError(f"{label} {phase} is not runner-E2E measured")
    rate = row.get("tokens_per_second_median")
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"{label} {phase} has invalid median rate: {rate!r}")
    return float(rate), int(row["rounds"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    phases = {}
    for phase in PHASES:
        candidate_rate, candidate_rounds = measured_rate(candidate, phase, "candidate")
        reference_rate, reference_rounds = measured_rate(reference, phase, "reference")
        phases[phase] = {
            "candidate_tokens_per_second_median": candidate_rate,
            "candidate_rounds": candidate_rounds,
            "reference_tokens_per_second_median": reference_rate,
            "reference_rounds": reference_rounds,
            "ratio_candidate_over_reference": candidate_rate / reference_rate,
            "percent_change": 100.0 * (candidate_rate / reference_rate - 1.0),
        }
    result = {
        "reference_id": args.reference_id,
        "reference_speed_json": str(args.reference.resolve()),
        "candidate_speed_json": str(args.candidate.resolve()),
        "acceptance": "informational; no speed threshold",
        "phases": phases,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for phase, row in phases.items():
        print(
            f"{phase}: {row['candidate_tokens_per_second_median']:.3f} vs "
            f"{row['reference_tokens_per_second_median']:.3f} token/s "
            f"({row['percent_change']:+.2f}%)"
        )


if __name__ == "__main__":
    main()
