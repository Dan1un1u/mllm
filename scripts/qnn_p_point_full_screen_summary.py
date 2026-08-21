#!/usr/bin/env python3
"""Summarize full-model A8/A16 P-point candidate E2E runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(args.result_root.glob("benchmark/round*/*/qnn_runner_e2e.csv")):
        label = path.parent.name
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                values[label][row["phase"]].append(float(row["tokens_per_second"]))

    rows = []
    for label, phases in sorted(values.items()):
        if set(phases) != {"prefill_e2e", "decode_e2e_after_first"}:
            raise ValueError(f"incomplete phases for {label}: {sorted(phases)}")
        kind = label.split("_", 1)[0]
        reference = f"{kind}_default"
        rows.append(
            {
                "label": label,
                "kind": kind,
                "prefill_tokens_per_second_median": statistics.median(phases["prefill_e2e"]),
                "decode_tokens_per_second_median": statistics.median(phases["decode_e2e_after_first"]),
                "prefill_rounds": phases["prefill_e2e"],
                "decode_rounds": phases["decode_e2e_after_first"],
                "reference": reference,
            }
        )
    by_label = {row["label"]: row for row in rows}
    for row in rows:
        reference = by_label[row["reference"]]
        row["prefill_vs_default_percent"] = 100.0 * (
            row["prefill_tokens_per_second_median"] / reference["prefill_tokens_per_second_median"] - 1.0
        )
        row["decode_vs_default_percent"] = 100.0 * (
            row["decode_tokens_per_second_median"] / reference["decode_tokens_per_second_median"] - 1.0
        )
        if len(row["prefill_rounds"]) != len(reference["prefill_rounds"]):
            raise ValueError(f"round count mismatch for {row['label']} and {row['reference']}")
        row["prefill_paired_median_vs_default_percent"] = statistics.median(
            100.0 * (candidate / baseline - 1.0)
            for candidate, baseline in zip(row["prefill_rounds"], reference["prefill_rounds"])
        )
        row["decode_paired_median_vs_default_percent"] = statistics.median(
            100.0 * (candidate / baseline - 1.0)
            for candidate, baseline in zip(row["decode_rounds"], reference["decode_rounds"])
        )

    args.output_json.write_text(json.dumps({"candidates": rows}, indent=2) + "\n")
    fieldnames = [
        "label", "kind", "prefill_tokens_per_second_median", "decode_tokens_per_second_median",
        "prefill_vs_default_percent", "decode_vs_default_percent",
        "prefill_paired_median_vs_default_percent", "decode_paired_median_vs_default_percent",
    ]
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["kind"], -item["decode_tokens_per_second_median"])):
            writer.writerow({key: f"{row[key]:.6f}" if isinstance(row[key], float) else row[key] for key in fieldnames})
    for kind in ("a8", "a16"):
        candidates = [row for row in rows if row["kind"] == kind]
        candidates.sort(key=lambda item: item["decode_tokens_per_second_median"], reverse=True)
        print(kind)
        for row in candidates:
            print(
                f"  {row['label']}: prefill {row['prefill_tokens_per_second_median']:.3f} "
                f"({row['prefill_vs_default_percent']:+.2f}%, "
                f"paired {row['prefill_paired_median_vs_default_percent']:+.2f}%), "
                f"decode {row['decode_tokens_per_second_median']:.3f} "
                f"({row['decode_vs_default_percent']:+.2f}%, "
                f"paired {row['decode_paired_median_vs_default_percent']:+.2f}%)"
            )


if __name__ == "__main__":
    main()
