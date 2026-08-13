#!/usr/bin/env python3
"""Summarize runner-level QNN AOT throughput, with an explicit QHAS fallback."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PHASES = ("prefill_e2e", "decode_e2e_after_first")


def read_runner_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def qhas_graph_execute_us(path):
    with path.open() as stream:
        document = json.load(stream)
    return float(document["data"]["htp_overall_summary"]["data"][0]["graph_execute_us"])


def summarize(paths, s1_qhas=None, s32_qhas=None, forbid_fallback=False):
    samples = {phase: [] for phase in PHASES}
    for path in paths:
        for row in read_runner_csv(path):
            phase = row.get("phase")
            if phase not in samples:
                continue
            tokens = int(row["tokens"])
            duration_us = float(row["duration_us"])
            if tokens <= 0 or not math.isfinite(duration_us) or duration_us <= 0:
                continue
            samples[phase].append({
                "file": str(path),
                "tokens": tokens,
                "duration_us": duration_us,
                "tokens_per_second": tokens * 1_000_000.0 / duration_us,
            })

    result = {"source": "runner_e2e_measured", "phases": {}, "samples": samples}
    for phase, rows in samples.items():
        if not rows:
            continue
        rates = [row["tokens_per_second"] for row in rows]
        durations = [row["duration_us"] for row in rows]
        tokens = [row["tokens"] for row in rows]
        result["phases"][phase] = {
            "rounds": len(rows),
            "source": "runner_e2e_measured",
            "tokens_median": statistics.median(tokens),
            "duration_us_median": statistics.median(durations),
            "tokens_per_second_median": statistics.median(rates),
            "tokens_per_second_min": min(rates),
            "tokens_per_second_max": max(rates),
        }

    missing = sorted(set(PHASES) - set(result["phases"]))
    if not missing:
        return result
    if forbid_fallback:
        raise ValueError(
            f"missing runner E2E samples for {missing}; QHAS fallback is forbidden for this baseline"
        )
    if not s1_qhas or not s32_qhas:
        raise ValueError(f"missing runner E2E samples for {missing}; QHAS fallback paths were not provided")

    fallback = {
        "prefill_e2e": (32, qhas_graph_execute_us(s32_qhas)),
        "decode_e2e_after_first": (1, qhas_graph_execute_us(s1_qhas)),
    }
    for phase in missing:
        tokens, duration_us = fallback[phase]
        rate = tokens * 1_000_000.0 / duration_us
        result["phases"][phase] = {
            "rounds": 0,
            "tokens_median": tokens,
            "duration_us_median": duration_us,
            "tokens_per_second_median": rate,
            "tokens_per_second_min": rate,
            "tokens_per_second_max": rate,
            "source": "qhas_graph_execute_derived",
        }
    if len(missing) == len(PHASES):
        result["source"] = "qhas_graph_execute_derived"
    else:
        result["source"] = "mixed_measured_and_derived"
    result["warning"] = (
        "At least one runner-level E2E phase was unavailable and was inferred from one "
        "traced graphExecute; derived values may include profiling perturbation and exclude "
        "CPU sampling, cache maintenance and runtime overhead."
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runner_csv", nargs="*", type=Path)
    parser.add_argument("--s1-qhas", type=Path)
    parser.add_argument("--s32-qhas", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--forbid-fallback", action="store_true")
    args = parser.parse_args()
    result = summarize(
        args.runner_csv,
        args.s1_qhas,
        args.s32_qhas,
        forbid_fallback=args.forbid_fallback,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    for phase in PHASES:
        row = result["phases"][phase]
        print(
            f"{phase}: {row['tokens_per_second_median']:.3f} token/s "
            f"({row['rounds']} measured rounds, source={result['source']})"
        )


if __name__ == "__main__":
    main()
