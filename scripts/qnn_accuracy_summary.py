#!/usr/bin/env python3
"""Summarize the lightweight Qwen3 accuracy sanity suite."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    nul_byte_count = raw.count(b"\0")
    # A severely degraded experimental model can emit token text containing
    # NUL. Preserve that fact in the result, but sanitize only the parser view
    # so a no-accuracy-gate baseline still completes its structural audit.
    text = raw.decode("utf-8", errors="replace").replace("\0", "�")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError("accuracy result CSV is empty")
    required = {"id", "match_mode", "expected", "output", "normalized_output", "pass"}
    if not required <= set(rows[0]):
        raise ValueError(f"accuracy result CSV is missing columns: {sorted(required - set(rows[0]))}")
    cases = []
    for row in rows:
        passed = row["pass"] == "1"
        cases.append({
            "id": row["id"],
            "match_mode": row["match_mode"],
            "expected": row["expected"],
            "output": row["output"],
            "normalized_output": row["normalized_output"],
            "pass": passed,
        })
    passed = sum(case["pass"] for case in cases)
    accuracy = 100.0 * passed / len(cases)
    if accuracy >= 80:
        risk = "pass"
        interpretation = "No obvious quality regression detected by the lightweight sanity suite."
    elif accuracy >= 65:
        risk = "warning"
        interpretation = "Possible quality regression; repeat and run a formal benchmark before drawing conclusions."
    else:
        risk = "fail"
        interpretation = "Likely material quality regression or runtime/model corruption."
    category_stats = {}
    for case in cases:
        category = case["id"].split("_", 1)[0]
        stats = category_stats.setdefault(category, {"passed": 0, "total": 0})
        stats["total"] += 1
        stats["passed"] += int(case["pass"])
    for stats in category_stats.values():
        stats["accuracy_percent"] = 100.0 * stats["passed"] / stats["total"]
    result = {
        "suite": "qwen3_sm8750_v79_accuracy_sanity_v2_100",
        "scope": (
            f"{len(cases)} deterministic short-answer checks covering arithmetic, logic, "
            "science, geography, language, code and instruction following; "
            "not a formal accuracy benchmark."
        ),
        "method": (
            "Greedy decoding with profiling disabled; mode-aware answer extraction removes "
            "special tokens and Markdown noise, handles numeric final answers, equivalent "
            "text alternatives, Chinese punctuation and brief explanations."
        ),
        "max_new_tokens": args.max_new_tokens,
        "passed": passed,
        "total": len(cases),
        "accuracy_percent": accuracy,
        "nul_byte_count": nul_byte_count,
        "structurally_parseable": True,
        "risk": risk,
        "interpretation": interpretation,
        "categories": category_stats,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"accuracy sanity: {passed}/{len(cases)} = {accuracy:.2f}% ({risk})")


if __name__ == "__main__":
    main()
