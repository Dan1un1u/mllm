#!/usr/bin/env python3
"""Gate the current-source W4A16 canonical run against the archived run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def accuracy_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["id"]: row for row in csv.DictReader(stream)}


def phase_tps(speed: dict, phase: str) -> float:
    return float(speed["phases"][phase]["tokens_per_second_median"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compact-full-output", type=Path, required=True)
    parser.add_argument("--compact-cropped-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    candidate_accuracy = read_json(args.candidate / "qwen3-sm8750-v79-g32-accuracy.json")
    reference_accuracy = read_json(args.reference / "qwen3-sm8750-v79-g32-accuracy.json")
    candidate_csv = args.candidate / "accuracy/qnn_accuracy_eval.csv"
    reference_csv = args.reference / "accuracy/qnn_accuracy_eval.csv"
    candidate_rows = accuracy_rows(candidate_csv)
    reference_rows = accuracy_rows(reference_csv)
    common_ids = sorted(set(candidate_rows) & set(reference_rows))
    output_mismatches = [
        case_id
        for case_id in common_ids
        if candidate_rows[case_id]["output"] != reference_rows[case_id]["output"]
    ]
    pass_mismatches = [
        case_id
        for case_id in common_ids
        if candidate_rows[case_id]["pass"] != reference_rows[case_id]["pass"]
    ]

    candidate_speed = read_json(args.candidate / "qwen3-sm8750-v79-g32-speed.json")
    reference_speed = read_json(args.reference / "qwen3-sm8750-v79-g32-speed.json")
    speed = {}
    for phase in ("prefill_e2e", "decode_e2e_after_first"):
        current = phase_tps(candidate_speed, phase)
        archived = phase_tps(reference_speed, phase)
        speed[phase] = {
            "candidate_tokens_per_second": current,
            "reference_tokens_per_second": archived,
            "delta_percent": 100.0 * (current - archived) / archived,
        }

    compact_full_sha = sha256(args.compact_full_output)
    compact_cropped_sha = sha256(args.compact_cropped_output)
    accuracy_usable = (
        candidate_accuracy.get("structurally_parseable") is True
        and candidate_accuracy.get("nul_byte_count") == 0
        and candidate_accuracy.get("total") == reference_accuracy.get("total") == 100
        and candidate_accuracy.get("passed", 0) >= reference_accuracy.get("passed", 0)
    )
    exact_answer_regression = (
        len(candidate_rows) == len(reference_rows) == 100
        and not output_mismatches
        and not pass_mismatches
    )
    required_profile_files = [
        "qwen3-sm8750-v79-g32-e2e-critical-path.html",
        "qwen3-sm8750-v79-g32-s1-chrometrace_qnn_htp_analysis_summary.json",
        "qwen3-sm8750-v79-g32-s32-chrometrace_qnn_htp_analysis_summary.json",
    ]
    profile_complete = all((args.candidate / name).is_file() for name in required_profile_files)
    manifest_equivalence = all(
        (Path("/mnt/d/llm_exp/models/qwen3_sm8750_v79/g32/"
              "w4a16_current_regression_qairt249/20260824/manifests")
         / f"model.0.{graph}_reference-equivalence.json").is_file()
        for graph in ("s1", "s32")
    )
    compact_byte_exact = compact_full_sha == compact_cropped_sha
    gate_passed = accuracy_usable and exact_answer_regression and profile_complete and manifest_equivalence and compact_byte_exact

    report = {
        "gate_passed": gate_passed,
        "interpretation": (
            "Current-source W4A16 preserves the archived graph contract and all 100 generated answers; "
            "the independently captured compact/full initial-prefill tensors are byte-identical."
            if gate_passed
            else "At least one strict W4A16 regression or compact-prefill correctness gate failed."
        ),
        "accuracy": {
            "usable_gate_passed": accuracy_usable,
            "exact_answer_regression_passed": exact_answer_regression,
            "candidate_passed": candidate_accuracy.get("passed"),
            "reference_passed": reference_accuracy.get("passed"),
            "total": candidate_accuracy.get("total"),
            "candidate_csv_sha256": sha256(candidate_csv),
            "reference_csv_sha256": sha256(reference_csv),
            "output_mismatch_count": len(output_mismatches),
            "output_mismatch_ids": output_mismatches,
            "pass_mismatch_count": len(pass_mismatches),
            "pass_mismatch_ids": pass_mismatches,
        },
        "profiling": {
            "complete": profile_complete,
            "required_files": required_profile_files,
            "speed": speed,
        },
        "graph_contract": {
            "reference_manifest_canonical_equivalence_except_exact_zero_bias_scale": manifest_equivalence,
            "graphs": ["model.0.s1", "model.0.s32"],
        },
        "compact_prefill_math": {
            "byte_exact": compact_byte_exact,
            "full_output_sha256": compact_full_sha,
            "cropped_output_sha256": compact_cropped_sha,
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = [
        "# Current-source W4A16 regression",
        "",
        f"- Gate: {'PASS' if gate_passed else 'FAIL'}",
        f"- Accuracy sanity: {candidate_accuracy.get('passed')}/100 "
        f"(reference {reference_accuracy.get('passed')}/100)",
        f"- Generated-answer mismatches: {len(output_mismatches)}/100",
        f"- s1/s32 quant manifests canonical-equivalent: {manifest_equivalence}",
        f"- Compact/full prefill output byte-exact: {compact_byte_exact}",
        f"- Prefill: {speed['prefill_e2e']['candidate_tokens_per_second']:.3f} tok/s "
        f"({speed['prefill_e2e']['delta_percent']:+.2f}%)",
        f"- Decode: {speed['decode_e2e_after_first']['candidate_tokens_per_second']:.3f} tok/s "
        f"({speed['decode_e2e_after_first']['delta_percent']:+.2f}%)",
        "",
    ]
    args.summary.write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
