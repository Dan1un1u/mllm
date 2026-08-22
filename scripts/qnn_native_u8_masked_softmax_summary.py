#!/usr/bin/env python3
"""Rank and summarize the isolated QAIRT native-U8 masked-Softmax experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


TAG = re.compile(r"qairt(?P<sdk>247|249)_s(?P<seq>1|32)_p(?P<p>default|\d+)$")


def _timing(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        values = [
            int(row["graph_execute_us"])
            for row in csv.DictReader(stream)
            if row["phase"] == "measured"
        ]
    if not values:
        raise AssertionError(f"no measured timings: {path}")
    return float(statistics.median(values))


def _identity(tag: str) -> tuple[str, int, str]:
    match = TAG.fullmatch(tag)
    if not match:
        raise AssertionError(f"invalid case tag: {tag}")
    return match["sdk"], int(match["seq"]), match["p"]


def _stage1(root: Path) -> dict:
    groups: dict[str, list[dict]] = {}
    for timing in sorted((root / "stage1").glob("*/timing.csv")):
        tag = timing.parent.name
        sdk, seq, point = _identity(tag)
        key = f"qairt{sdk}_s{seq}"
        groups.setdefault(key, []).append(
            {"tag": tag, "p": point, "median_us": _timing(timing)}
        )
    if set(groups) != {"qairt247_s1", "qairt247_s32", "qairt249_s1", "qairt249_s32"}:
        raise AssertionError(f"incomplete stage-1 groups: {sorted(groups)}")
    for values in groups.values():
        values.sort(key=lambda item: (item["median_us"], item["tag"]))
    return groups


def rank(root: Path, output: Path, top_file: Path, top: int) -> None:
    groups = _stage1(root)
    output.write_text(json.dumps(groups, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected = [item["tag"] for key in sorted(groups) for item in groups[key][:top]]
    top_file.write_text("\n".join(selected) + "\n", encoding="utf-8")


def _stage2(root: Path, stage1: dict) -> tuple[dict, dict]:
    candidates = {item["tag"] for values in stage1.values() for item in values[:3]}
    stage2: dict[str, dict] = {}
    for tag in sorted(candidates):
        process_medians = [
            _timing(path)
            for path in sorted((root / "stage2").glob(f"round*/{tag}/timing.csv"))
        ]
        if not process_medians:
            raise AssertionError(f"missing stage-2 timing: {tag}")
        stage2[tag] = {
            "process_medians_us": process_medians,
            "median_of_process_medians_us": statistics.median(process_medians),
            "minimum_us": min(process_medians),
            "maximum_us": max(process_medians),
        }
    winners: dict[str, str] = {}
    for key, values in stage1.items():
        eligible = [item["tag"] for item in values[:3]]
        winners[key] = min(
            eligible, key=lambda tag: (stage2[tag]["median_of_process_medians_us"], tag)
        )
    return stage2, winners


def select(root: Path, output: Path, winner_file: Path) -> None:
    stage1 = _stage1(root)
    stage2, winners = _stage2(root, stage1)
    output.write_text(
        json.dumps({"stage2": stage2, "winners": winners}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    winner_file.write_text("\n".join(winners[key] for key in sorted(winners)) + "\n", encoding="utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _deltas(left: bytes, right: bytes) -> dict:
    if len(left) != len(right):
        raise AssertionError(f"byte-count mismatch: {len(left)} versus {len(right)}")
    values = [abs(a - b) for a, b in zip(left, right, strict=True)]
    return {
        "bytes": len(values),
        "equal_fraction": sum(value == 0 for value in values) / len(values),
        "mean_abs_code_delta": statistics.fmean(values),
        "max_abs_code_delta": max(values),
    }


def _optrace_metrics(root: Path, tag: str) -> dict:
    qhas = (
        root
        / "optrace"
        / tag
        / f"{tag}-chrometrace_qnn_htp_analysis_summary.json"
    )
    document = json.loads(qhas.read_text(encoding="utf-8"))["data"]
    overall = document["htp_overall_summary"]["data"]
    if len(overall) != 1:
        raise AssertionError(f"expected one HTP overall record: {qhas}")
    kernels = [
        item
        for item in document["htp_op_types"]["data"]
        if item["op"] == "q::MaskedSoftmax_Crouton_Scratch"
    ]
    if len(kernels) != 1:
        raise AssertionError(f"expected one native masked-Softmax kernel type: {qhas}")
    kernel = kernels[0]
    return {
        "tag": tag,
        "kernel": kernel["op"],
        "instances": kernel["instances"],
        "work_cycles": kernel["cycles"],
        "dominant_path_cycles": kernel["num_dominant_path_cycles_htp_0"],
        "dram_read": kernel["dram_read"],
        "dram_write": kernel["dram_write"],
        "vtcm_read": kernel["vtcm_read"],
        "vtcm_write": kernel["vtcm_write"],
        "graph_execute_us": overall[0]["graph_execute_us"],
        "graph_timeline_cycles": overall[0]["timeline_cycles"],
    }


def _percent_change(candidate: int, reference: int) -> float:
    if reference == 0:
        raise AssertionError("cannot calculate a percentage change from zero")
    return (candidate / reference - 1.0) * 100.0


def final(root: Path, artifact_root: Path, output: Path) -> None:
    stage1 = _stage1(root)
    stage2, winners = _stage2(root, stage1)

    correctness: dict[str, dict] = {}
    outputs: dict[tuple[str, int], bytes] = {}
    for key, tag in winners.items():
        sdk, seq, _point = _identity(tag)
        first = (root / "correctness" / "first" / tag / "output.raw").read_bytes()
        repeat = (root / "correctness" / "repeat" / tag / "output.raw").read_bytes()
        expected = (artifact_root / f"host_expected_s{seq}_u8.raw").read_bytes()
        if first != repeat:
            raise AssertionError(f"non-repeatable output: {tag}")
        outputs[(sdk, seq)] = first
        correctness[key] = {
            "tag": tag,
            "repeatable": True,
            "output_sha256": _sha256(first),
            "versus_host_model": _deltas(first, expected),
        }
    cross_sdk = {
        f"s{seq}": {
            "qairt247_sha256": _sha256(outputs[("247", seq)]),
            "qairt249_sha256": _sha256(outputs[("249", seq)]),
            "comparison": _deltas(outputs[("247", seq)], outputs[("249", seq)]),
        }
        for seq in (1, 32)
    }
    optrace = {key: _optrace_metrics(root, tag) for key, tag in winners.items()}
    optrace_cross_sdk = {}
    for seq in (1, 32):
        reference = optrace[f"qairt247_s{seq}"]
        candidate = optrace[f"qairt249_s{seq}"]
        optrace_cross_sdk[f"s{seq}"] = {
            "work_cycles_percent_change": _percent_change(
                candidate["work_cycles"], reference["work_cycles"]
            ),
            "dominant_path_cycles_percent_change": _percent_change(
                candidate["dominant_path_cycles"], reference["dominant_path_cycles"]
            ),
            "graph_execute_us_percent_change": _percent_change(
                candidate["graph_execute_us"], reference["graph_execute_us"]
            ),
            "graph_timeline_cycles_percent_change": _percent_change(
                candidate["graph_timeline_cycles"], reference["graph_timeline_cycles"]
            ),
            "kernel_dram_unchanged": (
                candidate["dram_read"] == reference["dram_read"] == 0
                and candidate["dram_write"] == reference["dram_write"] == 0
            ),
            "kernel_vtcm_traffic_unchanged": (
                candidate["vtcm_read"] == reference["vtcm_read"]
                and candidate["vtcm_write"] == reference["vtcm_write"]
            ),
        }
    report = {
        "contract": {
            "activation": "asymmetric U8 input/output",
            "operation": "native qti.aisw masked-Softmax pattern",
            "shapes": [[1, 1, 1, 1024], [1, 1, 32, 1024]],
            "search": "independent top-3 finalize-P refinement per QAIRT and sequence",
        },
        "stage1": stage1,
        "stage2": stage2,
        "winners": winners,
        "correctness": correctness,
        "cross_sdk": cross_sdk,
        "optrace": optrace,
        "optrace_cross_sdk": optrace_cross_sdk,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    rank_parser = subparsers.add_parser("rank")
    rank_parser.add_argument("--results-root", type=Path, required=True)
    rank_parser.add_argument("--output", type=Path, required=True)
    rank_parser.add_argument("--top-file", type=Path, required=True)
    rank_parser.add_argument("--top", type=int, default=3)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--results-root", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument("--winner-file", type=Path, required=True)
    final_parser = subparsers.add_parser("final")
    final_parser.add_argument("--results-root", type=Path, required=True)
    final_parser.add_argument("--artifact-root", type=Path, required=True)
    final_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "rank":
        rank(args.results_root, args.output, args.top_file, args.top)
    elif args.command == "select":
        select(args.results_root, args.output, args.winner_file)
    else:
        final(args.results_root, args.artifact_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
