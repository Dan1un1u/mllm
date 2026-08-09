#!/usr/bin/env python3
"""Compare standalone Qwen3 Layer 5 A/B/C timing and graph contracts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


VARIANTS = ("A_original", "B_identity_fold", "C_hadamard_r1_r2")


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _timings(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load(path)
    return {item["workload"]: item for item in payload["results"]}


def _status(regression_percent: float) -> str:
    if regression_percent <= 3.0:
        return "pass"
    if regression_percent <= 5.0:
        return "inconclusive_rerun"
    return "fail"


def _manifest_signature(path: Path) -> dict[str, Any]:
    payload = _load(path)
    operations = payload.get("operations", [])
    names = [str(operation.get("name", "")) for operation in operations]
    op_types = [str(operation.get("qnn_op_type", "")) for operation in operations]
    return {
        "operation_count": len(operations),
        "operation_names_and_types": sorted(zip(names, op_types)),
        "operation_type_counts": dict(sorted(Counter(op_types).items())),
        "forbidden_named_ops": sorted(
            name for name in names if "hadamard" in name.lower() or "rotate" in name.lower()
        ),
    }


_MIR_OPERATION = re.compile(r'^\s+linalg\.[^.]+\.(?P<type>[A-Za-z0-9_]+Op) <name="(?P<name>[^"]+)">')


def _mir_signature(path: Path) -> dict[str, Any]:
    operations: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _MIR_OPERATION.match(line)
        if match and "using_qnn:true" in line:
            operations.append((match.group("name"), match.group("type")))
    names = [name for name, _ in operations]
    op_types = [op_type for _, op_type in operations]
    return {
        "operation_count": len(operations),
        "operation_names_and_types": operations,
        "operation_type_counts": dict(sorted(Counter(op_types).items())),
        "forbidden_named_ops": sorted(
            name for name in names if "hadamard" in name.lower() or "rotate" in name.lower()
        ),
    }


def _contract_comparison(manifest_root: Path | None) -> dict[str, Any]:
    if manifest_root is None:
        return {"available": False}
    result: dict[str, Any] = {"available": True, "workloads": {}}
    for workload in ("s1", "s32"):
        signatures: dict[str, Any] = {}
        for variant in VARIANTS:
            candidates = (
                ("manifest", manifest_root / variant / "manifests" / f"model.0.{workload}_quant_manifest.json"),
                ("manifest", manifest_root / variant / f"model.0.{workload}_quant_manifest.json"),
                ("mir", manifest_root / variant / f"qwen3_layer5_block_{workload}.mir"),
            )
            source = next(((kind, path) for kind, path in candidates if path.is_file()), None)
            if source is None:
                signatures[variant] = {"missing": True}
            else:
                kind, path = source
                signature = _manifest_signature(path) if kind == "manifest" else _mir_signature(path)
                signatures[variant] = {"path": str(path), "source": kind, **signature}
        baseline = signatures[VARIANTS[0]]
        exact_matches: dict[str, bool] = {}
        for variant in VARIANTS[1:]:
            candidate = signatures[variant]
            exact_matches[variant] = (
                not baseline.get("missing")
                and not candidate.get("missing")
                and baseline["operation_names_and_types"] == candidate["operation_names_and_types"]
                and not candidate["forbidden_named_ops"]
            )
        result["workloads"][workload] = {
            "variants": signatures,
            "exact_operation_contract_match": exact_matches,
        }
    result["pass"] = all(
        value
        for workload in result["workloads"].values()
        for value in workload["exact_operation_contract_match"].values()
    )
    return result


def build_report(timing_root: Path, manifest_root: Path | None) -> dict[str, Any]:
    timings = {
        variant: _timings(timing_root / variant / "timing.json")
        for variant in VARIANTS
    }
    comparisons: dict[str, Any] = {}
    for workload in ("s1_decode", "s32_prefill_chunk"):
        baseline = float(timings[VARIANTS[0]][workload]["median_us"])
        variants: dict[str, Any] = {}
        for variant in VARIANTS:
            item = timings[variant][workload]
            median = float(item["median_us"])
            regression = (median / baseline - 1.0) * 100.0
            variants[variant] = {
                "median_us": median,
                "p95_us": float(item["p95_us"]),
                "regression_percent_vs_A": regression,
                "status_vs_A": "baseline" if variant == VARIANTS[0] else _status(regression),
            }
        comparisons[workload] = variants

    graph_contract = _contract_comparison(manifest_root)
    candidate_statuses = [
        comparisons[workload]["C_hadamard_r1_r2"]["status_vs_A"]
        for workload in comparisons
    ]
    if "fail" in candidate_statuses:
        overall = "fail"
    elif "inconclusive_rerun" in candidate_statuses:
        overall = "inconclusive_rerun"
    elif graph_contract.get("available") and not graph_contract.get("pass"):
        overall = "fail_graph_contract"
    else:
        overall = "pass"
    return {
        "schema_version": 1,
        "layer": 5,
        "acceptance": {
            "pass_max_regression_percent": 3.0,
            "rerun_max_regression_percent": 5.0,
        },
        "comparisons": comparisons,
        "graph_contract": graph_contract,
        "overall_status": overall,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 Layer 5 offline R1/R2 performance report",
        "",
        "| Workload | Variant | Median (ms) | P95 (ms) | Regression vs A | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for workload, variants in report["comparisons"].items():
        for variant, item in variants.items():
            lines.append(
                f"| {workload} | {variant} | {item['median_us'] / 1000.0:.3f} | "
                f"{item['p95_us'] / 1000.0:.3f} | {item['regression_percent_vs_A']:+.2f}% | "
                f"{item['status_vs_A']} |"
            )
    lines.extend(("", f"Overall: **{report['overall_status']}**", ""))
    graph = report["graph_contract"]
    if graph.get("available"):
        lines.append(f"Graph operation contract: **{'pass' if graph.get('pass') else 'fail'}**")
    else:
        lines.append("Graph operation contract: not supplied")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.timing_root, args.manifest_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))


if __name__ == "__main__":
    main()
