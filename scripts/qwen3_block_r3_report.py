#!/usr/bin/env python3
"""Report exploratory Qwen3 Layer 5 C versus R3 device results.

R3 latency is exploratory. This report records repeated timing, single-capture
HTP diagnostics, stage attribution, and placement evidence without applying
the historical A-versus-C performance threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any


VARIANTS = OrderedDict((
    ("C", {
        "name": "C_hadamard_r1_r2", "optrace_dir": "C",
        "artifact_dir": "c_baseline", "realization": "offline R1/R2 control",
    }),
    ("D-Dense", {
        "name": "D_dense_r3", "optrace_dir": "D_dense",
        "artifact_dir": "dense", "realization": "post-RoPE R3 dense constant MatMul",
    }),
    ("D-FWHT-Graph", {
        "name": "D_fwht_graph_r3", "optrace_dir": "D_fwht",
        "artifact_dir": "fwht", "realization": "post-RoPE graph-native seven-stage FWHT",
    }),
))
WORKLOADS = OrderedDict((("s1", "s1_decode"), ("s32", "s32_prefill_chunk")))
TIMING_FIELDS = ("median_us", "p95_us", "mean_us", "min_us", "max_us")
OVERALL_FIELDS = (
    "graph_execute_us", "timeline_cycles", "qnn_nodes", "htp_nodes",
    "total_dram_read", "total_dram_write", "total_vtcm_read",
    "total_vtcm_write", "peak_vtcm_alloc",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _percent_change(candidate: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("cannot compute relative change from a zero baseline")
    return (candidate / baseline - 1.0) * 100.0


def _load_timing(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    results = {item["workload"]: item for item in payload["results"]}
    missing = set(WORKLOADS.values()) - set(results)
    if missing:
        raise ValueError(f"{path}: missing timing workloads: {sorted(missing)}")
    return {
        "path": str(path), "variant": payload["variant"],
        "warmup": int(payload["warmup"]), "iterations": int(payload["iterations"]),
        "timing_boundary": payload["timing_boundary"], "results": results,
    }


def _load_qhas(path: Path) -> dict[str, Any]:
    overall = _load_json(path)["data"]["htp_overall_summary"]["data"][0]
    return {field: overall[field] for field in OVERALL_FIELDS}


def _load_stages(path: Path) -> list[dict[str, Any]]:
    numeric = {
        "qnn_op_instances": int, "num_htp_ops": int, "cycles": int,
        "critical_path_us_estimate": float, "dram_read": int, "dram_write": int,
        "vtcm_read": int, "vtcm_write": int,
    }
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field, cast in numeric.items():
            row[field] = cast(row[field])
    return rows


def _load_placement(artifact_dir: Path, workload: str) -> dict[str, Any]:
    manifest_path = artifact_dir / "manifests" / f"model.0.{workload}_quant_manifest.json"
    mir_path = artifact_dir / "mir" / f"qwen3_layer5_block_{workload}.mir"
    operations = _load_json(manifest_path)["operations"]
    mir = mir_path.read_text(encoding="utf-8")
    qnn_true = mir.count("using_qnn:true")
    qnn_false = mir.count("using_qnn:false")
    return {
        "manifest_path": str(manifest_path), "mir_path": str(mir_path),
        "manifest_operation_count": len(operations),
        "package_counts": dict(sorted(Counter(op["package"] for op in operations).items())),
        "qnn_placed_mir_operations": qnn_true,
        "cpu_placed_mir_operations": qnn_false,
        "fully_qnn_placed": qnn_true > 0 and qnn_false == 0,
        "htp_optrace_captured": True,
    }


def _math_gate(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"available": False, "pass": False}
    payload = _load_json(path)
    results = payload.get("results", [])
    return {
        "available": True, "path": str(path),
        "pass": bool(results) and all(bool(item.get("pass")) for item in results),
        "results": results,
    }


def build_report(
    timing_root: Path,
    optrace_root: Path,
    artifact_root: Path,
    math_verification: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    timings = {
        short: _load_timing(timing_root / variant["name"] / "timing.json")
        for short, variant in VARIANTS.items()
    }
    timing_report: OrderedDict[str, Any] = OrderedDict()
    for workload, timing_name in WORKLOADS.items():
        baseline = float(timings["C"]["results"][timing_name]["median_us"])
        timing_report[workload] = OrderedDict()
        for short in VARIANTS:
            source = timings[short]["results"][timing_name]
            item = {field: source[field] for field in TIMING_FIELDS}
            item["samples"] = len(source.get("samples_us", []))
            item["median_change_percent_vs_c"] = _percent_change(
                float(source["median_us"]), baseline
            )
            timing_report[workload][short] = item

    internal: OrderedDict[str, Any] = OrderedDict()
    stage_rows: list[dict[str, Any]] = []
    placement_complete = True
    for workload in WORKLOADS:
        internal[workload] = OrderedDict()
        loaded_stages: dict[str, list[dict[str, Any]]] = {}
        for short, variant in VARIANTS.items():
            directory = optrace_root / variant["optrace_dir"] / workload
            qhas = _load_qhas(directory / "chrometrace_qnn_htp_analysis_summary.json")
            stages = _load_stages(directory / "structure-qwen3-stage-summary.csv")
            loaded_stages[short] = stages
            placement = _load_placement(artifact_root / variant["artifact_dir"], workload)
            placement_complete = placement_complete and placement["fully_qnn_placed"]
            internal[workload][short] = {
                "variant": variant["name"], "realization": variant["realization"],
                "directory": str(directory), "qhas": qhas, "placement": placement,
                "top_stages_by_critical_path": sorted(
                    ({
                        "stage": row["stage"],
                        "critical_path_us_estimate": row["critical_path_us_estimate"],
                        "qnn_op_instances": row["qnn_op_instances"],
                        "num_htp_ops": row["num_htp_ops"],
                        "kernel_resources": row["kernel_resources"],
                    } for row in stages),
                    key=lambda row: row["critical_path_us_estimate"], reverse=True,
                )[:5],
            }

        baseline_by_stage = {row["stage"]: row for row in loaded_stages["C"]}
        for short, rows in loaded_stages.items():
            for row in rows:
                baseline = baseline_by_stage.get(row["stage"])
                baseline_us = float(baseline["critical_path_us_estimate"]) if baseline else 0.0
                stage_rows.append({
                    "workload": workload, "variant": short, "stage": row["stage"],
                    "critical_path_us_estimate": row["critical_path_us_estimate"],
                    "critical_path_delta_us_vs_c": row["critical_path_us_estimate"] - baseline_us,
                    "qnn_op_instances": row["qnn_op_instances"],
                    "num_htp_ops": row["num_htp_ops"], "cycles": row["cycles"],
                    "dram_read": row["dram_read"], "dram_write": row["dram_write"],
                    "vtcm_read": row["vtcm_read"], "vtcm_write": row["vtcm_write"],
                    "kernel_resources": row["kernel_resources"],
                })

        baseline_qhas = internal[workload]["C"]["qhas"]
        for short in VARIANTS:
            qhas = internal[workload][short]["qhas"]
            qhas["graph_execute_change_percent_vs_c"] = _percent_change(
                float(qhas["graph_execute_us"]), float(baseline_qhas["graph_execute_us"])
            )
            qhas["timeline_cycles_change_percent_vs_c"] = _percent_change(
                float(qhas["timeline_cycles"]), float(baseline_qhas["timeline_cycles"])
            )

    math_gate = _math_gate(math_verification)
    evidence_complete = math_gate["pass"] and placement_complete
    report = {
        "schema_version": 1,
        "scope": "Qwen3 Layer 5 standalone C versus online R3; s1 and s32",
        "comparison_policy": "exploratory_no_performance_pass_fail",
        "canonical_timing_root": str(timing_root), "timing": timing_report,
        "internal_htp_optrace": internal,
        "mathematical_correctness_gate": math_gate,
        "prototype_completion": {
            "status": "complete" if evidence_complete else "incomplete_evidence",
            "all_mir_operations_qnn_placed": placement_complete,
            "all_s1_s32_timings_present": True,
            "all_s1_s32_htp_optraces_present": True,
            "performance_gate_applied": False,
        },
        "notes": [
            "Repeated profiling-disabled median is the latency metric.",
            "Single-capture QHAS and stage attribution are diagnostic only.",
            "R3 graph nodes may be attributed to adjacent Q/K/cache semantic stages.",
        ],
    }
    return report, stage_rows


def _markdown(report: dict[str, Any]) -> str:
    completion = report["prototype_completion"]
    lines = [
        "# Qwen3 Layer 5 R3 exploratory device report", "",
        f"Prototype evidence: **{completion['status']}**. R3 performance gate: **not applied**.",
        "", "## Profiling-disabled repeated timing", "",
        "| Workload | Variant | Median (ms) | P95 (ms) | Mean (ms) | Change vs C |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for workload, variants in report["timing"].items():
        for short, item in variants.items():
            lines.append(
                f"| {workload} | {short} | {item['median_us'] / 1000:.3f} | "
                f"{item['p95_us'] / 1000:.3f} | {item['mean_us'] / 1000:.3f} | "
                f"{item['median_change_percent_vs_c']:+.2f}% |"
            )
    lines += [
        "", "## Internal HTP Optrace (single capture, diagnostic)", "",
        "| Workload | Variant | graphExecute (ms) | Timeline cycles | QNN nodes | HTP ops |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for workload, variants in report["internal_htp_optrace"].items():
        for short, item in variants.items():
            qhas = item["qhas"]
            lines.append(
                f"| {workload} | {short} | {qhas['graph_execute_us'] / 1000:.3f} | "
                f"{qhas['timeline_cycles']} | {qhas['qnn_nodes']} | {qhas['htp_nodes']} |"
            )
    lines += [
        "", "## Backend placement", "",
        "| Workload | Variant | MIR QNN ops | MIR CPU ops | Manifest ops | Fully QNN |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for workload, variants in report["internal_htp_optrace"].items():
        for short, item in variants.items():
            placement = item["placement"]
            lines.append(
                f"| {workload} | {short} | {placement['qnn_placed_mir_operations']} | "
                f"{placement['cpu_placed_mir_operations']} | {placement['manifest_operation_count']} | "
                f"{'yes' if placement['fully_qnn_placed'] else 'no'} |"
            )
    lines += ["", "No R3 latency pass/fail threshold is applied.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-root", type=Path, required=True)
    parser.add_argument("--optrace-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--math-verification", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    report, stages = build_report(
        args.timing_root, args.optrace_root, args.artifact_root, args.math_verification
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{args.output_prefix}.json")
    markdown_path = Path(f"{args.output_prefix}.md")
    stage_path = Path(f"{args.output_prefix}_stages.csv")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    with stage_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(stages[0]))
        writer.writeheader()
        writer.writerows(stages)
    print(_markdown(report))
    print(f"generated: {json_path}, {markdown_path}, {stage_path}")


if __name__ == "__main__":
    main()
