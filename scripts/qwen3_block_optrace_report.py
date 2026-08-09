#!/usr/bin/env python3
"""Compare standalone Qwen3 block A/B/C HTP Optrace captures.

The timing acceptance decision comes from profiling-disabled repeated runs.
Optrace is a single-capture structural/bottleneck diagnostic and is not used as
the latency acceptance metric.
"""

import argparse
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path


VARIANTS = OrderedDict((
    ("A", "A_original"),
    ("B", "B_identity_fold"),
    ("C", "C_hadamard_r1_r2"),
))
WORKLOADS = OrderedDict((("s1", "s1 decode"), ("s32", "s32 prefill/chunk")))
GROUPS = OrderedDict((
    ("runtime_io", {"graph_input", "graph_output_runtime"}),
    ("norms", {"input_rmsnorm", "q_head_rmsnorm", "k_head_rmsnorm", "post_attention_rmsnorm"}),
    ("attention_qkv_projection", {"q_projection", "k_projection", "v_projection"}),
    ("qk_rope_cache_softmax", {"q_rope", "k_rope", "kv_cache_update", "qk_similarity", "scale_mask_softmax"}),
    ("attention_value_head_merge", {"attention_value", "head_merge"}),
    ("attention_o_projection", {"o_projection"}),
    ("residuals", {"attention_residual", "mlp_residual"}),
    ("mlp", {"mlp_gate_projection", "mlp_up_projection", "mlp_silu", "mlp_gate_product", "mlp_down_projection"}),
))
INTEGER_FIELDS = {
    "qnn_op_instances", "num_htp_ops", "cycles", "num_dominant_path_cycles_htp_0",
    "dram_read", "dram_write", "vtcm_read", "vtcm_write",
}
FORBIDDEN_ROTATION = re.compile(r"(?:hadamard|rotation|rotate|runtime_r[12])", re.IGNORECASE)


def load_capture(root, variant, workload):
    directory = root / variant / workload
    qhas_path = directory / "chrometrace_qnn_htp_analysis_summary.json"
    stages_path = directory / "structure-qwen3-stage-summary.csv"
    operators_path = directory / "structure-qwen3-operator-structure.csv"
    with qhas_path.open(encoding="utf-8") as stream:
        qhas = json.load(stream)
    overall = qhas["data"]["htp_overall_summary"]["data"][0]
    with stages_path.open(newline="", encoding="utf-8") as stream:
        stages = list(csv.DictReader(stream))
    with operators_path.open(newline="", encoding="utf-8") as stream:
        operators = list(csv.DictReader(stream))
    return directory, overall, stages, operators


def group_stages(rows):
    by_name = {row["stage"]: row for row in rows}
    grouped = []
    assigned = set()
    for group, stage_names in GROUPS.items():
        selected = [by_name[name] for name in stage_names if name in by_name]
        assigned.update(row["stage"] for row in selected)
        result = {"group": group, "stages": sorted(row["stage"] for row in selected)}
        for field in INTEGER_FIELDS:
            result[field] = sum(int(row[field]) for row in selected)
        result["critical_path_us_estimate"] = sum(float(row["critical_path_us_estimate"]) for row in selected)
        result["kernel_resources"] = sorted({item for row in selected for item in row["kernel_resources"].split("+") if item})
        grouped.append(result)
    unassigned = sorted(set(by_name) - assigned)
    if unassigned:
        raise ValueError(f"Unassigned block stages: {', '.join(unassigned)}")
    return grouped


def operator_signature(rows):
    fields = (
        "qnn_op", "qnn_op_type", "stage", "kernel_resources", "num_htp_ops",
        "dram_read", "dram_write", "vtcm_read", "vtcm_write",
    )
    return sorted(tuple(row[field] for field in fields) for row in rows)


def fmt(value, digits=3):
    return f"{value:.{digits}f}"


def build_report(root, timing_report=None):
    captures = OrderedDict()
    group_rows = []
    for workload in WORKLOADS:
        captures[workload] = OrderedDict()
        for short, variant in VARIANTS.items():
            directory, overall, stages, operators = load_capture(root, short, workload)
            forbidden = sorted({row["qnn_op"] for row in operators if FORBIDDEN_ROTATION.search(row["qnn_op"])})
            groups = group_stages(stages)
            for row in groups:
                group_rows.append({"workload": workload, "variant": short, **row})
            captures[workload][short] = {
                "variant": variant,
                "directory": str(directory),
                "graph_execute_us_single_optrace_capture": overall["graph_execute_us"],
                "timeline_cycles": overall["timeline_cycles"],
                "qnn_nodes": overall["qnn_nodes"],
                "htp_nodes": overall["htp_nodes"],
                "dram_read": overall["total_dram_read"],
                "dram_write": overall["total_dram_write"],
                "vtcm_read": overall["total_vtcm_read"],
                "vtcm_write": overall["total_vtcm_write"],
                "peak_vtcm_alloc": overall["peak_vtcm_alloc"],
                "groups": groups,
                "forbidden_rotation_ops": forbidden,
                "_operator_signature": operator_signature(operators),
            }

    contracts = OrderedDict()
    for workload, variants in captures.items():
        b, c = variants["B"], variants["C"]
        checks = OrderedDict((
            ("qnn_nodes_equal", b["qnn_nodes"] == c["qnn_nodes"]),
            ("htp_nodes_equal", b["htp_nodes"] == c["htp_nodes"]),
            ("dram_traffic_equal", (b["dram_read"], b["dram_write"]) == (c["dram_read"], c["dram_write"])),
            ("vtcm_traffic_equal", (b["vtcm_read"], b["vtcm_write"]) == (c["vtcm_read"], c["vtcm_write"])),
            ("peak_vtcm_equal", b["peak_vtcm_alloc"] == c["peak_vtcm_alloc"]),
            ("operator_contract_equal", b["_operator_signature"] == c["_operator_signature"]),
            ("no_runtime_rotation_ops", not b["forbidden_rotation_ops"] and not c["forbidden_rotation_ops"]),
        ))
        contracts[workload] = {"checks": checks, "pass": all(checks.values())}

    timing = None
    timing_pass = True
    if timing_report is not None:
        with timing_report.open(encoding="utf-8") as stream:
            source = json.load(stream)
        timing = {
            "path": str(timing_report),
            "overall_status": source["overall_status"],
            "comparisons": source["comparisons"],
        }
        timing_pass = source["overall_status"] == "pass"

    for variants in captures.values():
        for capture in variants.values():
            capture.pop("_operator_signature")
    return {
        "schema_version": 1,
        "scope": "Qwen3 Layer 5 standalone block; rotated-basis I/O; s1 decode and s32 prefill/chunk",
        "timing_acceptance": timing,
        "optrace_note": "Single-capture graphExecute is diagnostic only; pass/fail uses profiling-disabled repeated timing.",
        "captures": captures,
        "b_vs_c_static_contract": contracts,
        "overall_status": "pass" if timing_pass and all(item["pass"] for item in contracts.values()) else "fail",
    }, group_rows


def write_group_csv(path, rows):
    fields = (
        "workload", "variant", "group", "stages", "qnn_op_instances", "num_htp_ops",
        "cycles", "num_dominant_path_cycles_htp_0", "critical_path_us_estimate",
        "dram_read", "dram_write", "vtcm_read", "vtcm_write", "kernel_resources",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            row["stages"] = "+".join(row["stages"])
            row["kernel_resources"] = "+".join(row["kernel_resources"])
            writer.writerow({field: row[field] for field in fields})


def render_markdown(report):
    lines = [
        "# Qwen3 Layer 5 offline R1/R2 prototype report",
        "",
        f"Overall: **{report['overall_status'].upper()}**",
        "",
    ]
    timing = report["timing_acceptance"]
    if timing:
        lines += ["## Profiling-disabled repeated timing", "", "| Workload | A median | B vs A | C vs A |", "|---|---:|---:|---:|"]
        for workload, rows in timing["comparisons"].items():
            a, b, c = rows["A_original"], rows["B_identity_fold"], rows["C_hadamard_r1_r2"]
            lines.append(
                f"| {workload} | {a['median_us'] / 1000:.3f} ms | {b['regression_percent_vs_A']:+.2f}% | {c['regression_percent_vs_A']:+.2f}% |"
            )
        lines += [""]

    lines += [
        "## HTP Optrace overall (single capture, diagnostic)", "",
        "| Workload | Variant | graphExecute | QNN nodes | HTP nodes | DRAM R/W | VTCM R/W |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for workload, variants in report["captures"].items():
        for short, row in variants.items():
            lines.append(
                f"| {WORKLOADS[workload]} | {short} | {row['graph_execute_us_single_optrace_capture'] / 1000:.3f} ms | "
                f"{row['qnn_nodes']} | {row['htp_nodes']} | {row['dram_read']}/{row['dram_write']} | "
                f"{row['vtcm_read']}/{row['vtcm_write']} |"
            )
    lines += ["", "## B vs C static execution contract", "", "| Workload | Contract | Result |", "|---|---|---:|"]
    for workload, contract in report["b_vs_c_static_contract"].items():
        for name, passed in contract["checks"].items():
            lines.append(f"| {WORKLOADS[workload]} | {name} | {'PASS' if passed else 'FAIL'} |")

    lines += ["", "## Grouped block breakdown", ""]
    for workload, variants in report["captures"].items():
        lines += [f"### {WORKLOADS[workload]}", "", "Critical-path attribution from each single capture (us):", "", "| Group | A | B | C |", "|---|---:|---:|---:|"]
        by_variant = {short: {row["group"]: row for row in capture["groups"]} for short, capture in variants.items()}
        for group in GROUPS:
            values = [by_variant[short][group]["critical_path_us_estimate"] for short in VARIANTS]
            lines.append(f"| {group} | {fmt(values[0], 1)} | {fmt(values[1], 1)} | {fmt(values[2], 1)} |")
        lines.append("")
    lines += [
        "Optrace serialization inflates runner host wall time. The table uses QHAS graphExecute/critical-path data; latency acceptance remains the profiling-disabled 200-run median.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optrace-root", type=Path, required=True)
    parser.add_argument("--timing-report", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    report, groups = build_report(args.optrace_root, args.timing_report)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{args.output_prefix}.json")
    markdown_path = Path(f"{args.output_prefix}.md")
    csv_path = Path(f"{args.output_prefix}_groups.csv")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    write_group_csv(csv_path, groups)
    print(f"overall={report['overall_status']}")
    print(f"generated: {json_path}, {markdown_path}, {csv_path}")


if __name__ == "__main__":
    main()
