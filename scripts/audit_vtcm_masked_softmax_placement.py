#!/usr/bin/env python3

"""Audit the runtime Optrace contract for the VTCM Softmax placement gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OP_TYPE = "LLaMAPackage::VtcmMaskedSoftmaxPlacement"


def scalar(node: dict[str, Any], name: str) -> int:
    encoded = node["scalar_params"][name]
    if len(encoded) != 1:
        raise ValueError(f"{name} has an unexpected encoding: {encoded}")
    return int(next(iter(encoded.values())))


def audit(path: Path, expected_count: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes: dict[str, dict[str, Any]] = payload["graph"]["nodes"]
    custom = {node_id: node for node_id, node in nodes.items() if node.get("type") == OP_TYPE}
    custom_outputs = set(custom)
    custom_inputs = {name for node in custom.values() for name in node.get("input_names", [])}

    consumers = {
        output: [
            {"id": node_id, "type": node.get("type", ""), "grouping": node.get("grouping", "")}
            for node_id, node in nodes.items()
            if output in node.get("input_names", [])
        ]
        for output in custom_outputs
    }

    boundary_conversions = []
    for node_id, node in nodes.items():
        label = f"{node.get('type', '')} {node.get('grouping', '')}".lower()
        if "to_vtcm" not in label and "from_vtcm" not in label:
            continue
        inputs = set(node.get("input_names", []))
        outputs = set(node.get("output_names", []))
        if inputs & custom_outputs or outputs & custom_inputs:
            boundary_conversions.append(
                {"id": node_id, "type": node.get("type", ""), "grouping": node.get("grouping", "")}
            )

    totals = {
        name: sum(scalar(node, name) for node in custom.values())
        for name in (
            "mem_dram_read",
            "mem_dram_write",
            "mem_vtcm_read",
            "mem_vtcm_write",
            "cycles_duration",
            "cycles_dominant",
        )
    }
    consumer_types = sorted({consumer["type"] for items in consumers.values() for consumer in items})
    hvx_only = all("uses_hvx" in str(node["scalar_params"].get("op_flags", {})) for node in custom.values())
    every_output_consumed = all(consumers.values())
    passed = (
        len(custom) == expected_count
        and totals["mem_dram_read"] == 0
        and totals["mem_dram_write"] == 0
        and not boundary_conversions
        and every_output_consumed
        and hvx_only
    )
    return {
        "input": str(path),
        "op_type": OP_TYPE,
        "expected_count": expected_count,
        "observed_count": len(custom),
        "totals": totals,
        "consumer_types": consumer_types,
        "every_output_consumed": every_output_consumed,
        "boundary_vtcm_conversions": boundary_conversions,
        "all_custom_ops_use_hvx": hvx_only,
        "gate_passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("htp_json", type=Path)
    parser.add_argument("--expected-count", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit(args.htp_json, args.expected_count)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
