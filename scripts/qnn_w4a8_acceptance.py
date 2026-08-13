#!/usr/bin/env python3
"""Fail closed unless manifest + s1/s32 Optrace prove the W4A8G32 contract."""

import argparse
import json
from pathlib import Path

from qnn_optrace_quantization import load_manifests, load_runtime_events


def audit_graph(manifest_path, trace_path, qhas_path):
    graphs, tensors, operations = load_manifests([manifest_path])
    events = load_runtime_events(trace_path, qhas_path)
    traced_ops = {event["qnn_name"] for event in events}
    tensor_lookup = {(tensor["graph"], tensor["name"]): tensor for tensor in tensors.values()}
    targets = []
    failures = []
    for operation in operations.values():
        if operation.get("qnn_op_type") != "Conv2d":
            continue
        graph = operation["graph"]
        inputs = operation.get("inputs", [])
        outputs = operation.get("outputs", [])
        if len(inputs) < 2 or not outputs:
            continue
        activation = tensor_lookup.get((graph, inputs[0]), {})
        weight = tensor_lookup.get((graph, inputs[1]), {})
        output = tensor_lookup.get((graph, outputs[0]), {})
        recipe = weight.get("quant_recipe", {})
        # The pre-finalize manifest calls these logical operations Conv2d;
        # QAIRT is free to rename the physical LPBQ kernel during lowering.
        # Weight recipe identity is therefore the stable W4G32 selector.
        if recipe.get("type") != "lpbq":
            continue
        targets.append(operation["name"])
        if operation["name"] not in traced_ops:
            failures.append(f"{operation['name']}: not observed in HTP Optrace")
        for role, tensor in (("activation", activation), ("output", output)):
            quant = tensor.get("quant_recipe", {})
            if tensor.get("logical_quant_dtype") != "UInt8":
                failures.append(f"{operation['name']}: {role} logical dtype is not UInt8")
            if tensor.get("qnn_dtype") != "UFIXED_POINT_8":
                failures.append(f"{operation['name']}: {role} QNN dtype is not UFIXED_POINT_8")
            if quant.get("type") != "asymmetric_per_tensor" or quant.get("quant_max") != 255:
                failures.append(f"{operation['name']}: {role} is not asymmetric 0..255")
        if recipe.get("type") != "lpbq" or recipe.get("block_size") != 32:
            failures.append(
                f"{operation['name']}: weight is not LPBQ W4G32 "
                f"(type={recipe.get('type')}, block={recipe.get('block_size')})"
            )
    if len(targets) != 1009:
        failures.append(f"expected 1009 SHA LPBQ targets, found {len(targets)}")
    target_set = set(targets)
    physical_u16 = sorted(
        {
            event["qnn_name"]
            for event in events
            if event["qnn_name"] in target_set
            if "QUInt16" in event["data_type"] or "QUInt16" in event["input_data_types"]
        }
    )
    if physical_u16:
        failures.append(
            "post-lowering HTP trace contains QUInt16 on target graph: "
            + ", ".join(physical_u16[:20])
        )
    return {
        "graph": graphs[0] if graphs else manifest_path.stem,
        "target_operation_count": len(targets),
        "traced_target_count": sum(name in traced_ops for name in targets),
        "physical_u16_operation_count": len(physical_u16),
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-manifest", type=Path, required=True)
    parser.add_argument("--s1-trace", type=Path, required=True)
    parser.add_argument("--s1-qhas", type=Path, required=True)
    parser.add_argument("--s32-manifest", type=Path, required=True)
    parser.add_argument("--s32-trace", type=Path, required=True)
    parser.add_argument("--s32-qhas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [
        audit_graph(args.s1_manifest, args.s1_trace, args.s1_qhas),
        audit_graph(args.s32_manifest, args.s32_trace, args.s32_qhas),
    ]
    failures = [failure for report in reports for failure in report["failures"]]
    document = {"status": "pass" if not failures else "fail", "graphs": reports}
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
