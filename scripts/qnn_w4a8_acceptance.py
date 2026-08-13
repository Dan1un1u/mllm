#!/usr/bin/env python3
"""Fail closed unless manifest + s1/s32 Optrace prove the W4A8G32 contract."""

import argparse
import json
from pathlib import Path

from qnn_optrace_quantization import load_manifests, load_runtime_events


def _check_u8_tensor(tensor, label, failures):
    if tensor.get("logical_quant_dtype") != "UInt8":
        failures.append(f"{label}: logical dtype is not UInt8")
    if tensor.get("qnn_dtype") != "UFIXED_POINT_8":
        failures.append(f"{label}: QNN dtype is not UFIXED_POINT_8")
    quant = tensor.get("quant_recipe", {})
    if quant.get("type") != "asymmetric_per_tensor" or quant.get("quant_min") != 0 or quant.get("quant_max") != 255:
        failures.append(f"{label}: is not asymmetric 0..255")


def audit_graph(manifest_path, trace_path, qhas_path, rmsnorm_u8=False):
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
    rmsnorms = [operation for operation in operations.values() if operation.get("qnn_op_type") == "RmsNorm"]
    if len(rmsnorms) != 729:
        failures.append(f"expected 729 qti.aisw RmsNorm operations, found {len(rmsnorms)}")
    rmsnorm_bridges = {
        operation["name"]
        for operation in operations.values()
        if operation.get("qnn_op_type") == "Convert"
        and (operation["name"].endswith(".a8_to_a16") or operation["name"].endswith(".a16_to_a8"))
    }
    if rmsnorm_u8:
        if rmsnorm_bridges:
            failures.append(f"expected 0 explicit RMSNorm bridge conversions, found {len(rmsnorm_bridges)}")
        rmsnorm_names = {operation["name"] for operation in rmsnorms}
        traced_rmsnorm_names = rmsnorm_names & traced_ops
        missing_rmsnorms = sorted(rmsnorm_names - traced_ops)
        if missing_rmsnorms:
            failures.append(
                "RmsNorm operations missing from HTP Optrace: "
                + ", ".join(missing_rmsnorms[:20])
                + (" ..." if len(missing_rmsnorms) > 20 else "")
            )
        rmsnorm_physical_u16 = sorted(
            {
                event["qnn_name"]
                for event in events
                if event["qnn_name"] in rmsnorm_names
                if "QUInt16" in event["data_type"] or "QUInt16" in event["input_data_types"]
            }
        )
        if rmsnorm_physical_u16:
            failures.append(
                "post-lowering HTP trace contains QUInt16 on RMSNorm: "
                + ", ".join(rmsnorm_physical_u16[:20])
            )
        for operation in rmsnorms:
            inputs = operation.get("inputs", [])
            outputs = operation.get("outputs", [])
            if len(inputs) != 3 or len(outputs) != 1:
                failures.append(f"{operation['name']}: unexpected RmsNorm arity")
                continue
            for role, tensor_name in (("input", inputs[0]), ("gamma", inputs[1]), ("bias", inputs[2]), ("output", outputs[0])):
                tensor = tensor_lookup.get((operation["graph"], tensor_name), {})
                _check_u8_tensor(tensor, f"{operation['name']}: {role}", failures)
    return {
        "graph": graphs[0] if graphs else manifest_path.stem,
        "target_operation_count": len(targets),
        "traced_target_count": sum(name in traced_ops for name in targets),
        "physical_u16_operation_count": len(physical_u16),
        "rmsnorm_operation_count": len(rmsnorms),
        "rmsnorm_bridge_count": len(rmsnorm_bridges),
        "rmsnorm_traced_count": len(traced_rmsnorm_names) if rmsnorm_u8 else sum(
            operation["name"] in traced_ops for operation in rmsnorms
        ),
        "rmsnorm_physical_u16_count": len(rmsnorm_physical_u16) if rmsnorm_u8 else 0,
        "rmsnorm_u8_contract": rmsnorm_u8,
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
    parser.add_argument(
        "--rmsnorm-u8",
        action="store_true",
        help="Require all qti.aisw RmsNorm operands to be asymmetric UInt8 with zero explicit bridges.",
    )
    args = parser.parse_args()
    reports = [
        audit_graph(args.s1_manifest, args.s1_trace, args.s1_qhas, rmsnorm_u8=args.rmsnorm_u8),
        audit_graph(args.s32_manifest, args.s32_trace, args.s32_qhas, rmsnorm_u8=args.rmsnorm_u8),
    ]
    failures = [failure for report in reports for failure in report["failures"]]
    document = {"status": "pass" if not failures else "fail", "graphs": reports}
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
