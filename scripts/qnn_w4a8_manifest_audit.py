#!/usr/bin/env python3
"""Fail closed unless pre-finalize manifests prove W4A8G32 target linears."""

import argparse
import json
from pathlib import Path


def _check_u8_tensor(tensor: dict, label: str, failures: list[str]) -> None:
    if tensor.get("logical_quant_dtype") != "UInt8":
        failures.append(f"{label}: logical dtype is not UInt8")
    if tensor.get("qnn_dtype") != "UFIXED_POINT_8":
        failures.append(f"{label}: QNN dtype is not UFIXED_POINT_8")
    quant = tensor.get("quant_recipe", {})
    if quant.get("type") != "asymmetric_per_tensor" or quant.get("quant_min") != 0 or quant.get("quant_max") != 255:
        failures.append(f"{label}: is not asymmetric 0..255")


def audit(path: Path, rmsnorm_u8: bool = False) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    tensors = {tensor["name"]: tensor for tensor in document["tensors"]}
    targets = []
    failures = []
    for operation in document["operations"]:
        if operation.get("qnn_op_type") != "Conv2d":
            continue
        if len(operation.get("inputs", [])) < 2 or not operation.get("outputs"):
            continue
        activation = tensors[operation["inputs"][0]]
        weight = tensors[operation["inputs"][1]]
        output = tensors[operation["outputs"][0]]
        recipe = weight.get("quant_recipe", {})
        if recipe.get("type") != "lpbq":
            continue
        name = operation["name"]
        targets.append(name)
        for role, tensor in (("activation", activation), ("output", output)):
            if tensor.get("logical_quant_dtype") != "UInt8":
                failures.append(f"{name}: {role} logical dtype is not UInt8")
            if tensor.get("qnn_dtype") != "UFIXED_POINT_8":
                failures.append(f"{name}: {role} QNN dtype is not UFIXED_POINT_8")
            quant = tensor.get("quant_recipe", {})
            if quant.get("type") != "asymmetric_per_tensor" or quant.get("quant_max") != 255:
                failures.append(f"{name}: {role} is not asymmetric 0..255")
        if recipe.get("quant_to_dtype") != "Int4" or recipe.get("block_size") != 32:
            failures.append(f"{name}: weight is not LPBQ Int4 G32")
        encoding = weight.get("qnn_quantization", {})
        expected_blocks = weight.get("dimensions", [-1, -1, -1])[2] // 32
        if (
            encoding.get("encoding") != "blockwise_expansion"
            or encoding.get("num_blocks_per_axis") != expected_blocks
        ):
            failures.append(f"{name}: physical LPBQ block expansion is incomplete")

    if len(targets) != 1009:
        failures.append(f"expected 1009 SHA Conv2d LPBQ targets, found {len(targets)}")
    bridges = [
        op["name"]
        for op in document["operations"]
        if op.get("qnn_op_type") == "Convert"
        and (op["name"].endswith(".a8_to_a16") or op["name"].endswith(".a16_to_a8"))
    ]
    rmsnorms = [op for op in document["operations"] if op.get("qnn_op_type") == "RmsNorm"]
    if len(rmsnorms) != 729:
        failures.append(f"expected 729 qti.aisw RmsNorm operations, found {len(rmsnorms)}")
    if rmsnorm_u8:
        if bridges:
            failures.append(f"expected 0 explicit RMSNorm bridge conversions, found {len(bridges)}")
        for operation in rmsnorms:
            inputs = operation.get("inputs", [])
            outputs = operation.get("outputs", [])
            if len(inputs) != 3 or len(outputs) != 1:
                failures.append(f"{operation['name']}: unexpected RmsNorm arity")
                continue
            for role, tensor_id in (("input", inputs[0]), ("gamma", inputs[1]), ("bias", inputs[2]), ("output", outputs[0])):
                tensor = tensors.get(tensor_id)
                if tensor is None:
                    failures.append(f"{operation['name']}: missing {role} tensor {tensor_id}")
                else:
                    _check_u8_tensor(tensor, f"{operation['name']}: {role}", failures)
    elif len(bridges) != 1458:
        failures.append(f"expected 1458 explicit RMSNorm bridge conversions, found {len(bridges)}")
    return {
        "graph": document["graph"],
        "target_operation_count": len(targets),
        "rmsnorm_operation_count": len(rmsnorms),
        "explicit_rmsnorm_bridge_count": len(bridges),
        "rmsnorm_u8_contract": rmsnorm_u8,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--rmsnorm-u8",
        action="store_true",
        help="Require all qti.aisw RmsNorm operands to be asymmetric UInt8 with zero explicit bridges.",
    )
    args = parser.parse_args()
    reports = [audit(path, rmsnorm_u8=args.rmsnorm_u8) for path in args.manifest]
    failures = [failure for report in reports for failure in report["failures"]]
    result = {"status": "pass" if not failures else "fail", "graphs": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
