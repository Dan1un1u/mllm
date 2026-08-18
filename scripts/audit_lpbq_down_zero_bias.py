#!/usr/bin/env python3
"""Audit the single-variable contract of the LPBQ down-projection bias test."""

import argparse
import json
from pathlib import Path


OP_NAME = "model.layers.14.mlp.down_proj"


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def tensor_map(manifest: dict) -> dict[str, dict]:
    return {tensor["name"]: tensor for tensor in manifest["tensors"]}


def operation(manifest: dict) -> dict:
    matches = [op for op in manifest["operations"] if op["name"] == OP_NAME]
    assert len(matches) == 1, f"expected one {OP_NAME}, got {len(matches)}"
    return matches[0]


def role_signature(tensor: dict) -> dict:
    return {
        "ir_storage_dtype": tensor["ir_storage_dtype"],
        "logical_quant_dtype": tensor["logical_quant_dtype"],
        "qnn_dtype": tensor["qnn_dtype"],
        "tensor_type": tensor["tensor_type"],
        "dimensions": tensor["dimensions"],
        "quant_recipe": tensor["quant_recipe"],
        "qnn_quantization": tensor["qnn_quantization"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("omitted", type=Path)
    parser.add_argument("explicit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    omitted = load(args.omitted)
    explicit = load(args.explicit)
    omitted_op = operation(omitted)
    explicit_op = operation(explicit)
    omitted_tensors = tensor_map(omitted)
    explicit_tensors = tensor_map(explicit)

    assert omitted_op["package"] == explicit_op["package"] == "qti.aisw"
    assert omitted_op["qnn_op_type"] == explicit_op["qnn_op_type"] == "Conv2d"
    assert len(omitted_op["inputs"]) == 2
    assert len(explicit_op["inputs"]) == 3
    assert len(omitted_op["outputs"]) == len(explicit_op["outputs"]) == 1

    for role, omitted_name, explicit_name in (
        ("activation", omitted_op["inputs"][0], explicit_op["inputs"][0]),
        ("weight", omitted_op["inputs"][1], explicit_op["inputs"][1]),
        ("output", omitted_op["outputs"][0], explicit_op["outputs"][0]),
    ):
        assert role_signature(omitted_tensors[omitted_name]) == role_signature(
            explicit_tensors[explicit_name]
        ), f"{role} contract changed"

    bias = explicit_tensors[explicit_op["inputs"][2]]
    recipe = bias["quant_recipe"]
    assert bias["dimensions"] == [2048], bias
    assert bias["qnn_dtype"] == "UFIXED_POINT_8", bias
    assert bias["tensor_type"] == "STATIC", bias
    assert recipe["quant_to_dtype"] == "UInt8"
    assert recipe["quant_min"] == 0 and recipe["quant_max"] == 255
    assert bias["qnn_quantization"]["zero_point"] == 0

    output = explicit_tensors[explicit_op["outputs"][0]]
    assert bias["qnn_quantization"]["scale"] == output["qnn_quantization"]["scale"]

    report = {
        "status": "pass",
        "operation": OP_NAME,
        "only_logical_delta": "explicit static U8 zero bias input",
        "omitted_input_count": 2,
        "explicit_input_count": 3,
        "bias": role_signature(bias),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
