#!/usr/bin/env python3
"""Audit one standalone K/V-head projection quantization manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--projection", choices=("k_proj", "v_proj"), required=True)
    parser.add_argument("--seq", type=int, choices=(1, 32, 64), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_graph = f"model.0.s{args.seq}"
    if payload.get("graph") != expected_graph:
        raise AssertionError(f"graph mismatch: {payload.get('graph')} != {expected_graph}")
    operations = payload["operations"]
    expected_name = f"model.layers.14.self_attn.{args.projection}.0"
    matches = [op for op in operations if op["name"] == expected_name]
    if len(matches) != 1:
        raise AssertionError(f"expected one {expected_name}, got {len(matches)}")
    op = matches[0]
    if (op["package"], op["qnn_op_type"]) != ("qti.aisw", "Conv2d"):
        raise AssertionError(f"unexpected projection lowering: {op}")
    if len(op["inputs"]) != 2 or len(op["outputs"]) != 1:
        raise AssertionError(f"unexpected projection arity: {op}")

    tensors = {str(tensor["name"]): tensor for tensor in payload["tensors"]}
    input_tensor = tensors[str(op["inputs"][0])]
    weight_tensor = tensors[str(op["inputs"][1])]
    output_tensor = tensors[str(op["outputs"][0])]
    expected_input = [1, 1, args.seq, 2048]
    expected_weight = [1, 1, 2048, 128]
    expected_output = [1, 1, args.seq, 128]
    if input_tensor["dimensions"] != expected_input:
        raise AssertionError(f"input shape mismatch: {input_tensor['dimensions']}")
    if weight_tensor["dimensions"] != expected_weight:
        raise AssertionError(f"weight shape mismatch: {weight_tensor['dimensions']}")
    if output_tensor["dimensions"] != expected_output:
        raise AssertionError(f"output shape mismatch: {output_tensor['dimensions']}")
    for side, tensor in (("input", input_tensor), ("output", output_tensor)):
        if tensor["qnn_dtype"] != "UFIXED_POINT_8" or tensor["logical_quant_dtype"] != "UInt8":
            raise AssertionError(f"{side} is not native U8: {tensor}")
        quantization = tensor["qnn_quantization"]
        if not quantization.get("defined") or quantization.get("encoding") != "scale_offset":
            raise AssertionError(f"{side} qparam is not solved: {quantization}")

    recipe = weight_tensor["quant_recipe"]
    quantization = weight_tensor["qnn_quantization"]
    expected_recipe = {
        "type": "lpbq",
        "block_size": 32,
        "channel_axis": 3,
        "quant_min": -7,
        "quant_max": 7,
        "quant_to_dtype": "Int4",
    }
    for key, value in expected_recipe.items():
        if recipe.get(key) != value:
            raise AssertionError(f"weight recipe {key}: {recipe.get(key)} != {value}")
    expected_quantization = {
        "encoding": "blockwise_expansion",
        "axis": 3,
        "axis_size": 128,
        "num_blocks_per_axis": 64,
        "block_scale_count": 8192,
        "block_scale_bitwidth": 4,
    }
    for key, value in expected_quantization.items():
        if quantization.get(key) != value:
            raise AssertionError(
                f"weight quantization {key}: {quantization.get(key)} != {value}"
            )
    converts = [op["name"] for op in operations if op["qnn_op_type"] == "Convert"]
    if converts:
        raise AssertionError(f"standalone projection unexpectedly contains Convert: {converts}")

    report = {
        "graph": expected_graph,
        "projection": expected_name,
        "input": {
            "dimensions": input_tensor["dimensions"],
            "qnn_dtype": input_tensor["qnn_dtype"],
            "qnn_quantization": input_tensor["qnn_quantization"],
        },
        "weight": {
            "dimensions": weight_tensor["dimensions"],
            "logical_quant_dtype": weight_tensor["logical_quant_dtype"],
            "qnn_quantization": quantization,
            "quant_recipe": recipe,
        },
        "output": {
            "dimensions": output_tensor["dimensions"],
            "qnn_dtype": output_tensor["qnn_dtype"],
            "qnn_quantization": output_tensor["qnn_quantization"],
        },
        "convert_count": 0,
        "passed": True,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
