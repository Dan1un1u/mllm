#!/usr/bin/env python3
"""Audit the strict split-vs-packed K/V LPBQ graph contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _native_u8(tensor: dict[str, object], label: str) -> None:
    if tensor["qnn_dtype"] != "UFIXED_POINT_8" or tensor["logical_quant_dtype"] != "UInt8":
        raise AssertionError(f"{label} is not native U8: {tensor}")
    quantization = tensor["qnn_quantization"]
    if not quantization.get("defined") or quantization.get("encoding") != "scale_offset":
        raise AssertionError(f"{label} qparam is not solved: {quantization}")


def _lpbq_weight(tensor: dict[str, object], channels: int) -> None:
    if tensor["dimensions"] != [1, 1, 2048, channels]:
        raise AssertionError(f"weight shape mismatch: {tensor['dimensions']}")
    recipe = tensor["quant_recipe"]
    quantization = tensor["qnn_quantization"]
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
        "axis_size": channels,
        "num_blocks_per_axis": 64,
        "block_scale_count": channels * 64,
        "block_scale_bitwidth": 4,
    }
    for key, value in expected_quantization.items():
        if quantization.get(key) != value:
            raise AssertionError(
                f"weight quantization {key}: {quantization.get(key)} != {value}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--projection", choices=("k_proj", "v_proj"), required=True)
    parser.add_argument("--mode", choices=("split", "packed"), required=True)
    parser.add_argument("--seq", type=int, choices=(32, 64), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_graph = f"model.0.s{args.seq}"
    if payload.get("graph") != expected_graph:
        raise AssertionError(f"graph mismatch: {payload.get('graph')} != {expected_graph}")
    operations = payload["operations"]
    tensors = {str(tensor["name"]): tensor for tensor in payload["tensors"]}
    prefix = f"model.layers.14.self_attn.{args.projection}."
    projection_ops = [
        op for op in operations
        if op["name"].startswith(prefix) and op["qnn_op_type"] == "Conv2d"
    ]
    expected_count = 8 if args.mode == "split" else 1
    if len(projection_ops) != expected_count:
        raise AssertionError(f"projection count: {len(projection_ops)} != {expected_count}")
    expected_names = (
        {prefix + str(head) for head in range(8)}
        if args.mode == "split" else {prefix + "packed"}
    )
    if {op["name"] for op in projection_ops} != expected_names:
        raise AssertionError(f"projection names mismatch: {[op['name'] for op in projection_ops]}")

    channels = 128 if args.mode == "split" else 1024
    for op in projection_ops:
        if (op["package"], op["qnn_op_type"]) != ("qti.aisw", "Conv2d"):
            raise AssertionError(f"unexpected lowering: {op}")
        if len(op["inputs"]) != 2 or len(op["outputs"]) != 1:
            raise AssertionError(f"unexpected projection arity: {op}")
        input_tensor = tensors[str(op["inputs"][0])]
        weight_tensor = tensors[str(op["inputs"][1])]
        output_tensor = tensors[str(op["outputs"][0])]
        if input_tensor["dimensions"] != [1, 1, args.seq, 2048]:
            raise AssertionError(f"input shape mismatch: {input_tensor['dimensions']}")
        if output_tensor["dimensions"] != [1, 1, args.seq, channels]:
            raise AssertionError(f"projection output shape mismatch: {output_tensor['dimensions']}")
        _native_u8(input_tensor, "projection input")
        _native_u8(output_tensor, "projection output")
        _lpbq_weight(weight_tensor, channels)

    graph_outputs = [tensor for tensor in payload["tensors"] if tensor["tensor_type"] == "APP_READ"]
    if len(graph_outputs) != 8:
        raise AssertionError(f"graph output count: {len(graph_outputs)} != 8")
    for tensor in graph_outputs:
        if tensor["dimensions"] != [1, args.seq, 128]:
            raise AssertionError(f"head output shape mismatch: {tensor['dimensions']}")
        _native_u8(tensor, "head output")

    converts = [op["name"] for op in operations if op["qnn_op_type"] == "Convert"]
    if converts:
        raise AssertionError(f"graph unexpectedly contains Convert: {converts}")
    slicing_ops = [
        {"name": op["name"], "type": op["qnn_op_type"]}
        for op in operations
        if "slice" in op["qnn_op_type"].lower() or "split" in op["qnn_op_type"].lower()
    ]
    if args.mode == "packed" and len(slicing_ops) < 8:
        raise AssertionError(f"packed graph does not expose eight slice operations: {slicing_ops}")
    if args.mode == "split" and slicing_ops:
        raise AssertionError(f"split control unexpectedly contains slicing: {slicing_ops}")

    report = {
        "graph": expected_graph,
        "projection": args.projection,
        "mode": args.mode,
        "sequence": args.seq,
        "projection_count": len(projection_ops),
        "projection_channels": channels,
        "head_output_count": len(graph_outputs),
        "slice_operations": slicing_ops,
        "convert_count": 0,
        "passed": True,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
