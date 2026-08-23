#!/usr/bin/env python3
"""Audit W4A8/RMSNorm-U8 and W4A16 AR64 graphs before timing."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit(path: Path, activation_bits: int) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    tensors = {item["name"]: item for item in document["tensors"]}
    packages = collections.Counter(item["package"] for item in document["operations"])
    if set(packages) != {"qti.aisw"}:
        raise AssertionError(f"non-native package in {path}: {packages}")
    expected_dtype = "UFIXED_POINT_8" if activation_bits == 8 else "UFIXED_POINT_16"
    expected_max = 255 if activation_bits == 8 else 65535
    projections = []
    for operation in document["operations"]:
        if operation["qnn_op_type"] != "Conv2d" or len(operation["inputs"]) < 2:
            continue
        activation = tensors[operation["inputs"][0]]
        weight = tensors[operation["inputs"][1]]
        output = tensors[operation["outputs"][0]]
        recipe = weight["quant_recipe"]
        if recipe.get("type") != "lpbq":
            continue
        if recipe.get("quant_to_dtype") != "Int4" or recipe.get("block_size") != 32:
            raise AssertionError(f"{operation['name']}: not W4G32")
        for role, tensor in (("activation", activation), ("output", output)):
            if tensor["qnn_dtype"] != expected_dtype:
                raise AssertionError(f"{operation['name']} {role}: {tensor['qnn_dtype']}")
            quant = tensor["quant_recipe"]
            if quant.get("type") != "asymmetric_per_tensor" or quant.get("quant_max") != expected_max:
                raise AssertionError(f"{operation['name']} {role}: wrong activation contract")
        encoding = weight["qnn_quantization"]
        if encoding.get("encoding") != "blockwise_expansion" or encoding.get("num_blocks_per_axis") != 64:
            # q/k/v head projections also have In=2048 and therefore 64 G32 blocks;
            # MLP down projections have 6144/32=192 and are checked below.
            expected_blocks = weight["dimensions"][2] // 32
            if encoding.get("num_blocks_per_axis") != expected_blocks:
                raise AssertionError(f"{operation['name']}: wrong block expansion")
        projections.append(operation["name"])
    if len(projections) != 1009:
        raise AssertionError(f"expected 1009 LPBQ projections, got {len(projections)}")

    rmsnorms = [item for item in document["operations"] if item["qnn_op_type"] == "RmsNorm"]
    if len(rmsnorms) != 729:
        raise AssertionError(f"expected 729 RMSNorms, got {len(rmsnorms)}")
    rms_dtypes = collections.Counter()
    for operation in rmsnorms:
        operands = [tensors[name] for name in operation["inputs"] + operation["outputs"]]
        rms_dtypes.update(item["qnn_dtype"] for item in operands)
        if operands[0]["qnn_dtype"] != expected_dtype or operands[-1]["qnn_dtype"] != expected_dtype:
            raise AssertionError(f"{operation['name']}: RMSNorm activation width mismatch")
        if activation_bits == 8 and any(item["qnn_dtype"] != "UFIXED_POINT_8" for item in operands):
            raise AssertionError(f"{operation['name']}: A8 RMSNorm operand is not U8")
        if activation_bits == 16:
            if operands[1]["qnn_dtype"] != "UFIXED_POINT_16":
                raise AssertionError(f"{operation['name']}: A16 gamma is not U16")
            if operands[2]["qnn_dtype"] not in {"UFIXED_POINT_16", "SFIXED_POINT_32"}:
                raise AssertionError(f"{operation['name']}: A16 bias has unsupported dtype")

    op_types = collections.Counter(item["qnn_op_type"] for item in document["operations"])
    inputs = [item for item in document["tensors"] if item["tensor_type"] == "APP_WRITE"]
    outputs = [item for item in document["tensors"] if item["tensor_type"] == "APP_READ"]
    return {
        "graph": document["graph"],
        "manifest": str(path.resolve()),
        "manifest_sha256": _sha256(path),
        "activation_bits": activation_bits,
        "input_count": len(inputs),
        "output_count": len(outputs),
        "operation_count": len(document["operations"]),
        "operation_types": dict(sorted(op_types.items())),
        "operation_names": sorted(item["name"] for item in document["operations"]),
        "lpbq_projection_count": len(projections),
        "rmsnorm_count": len(rmsnorms),
        "rmsnorm_operand_dtypes": dict(sorted(rms_dtypes.items())),
        "native_packages": dict(packages),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a8-s1", type=Path, required=True)
    parser.add_argument("--a8-s64", type=Path, required=True)
    parser.add_argument("--a16-s1", type=Path, required=True)
    parser.add_argument("--a16-s64", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    pairs = {
        "s1": (_audit(args.a8_s1, 8), _audit(args.a16_s1, 16)),
        "s64": (_audit(args.a8_s64, 8), _audit(args.a16_s64, 16)),
    }
    report_pairs = {}
    for graph, (a8, a16) in pairs.items():
        if a8["input_count"] != a16["input_count"] or a8["output_count"] != a16["output_count"]:
            raise AssertionError(f"{graph}: graph I/O counts differ")
        type_delta = {
            key: a8["operation_types"].get(key, 0) - a16["operation_types"].get(key, 0)
            for key in sorted(set(a8["operation_types"]) | set(a16["operation_types"]))
            if a8["operation_types"].get(key, 0) != a16["operation_types"].get(key, 0)
        }
        a8_names = set(a8.pop("operation_names"))
        a16_names = set(a16.pop("operation_names"))
        report_pairs[graph] = {
            "a8": a8,
            "a16": a16,
            "operation_type_delta_a8_minus_a16": type_delta,
            "operation_names_only_in_a8": sorted(a8_names - a16_names),
            "operation_names_only_in_a16": sorted(a16_names - a8_names),
        }
    report = {
        "contract": {
            "weights": "W4 LPBQ G32 for both variants",
            "a8_activations": "asymmetric U8 with native U8 RMSNorm",
            "a16_activations": "asymmetric U16 with native U16 RMSNorm",
            "qairt_release": "2.49.0.260730",
            "a8_finalize": "P19",
            "a16_finalize": "default",
            "native_backend_only": True,
        },
        "graphs": report_pairs,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
