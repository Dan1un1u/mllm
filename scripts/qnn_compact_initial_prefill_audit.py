#!/usr/bin/env python3
"""Audit the full-model first-prefill A/B manifests."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


SEQ = 32
CONTEXT = 1024
LAYERS = 28
QUERY_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
VOCAB = 151936


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(tensor: dict) -> dict:
    return {
        "dimensions": tensor["dimensions"],
        "qnn_dtype": tensor["qnn_dtype"],
        "qnn_quantization": tensor["qnn_quantization"],
        "quant_recipe": tensor["quant_recipe"],
    }


def _shape_counts(tensors: list[dict]) -> dict[str, int]:
    counter = collections.Counter(
        f"{item['qnn_dtype']}:{'x'.join(map(str, item['dimensions']))}"
        for item in tensors
    )
    return dict(sorted(counter.items()))


def _audit(path: Path, width: int) -> tuple[dict, list[dict]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["graph"] != "model.0.s32":
        raise AssertionError(f"unexpected graph name: {document['graph']}")
    inputs = [x for x in document["tensors"] if x["tensor_type"] == "APP_WRITE"]
    outputs = [x for x in document["tensors"] if x["tensor_type"] == "APP_READ"]
    expected_inputs = 3 + (2 * LAYERS if width == CONTEXT else 0)
    if len(inputs) != expected_inputs:
        raise AssertionError(f"width {width}: expected {expected_inputs} inputs, got {len(inputs)}")
    if len(outputs) != 1 + 2 * LAYERS:
        raise AssertionError(f"width {width}: unexpected output count {len(outputs)}")

    input_shapes = _shape_counts(inputs)
    expected_shape_counts = {
        f"INT32:1x{SEQ}": 1,
        f"INT32:{SEQ}": 1,
        f"BOOL8:1x1x{SEQ}x{width}": 1,
    }
    if width == CONTEXT:
        expected_shape_counts.update(
            {
                f"UFIXED_POINT_8:1x{KV_HEADS}x{HEAD_DIM}x{CONTEXT - SEQ}": LAYERS,
                f"UFIXED_POINT_8:1x{KV_HEADS}x{CONTEXT - SEQ}x{HEAD_DIM}": LAYERS,
            }
        )
    if input_shapes != dict(sorted(expected_shape_counts.items())):
        raise AssertionError(f"width {width}: input signature mismatch: {input_shapes}")

    output_shapes = _shape_counts(outputs)
    expected_outputs = {
        f"UFIXED_POINT_8:1x1x{SEQ}x{VOCAB}": 1,
        f"UFIXED_POINT_8:1x{KV_HEADS}x{HEAD_DIM}x{SEQ}": LAYERS,
        f"UFIXED_POINT_8:1x{KV_HEADS}x{SEQ}x{HEAD_DIM}": LAYERS,
    }
    if output_shapes != dict(sorted(expected_outputs.items())):
        raise AssertionError(f"width {width}: output signature mismatch: {output_shapes}")

    packages = collections.Counter(op["package"] for op in document["operations"])
    if set(packages) != {"qti.aisw"}:
        raise AssertionError(f"width {width}: non-native package observed: {packages}")
    op_types = collections.Counter(op["qnn_op_type"] for op in document["operations"])
    if (
        op_types["Softmax"] != QUERY_HEADS * LAYERS
        or op_types["MatMul"] != 2 * QUERY_HEADS * LAYERS
    ):
        raise AssertionError(
            f"width {width}: unexpected attention op counts: "
            f"MatMul={op_types['MatMul']} Softmax={op_types['Softmax']}"
        )
    if op_types["ElementWiseEqual"]:
        raise AssertionError(f"width {width}: quantized-zero Equal remained in BOOL8 graph")

    return (
        {
            "manifest": str(path.resolve()),
            "manifest_sha256": _sha256(path),
            "width": width,
            "input_count": len(inputs),
            "output_count": len(outputs),
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "native_packages": dict(packages),
            "operation_count": len(document["operations"]),
            "operation_types": dict(sorted(op_types.items())),
        },
        outputs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--compact-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    full, full_outputs = _audit(args.full_manifest, CONTEXT)
    compact, compact_outputs = _audit(args.compact_manifest, SEQ)
    full_signatures = [_signature(item) for item in full_outputs]
    compact_signatures = [_signature(item) for item in compact_outputs]
    if full_signatures != compact_signatures:
        raise AssertionError("full and compact output dtype/qparam signatures differ")
    full_types = full["operation_types"]
    compact_types = compact["operation_types"]
    expected_cache_head_ops = 2 * KV_HEADS * LAYERS
    if full_types.get("Concat", 0) - compact_types.get("Concat", 0) != expected_cache_head_ops:
        raise AssertionError("compact SHA graph did not remove every per-head past-cache Concat")
    if full_types.get("StridedSlice", 0) - compact_types.get("StridedSlice", 0) != expected_cache_head_ops:
        raise AssertionError("compact SHA graph did not remove every per-head past-cache slice")

    report = {
        "contract": {
            "model": "Qwen3-1.7B W4A8G32 native-U8 RMSNorm split-head (SHA)",
            "sequence": SEQ,
            "full_width": CONTEXT,
            "compact_width": SEQ,
            "only_graph_variable": "semantically empty first-chunk past-cache and attention width",
            "backend": "native qti.aisw only",
            "output_signatures_identical": True,
        },
        "full": full,
        "compact": compact,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
