#!/usr/bin/env python3
"""Audit both attention manifests and generate a paired first-chunk fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SEQ = 32
CONTEXT = 1024
HEAD_DIM = 128
QUERY_HEADS = 16
KV_HEADS = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qparam(tensor: dict) -> dict[str, float | int]:
    quant = tensor["qnn_quantization"]
    if tensor["qnn_dtype"] != "UFIXED_POINT_8" or quant.get("encoding") != "scale_offset":
        raise AssertionError(f"not a quantized U8 tensor: {tensor['name']}")
    return {"scale": float(quant["scale"]), "zero_point": int(quant["zero_point"])}


def _all_equal(values: list[dict], label: str) -> dict:
    if not values or any(value != values[0] for value in values[1:]):
        raise AssertionError(f"{label} qparams are not common")
    return values[0]


def _contract(manifest: Path, width: int) -> dict:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    tensors = {item["name"]: item for item in document["tensors"]}
    app_write = [item for item in document["tensors"] if item["tensor_type"] == "APP_WRITE"]
    app_read = [item for item in document["tensors"] if item["tensor_type"] == "APP_READ"]
    if len(app_write) != QUERY_HEADS + 2 * KV_HEADS + 1:
        raise AssertionError(f"unexpected graph input count: {len(app_write)}")
    if len(app_read) != QUERY_HEADS:
        raise AssertionError(f"unexpected graph output count: {len(app_read)}")

    queries = app_write[:QUERY_HEADS]
    keys = app_write[QUERY_HEADS : QUERY_HEADS + KV_HEADS]
    values = app_write[QUERY_HEADS + KV_HEADS : QUERY_HEADS + 2 * KV_HEADS]
    mask = app_write[-1]
    expected_query = [1, 1, SEQ, HEAD_DIM]
    expected_key = [1, 1, HEAD_DIM, width]
    expected_value = [1, 1, width, HEAD_DIM]
    expected_mask = [1, 1, SEQ, width]
    for label, group, expected in (
        ("query", queries, expected_query),
        ("key", keys, expected_key),
        ("value", values, expected_value),
        ("output", app_read, expected_query),
    ):
        if any(item["dimensions"] != expected for item in group):
            raise AssertionError(f"unexpected {label} shape")
    if mask["dimensions"] != expected_mask:
        raise AssertionError(f"unexpected mask shape: {mask['dimensions']}")
    if mask["qnn_dtype"] != "BOOL8" or mask["qnn_quantization"].get("defined"):
        raise AssertionError("causal mask is not an unquantized BOOL8 graph input")

    matmuls = [op for op in document["operations"] if op["qnn_op_type"] == "MatMul"]
    softmaxes = [op for op in document["operations"] if op["qnn_op_type"] == "Softmax"]
    if len(matmuls) != 2 * QUERY_HEADS or len(softmaxes) != QUERY_HEADS:
        raise AssertionError(
            f"unexpected native-op counts: MatMul={len(matmuls)}, Softmax={len(softmaxes)}"
        )
    for op in matmuls + softmaxes:
        if op["package"] != "qti.aisw":
            raise AssertionError(f"non-native op: {op['name']} ({op['package']})")

    qk_outputs = []
    av_outputs = []
    for op in matmuls:
        output = tensors[op["outputs"][0]]
        if output["dimensions"] == expected_mask:
            qk_outputs.append(output)
        elif output["dimensions"] == expected_query:
            av_outputs.append(output)
    if len(qk_outputs) != QUERY_HEADS or len(av_outputs) != QUERY_HEADS:
        raise AssertionError(
            f"cannot classify MatMul outputs: qk={len(qk_outputs)}, av={len(av_outputs)}"
        )
    softmax_inputs = [tensors[op["inputs"][0]] for op in softmaxes]
    softmax_outputs = [tensors[op["outputs"][0]] for op in softmaxes]
    mul_ops = [
        next(op for op in document["operations"] if op["name"] == f"model.Mul.{head}")
        for head in range(QUERY_HEADS)
    ]
    reduce_ops = [
        next(op for op in document["operations"] if op["name"] == f"model.ReduceMin.{head}")
        for head in range(QUERY_HEADS)
    ]
    add_ops = [
        next(op for op in document["operations"] if op["name"] == f"model.Add.{head}")
        for head in range(QUERY_HEADS)
    ]

    return {
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "graph": document["graph"],
        "width": width,
        "native_ops": {"MatMul": len(matmuls), "Softmax": len(softmaxes)},
        "qparams": {
            "query": _all_equal([_qparam(x) for x in queries], "query"),
            "key": _all_equal([_qparam(x) for x in keys], "key"),
            "value": _all_equal([_qparam(x) for x in values], "value"),
            "mask": {"dtype": "BOOL8", "valid_code": 1, "invalid_code": 0},
            "qk": _all_equal([_qparam(x) for x in qk_outputs], "qk"),
            "scale_constant": _all_equal(
                [_qparam(tensors[op["inputs"][1]]) for op in mul_ops],
                "scale constant",
            ),
            "mul": _all_equal(
                [_qparam(tensors[op["outputs"][0]]) for op in mul_ops], "scaled logits"
            ),
            "reduce_min": _all_equal(
                [_qparam(tensors[op["outputs"][0]]) for op in reduce_ops], "reduce min"
            ),
            "minus_constant": _all_equal(
                [_qparam(tensors[op["inputs"][1]]) for op in add_ops],
                "minus-twenty constant",
            ),
            "masked_value": _all_equal(
                [_qparam(tensors[op["outputs"][0]]) for op in add_ops], "masked value"
            ),
            "softmax_input": _all_equal(
                [_qparam(x) for x in softmax_inputs], "softmax input"
            ),
            "softmax_output": _all_equal(
                [_qparam(x) for x in softmax_outputs], "softmax output"
            ),
            "output": _all_equal([_qparam(x) for x in av_outputs], "AV output"),
        },
        "shapes": {
            "query": expected_query,
            "key": expected_key,
            "value": expected_value,
            "mask": expected_mask,
            "output": expected_query,
        },
    }


def _quantize(values: np.ndarray, qparam: dict) -> np.ndarray:
    codes = np.floor(values / qparam["scale"] + qparam["zero_point"] + 0.5)
    return np.clip(codes, 0, 255).astype(np.uint8)


def _dequantize(codes: np.ndarray, qparam: dict) -> np.ndarray:
    return (codes.astype(np.float32) - qparam["zero_point"]) * qparam["scale"]


def _write(path: Path, data: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(data)
    path.write_bytes(contiguous.tobytes(order="C"))
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "shape": list(contiguous.shape),
    }


def _host_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    qparams: dict,
) -> np.ndarray:
    query_real = _dequantize(query, qparams["query"])
    key_real = _dequantize(key, qparams["key"])
    value_real = _dequantize(value, qparams["value"])
    outputs = np.empty((QUERY_HEADS, SEQ, HEAD_DIM), dtype=np.uint8)
    causal = np.tril(np.ones((SEQ, SEQ), dtype=bool))
    for head in range(QUERY_HEADS):
        kv_head = head // 2
        logits = query_real[head] @ key_real[kv_head]
        logits = _dequantize(_quantize(logits, qparams["qk"]), qparams["qk"])
        scale = _dequantize(
            _quantize(
                np.asarray([1.0 / math.sqrt(HEAD_DIM)], dtype=np.float32),
                qparams["scale_constant"],
            ),
            qparams["scale_constant"],
        )
        scaled = _dequantize(_quantize(logits * scale[0], qparams["mul"]), qparams["mul"])
        reduced = np.min(scaled, axis=-1, keepdims=True)
        reduced = _dequantize(
            _quantize(reduced, qparams["reduce_min"]), qparams["reduce_min"]
        )
        minus_twenty = _dequantize(
            _quantize(np.asarray([-20.0], dtype=np.float32), qparams["minus_constant"]),
            qparams["minus_constant"],
        )[0]
        masked = _dequantize(
            _quantize(reduced + minus_twenty, qparams["masked_value"]),
            qparams["masked_value"],
        )
        selected = np.where(causal, scaled, masked)
        selected = _dequantize(
            _quantize(selected, qparams["softmax_input"]), qparams["softmax_input"]
        )
        maximum = np.max(selected, axis=-1, keepdims=True)
        exponentials = np.exp(selected - maximum)
        probabilities = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
        probability_codes = _quantize(probabilities, qparams["softmax_output"])
        probabilities = _dequantize(probability_codes, qparams["softmax_output"])
        result = probabilities @ value_real[kv_head]
        outputs[head] = _quantize(result, qparams["output"])
    return outputs


def _fixtures(output_dir: Path, qparams: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260823)
    query_zp = int(qparams["query"]["zero_point"])
    query = np.clip(
        query_zp + rng.integers(-24, 25, size=(QUERY_HEADS, SEQ, HEAD_DIM)),
        0,
        255,
    ).astype(np.uint8)
    key_current = np.clip(
        128 + rng.integers(-24, 25, size=(KV_HEADS, HEAD_DIM, SEQ)), 0, 255
    ).astype(np.uint8)
    value_current = np.clip(
        128 + rng.integers(-40, 41, size=(KV_HEADS, SEQ, HEAD_DIM)), 0, 255
    ).astype(np.uint8)

    key_full = rng.integers(0, 256, size=(KV_HEADS, HEAD_DIM, CONTEXT), dtype=np.uint8)
    value_full = rng.integers(0, 256, size=(KV_HEADS, CONTEXT, HEAD_DIM), dtype=np.uint8)
    key_full[:, :, CONTEXT - SEQ :] = key_current
    value_full[:, CONTEXT - SEQ :, :] = value_current

    valid_code = int(qparams["mask"]["valid_code"])
    compact_mask = np.zeros((SEQ, SEQ), dtype=np.uint8)
    compact_mask[np.tril_indices(SEQ)] = valid_code
    full_mask = np.zeros((SEQ, CONTEXT), dtype=np.uint8)
    full_mask[:, CONTEXT - SEQ :] = compact_mask
    expected = _host_reference(query, key_current, value_current, qparams)

    return {
        "query": _write(output_dir / "query_s32_u8.raw", query),
        "full_key": _write(output_dir / "key_full_w1024_u8.raw", key_full),
        "full_value": _write(output_dir / "value_full_w1024_u8.raw", value_full),
        "full_mask": _write(output_dir / "mask_full_w1024_u8.raw", full_mask),
        "compact_key": _write(output_dir / "key_compact_w32_u8.raw", key_current),
        "compact_value": _write(output_dir / "value_compact_w32_u8.raw", value_current),
        "compact_mask": _write(output_dir / "mask_compact_w32_u8.raw", compact_mask),
        "host_expected": _write(output_dir / "host_expected_output_u8.raw", expected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--compact-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    full = _contract(args.full_manifest, CONTEXT)
    compact = _contract(args.compact_manifest, SEQ)
    if full["qparams"] != compact["qparams"]:
        raise AssertionError("full and compact graphs do not share identical qparams")
    fixtures = _fixtures(args.output_dir, compact["qparams"])
    report = {
        "contract": {
            "only_variable": "attention key/value width: 1024 masked columns versus 32 valid first-chunk columns",
            "layer": 14,
            "sequence": SEQ,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "backend": "native qti.aisw MatMul/Softmax/MatMul",
        },
        "full": full,
        "compact": compact,
        "fixtures": fixtures,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    report_path = args.report or args.output_dir / "fixture_report.json"
    report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
