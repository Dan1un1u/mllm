#!/usr/bin/env python3
"""Build a strict per-head versus packed K/V LPBQ micrograph artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from qnn_lpbq_a16_a8_projection_artifact import (
    DTYPE_INT8,
    TensorPayload,
    _dequantize,
    _qparam,
    _quantize,
    _sha256_bytes,
    _sha256_file,
    _tensor,
    read_descriptors,
    tensor_view,
    write_model,
)


LAYER = 14
INPUT_CHANNELS = 2048
HEAD_CHANNELS = 128
HEADS = 8
OUTPUT_CHANNELS = HEAD_CHANNELS * HEADS
GROUP = 32
PROJECTIONS = ("k_proj", "v_proj")
QPARAMS = {
    "k_proj": (
        "model.layers.14.self_attn.q_proj_input_qdq.fake_quant",
        "model.layers.14.self_attn.k_norm_input_qdq.fake_quant",
    ),
    "v_proj": (
        "model.layers.14.self_attn.q_proj_input_qdq.fake_quant",
        "model.layers.14.self_attn.v_cast_to_int16_qdq.fake_quant",
    ),
}


def _prefix(projection: str) -> str:
    return f"model.layers.{LAYER}.self_attn.{projection}"


def _payload(name: str, dtype_id: int, data: np.ndarray) -> TensorPayload:
    contiguous = np.ascontiguousarray(data)
    return TensorPayload(name, dtype_id, tuple(contiguous.shape), contiguous)


def _split_static_tensors(source: Path, descriptors: dict, projection: str) -> tuple[list[TensorPayload], dict]:
    prefix = _prefix(projection)
    weight_desc = descriptors[prefix + ".weight"]
    if weight_desc.dtype_id != DTYPE_INT8 or weight_desc.shape != (1, 1, INPUT_CHANNELS, OUTPUT_CHANNELS):
        raise ValueError(
            f"unexpected {projection} carrier: dtype={weight_desc.dtype_id}, shape={weight_desc.shape}"
        )
    weight = tensor_view(source, weight_desc)
    scale1_desc = descriptors[prefix + ".scale1"]
    scale2_desc = descriptors[prefix + ".scale2"]
    scale1 = tensor_view(source, scale1_desc).reshape(-1)
    scale2 = tensor_view(source, scale2_desc).reshape(-1)
    expected_scale1 = OUTPUT_CHANNELS * (INPUT_CHANNELS // GROUP)
    if scale1.size != expected_scale1 or scale2.size != OUTPUT_CHANNELS:
        raise ValueError(
            f"unexpected {projection} scale sizes: {scale1.size}, {scale2.size}"
        )

    tensors = [
        _tensor(source, descriptors, prefix + ".weight"),
        _tensor(source, descriptors, prefix + ".scale1"),
        _tensor(source, descriptors, prefix + ".scale2"),
    ]
    heads = []
    scale1_per_head = expected_scale1 // HEADS
    for head in range(HEADS):
        head_prefix = f"{prefix}.{head}"
        output_start = head * HEAD_CHANNELS
        output_end = output_start + HEAD_CHANNELS
        scale1_start = head * scale1_per_head
        scale1_end = scale1_start + scale1_per_head
        head_weight = np.ascontiguousarray(weight[:, :, :, output_start:output_end])
        head_scale1 = np.ascontiguousarray(scale1[scale1_start:scale1_end])
        head_scale2 = np.ascontiguousarray(scale2[output_start:output_end])
        tensors.extend(
            [
                _payload(head_prefix + ".weight", weight_desc.dtype_id, head_weight),
                _payload(head_prefix + ".scale1", scale1_desc.dtype_id, head_scale1),
                _payload(head_prefix + ".scale2", scale2_desc.dtype_id, head_scale2),
            ]
        )
        heads.append(
            {
                "head": head,
                "output_range": [output_start, output_end],
                "weight_sha256": _sha256_bytes(head_weight),
                "scale1_sha256": _sha256_bytes(head_scale1),
                "scale2_sha256": _sha256_bytes(head_scale2),
            }
        )

    reconstructed_weight = np.concatenate(
        [tensor.data for tensor in tensors if tensor.name.endswith(".weight") and tensor.name != prefix + ".weight"],
        axis=3,
    )
    reconstructed_scale1 = np.concatenate(
        [tensor.data for tensor in tensors if tensor.name.endswith(".scale1") and tensor.name != prefix + ".scale1"]
    )
    reconstructed_scale2 = np.concatenate(
        [tensor.data for tensor in tensors if tensor.name.endswith(".scale2") and tensor.name != prefix + ".scale2"]
    )
    if not np.array_equal(reconstructed_weight, weight):
        raise AssertionError(f"{projection}: split weights do not reconstruct packed payload")
    if not np.array_equal(reconstructed_scale1, scale1):
        raise AssertionError(f"{projection}: split scale1 does not reconstruct packed payload")
    if not np.array_equal(reconstructed_scale2, scale2):
        raise AssertionError(f"{projection}: split scale2 does not reconstruct packed payload")

    return tensors, {
        "packed": {
            "weight_shape": list(weight_desc.shape),
            "weight_sha256": _sha256_bytes(weight),
            "scale1_sha256": _sha256_bytes(scale1),
            "scale2_sha256": _sha256_bytes(scale2),
        },
        "heads": heads,
        "byte_exact_reconstruction": True,
    }


def _effective_weight(source: Path, descriptors: dict, projection: str,
                      output_indices: np.ndarray) -> np.ndarray:
    prefix = _prefix(projection)
    carrier = tensor_view(source, descriptors[prefix + ".weight"]).reshape(
        INPUT_CHANNELS, OUTPUT_CHANNELS
    )
    codes = np.ascontiguousarray(carrier[:, output_indices].T).astype(np.int16)
    signed = np.where(codes >= 8, codes - 16, codes)
    if int(signed.min()) < -7 or int(signed.max()) > 7:
        raise ValueError(f"invalid deployed W4 range: {signed.min()}, {signed.max()}")
    scale1 = np.asarray(
        tensor_view(source, descriptors[prefix + ".scale1"]), dtype=np.float32
    ).reshape(OUTPUT_CHANNELS, INPUT_CHANNELS // GROUP)[output_indices]
    scale2 = np.asarray(
        tensor_view(source, descriptors[prefix + ".scale2"]), dtype=np.float32
    ).reshape(OUTPUT_CHANNELS)[output_indices]
    block_scale = scale1 * scale2[:, None]
    return (
        signed.reshape(len(output_indices), INPUT_CHANNELS // GROUP, GROUP).astype(np.float32)
        * block_scale[:, :, None]
    ).reshape(len(output_indices), INPUT_CHANNELS)


def _fixture(source: Path, descriptors: dict, output_dir: Path, projection: str,
             qparams: dict, seq: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(20260822 + seq + PROJECTIONS.index(projection) * 100)
    input_codes = rng.integers(1, 255, size=(seq, INPUT_CHANNELS), dtype=np.uint8)
    zero_point = qparams[projection]["input"]["zero_point"]
    anchors = np.asarray([0, zero_point, 255, 1, 254], dtype=np.uint8)
    input_codes.reshape(-1)[: anchors.size] = anchors
    input_path = output_dir / f"input_{projection}_s{seq}_a8.raw"
    input_path.write_bytes(input_codes.tobytes(order="C"))

    output_indices = np.unique(np.linspace(0, OUTPUT_CHANNELS - 1, num=64, dtype=np.int64))
    rows = np.arange(min(seq, 4), dtype=np.int64)
    input_qparam = (qparams[projection]["input"]["scale"], zero_point)
    output_qparam = (
        qparams[projection]["output"]["scale"],
        qparams[projection]["output"]["zero_point"],
    )
    real_input = _dequantize(input_codes, input_qparam)
    weights = _effective_weight(source, descriptors, projection, output_indices)
    real_output = real_input[rows] @ weights.T
    expected_codes = _quantize(real_output, output_qparam, 8)
    fixture = {
        "path": os.fspath(input_path.resolve()),
        "bytes": input_path.stat().st_size,
        "sha256": _sha256_bytes(input_codes),
        "shape": [1, seq, INPUT_CHANNELS],
    }
    reference = {
        "rows": rows.tolist(),
        "output_indices": output_indices.tolist(),
        "expected_codes": expected_codes.astype(np.int64).tolist(),
        "expected_real": real_output.astype(np.float64).tolist(),
    }
    return fixture, reference


def build(source: Path, output_dir: Path) -> dict:
    descriptors = read_descriptors(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors: list[TensorPayload] = []
    weights = {}
    qparams = {}
    for projection in PROJECTIONS:
        projection_tensors, audit = _split_static_tensors(source, descriptors, projection)
        tensors.extend(projection_tensors)
        weights[projection] = audit
        qparams[projection] = {}
        for side, source_prefix in zip(("input", "output"), QPARAMS[projection], strict=True):
            target_prefix = f"diagnostic.a8.{projection}.{side}"
            scale, zero_point = _qparam(source, descriptors, source_prefix, 8)
            qparams[projection][side] = {"scale": scale, "zero_point": zero_point}
            tensors.extend(
                [
                    _tensor(source, descriptors, source_prefix + ".scale", target_prefix + ".scale"),
                    _tensor(source, descriptors, source_prefix + ".zero_point", target_prefix + ".zero_point"),
                ]
            )

    model_path = output_dir / "qwen3-kv-head-packing.mllm"
    write_model(model_path, tensors)
    fixtures = {}
    host_reference = {}
    for projection in PROJECTIONS:
        fixtures[projection] = {}
        host_reference[projection] = {}
        for seq in (1, 32):
            fixture, reference = _fixture(
                source, descriptors, output_dir, projection, qparams, seq
            )
            fixtures[projection][f"s{seq}"] = fixture
            host_reference[projection][f"s{seq}"] = reference

    return {
        "contract": {
            "only_variable": "8 independent 2048x128 Conv2d projections versus one 2048x1024 Conv2d projection",
            "activation": "asymmetric U8 input and output with identical qparams",
            "weight": "byte-equivalent signed W4[-7,7] G32 LPBQ payload and scales",
            "layer": LAYER,
            "heads": HEADS,
            "head_channels": HEAD_CHANNELS,
            "sequences": [1, 32],
        },
        "source": {
            "path": os.fspath(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": _sha256_file(source),
        },
        "artifact": {
            "path": os.fspath(model_path.resolve()),
            "bytes": model_path.stat().st_size,
            "sha256": _sha256_file(model_path),
        },
        "weights": weights,
        "qparams": qparams,
        "fixtures": fixtures,
        "host_reference": host_reference,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = build(args.source, args.output_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    report_path = args.report or args.output_dir / "artifact_report.json"
    report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
