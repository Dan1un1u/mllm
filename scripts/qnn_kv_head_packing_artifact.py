#!/usr/bin/env python3
"""Build an independent layer-14 K/V split-vs-packed LPBQ fixture.

The only source is the accepted native-U8 RMSNorm model.  The compact model
contains the same W4G32 payload twice: eight 2048x128 head projections for the
control and one 2048x1024 projection for the packing candidate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import qnn_kv_head_projection_artifact as base


HEADS = base.ALL_KV_CHANNELS // base.HEAD_DIM
SEQUENCES = (32, 64)


def _projection_payloads(
    source: Path,
    descriptors: dict[str, base.Descriptor],
    projection: str,
) -> tuple[list[base.TensorPayload], np.ndarray, dict[str, object]]:
    source_prefix = f"{base.LAYER_PREFIX}.{projection}"
    weight_desc = descriptors[source_prefix + ".weight"]
    scale1_desc = descriptors[source_prefix + ".scale1"]
    scale2_desc = descriptors[source_prefix + ".scale2"]
    expected_weight = (1, 1, base.HIDDEN_SIZE, base.ALL_KV_CHANNELS)
    if weight_desc.dtype_id != base.DTYPE_INT8 or weight_desc.shape != expected_weight:
        raise ValueError(f"unexpected {projection} weight: {weight_desc}")
    if scale1_desc.dtype_id != base.DTYPE_UINT8 or scale1_desc.shape != (
        base.ALL_KV_CHANNELS * base.HIDDEN_SIZE // base.GROUP_SIZE,
    ):
        raise ValueError(f"unexpected {projection} scale1: {scale1_desc}")
    if scale2_desc.dtype_id != base.DTYPE_FLOAT32 or scale2_desc.shape != (
        base.ALL_KV_CHANNELS,
    ):
        raise ValueError(f"unexpected {projection} scale2: {scale2_desc}")

    weight = np.ascontiguousarray(base.tensor_view(source, weight_desc))
    scale1_2d = np.ascontiguousarray(base.tensor_view(source, scale1_desc)).reshape(
        base.ALL_KV_CHANNELS, -1
    )
    scale2 = np.ascontiguousarray(base.tensor_view(source, scale2_desc))
    signed = np.where(weight.astype(np.int16) >= 8, weight.astype(np.int16) - 16,
                      weight.astype(np.int16))
    if int(signed.min()) < -7 or int(signed.max()) > 7:
        raise ValueError(
            f"invalid deployed W4 range for {projection}: {signed.min()}, {signed.max()}"
        )

    tensors: list[base.TensorPayload] = []
    split_metadata: list[dict[str, object]] = []
    for head in range(HEADS):
        begin = head * base.HEAD_DIM
        end = begin + base.HEAD_DIM
        prefix = f"{source_prefix}.{head}"
        head_weight = np.ascontiguousarray(weight[:, :, :, begin:end])
        head_scale1 = np.ascontiguousarray(scale1_2d[begin:end].reshape(-1))
        head_scale2 = np.ascontiguousarray(scale2[begin:end])
        tensors.extend(
            [
                base.TensorPayload(prefix + ".weight", base.DTYPE_INT8,
                                   head_weight.shape, head_weight),
                base.TensorPayload(prefix + ".scale1", base.DTYPE_UINT8,
                                   head_scale1.shape, head_scale1),
                base.TensorPayload(prefix + ".scale2", base.DTYPE_FLOAT32,
                                   head_scale2.shape, head_scale2),
            ]
        )
        split_metadata.append(
            {
                "head": head,
                "weight_sha256": base._sha256_bytes(head_weight),
                "scale1_sha256": base._sha256_bytes(head_scale1),
                "scale2_sha256": base._sha256_bytes(head_scale2),
            }
        )

    packed_prefix = source_prefix + ".packed"
    packed_scale1 = np.ascontiguousarray(scale1_2d.reshape(-1))
    tensors.extend(
        [
            base.TensorPayload(packed_prefix + ".weight", base.DTYPE_INT8,
                               weight.shape, weight),
            base.TensorPayload(packed_prefix + ".scale1", base.DTYPE_UINT8,
                               packed_scale1.shape, packed_scale1),
            base.TensorPayload(packed_prefix + ".scale2", base.DTYPE_FLOAT32,
                               scale2.shape, scale2),
        ]
    )

    signed_matrix = signed.reshape(base.HIDDEN_SIZE, base.ALL_KV_CHANNELS)
    block_scales = scale1_2d.astype(np.float32) * scale2[:, None]
    effective = (
        signed_matrix.T.reshape(
            base.ALL_KV_CHANNELS,
            base.HIDDEN_SIZE // base.GROUP_SIZE,
            base.GROUP_SIZE,
        ).astype(np.float32)
        * block_scales[:, :, None]
    ).reshape(base.ALL_KV_CHANNELS, base.HIDDEN_SIZE)
    metadata = {
        "split": split_metadata,
        "packed": {
            "weight_shape": list(weight.shape),
            "weight_sha256": base._sha256_bytes(weight),
            "scale1_shape": list(packed_scale1.shape),
            "scale1_sha256": base._sha256_bytes(packed_scale1),
            "scale2_shape": list(scale2.shape),
            "scale2_sha256": base._sha256_bytes(scale2),
        },
        "signed_w4_min": int(signed.min()),
        "signed_w4_max": int(signed.max()),
    }
    return tensors, effective, metadata


def build(source: Path, output_dir: Path) -> dict[str, object]:
    descriptors = base.read_descriptors(source)
    output_dir.mkdir(parents=True, exist_ok=False)
    input_qparam = base._qparam(source, descriptors, base.INPUT_QPARAM)
    tensors: list[base.TensorPayload] = []
    effective_weights: dict[str, np.ndarray] = {}
    projection_metadata: dict[str, object] = {}
    qparams: dict[str, object] = {}

    for projection, output_prefix in base.PROJECTIONS.items():
        projection_tensors, effective, metadata = _projection_payloads(
            source, descriptors, projection
        )
        tensors.extend(projection_tensors)
        effective_weights[projection] = effective
        output_qparam = base._qparam(source, descriptors, output_prefix)
        tensors.extend(
            base._qparam_payloads(
                source, descriptors, base.INPUT_QPARAM,
                f"diagnostic.{projection}.input",
            )
        )
        tensors.extend(
            base._qparam_payloads(
                source, descriptors, output_prefix,
                f"diagnostic.{projection}.output",
            )
        )
        qparams[projection] = {
            "input": {"scale": input_qparam[0], "zero_point": input_qparam[1]},
            "output": {"scale": output_qparam[0], "zero_point": output_qparam[1]},
        }
        projection_metadata[projection] = metadata

    model_path = output_dir / "qwen3-layer14-kv-head-packing-w4g32-a8.mllm"
    base.write_model(model_path, tensors)

    fixtures: dict[str, object] = {}
    for sequence in SEQUENCES:
        _, codes = base._fixture(input_qparam, sequence)
        input_path = output_dir / f"input_s{sequence}_a8.raw"
        input_path.write_bytes(codes.tobytes(order="C"))
        reconstructed = base._dequantize(codes, input_qparam)
        fixtures[f"s{sequence}"] = {
            "input": {
                "path": input_path.name,
                "bytes": input_path.stat().st_size,
                "sha256": base._sha256_bytes(codes),
            },
            "references": {},
        }
        for projection, output_prefix in base.PROJECTIONS.items():
            output_qparam = base._qparam(source, descriptors, output_prefix)
            values = reconstructed @ effective_weights[projection].T
            sequence_major = base._quantize(values, output_qparam)
            head_major = np.ascontiguousarray(
                sequence_major.reshape(sequence, HEADS, base.HEAD_DIM)
                .transpose(1, 0, 2)
                .reshape(-1)
            )
            reference_path = output_dir / f"reference_{projection}_s{sequence}_a8.raw"
            reference_path.write_bytes(head_major.tobytes(order="C"))
            fixtures[f"s{sequence}"]["references"][projection] = {
                "path": reference_path.name,
                "bytes": reference_path.stat().st_size,
                "sha256": base._sha256_bytes(head_major),
                "code_min": int(head_major.min()),
                "code_max": int(head_major.max()),
            }

    metadata = {
        "contract": {
            "source": os.fspath(source.resolve()),
            "source_sha256": base._sha256_file(source),
            "layer": 14,
            "heads": HEADS,
            "activation": "asymmetric UInt8 input and output",
            "weight": "signed W4 LPBQ, G32, Int8 carrier",
            "control": "eight independent 2048x128 Conv2d projections",
            "candidate": "one 2048x1024 Conv2d plus eight output slices",
            "sequences": list(SEQUENCES),
        },
        "model": {
            "path": model_path.name,
            "bytes": model_path.stat().st_size,
            "sha256": base._sha256_file(model_path),
        },
        "qparams": qparams,
        "projections": projection_metadata,
        "fixtures": fixtures,
    }
    (output_dir / "artifact.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
