#!/usr/bin/env python3
"""Extract strict layer-14 K/V-head W4G32/A8 projection fixtures.

Only the accepted native-U8 RMSNorm model is read.  The compact model carries
head 0 of K and V, the real activation qparams, deterministic inputs, and host
references.  It deliberately does not consume any earlier W4A8 experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = 0x519A
VERSION = 2
HEADER = struct.Struct("<II512sIQ")
PARAM = struct.Struct("<IIQQQ16i256s")
DTYPE_FLOAT32 = 0
DTYPE_INT8 = 16
DTYPE_INT32 = 18
DTYPE_UINT8 = 129
HIDDEN_SIZE = 2048
ALL_KV_CHANNELS = 1024
HEAD_DIM = 128
GROUP_SIZE = 32
LAYER_PREFIX = "model.layers.14.self_attn"
PROJECTIONS = {
    "k_proj": f"{LAYER_PREFIX}.k_norm_input_qdq.fake_quant",
    "v_proj": f"{LAYER_PREFIX}.v_cast_to_int16_qdq.fake_quant",
}
INPUT_QPARAM = f"{LAYER_PREFIX}.q_proj_input_qdq.fake_quant"
SEQUENCES = (1, 32, 64)


@dataclass(frozen=True)
class Descriptor:
    dtype_id: int
    size: int
    offset: int
    shape: tuple[int, ...]
    name: str


@dataclass
class TensorPayload:
    name: str
    dtype_id: int
    shape: tuple[int, ...]
    data: np.ndarray


def _cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8")


def _dtype(dtype_id: int) -> np.dtype:
    return {
        DTYPE_FLOAT32: np.dtype("<f4"),
        DTYPE_INT8: np.dtype("i1"),
        DTYPE_INT32: np.dtype("<i4"),
        DTYPE_UINT8: np.dtype("u1"),
    }[dtype_id]


def _sha256_bytes(data: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(data)).cast("B")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_descriptors(path: Path) -> dict[str, Descriptor]:
    result: dict[str, Descriptor] = {}
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError(f"truncated mllm V2 header: {path}")
        magic, version, _model, count, descriptor_offset = HEADER.unpack(raw)
        if (magic, version, descriptor_offset) != (MAGIC, VERSION, HEADER.size):
            raise ValueError(f"unexpected mllm V2 header: {(magic, version, descriptor_offset)}")
        for expected_id in range(count):
            fields = PARAM.unpack(stream.read(PARAM.size))
            param_id, dtype_id, size, offset, rank = fields[:5]
            name = _cstring(fields[21])
            if param_id != expected_id or rank > 16 or name in result:
                raise ValueError(f"invalid descriptor {expected_id}: {param_id}, {name!r}, {rank}")
            result[name] = Descriptor(
                dtype_id, size, offset, tuple(fields[5 : 5 + rank]), name
            )
    return result


def tensor_view(path: Path, descriptor: Descriptor) -> np.memmap:
    dtype = _dtype(descriptor.dtype_id)
    expected = int(np.prod(descriptor.shape, dtype=np.int64)) * dtype.itemsize
    if expected != descriptor.size:
        raise ValueError(f"payload size mismatch for {descriptor.name}: {descriptor.size} != {expected}")
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=descriptor.offset,
        shape=descriptor.shape,
        order="C",
    )


def write_model(path: Path, tensors: list[TensorPayload]) -> None:
    if path.exists():
        raise FileExistsError(path)
    model_name = b"Qwen3 layer-14 K/V head LPBQ scheduling diagnostic".ljust(512, b"\0")
    data_offset = HEADER.size + len(tensors) * PARAM.size
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(HEADER.pack(MAGIC, VERSION, model_name, len(tensors), HEADER.size))
        offset = data_offset
        for index, tensor in enumerate(tensors):
            data = np.ascontiguousarray(tensor.data)
            expected = int(np.prod(tensor.shape, dtype=np.int64)) * _dtype(tensor.dtype_id).itemsize
            if data.nbytes != expected:
                raise ValueError(f"size mismatch for {tensor.name}: {data.nbytes} != {expected}")
            raw_name = tensor.name.encode("utf-8")[:255].ljust(256, b"\0")
            shape = tensor.shape + (0,) * (16 - len(tensor.shape))
            stream.write(
                PARAM.pack(
                    index,
                    tensor.dtype_id,
                    data.nbytes,
                    offset,
                    len(tensor.shape),
                    *shape,
                    raw_name,
                )
            )
            offset += data.nbytes
        for tensor in tensors:
            stream.write(memoryview(np.ascontiguousarray(tensor.data)).cast("B"))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _qparam(source: Path, descriptors: dict[str, Descriptor], prefix: str) -> tuple[float, int]:
    scale = float(np.asarray(tensor_view(source, descriptors[prefix + ".scale"])).reshape(-1)[0])
    zero_point = int(
        np.asarray(tensor_view(source, descriptors[prefix + ".zero_point"])).reshape(-1)[0]
    )
    if not math.isfinite(scale) or scale <= 0 or not 0 <= zero_point <= 255:
        raise ValueError(f"invalid U8 qparam for {prefix}: {scale}, {zero_point}")
    return scale, zero_point


def _qparam_payloads(
    source: Path,
    descriptors: dict[str, Descriptor],
    source_prefix: str,
    target_prefix: str,
) -> list[TensorPayload]:
    payloads = []
    for suffix in (".scale", ".zero_point"):
        descriptor = descriptors[source_prefix + suffix]
        payloads.append(
            TensorPayload(
                target_prefix + suffix,
                descriptor.dtype_id,
                descriptor.shape,
                np.ascontiguousarray(tensor_view(source, descriptor)),
            )
        )
    return payloads


def _quantize(values: np.ndarray, qparam: tuple[float, int]) -> np.ndarray:
    scale, zero_point = qparam
    return np.rint(values / scale + zero_point).clip(0, 255).astype(np.uint8)


def _dequantize(values: np.ndarray, qparam: tuple[float, int]) -> np.ndarray:
    scale, zero_point = qparam
    return (values.astype(np.float32) - zero_point) * scale


def _fixture(qparam: tuple[float, int], sequence: int) -> tuple[np.ndarray, np.ndarray]:
    scale, zero_point = qparam
    lower = -zero_point * scale * 0.8
    upper = (255 - zero_point) * scale * 0.8
    rng = np.random.default_rng(20260824 + sequence)
    real = rng.uniform(lower, upper, size=(sequence, HIDDEN_SIZE)).astype(np.float32)
    anchors = np.asarray([0.0, lower, upper, lower / 2.0, upper / 2.0], dtype=np.float32)
    real.reshape(-1)[: len(anchors)] = anchors
    return real, _quantize(real, qparam)


def _projection_payloads(
    source: Path, descriptors: dict[str, Descriptor], projection: str
) -> tuple[list[TensorPayload], np.ndarray, dict[str, object]]:
    source_prefix = f"{LAYER_PREFIX}.{projection}"
    target_prefix = f"{LAYER_PREFIX}.{projection}.0"
    weight_desc = descriptors[source_prefix + ".weight"]
    scale1_desc = descriptors[source_prefix + ".scale1"]
    scale2_desc = descriptors[source_prefix + ".scale2"]
    expected_weight = (1, 1, HIDDEN_SIZE, ALL_KV_CHANNELS)
    if weight_desc.dtype_id != DTYPE_INT8 or weight_desc.shape != expected_weight:
        raise ValueError(f"unexpected {projection} weight: {weight_desc.dtype_id}, {weight_desc.shape}")
    if scale1_desc.dtype_id != DTYPE_UINT8 or scale1_desc.shape != (
        ALL_KV_CHANNELS * HIDDEN_SIZE // GROUP_SIZE,
    ):
        raise ValueError(f"unexpected {projection} scale1: {scale1_desc.dtype_id}, {scale1_desc.shape}")
    if scale2_desc.dtype_id != DTYPE_FLOAT32 or scale2_desc.shape != (ALL_KV_CHANNELS,):
        raise ValueError(f"unexpected {projection} scale2: {scale2_desc.dtype_id}, {scale2_desc.shape}")

    weight = np.ascontiguousarray(tensor_view(source, weight_desc)[:, :, :, :HEAD_DIM])
    scale1 = np.ascontiguousarray(
        tensor_view(source, scale1_desc).reshape(ALL_KV_CHANNELS, -1)[:HEAD_DIM].reshape(-1)
    )
    scale2 = np.ascontiguousarray(tensor_view(source, scale2_desc)[:HEAD_DIM])
    payloads = [
        TensorPayload(target_prefix + ".weight", DTYPE_INT8, weight.shape, weight),
        TensorPayload(target_prefix + ".scale1", DTYPE_UINT8, scale1.shape, scale1),
        TensorPayload(target_prefix + ".scale2", DTYPE_FLOAT32, scale2.shape, scale2),
    ]

    signed = np.where(weight.reshape(HIDDEN_SIZE, HEAD_DIM).astype(np.int16) >= 8,
                      weight.reshape(HIDDEN_SIZE, HEAD_DIM).astype(np.int16) - 16,
                      weight.reshape(HIDDEN_SIZE, HEAD_DIM).astype(np.int16))
    if int(signed.min()) < -7 or int(signed.max()) > 7:
        raise ValueError(f"invalid deployed W4 range for {projection}: {signed.min()}, {signed.max()}")
    block_scales = scale1.reshape(HEAD_DIM, HIDDEN_SIZE // GROUP_SIZE).astype(np.float32)
    block_scales *= scale2[:, None]
    effective_weight = (
        signed.T.reshape(HEAD_DIM, HIDDEN_SIZE // GROUP_SIZE, GROUP_SIZE).astype(np.float32)
        * block_scales[:, :, None]
    ).reshape(HEAD_DIM, HIDDEN_SIZE)
    metadata = {
        "weight_shape": list(weight.shape),
        "weight_sha256": _sha256_bytes(weight),
        "scale1_shape": list(scale1.shape),
        "scale1_sha256": _sha256_bytes(scale1),
        "scale2_shape": list(scale2.shape),
        "scale2_sha256": _sha256_bytes(scale2),
        "signed_w4_min": int(signed.min()),
        "signed_w4_max": int(signed.max()),
    }
    return payloads, effective_weight, metadata


def build(source: Path, output_dir: Path) -> dict[str, object]:
    descriptors = read_descriptors(source)
    output_dir.mkdir(parents=True, exist_ok=False)
    input_qparam = _qparam(source, descriptors, INPUT_QPARAM)
    tensors: list[TensorPayload] = []
    effective_weights: dict[str, np.ndarray] = {}
    projection_metadata: dict[str, object] = {}
    qparams: dict[str, object] = {}

    for projection, output_prefix in PROJECTIONS.items():
        payloads, effective_weight, metadata = _projection_payloads(
            source, descriptors, projection
        )
        tensors.extend(payloads)
        effective_weights[projection] = effective_weight
        output_qparam = _qparam(source, descriptors, output_prefix)
        tensors.extend(
            _qparam_payloads(
                source,
                descriptors,
                INPUT_QPARAM,
                f"diagnostic.{projection}.input",
            )
        )
        tensors.extend(
            _qparam_payloads(
                source,
                descriptors,
                output_prefix,
                f"diagnostic.{projection}.output",
            )
        )
        qparams[projection] = {
            "input": {"scale": input_qparam[0], "zero_point": input_qparam[1]},
            "output": {"scale": output_qparam[0], "zero_point": output_qparam[1]},
        }
        projection_metadata[projection] = metadata

    model_path = output_dir / "qwen3-layer14-kv-head0-w4g32-a8.mllm"
    write_model(model_path, tensors)

    fixtures: dict[str, object] = {}
    for sequence in SEQUENCES:
        real, codes = _fixture(input_qparam, sequence)
        input_path = output_dir / f"input_s{sequence}_a8.raw"
        input_path.write_bytes(codes.tobytes(order="C"))
        fixtures[f"s{sequence}"] = {
            "input": {
                "path": input_path.name,
                "bytes": input_path.stat().st_size,
                "sha256": _sha256_bytes(codes),
                "saturation_count": int(np.count_nonzero((codes == 0) | (codes == 255))),
            },
            "references": {},
        }
        reconstructed = _dequantize(codes, input_qparam)
        for projection, output_prefix in PROJECTIONS.items():
            output_qparam = _qparam(source, descriptors, output_prefix)
            values = reconstructed @ effective_weights[projection].T
            reference = _quantize(values, output_qparam)
            reference_path = output_dir / f"reference_{projection}_s{sequence}_a8.raw"
            reference_path.write_bytes(reference.tobytes(order="C"))
            fixtures[f"s{sequence}"]["references"][projection] = {
                "path": reference_path.name,
                "bytes": reference_path.stat().st_size,
                "sha256": _sha256_bytes(reference),
                "code_min": int(reference.min()),
                "code_max": int(reference.max()),
            }

    metadata = {
        "contract": {
            "source": os.fspath(source.resolve()),
            "source_sha256": _sha256_file(source),
            "layer": 14,
            "head": 0,
            "activation": "asymmetric UInt8 input and output",
            "weight": "signed W4 LPBQ, G32, Int8 carrier",
            "input_shape": [1, "S", HIDDEN_SIZE],
            "output_shape": [1, "S", HEAD_DIM],
            "sequences": list(SEQUENCES),
        },
        "model": {
            "path": model_path.name,
            "bytes": model_path.stat().st_size,
            "sha256": _sha256_file(model_path),
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
    metadata = build(args.source, args.output_dir)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
