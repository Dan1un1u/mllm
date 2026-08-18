#!/usr/bin/env python3
"""Build a compact, exact layer-14 gate/up output-channel split artifact.

The source is the accepted RMSNorm-U8 W4A8G32 model. This tool copies the
reference 2048x6144 carriers and creates two 2048x3072 carriers by slicing only
the output-channel axis. It does not recalibrate, requantize, or inspect any
historical W4A8 artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path


MAGIC = 0x519A
VERSION = 2
HEADER = struct.Struct("<II512sIQ")
PARAM = struct.Struct("<IIQQQ16i256s")
DTYPE_INT8 = 16
DTYPE_FLOAT32 = 0
DTYPE_UINT8 = 129
K = 2048
O = 6144
HALF_O = O // 2
GROUP = 32
PROJECTIONS = ("gate_proj", "up_proj")


@dataclass(frozen=True)
class Descriptor:
    dtype: int
    size: int
    offset: int
    shape: tuple[int, ...]
    name: str


@dataclass(frozen=True)
class Blob:
    dtype: int
    shape: tuple[int, ...]
    name: str
    data: bytes


def read_descriptors(path: Path) -> tuple[bytes, dict[str, Descriptor]]:
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
        magic, version, model_name, count, descriptor_offset = HEADER.unpack(raw)
        if (magic, version, descriptor_offset) != (MAGIC, VERSION, HEADER.size):
            raise ValueError("unsupported mllm V2 container")
        stream.seek(descriptor_offset)
        descriptors: dict[str, Descriptor] = {}
        for _ in range(count):
            fields = PARAM.unpack(stream.read(PARAM.size))
            dtype, size, offset, rank = fields[1:5]
            shape = tuple(int(value) for value in fields[5 : 5 + rank])
            name = fields[21].split(b"\0", 1)[0].decode("utf-8")
            descriptors[name] = Descriptor(dtype, size, offset, shape, name)
    return model_name, descriptors


def read_bytes(path: Path, descriptor: Descriptor) -> bytes:
    with path.open("rb") as stream:
        stream.seek(descriptor.offset)
        data = stream.read(descriptor.size)
    if len(data) != descriptor.size:
        raise ValueError(f"truncated tensor: {descriptor.name}")
    return data


def source_blob(path: Path, descriptors: dict[str, Descriptor], name: str) -> Blob:
    descriptor = descriptors[name]
    return Blob(descriptor.dtype, descriptor.shape, name, read_bytes(path, descriptor))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_model(path: Path, model_name: bytes, blobs: list[Blob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    data_offset = HEADER.size + len(blobs) * PARAM.size
    with temporary.open("wb") as stream:
        stream.write(HEADER.pack(MAGIC, VERSION, model_name, len(blobs), HEADER.size))
        offset = data_offset
        for index, blob in enumerate(blobs):
            if len(blob.shape) > 16 or len(blob.name.encode("utf-8")) >= 256:
                raise ValueError(f"invalid tensor descriptor: {blob.name}")
            shape = blob.shape + (0,) * (16 - len(blob.shape))
            name = blob.name.encode("utf-8") + b"\0" * (256 - len(blob.name.encode("utf-8")))
            stream.write(PARAM.pack(index, blob.dtype, len(blob.data), offset, len(blob.shape), *shape, name))
            offset += len(blob.data)
        for blob in blobs:
            stream.write(blob.data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    model_name, descriptors = read_descriptors(args.source)
    blobs: list[Blob] = []
    report: dict[str, object] = {
        "source": str(args.source.resolve()),
        "source_size": args.source.stat().st_size,
        "contract": {"w4_signed_range": [-7, 7], "group_size": GROUP, "activation": "asymmetric UInt8"},
        "projections": {},
    }

    for projection in PROJECTIONS:
        prefix = f"model.layers.14.mlp.{projection}"
        weight_desc = descriptors[prefix + ".weight"]
        scale1_desc = descriptors[prefix + ".scale1"]
        scale2_desc = descriptors[prefix + ".scale2"]
        if weight_desc.dtype != DTYPE_INT8 or weight_desc.shape != (1, 1, K, O):
            raise ValueError(f"unexpected weight contract: {weight_desc}")
        if scale1_desc.dtype != DTYPE_UINT8 or scale2_desc.dtype != DTYPE_FLOAT32:
            raise ValueError(f"unexpected LPBQ scale dtype for {prefix}")

        weight = read_bytes(args.source, weight_desc)
        scale1 = read_bytes(args.source, scale1_desc)
        scale2 = read_bytes(args.source, scale2_desc)
        signed_carrier = array("b", weight)
        signed_min = min(signed_carrier)
        signed_max = max(signed_carrier)
        if signed_min >= 0 and signed_max <= 15:
            signed_min = min(value - 16 if value >= 8 else value for value in signed_carrier)
            signed_max = max(value - 16 if value >= 8 else value for value in signed_carrier)
        if signed_min < -7 or signed_max > 7:
            raise ValueError(f"W4 carrier outside [-7,7] for {prefix}")

        full_weight = weight
        full_scale1 = scale1
        full_scale2 = scale2
        blobs.extend(
            [
                Blob(DTYPE_INT8, (1, 1, K, O), prefix + ".weight", full_weight),
                Blob(DTYPE_UINT8, (O * (K // GROUP),), prefix + ".scale1", full_scale1),
                Blob(DTYPE_FLOAT32, (O,), prefix + ".scale2", full_scale2),
            ]
        )

        split_weights: list[bytes] = []
        split_scale1: list[bytes] = []
        split_scale2: list[bytes] = []
        halves = []
        for index, (start, stop) in enumerate(((0, HALF_O), (HALF_O, O))):
            split_prefix = prefix + f".oc{index}"
            weight_part = b"".join(weight[row * O + start : row * O + stop] for row in range(K))
            scale1_part = scale1[start * (K // GROUP) : stop * (K // GROUP)]
            scale2_part = scale2[start * 4 : stop * 4]
            split_weights.append(weight_part)
            split_scale1.append(scale1_part)
            split_scale2.append(scale2_part)
            blobs.extend(
                [
                    Blob(DTYPE_INT8, (1, 1, K, HALF_O), split_prefix + ".weight", weight_part),
                    Blob(DTYPE_UINT8, (HALF_O * (K // GROUP),), split_prefix + ".scale1", scale1_part),
                    Blob(DTYPE_FLOAT32, (HALF_O,), split_prefix + ".scale2", scale2_part),
                ]
            )
            halves.append(
                {
                    "name": split_prefix,
                    "output_channels": HALF_O,
                    "weight_sha256": digest(weight_part),
                    "scale1_sha256": digest(scale1_part),
                    "scale2_sha256": digest(scale2_part),
                }
            )

        reconstructed_weight = b"".join(
            split_weights[0][row * HALF_O : (row + 1) * HALF_O]
            + split_weights[1][row * HALF_O : (row + 1) * HALF_O]
            for row in range(K)
        )
        if reconstructed_weight != weight:
            raise AssertionError(f"weight reconstruction failed for {prefix}")
        if b"".join(split_scale1) != scale1:
            raise AssertionError(f"scale1 reconstruction failed for {prefix}")
        if b"".join(split_scale2) != scale2:
            raise AssertionError(f"scale2 reconstruction failed for {prefix}")
        report["projections"][projection] = {
            "full_weight_sha256": digest(full_weight),
            "full_scale1_sha256": digest(full_scale1),
            "full_scale2_sha256": digest(full_scale2),
            "signed_code_min": signed_min,
            "signed_code_max": signed_max,
            "split_reconstructs_full_exactly": True,
            "halves": halves,
        }

    qparam_names = (
        "model.layers.14.mlp.up_proj_input_qdq.fake_quant.scale",
        "model.layers.14.mlp.up_proj_input_qdq.fake_quant.zero_point",
        "model.layers.14.mlp.gate_proj_output_qdq.fake_quant.scale",
        "model.layers.14.mlp.gate_proj_output_qdq.fake_quant.zero_point",
        "model.layers.14.mlp.up_proj_output_qdq.fake_quant.scale",
        "model.layers.14.mlp.up_proj_output_qdq.fake_quant.zero_point",
    )
    blobs.extend(source_blob(args.source, descriptors, name) for name in qparam_names)
    write_model(args.output, model_name, blobs)
    report["output"] = str(args.output.resolve())
    report["output_size"] = args.output.stat().st_size
    report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report["status"] = "pass"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(args.output), "bytes": args.output.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
