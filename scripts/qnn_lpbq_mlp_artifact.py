#!/usr/bin/env python3
"""Audit and relayout Qwen3 MLP LPBQ weights in an mllm V2 container.

The accepted Conv2D carrier is HWIO [1, 1, K, O].  This tool preserves the
deployed signed W4 codes and both LPBQ scale levels while producing either
FullyConnected OI [O, K] or MatMul IO [K, O] carriers.  It never recalibrates
or requantizes a tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


MAGIC = 0x519A
VERSION = 2
HEADER = struct.Struct("<II512sIQ")
PARAM = struct.Struct("<IIQQQ16i256s")
DTYPE_INT8 = 16
DTYPE_FLOAT32 = 0
DTYPE_UINT8 = 129
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8")


@dataclass(frozen=True)
class Descriptor:
    param_id: int
    dtype_id: int
    size: int
    offset: int
    shape: tuple[int, ...]
    name: str
    descriptor_offset: int
    raw_name: bytes


def read_descriptors(path: Path) -> dict[str, Descriptor]:
    result: dict[str, Descriptor] = {}
    with path.open("rb") as stream:
        raw_header = stream.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise ValueError("truncated V2 header")
        magic, version, _model_raw, num_params, desc_offset = HEADER.unpack(raw_header)
        if magic != MAGIC or version != VERSION or desc_offset != HEADER.size:
            raise ValueError(
                f"unexpected container header: magic={magic:#x}, version={version}, desc_offset={desc_offset}"
            )
        for expected_id in range(num_params):
            descriptor_offset = stream.tell()
            raw = stream.read(PARAM.size)
            if len(raw) != PARAM.size:
                raise ValueError(f"truncated descriptor {expected_id}")
            fields = PARAM.unpack(raw)
            param_id, dtype_id, size, offset, rank = fields[:5]
            shape_storage = fields[5:21]
            raw_name = fields[21]
            name = _cstring(raw_name)
            if param_id != expected_id:
                raise ValueError(f"descriptor id mismatch: {param_id} != {expected_id}")
            if not 0 <= rank <= 16:
                raise ValueError(f"invalid rank for {name}: {rank}")
            shape = tuple(int(value) for value in shape_storage[:rank])
            if name in result:
                raise ValueError(f"duplicate tensor: {name}")
            result[name] = Descriptor(
                param_id=param_id,
                dtype_id=dtype_id,
                size=size,
                offset=offset,
                shape=shape,
                name=name,
                descriptor_offset=descriptor_offset,
                raw_name=raw_name,
            )
    return result


def _array(path: Path, descriptor: Descriptor, dtype: np.dtype) -> np.memmap:
    expected = int(np.prod(descriptor.shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if descriptor.size != expected:
        raise ValueError(
            f"payload size mismatch for {descriptor.name}: descriptor={descriptor.size}, expected={expected}"
        )
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=descriptor.offset,
        shape=descriptor.shape,
        order="C",
    )


def _sha256_bytes(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _projection_prefix(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.{projection}"


def _canonical_projection(path: Path, descriptors: dict[str, Descriptor], prefix: str) -> dict[str, object]:
    weight_desc = descriptors[prefix + ".weight"]
    scale1_desc = descriptors[prefix + ".scale1"]
    scale2_desc = descriptors[prefix + ".scale2"]
    if weight_desc.dtype_id != DTYPE_INT8:
        raise ValueError(f"{weight_desc.name} is not an Int8 W4 carrier")
    if scale1_desc.dtype_id != DTYPE_UINT8 or scale2_desc.dtype_id != DTYPE_FLOAT32:
        raise ValueError(f"unexpected LPBQ scale dtypes for {prefix}")
    if len(weight_desc.shape) != 4 or weight_desc.shape[:2] != (1, 1):
        raise ValueError(f"expected Conv HWIO weight for {prefix}, got {weight_desc.shape}")

    k, o = weight_desc.shape[2:]
    if k % 32:
        raise ValueError(f"K is not divisible by G32 for {prefix}: {k}")
    weight_hwio = _array(path, weight_desc, np.int8)
    codes_io = np.asarray(weight_hwio).reshape(k, o)
    codes_oi = np.ascontiguousarray(codes_io.T)
    signed_oi = np.where(codes_oi >= 8, codes_oi.astype(np.int16) - 16, codes_oi).astype(np.int8)
    if int(signed_oi.min()) < -7 or int(signed_oi.max()) > 7:
        raise ValueError(
            f"signed W4 code outside [-7,7] for {prefix}: [{int(signed_oi.min())},{int(signed_oi.max())}]"
        )

    scale1 = np.asarray(_array(path, scale1_desc, np.uint8)).reshape(o, k // 32)
    scale2 = np.asarray(_array(path, scale2_desc, np.float32)).reshape(o)
    if np.any(scale1 < 1) or np.any(scale1 > 16):
        raise ValueError(f"UInt4 block scale outside [1,16] for {prefix}")
    if not np.all(np.isfinite(scale2)) or np.any(scale2 <= 0):
        raise ValueError(f"invalid level-2 scale for {prefix}")

    fc_roundtrip = np.ascontiguousarray(codes_oi).reshape(o, k)
    matmul_roundtrip = np.ascontiguousarray(codes_io).reshape(k, o).T
    if not np.array_equal(fc_roundtrip, codes_oi):
        raise AssertionError(f"FC OI roundtrip failed for {prefix}")
    if not np.array_equal(matmul_roundtrip, codes_oi):
        raise AssertionError(f"MatMul IO roundtrip failed for {prefix}")

    return {
        "prefix": prefix,
        "k": k,
        "o": o,
        "groups_per_channel": k // 32,
        "signed_code_min": int(signed_oi.min()),
        "signed_code_max": int(signed_oi.max()),
        "canonical_signed_oi_sha256": _sha256_bytes(signed_oi),
        "conv_hwio_carrier_sha256": _sha256_bytes(codes_io),
        "fc_oi_carrier_sha256": _sha256_bytes(codes_oi),
        "matmul_io_carrier_sha256": _sha256_bytes(codes_io),
        "scale1_sha256": _sha256_bytes(scale1),
        "scale2_sha256": _sha256_bytes(scale2),
        "scale1_min": int(scale1.min()),
        "scale1_max": int(scale1.max()),
        "scale2_min": float(scale2.min()),
        "scale2_max": float(scale2.max()),
    }


def audit(path: Path, layers: Iterable[int]) -> dict[str, object]:
    descriptors = read_descriptors(path)
    projections = [
        _canonical_projection(path, descriptors, _projection_prefix(layer, projection))
        for layer in layers
        for projection in PROJECTIONS
    ]
    return {
        "source": os.fspath(path.resolve()),
        "source_size": path.stat().st_size,
        "layers": list(layers),
        "projection_count": len(projections),
        "contract": {
            "w4_range": [-7, 7],
            "group_size": 32,
            "scale1_dtype": "UInt8",
            "scale2_dtype": "Float32",
        },
        "projections": projections,
    }


def _patch_descriptor(stream, descriptor: Descriptor, shape: tuple[int, ...]) -> None:
    if len(shape) > 16 or int(np.prod(shape, dtype=np.int64)) != int(np.prod(descriptor.shape, dtype=np.int64)):
        raise ValueError(f"invalid relayout shape for {descriptor.name}: {shape}")
    shape_storage = tuple(shape) + (0,) * (16 - len(shape))
    packed = PARAM.pack(
        descriptor.param_id,
        descriptor.dtype_id,
        descriptor.size,
        descriptor.offset,
        len(shape),
        *shape_storage,
        descriptor.raw_name,
    )
    stream.seek(descriptor.descriptor_offset)
    stream.write(packed)


def transform(source: Path, output: Path, layout: str, layers: Iterable[int]) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if layout not in {"fc", "matmul"}:
        raise ValueError(f"unsupported layout: {layout}")

    layers = list(layers)
    source_descriptors = read_descriptors(source)
    # Validate every selected source tensor before creating a multi-gigabyte copy.
    source_audit = audit(source, layers)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()

    try:
        shutil.copyfile(source, temporary)
        with temporary.open("r+b", buffering=0) as stream:
            for layer in layers:
                for projection in PROJECTIONS:
                    prefix = _projection_prefix(layer, projection)
                    descriptor = source_descriptors[prefix + ".weight"]
                    k, o = descriptor.shape[2:]
                    codes_io = np.asarray(_array(source, descriptor, np.int8)).reshape(k, o)
                    if layout == "fc":
                        carrier = np.ascontiguousarray(codes_io.T)
                        new_shape = (o, k)
                    else:
                        carrier = np.ascontiguousarray(codes_io)
                        new_shape = (k, o)
                    stream.seek(descriptor.offset)
                    stream.write(memoryview(carrier).cast("B"))
                    _patch_descriptor(stream, descriptor, new_shape)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    output_descriptors = read_descriptors(output)
    transformed: list[dict[str, object]] = []
    for layer in layers:
        for projection in PROJECTIONS:
            prefix = _projection_prefix(layer, projection)
            descriptor = output_descriptors[prefix + ".weight"]
            expected_shape = (
                (next(item["o"] for item in source_audit["projections"] if item["prefix"] == prefix),
                 next(item["k"] for item in source_audit["projections"] if item["prefix"] == prefix))
                if layout == "fc"
                else (next(item["k"] for item in source_audit["projections"] if item["prefix"] == prefix),
                      next(item["o"] for item in source_audit["projections"] if item["prefix"] == prefix))
            )
            if descriptor.shape != expected_shape:
                raise AssertionError(f"relayout shape mismatch for {prefix}: {descriptor.shape} != {expected_shape}")
            carrier = np.asarray(_array(output, descriptor, np.int8))
            canonical = np.ascontiguousarray(carrier if layout == "fc" else carrier.T)
            signed = np.where(canonical >= 8, canonical.astype(np.int16) - 16, canonical).astype(np.int8)
            expected_hash = next(
                item["canonical_signed_oi_sha256"] for item in source_audit["projections"] if item["prefix"] == prefix
            )
            actual_hash = _sha256_bytes(signed)
            if actual_hash != expected_hash:
                raise AssertionError(f"canonical W4 digest mismatch for {prefix}")
            transformed.append(
                {
                    "prefix": prefix,
                    "shape": list(descriptor.shape),
                    "canonical_signed_oi_sha256": actual_hash,
                }
            )

    return {
        "source": os.fspath(source.resolve()),
        "output": os.fspath(output.resolve()),
        "layout": layout,
        "layers": layers,
        "source_size": source.stat().st_size,
        "output_size": output.stat().st_size,
        "projection_count": len(transformed),
        "transformed": transformed,
    }


def _parse_layers(value: str) -> list[int]:
    if value == "all":
        return list(range(28))
    result = sorted({int(item) for item in value.split(",")})
    if not result or result[0] < 0 or result[-1] >= 28:
        raise argparse.ArgumentTypeError("layers must be 'all' or comma-separated indices in [0,27]")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("model", type=Path)
    audit_parser.add_argument("--layers", type=_parse_layers, default=_parse_layers("14"))
    audit_parser.add_argument("--output-json", type=Path)

    transform_parser = subparsers.add_parser("transform")
    transform_parser.add_argument("source", type=Path)
    transform_parser.add_argument("output", type=Path)
    transform_parser.add_argument("--layout", choices=("fc", "matmul"), required=True)
    transform_parser.add_argument("--layers", type=_parse_layers, default=_parse_layers("14"))
    transform_parser.add_argument("--output-json", type=Path)

    args = parser.parse_args()
    if args.command == "audit":
        summary = audit(args.model, args.layers)
    else:
        summary = transform(args.source, args.output, args.layout, args.layers)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
