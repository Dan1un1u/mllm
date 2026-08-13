#!/usr/bin/env python3
"""Validate an mllm V2 container without loading tensor payloads."""

from __future__ import annotations

import argparse
import collections
import json
import os
import struct
from pathlib import Path


MAGIC = 0x519A
VERSION = 2
HEADER = struct.Struct("<II512sIQ")
PARAM = struct.Struct("<IIQQQ16i256s")
DTYPES = {
    0: "Float32",
    1: "Float16",
    16: "Int8",
    17: "Int16",
    18: "Int32",
    128: "BFloat16",
    129: "UInt8",
    130: "UInt16",
    131: "UInt32",
    132: "Int64",
    133: "UInt64",
    134: "Byte",
    135: "MXFP4",
}


def _cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8")


def inspect(path: Path) -> dict:
    file_size = path.stat().st_size
    dtype_counts: collections.Counter[str] = collections.Counter()
    suffix_counts: collections.Counter[str] = collections.Counter()
    names: set[str] = set()
    ranges: list[tuple[int, int, str]] = []
    activation_zero_points: list[tuple[int, int, str]] = []

    with path.open("rb") as stream:
        raw_header = stream.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise ValueError("truncated V2 header")
        magic, version, model_raw, num_params, desc_offset = HEADER.unpack(raw_header)
        if magic != MAGIC or version != VERSION:
            raise ValueError(
                f"unexpected container signature: magic={magic:#x}, version={version}"
            )
        if desc_offset != HEADER.size:
            raise ValueError(f"unexpected descriptor offset: {desc_offset}")

        stream.seek(desc_offset)
        for expected_id in range(num_params):
            raw = stream.read(PARAM.size)
            if len(raw) != PARAM.size:
                raise ValueError(f"truncated descriptor {expected_id}")
            fields = PARAM.unpack(raw)
            param_id, dtype_id, size, offset, rank = fields[:5]
            shape = fields[5:21]
            name = _cstring(fields[21])
            if param_id != expected_id:
                raise ValueError(f"descriptor id mismatch: {param_id} != {expected_id}")
            if not name or name in names:
                raise ValueError(f"empty or duplicate tensor name at descriptor {param_id}: {name!r}")
            if rank > 16 or any(dim <= 0 for dim in shape[:rank]):
                raise ValueError(f"invalid shape for {name}: rank={rank}, shape={shape[:rank]}")
            if offset + size > file_size:
                raise ValueError(f"tensor payload exceeds file size: {name}")
            names.add(name)
            ranges.append((offset, offset + size, name))
            if (
                name.endswith(".fake_quant.zero_point")
                and ".weight_fake_quant." not in name
                and dtype_id == 18
                and size == 4
            ):
                return_position = stream.tell()
                stream.seek(offset)
                activation_zero_points.append((struct.unpack("<i", stream.read(4))[0], param_id, name))
                stream.seek(return_position)
            dtype_counts[DTYPES.get(dtype_id, f"Unknown({dtype_id})")] += 1
            suffix_counts[name.rsplit(".", 1)[-1]] += 1

    descriptor_end = desc_offset + num_params * PARAM.size
    previous_end = descriptor_end
    for start, end, name in sorted(ranges):
        if start < descriptor_end:
            raise ValueError(f"tensor payload overlaps descriptors: {name}")
        if start < previous_end:
            raise ValueError(f"tensor payload overlaps previous tensor: {name}")
        previous_end = end

    required_suffixes = {"weight", "scale1", "scale2", "scale", "zero_point"}
    missing_suffixes = sorted(required_suffixes - suffix_counts.keys())
    if missing_suffixes:
        raise ValueError(f"missing expected quantized tensor suffixes: {missing_suffixes}")
    if dtype_counts["UInt8"] == 0 or dtype_counts["UInt16"] == 0:
        raise ValueError(
            "expected both UInt8 LPBQ/activation state and preserved UInt16 non-target state"
        )
    if suffix_counts["scale1"] != 197 or suffix_counts["scale2"] != 197:
        raise ValueError(
            "expected exactly 197 W4G32 Linear/lm_head scale1 and scale2 tensors; "
            f"got scale1={suffix_counts['scale1']}, scale2={suffix_counts['scale2']}"
        )
    invalid_zero_points = [item for item in activation_zero_points if not 0 <= item[0] <= 255]
    if invalid_zero_points:
        value, _, name = invalid_zero_points[0]
        raise ValueError(f"activation zero-point outside UInt8 asymmetric range: {name}={value}")

    return {
        "path": os.fspath(path.resolve()),
        "model_name": _cstring(model_raw),
        "file_size_bytes": file_size,
        "num_params": num_params,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "activation_zero_point_count": len(activation_zero_points),
        "activation_zero_point_min": min(value for value, _, _ in activation_zero_points),
        "activation_zero_point_max": max(value for value, _, _ in activation_zero_points),
        "payload_end": previous_end,
        "trailing_bytes": file_size - previous_end,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    summary = inspect(args.model)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
