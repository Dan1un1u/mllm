#!/usr/bin/env python3
"""Convert one safetensors checkpoint to the raw mllm ModelFileV2 format."""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import torch
from safetensors import safe_open


MAGIC = 0x519A
VERSION = 2
MODEL_NAME_BYTES = 512
PARAM_NAME_BYTES = 256
MAX_RANK = 16
HEADER_SIZE = 532
PARAM_DESCRIPTOR_SIZE = 352

DTYPE_IDS = {
    torch.float32: 0,
    torch.float16: 1,
    torch.int8: 16,
    torch.int16: 17,
    torch.int32: 18,
    torch.bfloat16: 128,
    torch.uint8: 129,
    torch.uint16: 130,
    torch.int64: 132,
    torch.bool: 129,
}


def _fixed_utf8(value: str, width: int) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) > width:
        raise ValueError(f"UTF-8 value is longer than {width} bytes: {value!r}")
    return encoded.ljust(width, b"\0")


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.ndim == 0:
        value = value.reshape(1)
    return value.view(torch.uint8).numpy().tobytes()


def convert(input_path: Path, output_path: Path, model_name: str = "") -> None:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")

    with safe_open(str(input_path), framework="pt", device="cpu") as source:
        keys = list(source.keys())
        descriptors: list[tuple[int, int, int, int, list[int], str]] = []

        with temporary.open("wb+") as output:
            output.write(b"\0" * (HEADER_SIZE + len(keys) * PARAM_DESCRIPTOR_SIZE))
            for param_id, name in enumerate(keys):
                tensor = source.get_tensor(name)
                try:
                    dtype_id = DTYPE_IDS[tensor.dtype]
                except KeyError as exc:
                    raise TypeError(f"unsupported dtype {tensor.dtype} for {name}") from exc
                shape = list(tensor.shape)
                if len(shape) > MAX_RANK:
                    raise ValueError(f"rank {len(shape)} exceeds ModelFileV2 limit for {name}")
                data = _tensor_bytes(tensor)
                offset = output.tell()
                output.write(data)
                descriptors.append((param_id, dtype_id, len(data), offset, shape, name))

            output.seek(0)
            output.write(
                struct.pack(
                    f"<II{MODEL_NAME_BYTES}sIQ",
                    MAGIC,
                    VERSION,
                    _fixed_utf8(model_name, MODEL_NAME_BYTES),
                    len(descriptors),
                    HEADER_SIZE,
                )
            )
            for param_id, dtype_id, size, offset, shape, name in descriptors:
                shape_padded = shape + [0] * (MAX_RANK - len(shape))
                output.write(
                    struct.pack(
                        f"<IIQQQ{MAX_RANK}i{PARAM_NAME_BYTES}s",
                        param_id,
                        dtype_id,
                        size,
                        offset,
                        len(shape),
                        *shape_padded,
                        _fixed_utf8(name, PARAM_NAME_BYTES),
                    )
                )
            output.flush()
            os.fsync(output.fileno())

    os.replace(temporary, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        default="",
        help="Optional ModelFileV2 header name; upstream raw converter leaves it empty.",
    )
    args = parser.parse_args()
    convert(args.input, args.output, args.model_name)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
