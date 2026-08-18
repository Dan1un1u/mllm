#!/usr/bin/env python3
"""Generate deterministic native-U8 S=1 gate/up projection inputs."""

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path


ELEMENTS = 2048
HEADER = struct.Struct("<II512sIQ")
PARAM = struct.Struct("<IIQQQ16i256s")
ZP_NAME = "model.layers.14.mlp.up_proj_input_qdq.fake_quant.zero_point"
SCALE_NAME = "model.layers.14.mlp.up_proj_input_qdq.fake_quant.scale"


def scalar(model: Path, name: str, fmt: str):
    with model.open("rb") as stream:
        _magic, _version, _model, count, offset = HEADER.unpack(stream.read(HEADER.size))
        stream.seek(offset)
        for _ in range(count):
            fields = PARAM.unpack(stream.read(PARAM.size))
            tensor_name = fields[21].split(b"\0", 1)[0].decode("utf-8")
            if tensor_name == name:
                stream.seek(fields[3])
                return struct.unpack(fmt, stream.read(struct.calcsize(fmt)))[0]
    raise KeyError(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    zero_point = scalar(args.model, ZP_NAME, "<i")
    scale = scalar(args.model, SCALE_NAME, "<f")
    if not 0 <= zero_point <= 255 or not scale > 0:
        raise ValueError((scale, zero_point))
    rng = random.Random(0x61442048)
    fixtures = {
        "encoded_zero": bytes([zero_point]) * ELEMENTS,
        "qmin": bytes(ELEMENTS),
        "qmax": bytes([255]) * ELEMENTS,
        "alternating": bytes([0, 255]) * (ELEMENTS // 2),
        "ramp": bytes(range(256)) * (ELEMENTS // 256),
        "seeded_random": rng.randbytes(ELEMENTS),
    }
    manifest = {
        "shape": [1, 1, 2048],
        "dtype": "uint8",
        "input_scale": scale,
        "input_zero_point": zero_point,
        "fixtures": {},
    }
    for name, values in fixtures.items():
        path = args.output_dir / f"{name}.bin"
        path.write_bytes(values)
        manifest["fixtures"][name] = {
            "bytes": len(values),
            "sha256": hashlib.sha256(values).hexdigest(),
        }
    (args.output_dir / "fixtures.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
