#!/usr/bin/env python3
"""Generate deterministic native-U8 S=32 down-projection inputs."""

import argparse
import hashlib
import json
import random
from pathlib import Path


ELEMENTS = 32 * 6144
INPUT_ZERO_POINT = 109


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(0x5734A8)
    fixtures = {
        "encoded_zero": bytes([INPUT_ZERO_POINT]) * ELEMENTS,
        "qmin": bytes(ELEMENTS),
        "qmax": bytes([255]) * ELEMENTS,
        "alternating": bytes([0, 255]) * (ELEMENTS // 2),
        "ramp": bytes(range(256)) * (ELEMENTS // 256),
        "seeded_random": rng.randbytes(ELEMENTS),
    }
    manifest = {
        "shape": [1, 32, 6144],
        "dtype": "uint8",
        "input_scale": 0.4109087586402893,
        "input_zero_point": INPUT_ZERO_POINT,
        "fixtures": {},
    }
    for name, values in fixtures.items():
        path = args.output_dir / f"{name}.bin"
        path.write_bytes(values)
        manifest["fixtures"][name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.output_dir / "fixtures.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
