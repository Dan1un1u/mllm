#!/usr/bin/env python3
"""Verify logical and packed LPBQ code hashes against a base checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


PROJECTIONS = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with safe_open(str(args.base_checkpoint), framework="pt", device="cpu") as base:
        for training_dir in args.training_dir:
            manifest = json.loads((training_dir / "streaming-train.json").read_text())
            rows = {int(row["layer"]): row for row in manifest["rows"]}
            packed_mismatch = 0
            logical_mismatch = 0
            for layer_index in range(28):
                row = rows[layer_index]
                for projection, module_path in PROJECTIONS.items():
                    weight = base.get_tensor(
                        f"model.layers.{layer_index}.{module_path}.weight"
                    )
                    metadata = row["lpbq_scale_metadata"][projection]
                    packed_mismatch += int(
                        tensor_sha256(weight) != metadata["packed_codes_sha256"]
                    )
                    carrier = weight.reshape(
                        int(weight.shape[2]), int(weight.shape[3])
                    ).to(torch.int16)
                    nibble = carrier & 0x0F
                    codes = torch.where(nibble >= 8, nibble - 16, nibble)
                    codes = codes.transpose(0, 1).contiguous().to(torch.int8)
                    logical_mismatch += int(
                        tensor_sha256(codes) != metadata["codes_sha256"]
                    )
            print(
                json.dumps(
                    {
                        "training_dir": str(training_dir),
                        "complete": manifest["complete"],
                        "packed_mismatch": packed_mismatch,
                        "logical_mismatch": logical_mismatch,
                        "final_logits": manifest["final_eval"]["last_token_logits"],
                    },
                    sort_keys=True,
                )
            )
            if packed_mismatch or logical_mismatch:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
