#!/usr/bin/env python3
"""Merge learned per-layer LPBQ scales into an authoritative G32 base file.

The base checkpoint owns the packed HWIO INT4 codes.  A streaming training
directory owns only ``scale1`` and ``scale2`` for each decoder projection.  The
merge intentionally never re-quantizes BF16 teacher weights, which keeps the
VM/QNN artifact byte-aligned with ``Qwen3-1.7B-G32-base/model.safetensors``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


PROJECTIONS = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--scale-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    if not args.scale_dir.is_dir():
        raise FileNotFoundError(args.scale_dir)
    training_manifest = args.training_manifest or (args.scale_dir / "streaming-train.json")
    if not training_manifest.is_file():
        raise FileNotFoundError(training_manifest)
    report = json.loads(training_manifest.read_text(encoding="utf-8"))
    expected_base_sha = report.get("base_quant_checkpoint_sha256")
    base_sha = sha256_file(args.base_checkpoint)
    if expected_base_sha and expected_base_sha != base_sha:
        raise ValueError(
            "base checkpoint SHA mismatch: "
            f"manifest={expected_base_sha} actual={base_sha}"
        )

    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}
    with safe_open(str(args.base_checkpoint), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key).contiguous()

    rows = {int(row["layer"]): row for row in report.get("rows", [])}
    if set(rows) != set(range(28)):
        raise ValueError("training manifest must contain all 28 decoder layers")
    checked = 0
    for layer_index in range(28):
        row = rows[layer_index]
        scale_path = args.scale_dir / row["lpbq_scales_file"]
        if not scale_path.is_file():
            raise FileNotFoundError(scale_path)
        with safe_open(str(scale_path), framework="pt", device="cpu") as scale_file:
            for projection, module_path in PROJECTIONS.items():
                prefix = f"model.layers.{layer_index}.{module_path}"
                weight_key = prefix + ".weight"
                scale1_key = prefix + ".scale1"
                scale2_key = prefix + ".scale2"
                if weight_key not in tensors or scale1_key not in tensors or scale2_key not in tensors:
                    raise KeyError(f"base checkpoint is missing {prefix} LPBQ tensors")
                expected = row["lpbq_scale_metadata"][projection]
                actual_packed_sha = sha256_tensor(tensors[weight_key])
                if actual_packed_sha != expected["packed_codes_sha256"]:
                    raise ValueError(
                        f"{prefix}: packed code SHA mismatch: "
                        f"manifest={expected['packed_codes_sha256']} actual={actual_packed_sha}"
                    )
                learned_scale1 = scale_file.get_tensor(f"{projection}.scale1").contiguous()
                learned_scale2 = scale_file.get_tensor(f"{projection}.scale2").contiguous()
                base_scale1 = tensors[scale1_key]
                base_scale2 = tensors[scale2_key]
                if learned_scale1.numel() != base_scale1.numel():
                    raise ValueError(
                        f"{prefix}: scale1 elements {learned_scale1.numel()} != "
                        f"base {base_scale1.numel()}"
                    )
                if tuple(learned_scale2.shape) != tuple(base_scale2.shape):
                    raise ValueError(
                        f"{prefix}: scale2 shape {tuple(learned_scale2.shape)} != "
                        f"base {tuple(base_scale2.shape)}"
                    )
                tensors[scale1_key] = learned_scale1.reshape_as(base_scale1).to(torch.uint8)
                tensors[scale2_key] = learned_scale2.reshape_as(base_scale2).to(torch.float32)
                checked += 1

    metadata.update(
        {
            "mllm.lpbq.codes_source": str(args.base_checkpoint),
            "mllm.lpbq.scales_source": str(args.scale_dir),
            "mllm.lpbq.codes_sha256_domain": "packed HWIO carrier bytes",
            "mllm.lpbq.scale_training_manifest": str(training_manifest),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata=metadata)
    merge_manifest = {
        "schema_version": 1,
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": base_sha,
        "scale_dir": str(args.scale_dir),
        "training_manifest": str(training_manifest),
        "output": str(args.output),
        "replaced_scale_pairs": checked,
        "codes_unchanged": True,
        "contract": {
            "group_size": 32,
            "weight_layout": "HWIO [1,1,K,O]",
            "scale1_layout": "flattened [O,K/G] row-major",
            "scale2_layout": "[O] FP32",
        },
    }
    report_path = args.output.with_name("merge_manifest.json")
    report_path.write_text(json.dumps(merge_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {report_path}")
    print(f"checked {checked} LPBQ scale pairs; codes_unchanged=true")


if __name__ == "__main__":
    main()
