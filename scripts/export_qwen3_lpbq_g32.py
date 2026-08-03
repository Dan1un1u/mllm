#!/usr/bin/env python3
"""Export a complete Qwen3 LPBQ checkpoint with a true G32 contract.

The Qualcomm Python quantizer currently runs on CUDA and the VM used for
offline export does not expose the host GPU.  This exporter therefore keeps
the already calibrated A16/QDQ tensors from an existing checkpoint and
re-encodes only the Linear/lm_head weights from the original BF16 shards.

The emitted LPBQ tensors follow the same public contract as
``QLinearLPBQ.convert_to_conv2d_deploy_hwio``:

* signed symmetric INT4 codes in an INT8 carrier, packed as ``code & 0xf``;
* HWIO weight layout ``[1, 1, K, O]``;
* one UInt4 level-1 scale per ``(output, G32)`` group, flattened in row-major
  ``[O, K / 32]`` order;
* one FP32 level-2 scale per output channel.

This is intentionally a no-rotation export.  A rotated candidate needs a
fresh full-model activation calibration, so it is kept as the next artifact
instead of silently reusing non-rotated QDQ ranges.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open


LINEAR_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-model",
        type=Path,
        required=True,
        help="Original Qwen3 directory containing safetensors shards.",
    )
    parser.add_argument(
        "--base-quant-checkpoint",
        type=Path,
        required=True,
        help="Existing quantized model.safetensors with calibrated A16/QDQ tensors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the new model.safetensors and manifest.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=32,
        choices=(32,),
        help="LPBQ group size. Only the physical G32 contract is exported.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output directory.",
    )
    return parser.parse_args()


def _source_files(source_model: Path) -> list[Path]:
    files = sorted(source_model.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors shards found under {source_model}")
    return files


def _open_source_index(source_files: list[Path]) -> tuple[dict[str, int], list[set[str]]]:
    key_to_file: dict[str, int] = {}
    keys_by_file: list[set[str]] = []
    for file_index, path in enumerate(source_files):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        keys_by_file.append(keys)
        for key in keys:
            if key in key_to_file:
                raise ValueError(f"duplicate source tensor {key!r}")
            key_to_file[key] = file_index
    return key_to_file, keys_by_file


def _read_source_tensor(
    source_files: list[Path],
    key_to_file: dict[str, int],
    key: str,
) -> torch.Tensor:
    try:
        source_path = source_files[key_to_file[key]]
    except KeyError as exc:
        raise KeyError(f"source model has no tensor {key!r}") from exc
    with safe_open(str(source_path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _is_linear_weight(key: str) -> bool:
    return key == "lm_head.weight" or key.endswith(LINEAR_SUFFIXES)


def _quantize_lpbq_g32(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return HWIO uint4-carrier weight, scale1 and scale2."""

    if weight.ndim != 2:
        raise ValueError(f"expected OI matrix, got shape {tuple(weight.shape)}")
    out_features, in_features = map(int, weight.shape)
    if in_features % group_size:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={group_size}"
        )

    # The production quantizer constructs a float32 model from the BF16
    # checkpoint before observing parameters.  Match that dtype here so the
    # qparams and tie-breaking are reproducible.
    w = weight.detach().to(dtype=torch.float32)
    groups = w.reshape(out_features, in_features // group_size, group_size)
    eps = 0.0001 / 65535.0
    scale = torch.maximum(groups.amin(dim=-1).abs(), groups.amax(dim=-1)).div(7.0)
    scale = scale.clamp_min(eps)

    # torchao is the implementation used by QLinearLPBQ.  Keep a small
    # fallback for environments where torchao is not importable.
    try:
        from torchao.quantization.quant_primitives import _quantize_affine

        codes = _quantize_affine(
            w,
            [1, group_size],
            scale,
            torch.zeros_like(scale),
            torch.int32,
            quant_min=-7,
            quant_max=7,
        ).to(torch.int8)
    except (ImportError, RuntimeError):
        codes = torch.round(groups / scale.unsqueeze(-1)).clamp(-7, 7).to(torch.int8)

    # QNN's LPBQ level-1 scale is UInt4, with level-2 one scalar per output
    # channel.  Preserve the channel-major group order used by the runtime.
    scale2 = scale.amax(dim=1).div(16.0).to(torch.float32)
    scale1 = torch.round(scale / scale2.unsqueeze(1)).clamp(1, 16).to(torch.uint8)

    # Signed INT4 values are carried in int8 but must be explicitly masked to
    # their low nibble before entering QNN.  Transpose OI -> IO for HWIO.
    packed = torch.bitwise_and(codes.reshape(out_features, in_features).transpose(0, 1).contiguous(), 0x0F)
    packed = packed.reshape(1, 1, in_features, out_features).contiguous()

    # Decode the exact deployed representation for an artifact-local check.
    signed = packed.reshape(in_features, out_features).transpose(0, 1).to(torch.int16)
    signed = torch.where(signed >= 8, signed - 16, signed)
    decoded = signed.reshape(out_features, in_features // group_size, group_size).float()
    decoded = decoded * scale1.float().unsqueeze(-1) * scale2[:, None, None]
    original = groups
    error = decoded - original
    stats = {
        "weight_nmse": float(error.square().sum() / original.square().sum().clamp_min(1e-20)),
        "max_abs_error": float(error.abs().max()),
        "mean_abs_error": float(error.abs().mean()),
    }
    return packed, scale1.flatten().contiguous(), scale2.contiguous(), stats


def _load_base_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] | None = None
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key).contiguous()
    return tensors, metadata


def main() -> None:
    args = _parse_args()
    if args.group_size != 32:
        raise AssertionError("only G32 is supported by this exporter")
    base_path = args.base_quant_checkpoint
    if not base_path.is_file():
        raise FileNotFoundError(base_path)

    output_dir = args.output_dir
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing and not args.overwrite:
            raise FileExistsError(
                f"output directory is non-empty: {output_dir}; pass --overwrite"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    source_files = _source_files(args.source_model)
    key_to_file, _ = _open_source_index(source_files)
    tensors, base_metadata = _load_base_checkpoint(base_path)

    base_linear_keys = sorted(key for key in tensors if _is_linear_weight(key))
    if len(base_linear_keys) != 197:
        raise ValueError(
            f"expected 197 Qwen3 Linear/lm_head weights, found {len(base_linear_keys)}"
        )

    reports: dict[str, dict[str, object]] = {}
    for index, key in enumerate(base_linear_keys, start=1):
        source = _read_source_tensor(source_files, key_to_file, key)
        packed, scale1, scale2, stats = _quantize_lpbq_g32(source, args.group_size)
        tensors[key] = packed
        prefix = key[: -len(".weight")]
        tensors[prefix + ".scale1"] = scale1
        tensors[prefix + ".scale2"] = scale2
        reports[key] = {
            "source_dtype": str(source.dtype),
            "source_shape": list(source.shape),
            "weight_shape": list(packed.shape),
            "scale1_shape": list(scale1.shape),
            "scale2_shape": list(scale2.shape),
            **stats,
        }
        if index == 1 or index % 25 == 0 or index == len(base_linear_keys):
            print(f"[{index:3d}/{len(base_linear_keys)}] {key}", flush=True)

    output_path = output_dir / "model.safetensors"
    metadata = dict(base_metadata or {})
    metadata.update(
        {
            "mllm.quantization": "LPBQ W4A16, G32",
            "mllm.lpbq.group_size": "32",
            "mllm.lpbq.weight_layout": "HWIO [1,1,K,O], signed int4 in uint4 carrier",
            "mllm.lpbq.rotation": "none",
            "mllm.lpbq.source_checkpoint": str(base_path),
        }
    )
    save_file(tensors, str(output_path), metadata=metadata)

    manifest = {
        "schema_version": 1,
        "model": "Qwen3-1.7B",
        "source_model": str(args.source_model),
        "base_quant_checkpoint": str(base_path),
        "output": str(output_path),
        "group_size": args.group_size,
        "precision": "W4A16",
        "rotation": "none",
        "linear_weight_count": len(base_linear_keys),
        "contract": {
            "weight_dtype": "int8_carrier_uint4",
            "weight_layout": "HWIO [1,1,K,O]",
            "scale1_dtype": "uint8",
            "scale1_layout": "flattened [O,K/G] row-major",
            "scale2_dtype": "float32",
            "scale2_layout": "[O]",
            "signed_code_range": [-7, 7],
        },
        "weights": reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path} ({output_path.stat().st_size} bytes)")
    print(f"wrote {output_dir / 'export_manifest.json'}")


if __name__ == "__main__":
    main()
