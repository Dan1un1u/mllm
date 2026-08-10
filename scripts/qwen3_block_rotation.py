#!/usr/bin/env python3
"""Export standalone Qwen3 Layer 5 R1/R2/R3 LPBQ-G32 artifacts.

The exporter intentionally keeps the calibrated activation and KV-cache QDQ
parameters from an existing G32 checkpoint.  Only static Layer 5 parameters
are changed:

* A: the original quantized Layer 5 parameters;
* B: RMSNorm gamma folding with identity R1/R2;
* C: RMSNorm gamma folding with normalized Sylvester-Hadamard R1/R2;
* D-Dense: C plus an online post-RoPE R3 dense MatMul;
* D-FWHT-Graph: C plus an online post-RoPE seven-stage R3 FWHT graph.

All matrix transforms happen on canonical float OI weights before fresh G32
quantization and OI-to-HWIO conversion.  The emitted checkpoints contain only
the tensors needed to compile the standalone block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from export_qwen3_lpbq_g32 import (
    _open_source_index,
    _quantize_lpbq_g32,
    _read_source_tensor,
    _source_files,
)


VARIANT_A = "A_original"
VARIANT_B = "B_identity_fold"
VARIANT_C = "C_hadamard_r1_r2"
VARIANT_D_DENSE = "D_dense_r3"
VARIANT_D_FWHT_GRAPH = "D_fwht_graph_r3"
VARIANTS = (
    VARIANT_A,
    VARIANT_B,
    VARIANT_C,
    VARIANT_D_DENSE,
    VARIANT_D_FWHT_GRAPH,
)

LINEAR_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def normalized_fwht(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply the normalized Sylvester Walsh-Hadamard transform along ``dim``."""

    if tensor.ndim == 0:
        raise ValueError("FWHT requires a non-scalar tensor")
    dim %= tensor.ndim
    width = int(tensor.shape[dim])
    if not is_power_of_two(width):
        raise ValueError(f"Hadamard dimension must be a power of two, got {width}")

    moved = tensor.movedim(dim, -1).contiguous()
    original_shape = moved.shape
    work = moved.reshape(-1, width)
    half = 1
    while half < width:
        pairs = work.reshape(-1, width // (2 * half), 2, half)
        left = pairs[:, :, 0, :]
        right = pairs[:, :, 1, :]
        work = torch.stack((left + right, left - right), dim=2).reshape(-1, width)
        half *= 2
    work = work.reshape(original_shape).div_(math.sqrt(width))
    return work.movedim(-1, dim).contiguous()


def normalized_hadamard_matrix(order: int) -> torch.Tensor:
    """Return the normalized Sylvester matrix used by the online R3 paths."""

    return normalized_fwht(torch.eye(order, dtype=torch.float32), dim=-1)


def _headwise_output_rotation(weight: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """Compute ``(I_heads kron R2)^T @ weight`` without materializing R2."""

    if tuple(weight.shape[:1]) != (heads * head_dim,):
        raise ValueError(
            f"output dimension {weight.shape[0]} does not match heads*head_dim={heads * head_dim}"
        )
    return normalized_fwht(weight.reshape(heads, head_dim, -1), dim=1).reshape_as(weight)


def _headwise_input_rotation(weight: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """Compute ``weight @ (I_heads kron R2)`` without materializing R2."""

    if weight.shape[1] != heads * head_dim:
        raise ValueError(
            f"input dimension {weight.shape[1]} does not match heads*head_dim={heads * head_dim}"
        )
    return normalized_fwht(weight.reshape(weight.shape[0], heads, head_dim), dim=2).reshape_as(weight)


def fold_layer_weights(
    source: dict[str, torch.Tensor],
    *,
    hidden_size: int,
    head_dim: int,
    query_heads: int,
    kv_heads: int,
    rotate: bool,
) -> dict[str, torch.Tensor]:
    """Return canonical float32 OI weights after gamma/R1/R2 folding."""

    required = {f"{name}.weight" for name in LINEAR_NAMES}
    required.update(("input_layernorm.weight", "post_attention_layernorm.weight"))
    missing = sorted(required.difference(source))
    if missing:
        raise KeyError(f"missing source Layer tensors: {missing}")

    weights = {name: source[name].detach().to(torch.float32) for name in required}
    gamma_in = weights["input_layernorm.weight"]
    gamma_post = weights["post_attention_layernorm.weight"]
    if gamma_in.shape != (hidden_size,) or gamma_post.shape != (hidden_size,):
        raise ValueError("unexpected block RMSNorm gamma shape")

    def fold_input(weight: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        folded = weight * gamma.unsqueeze(0)
        return normalized_fwht(folded, dim=1) if rotate else folded

    result: dict[str, torch.Tensor] = {
        "self_attn.q_proj.weight": fold_input(weights["self_attn.q_proj.weight"], gamma_in),
        "self_attn.k_proj.weight": fold_input(weights["self_attn.k_proj.weight"], gamma_in),
        "mlp.gate_proj.weight": fold_input(weights["mlp.gate_proj.weight"], gamma_post),
        "mlp.up_proj.weight": fold_input(weights["mlp.up_proj.weight"], gamma_post),
    }

    value = fold_input(weights["self_attn.v_proj.weight"], gamma_in)
    if rotate:
        value = _headwise_output_rotation(value, kv_heads, head_dim)
    result["self_attn.v_proj.weight"] = value

    output = weights["self_attn.o_proj.weight"]
    if rotate:
        output = normalized_fwht(output, dim=0)
        output = _headwise_input_rotation(output, query_heads, head_dim)
    result["self_attn.o_proj.weight"] = output

    down = weights["mlp.down_proj.weight"]
    result["mlp.down_proj.weight"] = normalized_fwht(down, dim=0) if rotate else down
    return {key: value.contiguous() for key, value in result.items()}


def _load_source_layer(
    source_model: Path,
    layer: int,
) -> dict[str, torch.Tensor]:
    source_files = _source_files(source_model)
    key_to_file, _ = _open_source_index(source_files)
    prefix = f"model.layers.{layer}."
    relative_names = [f"{name}.weight" for name in LINEAR_NAMES]
    relative_names.extend(("input_layernorm.weight", "post_attention_layernorm.weight"))
    return {
        name: _read_source_tensor(source_files, key_to_file, prefix + name)
        for name in relative_names
    }


def _load_base_subset(base_path: Path, layer: int) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    prefix = f"model.layers.{layer}."
    rope_names = ("model.mllm_max_sin_embedding", "model.mllm_max_cos_embedding")
    rope_qdq_names = ("model.sin_embedding_input_qdq.", "model.cos_embedding_input_qdq.")
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(base_path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            if key.startswith(prefix) or key.startswith(rope_names) or key.startswith(rope_qdq_names):
                tensors[key] = handle.get_tensor(key).contiguous()
    if not tensors:
        raise ValueError(f"no Layer {layer} tensors found in {base_path}")
    return tensors, metadata


def _encode_unit_rmsnorm(tensors: dict[str, torch.Tensor], prefix: str) -> None:
    """Encode an exact unit gamma using the existing UInt16 RMSNorm contract."""

    weight_key = prefix + ".weight"
    scale_key = prefix + ".scale"
    zero_key = prefix + ".zero_point"
    for key in (weight_key, scale_key, zero_key):
        if key not in tensors:
            raise KeyError(f"candidate checkpoint needs {key}")

    weight = tensors[weight_key]
    if weight.dtype == torch.uint16:
        tensors[weight_key] = torch.full_like(weight, torch.iinfo(torch.uint16).max)
        tensors[scale_key] = torch.full_like(tensors[scale_key], 1.0 / 65535.0)
        tensors[zero_key] = torch.zeros_like(tensors[zero_key])
    elif weight.is_floating_point():
        tensors[weight_key] = torch.ones_like(weight)
    else:
        raise TypeError(f"unsupported RMSNorm carrier dtype {weight.dtype} for {weight_key}")


def _clone_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone().contiguous() for key, value in tensors.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_variant(
    output_root: Path,
    variant: str,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
    manifest: dict[str, object],
) -> None:
    variant_dir = output_root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = variant_dir / "model.safetensors"
    save_file(tensors, str(checkpoint), metadata=metadata)
    manifest["checkpoint"] = str(checkpoint)
    manifest["checkpoint_sha256"] = _sha256(checkpoint)
    manifest["tensor_count"] = len(tensors)
    (variant_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def export_variants(
    *,
    source_model: Path,
    base_quant_checkpoint: Path,
    output_dir: Path,
    layer: int = 5,
    hidden_size: int = 2048,
    head_dim: int = 128,
    query_heads: int = 16,
    kv_heads: int = 8,
    group_size: int = 32,
    variants: Iterable[str] = VARIANTS,
    overwrite: bool = False,
) -> None:
    selected = tuple(variants)
    unknown = sorted(set(selected).difference(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    if group_size != 32:
        raise ValueError("the QNN prototype supports only physical LPBQ G32")
    if not is_power_of_two(hidden_size) or not is_power_of_two(head_dim):
        raise ValueError("fixed Hadamard R1/R2 dimensions must be powers of two")
    if query_heads * head_dim != hidden_size:
        raise ValueError("query_heads * head_dim must equal hidden_size")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {output_dir}")
        for variant in selected:
            target = output_dir / variant
            if target.exists():
                shutil.rmtree(target)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    source = _load_source_layer(source_model, layer)
    base_tensors, base_metadata = _load_base_subset(base_quant_checkpoint, layer)
    layer_prefix = f"model.layers.{layer}."

    common = {
        "schema_version": 1,
        "model": "Qwen3-1.7B",
        "layer": layer,
        "source_model": str(source_model),
        "base_quant_checkpoint": str(base_quant_checkpoint),
        "graph_style": "SHA per-head Q/K/V projections",
        "activation_qdq": "reused bit-exact from baseline; speed-only experiment",
        "group_size": group_size,
        "contract": {
            "weight_dtype": "int8_carrier_uint4",
            "weight_layout": "HWIO [1,1,K,O]",
            "scale1_dtype": "uint8",
            "scale1_layout": "flattened [O,K/G] row-major",
            "scale2_dtype": "float32",
            "scale2_layout": "[O]",
            "hidden_boundary": "R1 for C/D; baseline basis for A/B",
            "value_cache_boundary": "per-head R2 for C/D; baseline basis for A/B",
            "key_cache_boundary": "R3 for D; baseline basis for A/B/C",
        },
    }

    if VARIANT_A in selected:
        metadata = dict(base_metadata)
        metadata.update(
            {
                "mllm.block.variant": VARIANT_A,
                "mllm.block.layer": str(layer),
                "mllm.lpbq.rotation": "none",
            }
        )
        _write_variant(
            output_dir,
            VARIANT_A,
            _clone_tensors(base_tensors),
            metadata,
            {**common, "variant": VARIANT_A, "rotation": "none", "weights": "baseline bit-exact"},
        )

    candidate_variants = (
        (VARIANT_B, False, "none"),
        (VARIANT_C, True, "none"),
        (VARIANT_D_DENSE, True, "dense"),
        (VARIANT_D_FWHT_GRAPH, True, "fwht-graph"),
    )
    for variant, rotate, r3_mode in candidate_variants:
        if variant not in selected:
            continue
        folded = fold_layer_weights(
            source,
            hidden_size=hidden_size,
            head_dim=head_dim,
            query_heads=query_heads,
            kv_heads=kv_heads,
            rotate=rotate,
        )
        tensors = _clone_tensors(base_tensors)
        reports: dict[str, object] = {}
        for relative_name, canonical_weight in folded.items():
            packed, scale1, scale2, stats = _quantize_lpbq_g32(canonical_weight, group_size)
            weight_key = layer_prefix + relative_name
            projection_prefix = weight_key[: -len(".weight")]
            tensors[weight_key] = packed
            tensors[projection_prefix + ".scale1"] = scale1
            tensors[projection_prefix + ".scale2"] = scale2
            reports[weight_key] = {
                "canonical_shape": list(canonical_weight.shape),
                "deployed_shape": list(packed.shape),
                **stats,
            }

        if r3_mode == "dense":
            r3 = normalized_hadamard_matrix(head_dim)
            packed, scale1, scale2, stats = _quantize_lpbq_g32(r3, group_size)
            r3_prefix = layer_prefix + "self_attn.r3_dense"
            # QNN MatMul consumes a [K,O] static input. The shared LPBQ
            # encoder emits [1,1,K,O], so remove only the singleton dims.
            tensors[r3_prefix + ".weight"] = packed.reshape(head_dim, head_dim)
            tensors[r3_prefix + ".scale1"] = scale1
            tensors[r3_prefix + ".scale2"] = scale2
            reports[r3_prefix + ".weight"] = {
                "canonical_shape": [head_dim, head_dim],
                "deployed_shape": [head_dim, head_dim],
                "shared_by": f"{query_heads} Q and {kv_heads} K R3 MatMuls",
                **stats,
            }

        if r3_mode == "fwht-graph":
            # Encode 1/sqrt(2) exactly at the top UInt16 code. Every FWHT
            # stage normalizes its butterfly and keeps the C Q/K encoding.
            norm_prefix = (
                layer_prefix
                + "self_attn.r3_fwht_norm_constant_qdq.fake_quant"
            )
            tensors[norm_prefix + ".scale"] = torch.tensor(
                [1.0 / (math.sqrt(2.0) * 65535.0)], dtype=torch.float32
            )
            tensors[norm_prefix + ".zero_point"] = torch.zeros(1, dtype=torch.int32)

        _encode_unit_rmsnorm(tensors, layer_prefix + "input_layernorm")
        _encode_unit_rmsnorm(tensors, layer_prefix + "post_attention_layernorm")
        rotation = "identity gamma fold" if not rotate else "R1=H2048/sqrt(2048), R2=H128/sqrt(128)"
        if r3_mode != "none":
            rotation += ", R3=H128/sqrt(128) post-RoPE Q/current-K"
        metadata = dict(base_metadata)
        metadata.update(
            {
                "mllm.block.variant": variant,
                "mllm.block.layer": str(layer),
                "mllm.lpbq.rotation": rotation,
                "mllm.block.r3_mode": r3_mode,
                "mllm.activation_qdq": "reused baseline values; speed-only",
            }
        )
        _write_variant(
            output_dir,
            variant,
            tensors,
            metadata,
            {
                **common,
                "variant": variant,
                "rotation": rotation,
                "r3_mode": r3_mode,
                "key_cache_boundary": "R3" if r3_mode != "none" else "baseline",
                "rmsnorm_gamma": "folded into q/k/v and gate/up; deployed gamma encodes exact one",
                "weights": reports,
            },
        )

    root_manifest = {
        **common,
        "variants": list(selected),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(root_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--base-quant-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=32, choices=(32,))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_variants(
        source_model=args.source_model,
        base_quant_checkpoint=args.base_quant_checkpoint,
        output_dir=args.output_dir,
        layer=args.layer,
        hidden_size=args.hidden_size,
        head_dim=args.head_dim,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        group_size=args.group_size,
        variants=args.variants,
        overwrite=args.overwrite,
    )
    print(f"wrote standalone block variants under {args.output_dir}")


if __name__ == "__main__":
    main()
