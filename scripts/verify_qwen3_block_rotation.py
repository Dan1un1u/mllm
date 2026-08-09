#!/usr/bin/env python3
"""Verify pre-quantization Qwen3 Layer 5 R1/R2 folding for s1 and s32."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen3_block_rotation import (
    LINEAR_NAMES,
    _load_source_layer,
    fold_layer_weights,
    normalized_fwht,
)
from export_qwen3_lpbq_g32 import _open_source_index, _read_source_tensor, _source_files


def rms_norm(value: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps) * gamma


def rope(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
    return value * cos[:, None, :, :] + rotated * sin[:, None, :, :]


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).double().norm() / expected.double().norm().clamp_min(1e-30))


def block_forward(
    x: torch.Tensor,
    past_key: torch.Tensor,
    past_value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    weights: dict[str, torch.Tensor],
    gamma_in: torch.Tensor,
    gamma_post: torch.Tensor,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    *,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, _ = x.shape
    residual = x
    normed = rms_norm(x, gamma_in, eps)
    query = F.linear(normed, weights["self_attn.q_proj.weight"])
    key = F.linear(normed, weights["self_attn.k_proj.weight"])
    value = F.linear(normed, weights["self_attn.v_proj.weight"])
    query = query.reshape(batch, seq_len, query_heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
    query = rope(rms_norm(query, gamma_q, eps), cos, sin)
    key = rope(rms_norm(key, gamma_k, eps), cos, sin)
    present_key = key.transpose(2, 3)
    present_value = value

    key_cache = torch.cat((past_key, present_key), dim=-1)
    value_cache = torch.cat((past_value, present_value), dim=2)
    groups = query_heads // kv_heads
    key_cache = key_cache.repeat_interleave(groups, dim=1)
    value_cache = value_cache.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query, key_cache) / math.sqrt(head_dim)
    past_len = past_key.shape[-1]
    current_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
    )
    scores[:, :, :, past_len:] = scores[:, :, :, past_len:].masked_fill(
        current_mask[None, None, :, :], float("-inf")
    )
    probabilities = torch.softmax(scores, dim=-1)
    attention = torch.matmul(probabilities, value_cache)
    attention = attention.transpose(1, 2).reshape(batch, seq_len, query_heads * head_dim)
    x = residual + F.linear(attention, weights["self_attn.o_proj.weight"])

    residual = x
    normed = rms_norm(x, gamma_post, eps)
    gate = F.silu(F.linear(normed, weights["mlp.gate_proj.weight"]))
    up = F.linear(normed, weights["mlp.up_proj.weight"])
    x = residual + F.linear(gate * up, weights["mlp.down_proj.weight"])
    return x, present_key, present_value


def verify(
    source_model: Path,
    seq_lens: list[int],
    *,
    layer: int = 5,
    context_len: int = 1024,
    hidden_size: int = 2048,
    intermediate_size: int = 6144,
    query_heads: int = 16,
    kv_heads: int = 8,
    head_dim: int = 128,
    rope_theta: float = 1_000_000.0,
    eps: float = 1e-6,
    seed: int = 20260809,
) -> dict[str, object]:
    source = _load_source_layer(source_model, layer)
    source_files = _source_files(source_model)
    key_to_file, _ = _open_source_index(source_files)
    prefix = f"model.layers.{layer}.self_attn."
    gamma_q = _read_source_tensor(source_files, key_to_file, prefix + "q_norm.weight").float()
    gamma_k = _read_source_tensor(source_files, key_to_file, prefix + "k_norm.weight").float()
    original_weights = {
        f"{name}.weight": source[f"{name}.weight"].float().contiguous()
        for name in LINEAR_NAMES
    }
    rotated_weights = fold_layer_weights(
        source,
        hidden_size=hidden_size,
        head_dim=head_dim,
        query_heads=query_heads,
        kv_heads=kv_heads,
        rotate=True,
    )
    gamma_in = source["input_layernorm.weight"].float()
    gamma_post = source["post_attention_layernorm.weight"].float()
    unit_gamma = torch.ones_like(gamma_in)

    results = []
    for seq_len in seq_lens:
        if seq_len <= 0 or seq_len >= context_len:
            raise ValueError(f"invalid seq_len {seq_len}")
        generator = torch.Generator(device="cpu").manual_seed(seed + seq_len)
        past_len = context_len - seq_len
        x = torch.randn((1, seq_len, hidden_size), generator=generator) * 0.2
        past_key = torch.randn((1, kv_heads, head_dim, past_len), generator=generator) * 0.2
        past_value = torch.randn((1, kv_heads, past_len, head_dim), generator=generator) * 0.2

        positions = torch.arange(past_len, context_len, dtype=torch.float32)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        frequencies = torch.outer(positions, inv_freq)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cos = embedding.cos()[None, :, :]
        sin = embedding.sin()[None, :, :]

        expected, expected_key, expected_value = block_forward(
            x,
            past_key,
            past_value,
            cos,
            sin,
            original_weights,
            gamma_in,
            gamma_post,
            gamma_q,
            gamma_k,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            eps=eps,
        )
        x_rotated = normalized_fwht(x, dim=-1)
        past_value_rotated = normalized_fwht(past_value, dim=-1)
        actual, actual_key, actual_value = block_forward(
            x_rotated,
            past_key,
            past_value_rotated,
            cos,
            sin,
            rotated_weights,
            unit_gamma,
            unit_gamma,
            gamma_q,
            gamma_k,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            eps=eps,
        )
        expected_rotated = normalized_fwht(expected, dim=-1)
        expected_value_rotated = normalized_fwht(expected_value, dim=-1)
        metrics = {
            "seq_len": seq_len,
            "past_len": past_len,
            "hidden_relative_l2": relative_l2(actual, expected_rotated),
            "key_relative_l2": relative_l2(actual_key, expected_key),
            "value_relative_l2": relative_l2(actual_value, expected_value_rotated),
            "hidden_max_abs": float((actual - expected_rotated).abs().max()),
        }
        metrics["pass"] = (
            metrics["hidden_relative_l2"] <= 1e-4
            and metrics["key_relative_l2"] <= 1e-4
            and metrics["value_relative_l2"] <= 1e-4
        )
        results.append(metrics)
    return {
        "schema_version": 1,
        "layer": layer,
        "dtype": "float32 pre-quantization reference",
        "threshold_relative_l2": 1e-4,
        "results": results,
        "pass": all(item["pass"] for item in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=(1, 32))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.source_model, list(args.seq_lens))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
