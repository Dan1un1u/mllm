#!/usr/bin/env python3
"""Independent full-model mathematical oracle for folded D-Dense R3.

The oracle replays the deployed LPBQ-G32/QDQ graph from the folded
safetensors checkpoint, including all 28 decoder layers, per-layer R3
carriers, KV-cache quantization, the folded embedding/final boundary, and the
lm_head.  It is deliberately separate from QNN execution: the probe JSON
supplies only token IDs and (optionally) a raw final-logits dump for a
numerical cross-check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open


UINT16_MAX = 65535
DEFAULT_LAYERS = 28
DEFAULT_HEADS = 16
DEFAULT_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128
DEFAULT_HIDDEN = 2048
DEFAULT_VOCAB = 151936


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fnv1a_u16(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().numpy().tobytes()
    acc = 1469598103934665603
    for byte in raw:
        acc ^= byte
        acc = (acc * 1099511628211) & ((1 << 64) - 1)
    return f"{acc:016x}"


class QuantizedCheckpoint:
    """Safetensors reader with the same LPBQ/QDQ decode as the block oracle."""

    def __init__(self, handle: Any):
        self.handle = handle
        self.keys = set(handle.keys())
        self.cache: dict[str, torch.Tensor] = {}
        self.weight_cache: dict[str, torch.Tensor] = {}

    def tensor(self, key: str) -> torch.Tensor:
        if key not in self.keys:
            shared_key = re.sub(r"_h\d+(?=\.fake_quant\.)", "", key)
            if shared_key not in self.keys:
                raise KeyError(f"checkpoint has no tensor {key!r} (or shared {shared_key!r})")
            key = shared_key
        if key not in self.cache:
            self.cache[key] = self.handle.get_tensor(key).contiguous()
        return self.cache[key]

    def scalar(self, key: str) -> float:
        value = self.tensor(key)
        if value.numel() != 1:
            raise ValueError(f"expected scalar {key}, got {tuple(value.shape)}")
        return float(value.item())

    def dequantize_codes(
        self,
        codes: torch.Tensor,
        qdq_prefix: str,
        *,
        fake_quant: bool = True,
        zero_point_override: int | None = None,
    ) -> torch.Tensor:
        middle = ".fake_quant" if fake_quant else ""
        scale = self.scalar(qdq_prefix + middle + ".scale")
        zero_point = (
            float(zero_point_override)
            if zero_point_override is not None
            else self.scalar(qdq_prefix + middle + ".zero_point")
        )
        return (codes.to(torch.float32) - zero_point) * scale

    def qdq(
        self,
        value: torch.Tensor,
        qdq_prefix: str,
        *,
        qmin: int = 0,
        qmax: int = UINT16_MAX,
        zero_point_override: int | None = None,
    ) -> torch.Tensor:
        scale = self.scalar(qdq_prefix + ".fake_quant.scale")
        zero_point = (
            float(zero_point_override)
            if zero_point_override is not None
            else self.scalar(qdq_prefix + ".fake_quant.zero_point")
        )
        codes = torch.round(value / scale + zero_point).clamp(qmin, qmax)
        return (codes - zero_point) * scale

    def rms_weight(self, prefix: str) -> torch.Tensor:
        weight = self.tensor(prefix + ".weight")
        if weight.is_floating_point():
            return weight.float()
        scale = self.scalar(prefix + ".scale")
        zero_point = self.scalar(prefix + ".zero_point")
        return (weight.float() - zero_point) * scale

    def lpbq_weight(self, prefix: str, group_size: int = 32) -> torch.Tensor:
        if prefix in self.weight_cache:
            return self.weight_cache[prefix]
        packed = self.tensor(prefix + ".weight")
        scale1 = self.tensor(prefix + ".scale1").float()
        scale2 = self.tensor(prefix + ".scale2").float()
        if packed.ndim == 4:
            in_features, out_features = int(packed.shape[2]), int(packed.shape[3])
            carrier = packed.reshape(in_features, out_features)
        elif packed.ndim == 2:
            in_features, out_features = map(int, packed.shape)
            carrier = packed
        else:
            raise ValueError(f"unsupported LPBQ carrier shape {tuple(packed.shape)} for {prefix}")
        if in_features % group_size:
            raise ValueError(f"{prefix}: input width {in_features} is not divisible by G{group_size}")
        signed = carrier.to(torch.int16)
        signed = torch.where(signed >= 8, signed - 16, signed)
        signed_oi = signed.transpose(0, 1).reshape(
            out_features, in_features // group_size, group_size
        )
        decoded = signed_oi.float()
        decoded *= scale1.reshape(out_features, in_features // group_size).unsqueeze(-1)
        decoded *= scale2.reshape(out_features, 1, 1)
        decoded = decoded.reshape(out_features, in_features)
        self.weight_cache[prefix] = decoded
        return decoded


def rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps) * weight


def head_rope(
    checkpoint: QuantizedCheckpoint,
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head: int,
    kind: str,
    attn: str,
    norm_weight: torch.Tensor,
) -> torch.Tensor:
    prefix = f"{attn}.{kind}"
    value = checkpoint.qdq(value, f"{prefix}_norm_input_qdq_h{head}")
    value = rms_norm(value, norm_weight)
    value = checkpoint.qdq(value, f"{prefix}_norm_output_qdq_h{head}")
    negative_half = checkpoint.qdq(
        -value[..., value.shape[-1] // 2 :],
        f"{prefix}_rope_neg_half_qdq_h{head}",
    )
    rotated = torch.cat((negative_half, value[..., : value.shape[-1] // 2]), dim=-1)
    lhs = checkpoint.qdq(
        value * cos, f"{prefix}_rope_mul_0_output_qdq_h{head}"
    )
    rhs = checkpoint.qdq(
        rotated * sin, f"{prefix}_rope_mul_1_output_qdq_h{head}"
    )
    return checkpoint.qdq(
        lhs + rhs, f"{prefix}_rope_add_0_output_qdq_h{head}"
    )


def rope_inputs(
    checkpoint: QuantizedCheckpoint,
    position: int,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    indices = torch.arange(half, dtype=torch.float32)
    inv_freq = torch.pow(torch.tensor(1_000_000.0), -indices / half)
    sin_values = torch.sin(position * inv_freq)
    cos_values = torch.cos(position * inv_freq)
    sin_codes = torch.round(sin_values * 32768.0 + 32768.0).clamp(0, UINT16_MAX)
    cos_codes = torch.round(cos_values * 32768.0 + 32768.0).clamp(0, UINT16_MAX)
    sin_codes = torch.cat((sin_codes, sin_codes)).to(torch.uint16).reshape(1, 1, head_dim)
    cos_codes = torch.cat((cos_codes, cos_codes)).to(torch.uint16).reshape(1, 1, head_dim)
    sin = checkpoint.dequantize_codes(sin_codes, "model.sin_embedding_input_qdq")
    cos = checkpoint.dequantize_codes(cos_codes, "model.cos_embedding_input_qdq")
    return sin[:, None], cos[:, None]


def layer_forward(
    checkpoint: QuantizedCheckpoint,
    hidden: torch.Tensor,
    past_key: torch.Tensor | None,
    past_value: torch.Tensor | None,
    layer: int,
    position: int,
    *,
    layers: int = DEFAULT_LAYERS,
    query_heads: int = DEFAULT_HEADS,
    kv_heads: int = DEFAULT_KV_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    padded_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer_prefix = f"model.layers.{layer}"
    attn = layer_prefix + ".self_attn"
    mlp = layer_prefix + ".mlp"
    if layer != 0:
        hidden = checkpoint.qdq(
            hidden, layer_prefix + ".input_layernorm_input_qdq"
        )
    residual = hidden
    hidden = rms_norm(
        hidden, checkpoint.rms_weight(layer_prefix + ".input_layernorm")
    )
    projected_input = checkpoint.qdq(hidden, attn + ".q_proj_input_qdq")
    query = F.linear(projected_input, checkpoint.lpbq_weight(attn + ".q_proj"))
    key = F.linear(projected_input, checkpoint.lpbq_weight(attn + ".k_proj"))
    value = F.linear(projected_input, checkpoint.lpbq_weight(attn + ".v_proj"))
    batch, seq_len, _ = query.shape
    query = query.reshape(batch, seq_len, query_heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, seq_len, kv_heads, head_dim).transpose(1, 2)

    q_norm = checkpoint.rms_weight(attn + ".q_norm")
    k_norm = checkpoint.rms_weight(attn + ".k_norm")
    sin, cos = rope_inputs(checkpoint, position, head_dim=head_dim)
    query_heads_values = [
        head_rope(checkpoint, query[:, h : h + 1], cos, sin, h, "q", attn, q_norm)
        for h in range(query_heads)
    ]
    key_heads_values = [
        head_rope(checkpoint, key[:, h : h + 1], cos, sin, h, "k", attn, k_norm)
        for h in range(kv_heads)
    ]
    r3 = checkpoint.lpbq_weight(attn + ".r3_dense")
    query_heads_values = [
        checkpoint.qdq(
            F.linear(item, r3), f"{attn}.q_rope_add_0_output_qdq_h{h}"
        )
        for h, item in enumerate(query_heads_values)
    ]
    key_heads_values = [
        checkpoint.qdq(
            F.linear(item, r3), f"{attn}.k_rope_add_0_output_qdq_h{h}"
        )
        for h, item in enumerate(key_heads_values)
    ]

    key_caches: list[torch.Tensor] = []
    value_caches: list[torch.Tensor] = []
    if padded_cache and past_key is None:
        zero_key = torch.zeros(
            (batch, kv_heads, head_dim, 1023), dtype=torch.uint8
        )
        zero_value = torch.zeros(
            (batch, kv_heads, 1023, head_dim), dtype=torch.uint8
        )
        past_key = checkpoint.dequantize_codes(
            zero_key, f"{attn}.k_cast_to_int8_qdq", zero_point_override=128
        )
        past_value = checkpoint.dequantize_codes(
            zero_value, f"{attn}.v_cast_to_int8_qdq", zero_point_override=128
        )
    for h in range(kv_heads):
        current_key = checkpoint.qdq(
            key_heads_values[h],
            f"{attn}.k_cast_to_int8_qdq_h{h}",
            qmax=255,
            zero_point_override=128,
        ).transpose(2, 3)
        current_value = checkpoint.qdq(
            value[:, h : h + 1], f"{attn}.v_cast_to_int16_qdq_h{h}"
        )
        current_value = checkpoint.qdq(
            current_value,
            f"{attn}.v_cast_to_int8_qdq_h{h}",
            qmax=255,
            zero_point_override=128,
        )
        if past_key is None:
            key_cache = current_key
        else:
            key_cache = torch.cat((past_key[:, h : h + 1], current_key), dim=-1)
        if past_value is None:
            value_cache = current_value
        else:
            value_cache = torch.cat((past_value[:, h : h + 1], current_value), dim=2)
        key_caches.append(key_cache)
        value_caches.append(value_cache)

    valid_positions: torch.Tensor | None = None
    if padded_cache:
        valid_positions = torch.zeros((batch, 1, 1, 1024), dtype=torch.bool)
        valid_positions[..., :position] = True
        valid_positions[..., -1] = True

    attention_outputs: list[torch.Tensor] = []
    for head, query_head in enumerate(query_heads_values):
        kv_head = head // (query_heads // kv_heads)
        scores = checkpoint.qdq(
            torch.matmul(query_head, key_caches[kv_head]),
            f"{attn}.qk_matmul_output_qdq_h{head}",
        )
        scale = checkpoint.qdq(
            torch.tensor([1.0 / math.sqrt(head_dim)]),
            f"{attn}.scaling_qdq_h{head}",
        )
        scores = checkpoint.qdq(scores * scale, f"{attn}.mul_0_output_qdq_h{head}")
        if valid_positions is not None:
            minimum = checkpoint.qdq(
                scores.amin(dim=-1, keepdim=True),
                f"{attn}.reduce_min_output_qdq_h{head}",
            )
            minus_twenty = checkpoint.qdq(
                torch.tensor([-20.0]), f"{attn}.neg_20_qdq_h{head}"
            )
            masked_value = checkpoint.qdq(
                minimum + minus_twenty, f"{attn}.minus_0_output_qdq_h{head}"
            )
            scores = torch.where(valid_positions, scores, masked_value)
        scores = checkpoint.qdq(scores, f"{attn}.where_attn_qdq_h{head}")
        probabilities = checkpoint.qdq(
            F.softmax(scores, dim=-1), f"{attn}.softmax_output_qdq_h{head}"
        )
        attention_outputs.append(
            checkpoint.qdq(
                torch.matmul(probabilities, value_caches[kv_head]),
                f"{attn}.attn_value_matmul_output_qdq_h{head}",
            )
        )

    attention = (
        torch.cat(attention_outputs, dim=1)
        .transpose(1, 2)
        .reshape(batch, seq_len, -1)
    )
    attention = F.linear(attention, checkpoint.lpbq_weight(attn + ".o_proj"))
    attention = checkpoint.qdq(attention, layer_prefix + ".add_0_lhs_input_qdq")
    hidden = checkpoint.qdq(
        residual + attention, layer_prefix + ".add_0_output_qdq"
    )

    residual = hidden
    hidden = rms_norm(
        hidden, checkpoint.rms_weight(layer_prefix + ".post_attention_layernorm")
    )
    hidden = checkpoint.qdq(hidden, mlp + ".up_proj_input_qdq")
    up = checkpoint.qdq(
        F.linear(hidden, checkpoint.lpbq_weight(mlp + ".up_proj")),
        mlp + ".up_proj_output_qdq",
    )
    gate = checkpoint.qdq(
        F.linear(hidden, checkpoint.lpbq_weight(mlp + ".gate_proj")),
        mlp + ".gate_proj_output_qdq",
    )
    sigmoid = checkpoint.qdq(
        torch.sigmoid(gate), mlp + ".sigmoid_output_qdq"
    )
    gate = checkpoint.qdq(gate * sigmoid, mlp + ".act_output_qdq")
    down_input = checkpoint.qdq(
        gate * up, mlp + ".down_proj_input_qdq"
    )
    down = F.linear(down_input, checkpoint.lpbq_weight(mlp + ".down_proj"))
    down = checkpoint.qdq(
        down, layer_prefix + ".add_1_lhs_input_qdq"
    )
    hidden = residual + down
    if padded_cache:
        new_key = past_key.clone()
        new_value = past_value.clone()
        for h in range(kv_heads):
            new_key[:, h : h + 1, :, position : position + 1] = key_caches[h][..., -1:]
            new_value[:, h : h + 1, position : position + 1, :] = value_caches[h][..., -1:, :]
    else:
        new_key = torch.cat(key_caches, dim=1)
        new_value = torch.cat(value_caches, dim=1)
    return hidden, new_key, new_value


def embedding(checkpoint: QuantizedCheckpoint, token: int) -> torch.Tensor:
    codes = checkpoint.tensor("model.embed_tokens.weight")[token].reshape(1, 1, -1)
    return checkpoint.dequantize_codes(
        codes, "model.embed_tokens", fake_quant=False
    )


def final_logits(
    checkpoint: QuantizedCheckpoint,
    hidden: torch.Tensor,
) -> torch.Tensor:
    hidden = checkpoint.qdq(hidden, "model.norm_input_qdq")
    hidden = rms_norm(hidden, checkpoint.rms_weight("model.norm"))
    hidden = checkpoint.qdq(hidden, "lm_head_input_qdq")
    logits = F.linear(hidden, checkpoint.lpbq_weight("lm_head"))
    return checkpoint.qdq(logits, "lm_head_output_qdq")


def replay(
    checkpoint: QuantizedCheckpoint,
    prompt_tokens: list[int],
    expected_tokens: list[int],
    *,
    layers: int,
    threads: int,
) -> dict[str, Any]:
    if threads > 0:
        torch.set_num_threads(threads)
    caches: list[tuple[torch.Tensor | None, torch.Tensor | None]] = [
        (None, None) for _ in range(layers)
    ]
    hidden: torch.Tensor | None = None
    position = 0
    layer_timings: list[float] = []
    for token in prompt_tokens:
        hidden = embedding(checkpoint, int(token))
        for layer in range(layers):
            start = time.perf_counter()
            hidden, key, value = layer_forward(
                checkpoint, hidden, caches[layer][0], caches[layer][1],
                layer, position, layers=layers,
                padded_cache=True,
            )
            layer_timings.append(time.perf_counter() - start)
            caches[layer] = (key, value)
        position += 1
    generated: list[int] = []
    last_logits: torch.Tensor | None = None
    token_timings: list[float] = []
    for _ in range(len(expected_tokens)):
        if hidden is None:
            raise ValueError("prompt must contain at least one token")
        start = time.perf_counter()
        last_logits = final_logits(checkpoint, hidden)
        next_token = int(last_logits.reshape(-1).argmax().item())
        token_timings.append(time.perf_counter() - start)
        generated.append(next_token)
        if len(generated) == len(expected_tokens):
            break
        hidden = embedding(checkpoint, next_token)
        for layer in range(layers):
            start = time.perf_counter()
            hidden, key, value = layer_forward(
                checkpoint, hidden, caches[layer][0], caches[layer][1],
                layer, position, layers=layers,
                padded_cache=True,
            )
            layer_timings.append(time.perf_counter() - start)
            caches[layer] = (key, value)
        position += 1
    if last_logits is None:
        raise AssertionError("no logits were produced")
    return {
        "generated_tokens": generated,
        "last_logits": last_logits.detach().cpu(),
        "token_wall_s": token_timings,
        "layer_wall_s": layer_timings,
        "final_hidden": hidden.detach().cpu() if hidden is not None else None,
    }


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual64 = actual.double().reshape(-1)
    expected64 = expected.double().reshape(-1)
    error = actual64 - expected64
    return {
        "relative_l2": float(error.norm() / expected64.norm().clamp_min(1e-30)),
        "max_abs": float(error.abs().max()),
        "mean_abs": float(error.abs().mean()),
        "cosine_similarity": float(F.cosine_similarity(actual64, expected64, dim=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qnn-logits-u16", type=Path)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    probe = json.loads(args.probe_json.read_text(encoding="utf-8"))
    prompt_tokens = [int(value) for value in probe.get("prompt_tokens", [])]
    expected_tokens = [int(value) for value in probe.get("generated_tokens", [])]
    if not prompt_tokens:
        raise SystemExit("probe JSON has no prompt_tokens; rebuild split runner and rerun it")
    if not expected_tokens:
        raise SystemExit("probe JSON has no generated_tokens")
    if args.layers <= 0:
        raise SystemExit("--layers must be positive")

    started = time.perf_counter()
    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as handle:
        checkpoint = QuantizedCheckpoint(handle)
        replay_result = replay(
            checkpoint, prompt_tokens, expected_tokens,
            layers=args.layers, threads=args.threads,
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "oracle": "qwen3_full_model_d_dense_r3_torch_replay_v1",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "probe_json": str(args.probe_json.resolve()),
            "probe_json_sha256": sha256(args.probe_json),
            "layers": args.layers,
            "prompt_tokens": prompt_tokens,
            "expected_generated_tokens": expected_tokens,
            "oracle_generated_tokens": replay_result["generated_tokens"],
            "token_match": replay_result["generated_tokens"] == expected_tokens,
            "token_wall_ms": [round(1000.0 * value, 3) for value in replay_result["token_wall_s"]],
            "layer_wall_ms_total": round(1000.0 * sum(replay_result["layer_wall_s"]), 3),
            "last_logits_fnv1a_f32": fnv1a_u16(replay_result["last_logits"].to(torch.float32)),
            "numerical_oracle": "independent_replay_token_match",
        }
        if args.qnn_logits_u16 is not None:
            raw = torch.from_file(
                str(args.qnn_logits_u16), shared=False, size=DEFAULT_VOCAB, dtype=torch.uint16
            ).clone()
            qnn_logits = checkpoint.dequantize_codes(
                raw, "lm_head_output_qdq"
            )
            oracle_logits = replay_result["last_logits"].reshape(-1)
            logit_metrics = metrics(oracle_logits, qnn_logits)
            report["qnn_logits_u16"] = str(args.qnn_logits_u16.resolve())
            report["qnn_logits_u16_sha256"] = sha256(args.qnn_logits_u16)
            report["qnn_logits_top1"] = int(qnn_logits.argmax().item())
            report["oracle_logits_top1"] = int(oracle_logits.argmax().item())
            report["qnn_logits_metrics_vs_oracle"] = logit_metrics
            report["qnn_logits_fnv1a_u16"] = fnv1a_u16(raw)
            report["qnn_logits_gate"] = bool(
                report["qnn_logits_top1"] == report["oracle_logits_top1"]
                and logit_metrics["relative_l2"] <= 0.02
                and logit_metrics["cosine_similarity"] >= 0.999
            )
        else:
            report["qnn_logits_gate"] = "not_run"

        report["numerical_gate"] = bool(
            report["token_match"]
            and report["qnn_logits_gate"] is True
        )
        report["gate_status"] = (
            "numerical_gate_pass" if report["numerical_gate"]
            else "numerical_gate_fail_or_not_run"
        )
        report["elapsed_s"] = round(time.perf_counter() - started, 3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["numerical_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
