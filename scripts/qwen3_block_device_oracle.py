#!/usr/bin/env python3
"""Recompute Qwen3 Layer 5 D-Dense from raw QNN boundary tensors.

The reference decodes the deployed LPBQ-G32 weights and replays the explicit
activation QDQ points in the SHA graph.  It also evaluates a deliberately
incorrect no-R3 counterfactual so that a passing result demonstrates that the
post-RoPE R3 was actually applied, rather than merely producing plausible
block outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open


LAYER = "model.layers.5"
ATTN = LAYER + ".self_attn"
MLP = LAYER + ".mlp"
UINT16_MAX = 65535
GATES = {
    "hidden_relative_l2_max": 0.002,
    "present_key_relative_l2_max": 0.03,
    "present_value_relative_l2_max": 0.01,
    "r3_key_error_ratio_vs_disabled_max": 0.05,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_tensor(path: Path, dtype: torch.dtype, shape: list[int]) -> torch.Tensor:
    expected = math.prod(shape)
    element_size = torch.empty((), dtype=dtype).element_size()
    actual_bytes = path.stat().st_size
    if actual_bytes != expected * element_size:
        raise ValueError(
            f"{path}: expected {expected * element_size} bytes for {shape}/{dtype}, "
            f"got {actual_bytes}"
        )
    return torch.from_file(str(path), shared=False, size=expected, dtype=dtype).clone().reshape(shape)


def load_dump(metadata_path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("contract") != "qwen3_layer5_block_raw_quantized_v1":
        raise ValueError(f"unsupported dump contract in {metadata_path}")
    root = metadata_path.with_suffix("")
    dtype_map = {"uint16": torch.uint16, "uint8": torch.uint8}
    tensors: dict[str, torch.Tensor] = {}
    for item in metadata["tensors"]:
        path = Path(str(root) + item["file_suffix"])
        tensors[item["name"]] = _raw_tensor(path, dtype_map[item["dtype"]], item["shape"])
    return metadata, tensors


class QuantizedCheckpoint:
    def __init__(self, handle: Any):
        self.handle = handle
        self.keys = set(handle.keys())
        self.cache: dict[str, torch.Tensor] = {}

    def tensor(self, key: str) -> torch.Tensor:
        if key not in self.keys:
            # prepareParametersForSHA duplicates shared MHA QDQ tensors into
            # per-head names in memory. The artifact intentionally stores only
            # the source shared tensors, so mirror that aliasing here.
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
        signed_oi = signed.transpose(0, 1).reshape(out_features, in_features // group_size, group_size)
        decoded = signed_oi.float()
        decoded *= scale1.reshape(out_features, in_features // group_size).unsqueeze(-1)
        decoded *= scale2.reshape(out_features, 1, 1)
        return decoded.reshape(out_features, in_features)


def rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps) * weight


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _head_rope(
    checkpoint: QuantizedCheckpoint,
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head: int,
    kind: str,
    norm_weight: torch.Tensor,
) -> torch.Tensor:
    prefix = f"{ATTN}.{kind}"
    value = checkpoint.qdq(value, f"{prefix}_norm_input_qdq_h{head}")
    value = rms_norm(value, norm_weight)
    value = checkpoint.qdq(value, f"{prefix}_norm_output_qdq_h{head}")
    negative_half = checkpoint.qdq(
        -value[..., value.shape[-1] // 2 :], f"{prefix}_rope_neg_half_qdq_h{head}"
    )
    rotated = torch.cat((negative_half, value[..., : value.shape[-1] // 2]), dim=-1)
    lhs = checkpoint.qdq(value * cos, f"{prefix}_rope_mul_0_output_qdq_h{head}")
    rhs = checkpoint.qdq(rotated * sin, f"{prefix}_rope_mul_1_output_qdq_h{head}")
    return checkpoint.qdq(lhs + rhs, f"{prefix}_rope_add_0_output_qdq_h{head}")


def block_forward(
    checkpoint: QuantizedCheckpoint,
    raw: dict[str, torch.Tensor],
    *,
    apply_r3: bool,
    query_heads: int = 16,
    kv_heads: int = 8,
    head_dim: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_qdq = LAYER + ".input_layernorm_input_qdq"
    key_boundary_qdq = ATTN + ".k_cast_to_int8_qdq"
    value_boundary_qdq = ATTN + ".v_cast_to_int8_qdq"
    x = checkpoint.dequantize_codes(raw["hidden"], hidden_qdq)
    sin = checkpoint.dequantize_codes(raw["sin"], "model.sin_embedding_input_qdq")[:, None]
    cos = checkpoint.dequantize_codes(raw["cos"], "model.cos_embedding_input_qdq")[:, None]
    past_key = checkpoint.dequantize_codes(
        raw["past_key"], key_boundary_qdq, zero_point_override=128
    )
    past_value = checkpoint.dequantize_codes(
        raw["past_value"], value_boundary_qdq, zero_point_override=128
    )
    # These QDQ constants are injected by compile_common.hpp after the
    # safetensors checkpoint is loaded, so reproduce them explicitly here.
    mask = (raw["mask"].float() - UINT16_MAX) * (0.001 / UINT16_MAX)

    residual = x
    x = rms_norm(x, checkpoint.rms_weight(LAYER + ".input_layernorm"))
    projected_input = checkpoint.qdq(x, ATTN + ".q_proj_input_qdq")
    query = F.linear(projected_input, checkpoint.lpbq_weight(ATTN + ".q_proj"))
    key = F.linear(projected_input, checkpoint.lpbq_weight(ATTN + ".k_proj"))
    value = F.linear(projected_input, checkpoint.lpbq_weight(ATTN + ".v_proj"))
    batch, seq_len, _ = query.shape
    query = query.reshape(batch, seq_len, query_heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
    q_norm_weight = checkpoint.rms_weight(ATTN + ".q_norm")
    k_norm_weight = checkpoint.rms_weight(ATTN + ".k_norm")
    query_heads_values = [
        _head_rope(checkpoint, query[:, h : h + 1], cos, sin, h, "q", q_norm_weight)
        for h in range(query_heads)
    ]
    key_heads_values = [
        _head_rope(checkpoint, key[:, h : h + 1], cos, sin, h, "k", k_norm_weight)
        for h in range(kv_heads)
    ]
    if apply_r3:
        r3_weight = checkpoint.lpbq_weight(ATTN + ".r3_dense")
        # QNN uses matmul(x, R3), not a linear layer with an implicit
        # transpose.  LPBQ quantization makes the carrier only approximately
        # symmetric, so preserve the deployed right-multiplication contract.
        query_heads_values = [
            checkpoint.qdq(
                torch.matmul(item, r3_weight), f"{ATTN}.q_rope_add_0_output_qdq_h{h}"
            )
            for h, item in enumerate(query_heads_values)
        ]
        key_heads_values = [
            checkpoint.qdq(
                torch.matmul(item, r3_weight), f"{ATTN}.k_rope_add_0_output_qdq_h{h}"
            )
            for h, item in enumerate(key_heads_values)
        ]

    current_keys: list[torch.Tensor] = []
    current_values: list[torch.Tensor] = []
    key_cache: list[torch.Tensor] = []
    value_cache: list[torch.Tensor] = []
    for h in range(kv_heads):
        current_key = checkpoint.qdq(
            key_heads_values[h],
            f"{ATTN}.k_cast_to_int8_qdq_h{h}",
            qmax=255,
            zero_point_override=128,
        ).transpose(2, 3)
        current_value = checkpoint.qdq(
            value[:, h : h + 1], f"{ATTN}.v_cast_to_int16_qdq_h{h}"
        )
        current_value = checkpoint.qdq(
            current_value,
            f"{ATTN}.v_cast_to_int8_qdq_h{h}",
            qmax=255,
            zero_point_override=128,
        )
        current_keys.append(current_key)
        current_values.append(current_value)
        key_cache.append(torch.cat((past_key[:, h : h + 1], current_key), dim=-1))
        value_cache.append(torch.cat((past_value[:, h : h + 1], current_value), dim=2))

    attention_outputs: list[torch.Tensor] = []
    for h, q_head in enumerate(query_heads_values):
        kv_head = h // (query_heads // kv_heads)
        scores = checkpoint.qdq(
            torch.matmul(q_head, key_cache[kv_head]), f"{ATTN}.qk_matmul_output_qdq_h{h}"
        )
        scale = checkpoint.qdq(
            torch.tensor([1.0 / math.sqrt(head_dim)]), f"{ATTN}.scaling_qdq_h{h}"
        )
        scores = checkpoint.qdq(scores * scale, f"{ATTN}.mul_0_output_qdq_h{h}")
        minimum = checkpoint.qdq(scores.amin(dim=-1, keepdim=True), f"{ATTN}.reduce_min_output_qdq_h{h}")
        minus_twenty = checkpoint.qdq(torch.tensor([-20.0]), f"{ATTN}.neg_20_qdq_h{h}")
        masked_value = checkpoint.qdq(minimum + minus_twenty, f"{ATTN}.minus_0_output_qdq_h{h}")
        scores = torch.where(mask == 0, scores, masked_value)
        scores = checkpoint.qdq(scores, f"{ATTN}.where_attn_qdq_h{h}")
        probabilities = checkpoint.qdq(F.softmax(scores, dim=-1), f"{ATTN}.softmax_output_qdq_h{h}")
        attention_outputs.append(
            checkpoint.qdq(
                torch.matmul(probabilities, value_cache[kv_head]),
                f"{ATTN}.attn_value_matmul_output_qdq_h{h}",
            )
        )

    attention = torch.cat(attention_outputs, dim=1).transpose(1, 2).reshape(batch, seq_len, -1)
    attention = F.linear(attention, checkpoint.lpbq_weight(ATTN + ".o_proj"))
    attention = checkpoint.qdq(attention, LAYER + ".add_0_lhs_input_qdq")
    x = checkpoint.qdq(residual + attention, LAYER + ".add_0_output_qdq")

    residual = x
    x = rms_norm(x, checkpoint.rms_weight(LAYER + ".post_attention_layernorm"))
    x = checkpoint.qdq(x, MLP + ".up_proj_input_qdq")
    up = checkpoint.qdq(
        F.linear(x, checkpoint.lpbq_weight(MLP + ".up_proj")), MLP + ".up_proj_output_qdq"
    )
    gate = checkpoint.qdq(
        F.linear(x, checkpoint.lpbq_weight(MLP + ".gate_proj")), MLP + ".gate_proj_output_qdq"
    )
    sigmoid = checkpoint.qdq(torch.sigmoid(gate), MLP + ".sigmoid_output_qdq")
    gate = checkpoint.qdq(gate * sigmoid, MLP + ".act_output_qdq")
    down_input = checkpoint.qdq(gate * up, MLP + ".down_proj_input_qdq")
    down = F.linear(down_input, checkpoint.lpbq_weight(MLP + ".down_proj"))
    down = checkpoint.qdq(down, LAYER + ".add_1_lhs_input_qdq")
    x = checkpoint.qdq(residual + down, hidden_qdq)
    return x, torch.cat(current_keys, dim=1), torch.cat(current_values, dim=1)


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


def compare_input_hashes(dump_root: Path, workload: str) -> dict[str, Any]:
    suffixes = (
        "hidden_u16.bin",
        "sin_u16.bin",
        "cos_u16.bin",
        "mask_u16.bin",
        "past_key_u8.bin",
        "past_value_u8.bin",
    )
    records = {}
    for suffix in suffixes:
        c_path = dump_root / f"C_hadamard_r1_r2.{workload}.{suffix}"
        d_path = dump_root / f"D_dense_r3.{workload}.{suffix}"
        c_hash, d_hash = _sha256(c_path), _sha256(d_path)
        records[suffix] = {"c_sha256": c_hash, "d_sha256": d_hash, "equal": c_hash == d_hash}
    return {"files": records, "pass": all(item["equal"] for item in records.values())}


def run(checkpoint_path: Path, dump_root: Path) -> dict[str, Any]:
    results = []
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        checkpoint = QuantizedCheckpoint(handle)
        for workload in ("s1_decode", "s32_prefill_chunk"):
            metadata, raw = load_dump(dump_root / f"D_dense_r3.{workload}.json")
            actual_hidden = checkpoint.dequantize_codes(
                raw["hidden_out"], LAYER + ".input_layernorm_input_qdq"
            )
            actual_key = checkpoint.dequantize_codes(
                raw["present_key"], ATTN + ".k_cast_to_int8_qdq", zero_point_override=128
            )
            actual_value = checkpoint.dequantize_codes(
                raw["present_value"], ATTN + ".v_cast_to_int8_qdq", zero_point_override=128
            )
            expected = block_forward(checkpoint, raw, apply_r3=True)
            no_r3 = block_forward(checkpoint, raw, apply_r3=False)
            tensor_results = {
                "hidden_out": metrics(actual_hidden, expected[0]),
                "present_key": metrics(actual_key, expected[1]),
                "present_value": metrics(actual_value, expected[2]),
                "no_r3_hidden_out": metrics(actual_hidden, no_r3[0]),
                "no_r3_present_key": metrics(actual_key, no_r3[1]),
            }
            key_ratio = tensor_results["present_key"]["relative_l2"] / max(
                tensor_results["no_r3_present_key"]["relative_l2"], 1e-30
            )
            input_hashes = compare_input_hashes(dump_root, workload)
            checks = {
                "input_hashes_equal": input_hashes["pass"],
                "hidden_relative_l2": tensor_results["hidden_out"]["relative_l2"]
                <= GATES["hidden_relative_l2_max"],
                "present_key_relative_l2": tensor_results["present_key"]["relative_l2"]
                <= GATES["present_key_relative_l2_max"],
                "present_value_relative_l2": tensor_results["present_value"]["relative_l2"]
                <= GATES["present_value_relative_l2_max"],
                "r3_discrimination": key_ratio <= GATES["r3_key_error_ratio_vs_disabled_max"],
            }
            results.append(
                {
                    "workload": workload,
                    "seq_len": metadata["seq_len"],
                    "past_len": metadata["past_len"],
                    "input_hashes": input_hashes,
                    "tensors": tensor_results,
                    "r3_key_error_ratio_vs_disabled": key_ratio,
                    "checks": checks,
                    "pass": all(checks.values()),
                }
            )
    return {
        "schema_version": 1,
        "oracle": "pytorch_lpbq_g32_explicit_qdq_vs_qnn_raw_boundary",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dump_root": str(dump_root.resolve()),
        "gates": GATES,
        "results": results,
        "pass": all(item["pass"] for item in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.checkpoint, args.dump_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
