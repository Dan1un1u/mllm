#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen3_block_rotation import fold_layer_weights, normalized_fwht  # noqa: E402


def hadamard(order: int) -> torch.Tensor:
    result = torch.ones((1, 1), dtype=torch.float64)
    while result.shape[0] < order:
        result = torch.cat(
            (torch.cat((result, result), dim=1), torch.cat((result, -result), dim=1)),
            dim=0,
        )
    return result / math.sqrt(order)


class Qwen3BlockRotationTest(unittest.TestCase):
    def test_fwht_matches_explicit_sylvester_matrix(self) -> None:
        torch.manual_seed(7)
        value = torch.randn(3, 8, 5, dtype=torch.float64)
        expected = torch.einsum("aib,ij->ajb", value, hadamard(8))
        actual = normalized_fwht(value, dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_folded_linear_algebra(self) -> None:
        torch.manual_seed(11)
        hidden = 8
        head_dim = 4
        query_heads = 2
        kv_heads = 1
        intermediate = 12
        source = {
            "input_layernorm.weight": torch.rand(hidden, dtype=torch.float64) + 0.5,
            "post_attention_layernorm.weight": torch.rand(hidden, dtype=torch.float64) + 0.5,
            "self_attn.q_proj.weight": torch.randn(hidden, hidden, dtype=torch.float64),
            "self_attn.k_proj.weight": torch.randn(head_dim, hidden, dtype=torch.float64),
            "self_attn.v_proj.weight": torch.randn(head_dim, hidden, dtype=torch.float64),
            "self_attn.o_proj.weight": torch.randn(hidden, hidden, dtype=torch.float64),
            "mlp.gate_proj.weight": torch.randn(intermediate, hidden, dtype=torch.float64),
            "mlp.up_proj.weight": torch.randn(intermediate, hidden, dtype=torch.float64),
            "mlp.down_proj.weight": torch.randn(hidden, intermediate, dtype=torch.float64),
        }
        folded = fold_layer_weights(
            source,
            hidden_size=hidden,
            head_dim=head_dim,
            query_heads=query_heads,
            kv_heads=kv_heads,
            rotate=True,
        )
        r1 = hadamard(hidden).to(torch.float32)
        r2 = hadamard(head_dim).to(torch.float32)
        x = torch.randn(3, hidden)
        x_rotated = x @ r1
        gamma_in = source["input_layernorm.weight"].float()
        gamma_post = source["post_attention_layernorm.weight"].float()

        for name in ("self_attn.q_proj.weight", "self_attn.k_proj.weight"):
            expected = (x * gamma_in) @ source[name].float().T
            actual = x_rotated @ folded[name].T
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

        expected_v = (x * gamma_in) @ source["self_attn.v_proj.weight"].float().T @ r2
        actual_v = x_rotated @ folded["self_attn.v_proj.weight"].T
        torch.testing.assert_close(actual_v, expected_v, rtol=2e-5, atol=2e-5)

        attention_heads = torch.randn(3, hidden)
        expected_o = attention_heads @ source["self_attn.o_proj.weight"].float().T @ r1
        actual_o = (attention_heads.reshape(3, query_heads, head_dim) @ r2).reshape(3, hidden)
        actual_o = actual_o @ folded["self_attn.o_proj.weight"].T
        torch.testing.assert_close(actual_o, expected_o, rtol=2e-5, atol=2e-5)

        for name in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            expected = (x * gamma_post) @ source[name].float().T
            actual = x_rotated @ folded[name].T
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

        mlp_hidden = torch.randn(3, intermediate)
        expected_down = mlp_hidden @ source["mlp.down_proj.weight"].float().T @ r1
        actual_down = mlp_hidden @ folded["mlp.down_proj.weight"].T
        torch.testing.assert_close(actual_down, expected_down, rtol=2e-5, atol=2e-5)

    def test_identity_control_only_folds_gamma(self) -> None:
        torch.manual_seed(13)
        hidden = 8
        source = {
            "input_layernorm.weight": torch.rand(hidden) + 0.5,
            "post_attention_layernorm.weight": torch.rand(hidden) + 0.5,
            "self_attn.q_proj.weight": torch.randn(hidden, hidden),
            "self_attn.k_proj.weight": torch.randn(4, hidden),
            "self_attn.v_proj.weight": torch.randn(4, hidden),
            "self_attn.o_proj.weight": torch.randn(hidden, hidden),
            "mlp.gate_proj.weight": torch.randn(12, hidden),
            "mlp.up_proj.weight": torch.randn(12, hidden),
            "mlp.down_proj.weight": torch.randn(hidden, 12),
        }
        folded = fold_layer_weights(
            source,
            hidden_size=hidden,
            head_dim=4,
            query_heads=2,
            kv_heads=1,
            rotate=False,
        )
        torch.testing.assert_close(
            folded["self_attn.q_proj.weight"],
            source["self_attn.q_proj.weight"] * source["input_layernorm.weight"],
        )
        torch.testing.assert_close(
            folded["self_attn.o_proj.weight"], source["self_attn.o_proj.weight"]
        )
        torch.testing.assert_close(
            folded["mlp.down_proj.weight"], source["mlp.down_proj.weight"]
        )


if __name__ == "__main__":
    unittest.main()
