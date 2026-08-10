from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen3_block_device_oracle import QuantizedCheckpoint, metrics


class FakeHandle:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.tensors[key]


class DeviceOracleTest(unittest.TestCase):
    def test_per_head_qdq_alias_uses_shared_checkpoint_tensor(self) -> None:
        tensors = {
            "model.layers.5.self_attn.q_norm_input_qdq.fake_quant.scale": torch.tensor([0.25]),
            "model.layers.5.self_attn.q_norm_input_qdq.fake_quant.zero_point": torch.tensor([7]),
        }
        checkpoint = QuantizedCheckpoint(FakeHandle(tensors))
        actual = checkpoint.qdq(
            torch.tensor([0.0, 0.26]),
            "model.layers.5.self_attn.q_norm_input_qdq_h13",
        )
        torch.testing.assert_close(actual, torch.tensor([0.0, 0.25]))

    def test_lpbq_decoder_reconstructs_signed_g32_carrier(self) -> None:
        carrier = torch.zeros((1, 1, 32, 2), dtype=torch.uint8)
        carrier[0, 0, :, 0] = 0x01
        carrier[0, 0, :, 1] = 0x0F
        tensors = {
            "linear.weight": carrier,
            "linear.scale1": torch.tensor([2, 3], dtype=torch.uint8),
            "linear.scale2": torch.tensor([0.5, 0.25], dtype=torch.float32),
        }
        decoded = QuantizedCheckpoint(FakeHandle(tensors)).lpbq_weight("linear")
        torch.testing.assert_close(decoded[0], torch.ones(32))
        torch.testing.assert_close(decoded[1], torch.full((32,), -0.75))

    def test_metrics_identical_tensor_is_exact(self) -> None:
        result = metrics(torch.arange(8.0), torch.arange(8.0))
        self.assertEqual(result["relative_l2"], 0.0)
        self.assertEqual(result["max_abs"], 0.0)
        self.assertEqual(result["cosine_similarity"], 1.0)


if __name__ == "__main__":
    unittest.main()
