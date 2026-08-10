#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from safetensors.torch import save_file


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_qwen3_lpbq_g32 import _quantize_lpbq_g32  # noqa: E402
from qwen3_rotation_manifest import (  # noqa: E402
    build_export_manifest_v2,
    build_input_records,
    finalize_manifest,
    independent_hadamard_matrix,
    validate_export_manifest,
)


class Qwen3RotationManifestTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        source_dir = root / "source"
        source_dir.mkdir()
        source_shard = source_dir / "model-00001-of-00001.safetensors"
        source_shard.write_bytes(b"source-shard")
        base = root / "base.safetensors"
        base.write_bytes(b"base-checkpoint")

        order = 8
        group_size = 4
        r3 = independent_hadamard_matrix(order)
        packed, scale1, scale2, _ = _quantize_lpbq_g32(r3, group_size)
        prefix = "model.layers.5.self_attn.r3_dense"
        checkpoint = root / "D_dense_r3" / "model.safetensors"
        checkpoint.parent.mkdir()
        save_file(
            {
                prefix + ".weight": packed.reshape(order, order),
                prefix + ".scale1": scale1,
                prefix + ".scale2": scale2,
                "model.layers.5.test": r3[:1].clone(),
                "model.layers.5.scalar": r3[0, 0].clone(),
            },
            str(checkpoint),
            metadata={
                "mllm.block.variant": "D_dense_r3",
                "mllm.block.r3_mode": "dense",
            },
        )

        legacy = {
            "model": "Qwen3-test",
            "layer": 5,
            "variant": "D_dense_r3",
            "rotation": "R1/R2/R3 normalized Sylvester Hadamard",
            "r3_mode": "dense",
            "key_cache_boundary": "R3",
            "group_size": group_size,
            "dimensions": {
                "hidden_size": order,
                "head_dim": order,
                "query_heads": 2,
                "kv_heads": 1,
            },
            "activation_qdq": "test",
            "contract": {
                "weight_dtype": "int8_carrier_uint4",
                "weight_layout": "IO",
                "scale1_dtype": "uint8",
                "scale1_layout": "flattened",
                "scale2_dtype": "float32",
                "scale2_layout": "output",
                "hidden_boundary": "R1",
                "value_cache_boundary": "R2",
                "key_cache_boundary": "R3",
            },
            "weights": {
                prefix + ".weight": {
                    "canonical_shape": [order, order],
                    "deployed_shape": [order, order],
                }
            },
        }
        manifest = build_export_manifest_v2(
            legacy,
            checkpoint,
            inputs=build_input_records(source_dir, [source_shard], base),
            provenance={"test": True},
        )
        manifest_path = checkpoint.parent / "export_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest_path

    def test_validates_tensor_inventory_inputs_and_exact_r3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._fixture(Path(temporary))
            report = validate_export_manifest(manifest_path)
            self.assertTrue(report["pass"], report["errors"])
            self.assertTrue(report["checks"]["r3_dense_carrier"]["exact"])
            self.assertEqual(
                report["checks"]["r3_dense_carrier"]["max_abs_error"], 0.0
            )

    def test_rejects_self_consistent_manifest_with_wrong_basis_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._fixture(Path(temporary))
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["rotation_contract"]["r3"]["float32_matrix_sha256"] = "0" * 64
            payload = finalize_manifest(payload)
            manifest_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            report = validate_export_manifest(manifest_path)
            self.assertFalse(report["pass"])
            self.assertTrue(
                any("rotation_contract.r3" in error for error in report["errors"])
            )

    def test_rejects_checkpoint_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._fixture(Path(temporary))
            checkpoint = manifest_path.parent / "model.safetensors"
            with checkpoint.open("ab") as stream:
                stream.write(b"tamper")
            report = validate_export_manifest(manifest_path)
            self.assertFalse(report["pass"])
            self.assertTrue(
                any("artifact.checkpoint" in error for error in report["errors"])
            )


if __name__ == "__main__":
    unittest.main()
