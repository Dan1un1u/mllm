#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen3_block_optrace_report import (  # noqa: E402
    FORBIDDEN_ROTATION,
    group_stages,
    operator_signature,
)


class Qwen3BlockOptraceReportTest(unittest.TestCase):
    def test_stage_group_sums_static_and_dynamic_counters(self) -> None:
        source = []
        for stage, cycles in (("q_projection", 11), ("k_projection", 13), ("v_projection", 17)):
            source.append({
                "stage": stage,
                "qnn_op_instances": "1",
                "num_htp_ops": "2",
                "cycles": str(cycles),
                "num_dominant_path_cycles_htp_0": "3",
                "dram_read": "5",
                "dram_write": "7",
                "vtcm_read": "11",
                "vtcm_write": "13",
                "critical_path_us_estimate": "1.5",
                "kernel_resources": "DMA+HMX",
            })
        grouped = {row["group"]: row for row in group_stages(source)}
        projection = grouped["attention_qkv_projection"]
        self.assertEqual(projection["cycles"], 41)
        self.assertEqual(projection["qnn_op_instances"], 3)
        self.assertEqual(projection["num_htp_ops"], 6)
        self.assertEqual(projection["critical_path_us_estimate"], 4.5)
        self.assertEqual(projection["kernel_resources"], ["DMA", "HMX"])

    def test_operator_contract_excludes_single_capture_cycle_noise(self) -> None:
        base = {
            "qnn_op": "model.layers.5.self_attn.q_proj.0",
            "qnn_op_type": "Conv2d_w_blk_exp_scale",
            "stage": "q_projection",
            "kernel_resources": "DMA+HMX+HVX",
            "num_htp_ops": "12",
            "dram_read": "4096",
            "dram_write": "0",
            "vtcm_read": "1024",
            "vtcm_write": "2048",
            "cycles": "100",
        }
        noisy = dict(base, cycles="125")
        self.assertEqual(operator_signature([base]), operator_signature([noisy]))

    def test_runtime_rotation_name_is_rejected(self) -> None:
        self.assertIsNotNone(FORBIDDEN_ROTATION.search("runtime_hadamard_r1"))
        self.assertIsNone(FORBIDDEN_ROTATION.search("model.layers.5.self_attn.q_proj.0"))


if __name__ == "__main__":
    unittest.main()
