#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen3_block_r3_report import VARIANTS, WORKLOADS, build_report  # noqa: E402


class Qwen3BlockR3ReportTest(unittest.TestCase):
    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_combines_timing_optrace_and_placement_without_perf_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timing_root = root / "timing"
            optrace_root = root / "optrace"
            artifact_root = root / "artifacts"
            medians = {
                "C": (100.0, 200.0),
                "D-Dense": (99.0, 202.0),
                "D-FWHT-Graph": (130.0, 800.0),
            }

            for short, variant in VARIANTS.items():
                results = []
                for index, (_, timing_name) in enumerate(WORKLOADS.items()):
                    median = medians[short][index]
                    results.append({
                        "workload": timing_name,
                        "median_us": median,
                        "p95_us": median + 2,
                        "mean_us": median + 1,
                        "min_us": median - 1,
                        "max_us": median + 3,
                        "samples_us": [int(median)] * 200,
                    })
                self._write_json(
                    timing_root / variant["name"] / "timing.json",
                    {
                        "variant": variant["name"],
                        "warmup": 20,
                        "iterations": 200,
                        "timing_boundary": "QnnGraph_execute",
                        "results": results,
                    },
                )

                for workload in WORKLOADS:
                    directory = optrace_root / variant["optrace_dir"] / workload
                    self._write_json(
                        directory / "chrometrace_qnn_htp_analysis_summary.json",
                        {
                            "data": {
                                "htp_overall_summary": {
                                    "data": [{
                                        "graph_execute_us": 500,
                                        "timeline_cycles": 1000,
                                        "qnn_nodes": 10,
                                        "htp_nodes": 20,
                                        "total_dram_read": 1,
                                        "total_dram_write": 2,
                                        "total_vtcm_read": 3,
                                        "total_vtcm_write": 4,
                                        "peak_vtcm_alloc": 5,
                                    }]
                                }
                            }
                        },
                    )
                    directory.mkdir(parents=True, exist_ok=True)
                    stage_path = directory / "structure-qwen3-stage-summary.csv"
                    fields = (
                        "stage", "qnn_op_instances", "num_htp_ops", "cycles",
                        "critical_path_us_estimate", "dram_read", "dram_write",
                        "vtcm_read", "vtcm_write", "kernel_resources",
                    )
                    with stage_path.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=fields)
                        writer.writeheader()
                        writer.writerow({
                            "stage": "q_rope",
                            "qnn_op_instances": 2,
                            "num_htp_ops": 3,
                            "cycles": 4,
                            "critical_path_us_estimate": 6.0,
                            "dram_read": 7,
                            "dram_write": 8,
                            "vtcm_read": 9,
                            "vtcm_write": 10,
                            "kernel_resources": "HVX",
                        })

                    artifacts = artifact_root / variant["artifact_dir"]
                    self._write_json(
                        artifacts / "manifests" / f"model.0.{workload}_quant_manifest.json",
                        {"operations": [{"package": "qti.aisw"}]},
                    )
                    mir = artifacts / "mir" / f"qwen3_layer5_block_{workload}.mir"
                    mir.parent.mkdir(parents=True, exist_ok=True)
                    mir.write_text("using_qnn:true\n", encoding="utf-8")

            math_path = root / "math.json"
            self._write_json(math_path, {"results": [{"pass": True}, {"pass": True}]})
            report, stages = build_report(
                timing_root, optrace_root, artifact_root, math_path
            )

            self.assertEqual(
                report["comparison_policy"], "exploratory_no_performance_pass_fail"
            )
            self.assertFalse(
                report["prototype_completion"]["performance_gate_applied"]
            )
            self.assertEqual(report["prototype_completion"]["status"], "complete")
            self.assertAlmostEqual(
                report["timing"]["s1"]["D-Dense"]["median_change_percent_vs_c"],
                -1.0,
            )
            self.assertAlmostEqual(
                report["timing"]["s32"]["D-FWHT-Graph"][
                    "median_change_percent_vs_c"
                ],
                300.0,
            )
            self.assertTrue(
                report["internal_htp_optrace"]["s1"]["D-FWHT-Graph"][
                    "placement"
                ]["fully_qnn_placed"]
            )
            self.assertEqual(len(stages), len(VARIANTS) * len(WORKLOADS))


if __name__ == "__main__":
    unittest.main()
