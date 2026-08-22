#!/usr/bin/env python3
"""Audit and summarize the K/V per-head versus packed LPBQ experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


PROJECTIONS = ("k_proj", "v_proj")
VARIANTS = ("per_head", "packed")
SEQUENCES = (1, 32)
HEADS = 8
INPUT_CHANNELS = 2048
HEAD_CHANNELS = 128
OUTPUT_CHANNELS = HEADS * HEAD_CHANNELS


def _tag(projection: str, variant: str, seq: int) -> str:
    return f"a8_{projection}_{variant}_split8_s{seq}_p19"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _targets(projection: str, variant: str) -> list[str]:
    prefix = f"model.layers.14.self_attn.{projection}"
    if variant == "packed":
        return [prefix]
    return [f"{prefix}.{head}" for head in range(HEADS)]


def _manifest(artifact_root: Path, projection: str, variant: str, seq: int) -> dict:
    tag = _tag(projection, variant, seq)
    case = artifact_root / "contexts" / tag
    path = case / "manifests" / f"model.0.s{seq}_quant_manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    operations = {item["name"]: item for item in document["operations"]}
    tensors = {item["name"]: item for item in document["tensors"]}
    targets = _targets(projection, variant)
    signatures = []
    output_qparams = []
    for target in targets:
        operation = operations.get(target)
        if operation is None:
            raise AssertionError(f"{tag}: missing target {target}")
        if operation["package"] != "qti.aisw" or operation["qnn_op_type"] != "Conv2d":
            raise AssertionError(f"{tag}: {target} is not qti.aisw::Conv2d")
        if len(operation["inputs"]) != 2 or len(operation["outputs"]) != 1:
            raise AssertionError(f"{tag}: unexpected target arity for {target}")
        weight = tensors[operation["inputs"][1]]
        recipe = weight["quant_recipe"]
        encoding = weight["qnn_quantization"]
        expected_output = HEAD_CHANNELS if variant == "per_head" else OUTPUT_CHANNELS
        if (
            weight["qnn_dtype"] != "SFIXED_POINT_8"
            or weight["dimensions"] != [1, 1, INPUT_CHANNELS, expected_output]
            or recipe.get("type") != "lpbq"
            or recipe.get("quant_min") != -7
            or recipe.get("quant_max") != 7
            or recipe.get("block_size") != 32
            or encoding.get("encoding") != "blockwise_expansion"
            or encoding.get("block_scale_bitwidth") != 4
        ):
            raise AssertionError(f"{tag}: invalid W4G32 LPBQ target weight for {target}")
        output = tensors[operation["outputs"][0]]
        if output["qnn_dtype"] != "UFIXED_POINT_8":
            raise AssertionError(f"{tag}: output is not asymmetric U8 for {target}")
        output_qparams.append(output["qnn_quantization"])
        signatures.append(
            {
                "name": target,
                "weight_dimensions": weight["dimensions"],
                "weight_dtype": weight["qnn_dtype"],
                "weight_recipe": recipe,
                "weight_encoding": encoding,
                "output_quantization": output["qnn_quantization"],
            }
        )
    if any(item != output_qparams[0] for item in output_qparams[1:]):
        raise AssertionError(f"{tag}: per-head outputs do not share one qparam")

    dynamic = [
        item
        for item in document["tensors"]
        if item["tensor_type"] != "STATIC" and "FIXED_POINT" in item["qnn_dtype"]
    ]
    for tensor in dynamic:
        recipe = tensor["quant_recipe"]
        if (
            tensor["qnn_dtype"] != "UFIXED_POINT_8"
            or recipe.get("type") != "asymmetric_per_tensor"
            or recipe.get("quant_to_dtype") != "UInt8"
            or recipe.get("quant_max") != 255
        ):
            raise AssertionError(f"{tag}: dynamic activation contract mismatch: {tensor['name']}")

    context = case / f"{tag}.bin"
    schematic = case / "schematics" / f"model.0.s{seq}_schematic.bin"
    return {
        "manifest": str(path.resolve()),
        "manifest_sha256": _sha256(path),
        "operation_count": len(document["operations"]),
        "target_count": len(targets),
        "targets": signatures,
        "common_output_qparam": output_qparams[0],
        "dynamic_u8_tensor_count": len(dynamic),
        "context_bytes": context.stat().st_size,
        "context_sha256": _sha256(context),
        "schematic_bytes": schematic.stat().st_size,
        "schematic_sha256": _sha256(schematic),
    }


def _codes(path: Path) -> bytes:
    data = path.read_bytes()
    if not data:
        raise AssertionError(f"empty output: {path}")
    return data


def _correctness(artifact_report: dict, result_root: Path, projection: str, seq: int) -> dict:
    outputs = {}
    for variant in VARIANTS:
        tag = _tag(projection, variant, seq)
        first_path = result_root / "correctness" / "first" / tag / "output.raw"
        repeat_path = result_root / "correctness" / "repeat" / tag / "output.raw"
        first = _codes(first_path)
        repeat = _codes(repeat_path)
        if len(first) != seq * OUTPUT_CHANNELS:
            raise AssertionError(f"{tag}: output byte count mismatch")
        if first != repeat:
            raise AssertionError(f"{tag}: repeated execution is not byte-exact")
        outputs[variant] = first
    if outputs["per_head"] != outputs["packed"]:
        deltas = [abs(left - right) for left, right in zip(outputs["per_head"], outputs["packed"], strict=True)]
        raise AssertionError(
            f"{projection} s{seq}: packed output differs; max code delta={max(deltas)}"
        )

    reference = artifact_report["host_reference"][projection][f"s{seq}"]
    observed = outputs["packed"]
    deltas = []
    for row, expected_row in zip(reference["rows"], reference["expected_codes"], strict=True):
        for output_index, expected in zip(reference["output_indices"], expected_row, strict=True):
            deltas.append(abs(observed[row * OUTPUT_CHANNELS + output_index] - expected))
    return {
        "bytes": len(outputs["packed"]),
        "sha256": hashlib.sha256(outputs["packed"]).hexdigest(),
        "repeatable": True,
        "per_head_equals_packed": True,
        "sampled_host_reference": {
            "samples": len(deltas),
            "equal_fraction": sum(delta == 0 for delta in deltas) / len(deltas),
            "mean_abs_code_delta": statistics.fmean(deltas),
            "max_abs_code_delta": max(deltas),
        },
    }


def _timing(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        values = [
            int(row["graph_execute_us"])
            for row in csv.DictReader(stream)
            if row["phase"] == "measured"
        ]
    if len(values) != 1000:
        raise AssertionError(f"expected 1000 measured samples in {path}, got {len(values)}")
    return statistics.median(values)


def _speed(result_root: Path, projection: str, seq: int) -> dict:
    rounds = {}
    for variant in VARIANTS:
        tag = _tag(projection, variant, seq)
        rounds[variant] = [
            _timing(result_root / "speed" / f"round{index}" / tag / "timing.csv")
            for index in range(1, 11)
        ]
    medians = {variant: statistics.median(values) for variant, values in rounds.items()}
    paired_delta = [
        packed - per_head
        for packed, per_head in zip(rounds["packed"], rounds["per_head"], strict=True)
    ]
    return {
        variant: {
            "process_medians_us": rounds[variant],
            "median_of_process_medians_us": medians[variant],
            "minimum_us": min(rounds[variant]),
            "maximum_us": max(rounds[variant]),
        }
        for variant in VARIANTS
    } | {
        "comparison": {
            "packed_minus_per_head_us": medians["packed"] - medians["per_head"],
            "packed_over_per_head": medians["packed"] / medians["per_head"],
            "packed_speedup_percent": (medians["per_head"] / medians["packed"] - 1.0) * 100.0,
            "paired_process_delta_us": paired_delta,
            "paired_median_delta_us": statistics.median(paired_delta),
            "packed_wins": sum(delta < 0 for delta in paired_delta),
            "ties": sum(delta == 0 for delta in paired_delta),
        }
    }


def _sum_physical(items: list[dict], predicate) -> dict:
    selected = [item for item in items if predicate(item["op"])]
    return {
        "physical_names": [item["op"] for item in selected],
        "instances": sum(item["instances"] for item in selected),
        "work_cycles": sum(item["cycles"] for item in selected),
        "dominant_path_cycles": sum(item["num_dominant_path_cycles_htp_0"] for item in selected),
        "dram_read": sum(item["dram_read"] for item in selected),
        "dram_write": sum(item["dram_write"] for item in selected),
        "vtcm_read": sum(item["vtcm_read"] for item in selected),
        "vtcm_write": sum(item["vtcm_write"] for item in selected),
    }


def _event_sum(items: list[dict], predicate) -> dict:
    selected = [item for item in items if predicate(item)]
    return {
        "instances": len(selected),
        "work_cycles": sum(item["cycles"] for item in selected),
        "dominant_path_cycles": sum(item["num_dominant_path_cycles"] for item in selected),
        "dram_read": sum(item["dram_read"] for item in selected),
        "dram_write": sum(item["dram_write"] for item in selected),
        "vtcm_read": sum(item["vtcm_read"] for item in selected),
        "vtcm_write": sum(item["vtcm_write"] for item in selected),
    }


def _optrace(result_root: Path, projection: str, variant: str, seq: int) -> dict:
    tag = _tag(projection, variant, seq)
    directory = result_root / "optrace" / tag
    base = directory / tag
    qhas_path = base.with_name(base.name + "-chrometrace_qnn_htp_analysis_summary.json")
    qhas = json.loads(qhas_path.read_text(encoding="utf-8"))["data"]
    overall = qhas["htp_overall_summary"]["data"][0]
    physical = qhas["htp_op_types"]["data"]
    events = qhas["htp_op_instances"]["data"]
    target_prefix = f"model.layers.14.self_attn.{projection}"
    qnn_nodes = [
        item
        for item in qhas["qnn_op_instances_nodes"]["data"]
        if item["qnn_op"].startswith(target_prefix)
    ]
    weight_events = [item for item in events if "weights_to_vtcm" in item["htp_op"]]
    return {
        "overall": {
            key: overall[key]
            for key in (
                "graph_execute_us", "timeline_cycles", "peak_vtcm_alloc", "total_dram_read",
                "total_dram_write", "total_vtcm_read", "total_vtcm_write", "qnn_nodes", "htp_nodes"
            )
        },
        "logical_projection": {
            "qnn_nodes": len(qnn_nodes),
            "work_cycles": sum(item["cycles"] for item in qnn_nodes),
            "dominant_path_cycles": sum(item["num_dominant_path_cycles_htp_0"] for item in qnn_nodes),
            "physical_nodes": sum(item["num_htp_ops"] for item in qnn_nodes),
            "dram_read": sum(item["dram_read"] for item in qnn_nodes),
            "dram_write": sum(item["dram_write"] for item in qnn_nodes),
        },
        "physical": {
            "weights_to_vtcm": _sum_physical(physical, lambda name: "weights_to_vtcm" in name),
            "w4_expand": _sum_physical(physical, lambda name: "expand_block_quant_to_pc_int8_weights" in name),
            "hmx": _sum_physical(
                physical, lambda name: name.startswith("q::ConvLayer_s") and name.endswith(".opt")
            ),
            "bias_to_vtcm": _sum_physical(physical, lambda name: "bias_to_vtcm" in name),
            "checkpoint_sync": _sum_physical(
                physical, lambda name: "Checkpoint" in name or "Sync" in name or "Wait" in name
            ),
            "output": _sum_physical(physical, lambda name: "OutputSlice" in name),
        },
        "weight_dma_events": {
            "all": _event_sum(weight_events, lambda _item: True),
            "payload": _event_sum(weight_events, lambda item: item["dma"] and not item["dma_wait"]),
            "dependency_wait": _event_sum(weight_events, lambda item: item["dma_wait"]),
        },
        "checkpoint_events": _event_sum(
            events, lambda item: item["dma_set"] or "Checkpoint" in item["htp_op"]
        ),
        "raw_optrace": str((directory / "qnn-profiling-data.log").resolve()),
    }


def _ratio(packed: int | float, per_head: int | float) -> float | None:
    return packed / per_head if per_head else None


def _optrace_comparison(per_head: dict, packed: dict) -> dict:
    result = {
        "overall": {
            field: {
                "per_head": per_head["overall"][field],
                "packed": packed["overall"][field],
                "packed_over_per_head": _ratio(packed["overall"][field], per_head["overall"][field]),
            }
            for field in ("graph_execute_us", "timeline_cycles", "peak_vtcm_alloc", "total_dram_read", "total_dram_write")
        },
        "logical_projection": {},
        "physical": {},
        "weight_dma_events": {},
        "checkpoint_events": {},
    }
    for field in ("qnn_nodes", "work_cycles", "dominant_path_cycles", "physical_nodes", "dram_read", "dram_write"):
        result["logical_projection"][field] = {
            "per_head": per_head["logical_projection"][field],
            "packed": packed["logical_projection"][field],
            "packed_over_per_head": _ratio(packed["logical_projection"][field], per_head["logical_projection"][field]),
        }
    for category in ("weights_to_vtcm", "w4_expand", "hmx", "bias_to_vtcm", "checkpoint_sync", "output"):
        result["physical"][category] = {}
        for field in ("instances", "work_cycles", "dominant_path_cycles", "dram_read", "dram_write", "vtcm_read", "vtcm_write"):
            result["physical"][category][field] = {
                "per_head": per_head["physical"][category][field],
                "packed": packed["physical"][category][field],
                "packed_over_per_head": _ratio(packed["physical"][category][field], per_head["physical"][category][field]),
            }
    for category in ("all", "payload", "dependency_wait"):
        result["weight_dma_events"][category] = {}
        for field in ("instances", "work_cycles", "dominant_path_cycles", "dram_read", "vtcm_write"):
            result["weight_dma_events"][category][field] = {
                "per_head": per_head["weight_dma_events"][category][field],
                "packed": packed["weight_dma_events"][category][field],
                "packed_over_per_head": _ratio(
                    packed["weight_dma_events"][category][field], per_head["weight_dma_events"][category][field]
                ),
            }
    for field in ("instances", "work_cycles", "dominant_path_cycles"):
        result["checkpoint_events"][field] = {
            "per_head": per_head["checkpoint_events"][field],
            "packed": packed["checkpoint_events"][field],
            "packed_over_per_head": _ratio(packed["checkpoint_events"][field], per_head["checkpoint_events"][field]),
        }
    return result


def summarize(artifact_root: Path, result_root: Path) -> dict:
    artifact_report = json.loads((artifact_root / "artifact_report.json").read_text(encoding="utf-8"))
    manifests = {}
    correctness = {}
    speed = {}
    optrace = {}
    comparisons = {}
    for projection in PROJECTIONS:
        for seq in SEQUENCES:
            key = f"{projection}_s{seq}"
            manifests[key] = {
                variant: _manifest(artifact_root, projection, variant, seq)
                for variant in VARIANTS
            }
            correctness[key] = _correctness(artifact_report, result_root, projection, seq)
            speed[key] = _speed(result_root, projection, seq)
            for variant in VARIANTS:
                optrace[_tag(projection, variant, seq)] = _optrace(
                    result_root, projection, variant, seq
                )
            comparisons[key] = _optrace_comparison(
                optrace[_tag(projection, "per_head", seq)],
                optrace[_tag(projection, "packed", seq)],
            )
    non_regression = all(
        item["packed"]["median_of_process_medians_us"]
        <= item["per_head"]["median_of_process_medians_us"]
        for item in speed.values()
    )
    speedups = [item["comparison"]["packed_speedup_percent"] for item in speed.values()]
    paired_wins = [item["comparison"]["packed_wins"] for item in speed.values()]
    mixed_optrace_direction = any(
        item["overall"]["timeline_cycles"]["packed_over_per_head"] < 1.0
        for item in comparisons.values()
    ) and any(
        item["overall"]["timeline_cycles"]["packed_over_per_head"] > 1.0
        for item in comparisons.values()
    )
    return {
        "gate": {
            "correctness": "PASS",
            "literal_non_regression": "PASS" if non_regression else "FAIL",
            "optimization_signal": "INCONCLUSIVE",
            "full_model_migration": "STOP_PENDING_DISCUSSION",
            "reason": (
                "All four aggregate medians narrowly favor packed, but gains are only "
                f"{min(speedups):.2f}% to {max(speedups):.2f}%, paired wins are "
                f"{min(paired_wins)} to {max(paired_wins)} of ten, and Optrace timeline "
                f"direction is {'mixed' if mixed_optrace_direction else 'uniform'}."
            ),
        },
        "contract": {
            "qairt": "2.47.0.260601",
            "target": "SM8750 / V79",
            "finalize_p": 19,
            "activation": "asymmetric U8, identical input/output qparams",
            "static_weight": "byte-equivalent signed W4[-7,7] G32 LPBQ",
            "only_variable": "8x 2048x128 Conv2d versus 1x 2048x1024 Conv2d followed by an eight-head split",
            "graph_boundary": "both variants expose eight [1,S,128] outputs with identical qparams",
            "timing": "ten paired fresh-process rounds, alternating order, 50 warmup + 1000 measured",
        },
        "artifact": artifact_report,
        "manifests": manifests,
        "correctness": correctness,
        "speed": speed,
        "optrace": optrace,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = summarize(args.artifact_root, args.result_root)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if not args.quiet:
        print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
