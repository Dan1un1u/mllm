#!/usr/bin/env python3
"""Audit and summarize the strict LPBQ A16-versus-A8 projection experiment."""

from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path


PROJECTIONS = {
    "gate_proj": (2048, 6144),
    "up_proj": (2048, 6144),
    "down_proj": (6144, 2048),
    "lm_head": (2048, 151936),
}
ACTIVATIONS = ("a16", "a8")
SEQUENCES = (1, 32)


def _tag(activation: str, projection: str, seq: int) -> str:
    return f"{activation}_{projection}_s{seq}"


def _op_name(projection: str) -> str:
    return "model.lm_head" if projection == "lm_head" else f"model.layers.14.mlp.{projection}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(artifact_root: Path, activation: str, projection: str, seq: int) -> dict[str, object]:
    tag = _tag(activation, projection, seq)
    case = artifact_root / "contexts" / tag
    path = case / "manifests" / f"model.0.s{seq}_quant_manifest.json"
    context = case / f"{tag}.bin"
    schematic = case / "schematics" / f"model.0.s{seq}_schematic.bin"
    document = json.loads(path.read_text(encoding="utf-8"))
    operations = {item["name"]: item for item in document["operations"]}
    target = _op_name(projection)
    if target not in operations:
        raise AssertionError(f"{tag}: target operation is missing: {target}")
    if any(item["package"] != "qti.aisw" for item in document["operations"]):
        raise AssertionError(f"{tag}: non-QNN package observed")
    operation = operations[target]
    if operation["qnn_op_type"] != "Conv2d" or len(operation["inputs"]) != 2:
        raise AssertionError(f"{tag}: target is not a two-input QNN Conv2d")
    tensors = {item["name"]: item for item in document["tensors"]}
    weight = tensors[operation["inputs"][1]]
    recipe = weight["quant_recipe"]
    encoding = weight["qnn_quantization"]
    if (
        weight["qnn_dtype"] != "SFIXED_POINT_8"
        or recipe.get("type") != "lpbq"
        or recipe.get("quant_min") != -7
        or recipe.get("quant_max") != 7
        or recipe.get("block_size") != 32
        or encoding.get("encoding") != "blockwise_expansion"
        or encoding.get("block_scale_bitwidth") != 4
    ):
        raise AssertionError(f"{tag}: static weight is not signed W4G32 LPBQ")

    expected_dtype = "UFIXED_POINT_16" if activation == "a16" else "UFIXED_POINT_8"
    expected_quant_to = "UInt16" if activation == "a16" else "UInt8"
    expected_max = 65535 if activation == "a16" else 255
    dynamic = [
        item for item in document["tensors"]
        if item["tensor_type"] != "STATIC" and "FIXED_POINT" in item["qnn_dtype"]
    ]
    if not dynamic:
        raise AssertionError(f"{tag}: no dynamic quantized tensors")
    for tensor in dynamic:
        qrecipe = tensor["quant_recipe"]
        if (
            tensor["qnn_dtype"] != expected_dtype
            or qrecipe.get("type") != "asymmetric_per_tensor"
            or qrecipe.get("quant_to_dtype") != expected_quant_to
            or qrecipe.get("quant_max") != expected_max
        ):
            raise AssertionError(f"{tag}: activation tensor contract mismatch: {tensor['name']}")

    weight_signature = {
        "dimensions": weight["dimensions"],
        "qnn_dtype": weight["qnn_dtype"],
        "quant_recipe": recipe,
        "qnn_quantization": encoding,
    }
    return {
        "manifest": str(path.resolve()),
        "manifest_sha256": _sha256(path),
        "context_bytes": context.stat().st_size,
        "context_sha256": _sha256(context),
        "schematic_bytes": schematic.stat().st_size,
        "schematic_sha256": _sha256(schematic),
        "operation_count": len(document["operations"]),
        "target": {"name": target, "type": operation["qnn_op_type"], "package": operation["package"]},
        "activation_dtype": expected_dtype,
        "dynamic_tensor_count": len(dynamic),
        "weight_signature": weight_signature,
    }


def _read_codes(path: Path, activation: str) -> array.array:
    typecode = "H" if activation == "a16" else "B"
    result = array.array(typecode)
    with path.open("rb") as stream:
        result.frombytes(stream.read())
    if activation == "a16" and sys.byteorder != "little":
        result.byteswap()
    return result


def _correctness(artifact_report: dict[str, object], result_root: Path,
                 activation: str, projection: str, seq: int) -> dict[str, object]:
    tag = _tag(activation, projection, seq)
    first_path = result_root / "correctness" / "first" / tag / "output.raw"
    repeat_path = result_root / "correctness" / "repeat" / tag / "output.raw"
    first = first_path.read_bytes()
    repeat = repeat_path.read_bytes()
    output_channels = PROJECTIONS[projection][1]
    expected_bytes = seq * output_channels * (2 if activation == "a16" else 1)
    if len(first) != expected_bytes or len(repeat) != expected_bytes:
        raise AssertionError(f"{tag}: output size mismatch")
    if first != repeat:
        raise AssertionError(f"{tag}: repeated execution is not byte-exact")

    reference = artifact_report["host_reference"][projection][f"s{seq}_{activation}"]
    observed = _read_codes(first_path, activation)
    deltas = []
    for row, expected_row in zip(reference["rows"], reference["expected_codes"], strict=True):
        for output_index, expected in zip(reference["output_indices"], expected_row, strict=True):
            actual = observed[row * output_channels + output_index]
            deltas.append(abs(int(actual) - int(expected)))
    return {
        "bytes": len(first),
        "sha256": _sha256(first_path),
        "repeatable": True,
        "sample_count": len(deltas),
        "host_reference_equal_fraction": sum(delta == 0 for delta in deltas) / len(deltas),
        "host_reference_mean_abs_code_delta": statistics.fmean(deltas),
        "host_reference_max_abs_code_delta": max(deltas),
    }


def _timing(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        values = [
            int(row["graph_execute_us"])
            for row in csv.DictReader(stream)
            if row["phase"] == "measured"
        ]
    if len(values) != 500:
        raise AssertionError(f"expected 500 measured samples in {path}, got {len(values)}")
    return statistics.median(values)


def _speed(result_root: Path, projection: str, seq: int) -> dict[str, object]:
    result: dict[str, object] = {}
    aggregate: dict[str, float] = {}
    for activation in ACTIVATIONS:
        tag = _tag(activation, projection, seq)
        rounds = [
            _timing(result_root / "speed" / f"round{index}" / tag / "timing.csv")
            for index in range(1, 6)
        ]
        aggregate[activation] = statistics.median(rounds)
        result[activation] = {
            "process_medians_us": rounds,
            "median_of_process_medians_us": aggregate[activation],
            "min_process_median_us": min(rounds),
            "max_process_median_us": max(rounds),
        }
    result["comparison"] = {
        "a8_minus_a16_us": aggregate["a8"] - aggregate["a16"],
        "a8_over_a16": aggregate["a8"] / aggregate["a16"],
        "a8_speedup_percent": (aggregate["a16"] / aggregate["a8"] - 1.0) * 100.0,
    }
    return result


def _aggregate_resources(items: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, dict[str, int | float]] = {}
    for item in items:
        resource = result.setdefault(
            item["type"],
            {"lanes": 0, "timeline_max_cycles": 0, "cycles_used": 0,
             "dram_read": 0, "dram_write": 0, "vtcm_read": 0, "vtcm_write": 0},
        )
        resource["lanes"] += 1
        resource["timeline_max_cycles"] = max(resource["timeline_max_cycles"], item["timeline_cycles"])
        for field in ("cycles_used", "dram_read", "dram_write", "vtcm_read", "vtcm_write"):
            resource[field] += item[field]
    for resource in result.values():
        capacity = resource["timeline_max_cycles"] * resource["lanes"]
        resource["utilization_percent"] = 100.0 * resource["cycles_used"] / capacity if capacity else 0.0
        resource["idle_capacity_cycles"] = capacity - resource["cycles_used"]
    return result


def _kernel(item: dict[str, object]) -> dict[str, object]:
    return {
        "physical_name": item["op"],
        "instances": item["instances"],
        "work_cycles": item["cycles"],
        "dominant_path_cycles": item["num_dominant_path_cycles_htp_0"],
        "dram_read": item["dram_read"],
        "dram_write": item["dram_write"],
        "vtcm_read": item["vtcm_read"],
        "vtcm_write": item["vtcm_write"],
    }


def _optrace(result_root: Path, activation: str, projection: str, seq: int) -> dict[str, object]:
    tag = _tag(activation, projection, seq)
    directory = result_root / "optrace" / tag
    base = directory / tag
    qhas_path = base.with_name(base.name + "-chrometrace_qnn_htp_analysis_summary.json")
    operators_path = base.with_name(base.name + "-operators.csv")
    raw_path = directory / "qnn-profiling-data.log"
    qhas = json.loads(qhas_path.read_text(encoding="utf-8"))["data"]
    op_types = qhas["htp_op_types"]["data"]

    selected: dict[str, object] = {}
    dma_sync = []
    input_boundary = []
    output_boundary = []
    physical_names = []
    for item in op_types:
        name = item["op"]
        physical_names.append(name)
        if "weights_to_vtcm" in name:
            selected["weights_to_vtcm"] = _kernel(item)
        elif "expand_block_quant_to_pc_int8_weights" in name:
            selected["w4_expand"] = _kernel(item)
        elif name.startswith("q::ConvLayer_s") and name.endswith(".opt"):
            selected["hmx_projection"] = _kernel(item)
        if "Checkpoint" in name or "Sync" in name or "Wait" in name:
            dma_sync.append(_kernel(item))
        if "InputSlice" in name or "activations_to_vtcm" in name or "ForceFormat_Crouton" in name:
            input_boundary.append(_kernel(item))
        if "OutputSlice" in name or "ForceFormat_Flat" in name or "requant" in name.lower():
            output_boundary.append(_kernel(item))
    for required in ("weights_to_vtcm", "w4_expand", "hmx_projection"):
        if required not in selected:
            raise AssertionError(f"{tag}: missing physical kernel category {required}")

    target = _op_name(projection)
    operator = None
    with operators_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["qnn_op_name"] == target:
                operator = {
                    "qnn_op_type": row["qnn_op_type"],
                    "execution_domain": row["execution_domain"],
                    "kernel_resources": row["kernel_resources"],
                    "trace_lanes": row["trace_lanes"],
                    "wall_span_cycles": int(row["wall_span_cycles"]),
                    "active_union_cycles": int(row["active_union_cycles"]),
                    "total_work_cycles": int(row["total_work_cycles"]),
                    "overlapped_work_cycles": int(row["overlapped_work_cycles"]),
                    "max_parallelism": int(row["max_parallelism"]),
                }
                break
    if operator is None:
        raise AssertionError(f"{tag}: projection missing from operator CSV")
    return {
        "raw_bytes": raw_path.stat().st_size,
        "raw_sha256": _sha256(raw_path),
        "resources": _aggregate_resources(qhas["htp_overall_summary"]["data"][0]["htp_resources"]["data"]),
        "selected_kernels": selected,
        "dma_sync_kernels": dma_sync,
        "input_boundary_kernels": input_boundary,
        "output_boundary_kernels": output_boundary,
        "physical_kernel_names": physical_names,
        "projection_operator": operator,
    }


def _ratio(a8: float, a16: float) -> float | None:
    return a8 / a16 if a16 else None


def _pair_analysis(a16: dict[str, object], a8: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for category in ("weights_to_vtcm", "w4_expand", "hmx_projection"):
        left = a16["selected_kernels"][category]
        right = a8["selected_kernels"][category]
        result[category] = {
            "a16": left,
            "a8": right,
            "work_ratio_a8_over_a16": _ratio(right["work_cycles"], left["work_cycles"]),
            "dominant_ratio_a8_over_a16": _ratio(right["dominant_path_cycles"], left["dominant_path_cycles"]),
            "instance_delta_a8_minus_a16": right["instances"] - left["instances"],
            "dram_read_delta_a8_minus_a16": right["dram_read"] - left["dram_read"],
            "vtcm_read_delta_a8_minus_a16": right["vtcm_read"] - left["vtcm_read"],
            "vtcm_write_delta_a8_minus_a16": right["vtcm_write"] - left["vtcm_write"],
        }
    for resource in ("HMX", "HVX"):
        left = a16["resources"][resource]
        right = a8["resources"][resource]
        result[resource.lower() + "_resource"] = {
            "a16": left,
            "a8": right,
            "timeline_ratio_a8_over_a16": _ratio(right["timeline_max_cycles"], left["timeline_max_cycles"]),
            "busy_ratio_a8_over_a16": _ratio(right["cycles_used"], left["cycles_used"]),
            "idle_capacity_ratio_a8_over_a16": _ratio(right["idle_capacity_cycles"], left["idle_capacity_cycles"]),
        }
    left_op = a16["projection_operator"]
    right_op = a8["projection_operator"]
    result["projection_operator"] = {
        "a16": left_op,
        "a8": right_op,
        "wall_ratio_a8_over_a16": _ratio(right_op["wall_span_cycles"], left_op["wall_span_cycles"]),
        "active_ratio_a8_over_a16": _ratio(right_op["active_union_cycles"], left_op["active_union_cycles"]),
        "work_ratio_a8_over_a16": _ratio(right_op["total_work_cycles"], left_op["total_work_cycles"]),
    }
    result["dma_sync"] = {"a16": a16["dma_sync_kernels"], "a8": a8["dma_sync_kernels"]}
    result["input_boundary"] = {
        "a16": a16["input_boundary_kernels"],
        "a8": a8["input_boundary_kernels"],
    }
    result["output_boundary"] = {
        "a16": a16["output_boundary_kernels"],
        "a8": a8["output_boundary_kernels"],
        "explicit_requant_kernel_a16": any("requant" in item["physical_name"].lower() for item in a16["output_boundary_kernels"]),
        "explicit_requant_kernel_a8": any("requant" in item["physical_name"].lower() for item in a8["output_boundary_kernels"]),
    }
    return result


def summarize(artifact_root: Path, result_root: Path) -> dict[str, object]:
    artifact_report = json.loads((artifact_root / "artifact_report.json").read_text(encoding="utf-8"))
    manifests: dict[str, object] = {}
    correctness: dict[str, object] = {}
    optrace: dict[str, object] = {}
    pair_analysis: dict[str, object] = {}
    speed: dict[str, object] = {}
    for projection in PROJECTIONS:
        for seq in SEQUENCES:
            pair_key = f"{projection}_s{seq}"
            pair_manifests = {
                activation: _manifest(artifact_root, activation, projection, seq)
                for activation in ACTIVATIONS
            }
            if pair_manifests["a16"]["weight_signature"] != pair_manifests["a8"]["weight_signature"]:
                raise AssertionError(f"{pair_key}: finalized LPBQ weight signatures differ")
            manifests[pair_key] = {
                **pair_manifests,
                "finalized_weight_signature_identical": True,
            }
            speed[pair_key] = _speed(result_root, projection, seq)
            for activation in ACTIVATIONS:
                tag = _tag(activation, projection, seq)
                correctness[tag] = _correctness(
                    artifact_report, result_root, activation, projection, seq
                )
                optrace[tag] = _optrace(result_root, activation, projection, seq)
            pair_analysis[pair_key] = _pair_analysis(
                optrace[_tag("a16", projection, seq)],
                optrace[_tag("a8", projection, seq)],
            )
    return {
        "gate": "PASS",
        "gate_scope": "strict single-variable integrity, deterministic execution, sampled host math; no speed or accuracy threshold",
        "contract": {
            "qairt": "2.47.0.260601",
            "device": "SM8750 / V79",
            "static_weight": "one common signed W4[-7,7] G32 LPBQ carrier + UInt4 block scales + FP32 channel scales",
            "physical_expression": "qti.aisw::Conv2d with NHWC activation and HWIO static weight",
            "only_variable": "asymmetric activation input/output contract and formal qparams: U16 versus U8",
            "timing": "profiling off; five fresh-process paired rounds; alternating order; 20 warmup + 500 measured",
            "optrace": "one fresh-process capture per activation/projection/sequence case",
        },
        "artifact_audit": {
            "common_artifact": artifact_report["artifact"],
            "archived_model_weights_byte_identical": {
                projection: artifact_report["weights"][projection]["archived_models_byte_identical"]
                for projection in PROJECTIONS
            },
            "common_weight_source": artifact_report["contract"]["common_static_weight_source"],
        },
        "manifests": manifests,
        "correctness": correctness,
        "speed": speed,
        "optrace": optrace,
        "pair_analysis": pair_analysis,
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
