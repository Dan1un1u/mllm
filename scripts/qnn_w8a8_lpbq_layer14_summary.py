#!/usr/bin/env python3
"""Audit and summarize the layer-14 W8A8 versus LPBQ experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path, PureWindowsPath

import numpy as np

import qnn_w8a8_lpbq_layer14_artifact as artifact


PROJECTIONS = (
    "model.layers.14.mlp.gate_proj",
    "model.layers.14.mlp.up_proj",
    "model.layers.14.mlp.down_proj",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_audit(artifact_root: Path, variant: str, seq: int) -> dict[str, object]:
    case = artifact_root / "contexts" / f"{variant}_s{seq}"
    manifest_path = case / "manifests" / f"model.0.s{seq}_quant_manifest.json"
    context_path = case / f"{variant}_s{seq}.bin"
    schematic_path = case / "schematics" / f"model.0.s{seq}_schematic.bin"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensors = {tensor["name"]: tensor for tensor in manifest["tensors"]}
    operations = {operation["name"]: operation for operation in manifest["operations"]}
    if set(PROJECTIONS) - operations.keys():
        raise AssertionError(f"{variant} s{seq}: missing projection operations")
    if any(operation["package"] != "qti.aisw" for operation in manifest["operations"]):
        raise AssertionError(f"{variant} s{seq}: non-QNN package observed")

    weights: dict[str, object] = {}
    for name in PROJECTIONS:
        operation = operations[name]
        if operation["qnn_op_type"] != "Conv2d" or len(operation["inputs"]) != 2:
            raise AssertionError(f"{variant} s{seq}: malformed projection {name}")
        weight = tensors[operation["inputs"][1]]
        recipe = weight["quant_recipe"]
        encoding = weight["qnn_quantization"]
        if variant == "lpbq":
            if recipe.get("type") != "lpbq" or recipe.get("block_size") != 32:
                raise AssertionError(f"{variant} s{seq}: {name} is not W4G32 LPBQ")
            if recipe.get("quant_min") != -7 or recipe.get("quant_max") != 7:
                raise AssertionError(f"{variant} s{seq}: {name} W4 range changed")
            if encoding.get("encoding") != "blockwise_expansion":
                raise AssertionError(f"{variant} s{seq}: {name} lacks blockwise expansion")
        else:
            if recipe.get("type") != "symmetric_per_tensor":
                raise AssertionError(f"{variant} s{seq}: {name} is not symmetric per-tensor")
            if weight.get("qnn_dtype") != "SFIXED_POINT_8" or encoding.get("zero_point") != 0:
                raise AssertionError(f"{variant} s{seq}: {name} is not native signed QNN W8")
        weights[name.rsplit(".", 1)[-1]] = {
            "dimensions": weight["dimensions"],
            "qnn_dtype": weight["qnn_dtype"],
            "recipe": recipe,
            "encoding": encoding.get("encoding"),
            "scale": encoding.get("scale"),
            "zero_point": encoding.get("zero_point"),
        }

    activation_tensors = [
        tensor for tensor in manifest["tensors"]
        if tensor["tensor_type"] != "STATIC" and "FIXED_POINT" in tensor["qnn_dtype"]
    ]
    if not activation_tensors:
        raise AssertionError(f"{variant} s{seq}: no quantized activation tensors")
    if any(
        tensor["qnn_dtype"] != "UFIXED_POINT_8"
        or tensor["quant_recipe"].get("type") != "asymmetric_per_tensor"
        for tensor in activation_tensors
    ):
        raise AssertionError(f"{variant} s{seq}: activation contract is not uniformly asymmetric A8")

    activation_signature = sorted(
        (
            tuple(tensor["dimensions"]), tensor["tensor_type"], tensor.get("producer"),
            tuple(sorted(tensor.get("consumers", []))),
            tensor["qnn_quantization"].get("scale"), tensor["qnn_quantization"].get("zero_point"),
        )
        for tensor in activation_tensors
    )
    return {
        "graph": manifest["graph"],
        "operation_count": len(manifest["operations"]),
        "projection_count": len(PROJECTIONS),
        "activation_tensor_count": len(activation_tensors),
        "weights": weights,
        "activation_signature": activation_signature,
        "context_bytes": context_path.stat().st_size,
        "context_sha256": _sha256(context_path),
        "schematic_bytes": schematic_path.stat().st_size,
        "schematic_sha256": _sha256(schematic_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def _timing(path: Path) -> tuple[float, int]:
    with path.open(newline="", encoding="utf-8") as stream:
        values = [
            int(row["graph_execute_us"])
            for row in csv.DictReader(stream)
            if row["phase"] == "measured"
        ]
    if len(values) != 500:
        raise AssertionError(f"expected 500 measured samples in {path}, got {len(values)}")
    return statistics.median(values), len(values)


def _speed(result_root: Path, seq: int) -> dict[str, object]:
    per_variant: dict[str, object] = {}
    medians: dict[str, float] = {}
    for variant in ("lpbq", "w8a8"):
        rounds = []
        for round_index in range(1, 6):
            median_us, samples = _timing(
                result_root / "speed" / f"round{round_index}" / f"{variant}_s{seq}" / "timing.csv"
            )
            rounds.append({"round": round_index, "median_us": median_us, "samples": samples})
        process_medians = [item["median_us"] for item in rounds]
        aggregate = statistics.median(process_medians)
        medians[variant] = aggregate
        per_variant[variant] = {
            "rounds": rounds,
            "median_of_process_medians_us": aggregate,
            "min_process_median_us": min(process_medians),
            "max_process_median_us": max(process_medians),
        }
    per_variant["comparison"] = {
        "w8_minus_lpbq_us": medians["w8a8"] - medians["lpbq"],
        "w8_over_lpbq": medians["w8a8"] / medians["lpbq"],
        "w8_speedup_percent": (medians["lpbq"] / medians["w8a8"] - 1.0) * 100.0,
    }
    return per_variant


def _correctness(result_root: Path, seq: int) -> dict[str, object]:
    expected_size = seq * 2048
    outputs: dict[str, bytes] = {}
    result: dict[str, object] = {}
    for variant in ("lpbq", "w8a8"):
        first_path = result_root / "correctness" / "first" / f"{variant}_s{seq}" / "output.raw"
        repeat_path = result_root / "correctness" / "repeat" / f"{variant}_s{seq}" / "output.raw"
        first = first_path.read_bytes()
        repeat = repeat_path.read_bytes()
        if len(first) != expected_size or len(repeat) != expected_size:
            raise AssertionError(f"{variant} s{seq}: output size mismatch")
        if first != repeat:
            raise AssertionError(f"{variant} s{seq}: output is not repeatable")
        outputs[variant] = first
        result[variant] = {"repeatable": True, "bytes": len(first), "sha256": _sha256(first_path)}
    deltas = [abs(left - right) for left, right in zip(outputs["lpbq"], outputs["w8a8"], strict=True)]
    result["cross_variant"] = {
        "equal_fraction": sum(delta == 0 for delta in deltas) / len(deltas),
        "max_abs_code_delta": max(deltas),
        "mean_abs_code_delta": statistics.fmean(deltas),
    }
    return result


def _host_reference(artifact_root: Path, result_root: Path) -> dict[str, object]:
    report = json.loads((artifact_root / "artifact_report.json").read_text(encoding="utf-8"))
    source = Path(report["source"])
    if not source.is_file():
        windows_source = PureWindowsPath(report["source"])
        if windows_source.drive and windows_source.drive[0].isalpha():
            source = Path("/mnt") / windows_source.drive[0].lower() / Path(*windows_source.parts[1:])
    descriptors = artifact.read_descriptors(source)
    result: dict[str, object] = {}
    for seq in (1, 32):
        input_codes = np.fromfile(artifact_root / f"input_s{seq}.raw", dtype=np.uint8).reshape(seq, 2048)
        cases: dict[str, object] = {}
        for variant in ("lpbq", "w8a8"):
            expected = artifact.simulate(source, descriptors, input_codes, variant)["output_code"].reshape(-1)
            observed = np.fromfile(
                result_root / "correctness" / "first" / f"{variant}_s{seq}" / "output.raw",
                dtype=np.uint8,
            )
            delta = np.abs(expected.astype(np.int16) - observed.astype(np.int16))
            cases[variant] = {
                "equal_fraction": float(np.mean(delta == 0)),
                "mean_abs_code_delta": float(np.mean(delta)),
                "p99_abs_code_delta": float(np.quantile(delta, 0.99)),
                "max_abs_code_delta": int(np.max(delta)),
            }
        result[f"s{seq}"] = cases
    return result


def _optrace_presence(result_root: Path, variant: str, seq: int) -> dict[str, object]:
    directory = result_root / "optrace" / f"{variant}_s{seq}"
    raw = directory / "qnn-profiling-data.log"
    detail = directory / "qnn_detail_profile.txt"
    if not raw.is_file() or not detail.is_file():
        raise AssertionError(f"missing Optrace capture for {variant} s{seq}")
    return {
        "raw_bytes": raw.stat().st_size,
        "raw_sha256": _sha256(raw),
        "detail_bytes": detail.stat().st_size,
    }


def _optrace_analysis(result_root: Path, variant: str, seq: int) -> dict[str, object]:
    directory = result_root / "optrace" / f"{variant}_s{seq}"
    base = directory / f"layer14-{variant}-s{seq}"
    qhas_path = base.with_name(base.name + "-chrometrace_qnn_htp_analysis_summary.json")
    operators_path = base.with_name(base.name + "-operators.csv")
    qhas = json.loads(qhas_path.read_text(encoding="utf-8"))["data"]

    resources: dict[str, dict[str, int]] = {}
    for item in qhas["htp_overall_summary"]["data"][0]["htp_resources"]["data"]:
        summary = resources.setdefault(
            item["type"],
            {"lanes": 0, "timeline_max_cycles": 0, "cycles_used": 0,
             "dram_read": 0, "dram_write": 0, "vtcm_read": 0, "vtcm_write": 0},
        )
        summary["lanes"] += 1
        summary["timeline_max_cycles"] = max(summary["timeline_max_cycles"], item["timeline_cycles"])
        for field in ("cycles_used", "dram_read", "dram_write", "vtcm_read", "vtcm_write"):
            summary[field] += item[field]

    selected_names = {
        "weights_to_vtcm": "q::ConvLayer.opt.weights_to_vtcm",
        "lpbq_expand": "q::ConvLayer.opt.expand_block_quant_to_pc_int8_weights",
        "conv_hmx": "q::ConvLayer_s1.opt",
    }
    by_name = {item["op"]: item for item in qhas["htp_op_types"]["data"]}
    kernels = {
        label: {
            "cycles": by_name[name]["cycles"],
            "dominant_path_cycles": by_name[name]["num_dominant_path_cycles_htp_0"],
            "instances": by_name[name]["instances"],
            "dram_read": by_name[name]["dram_read"],
            "dram_write": by_name[name]["dram_write"],
            "vtcm_read": by_name[name]["vtcm_read"],
            "vtcm_write": by_name[name]["vtcm_write"],
        }
        for label, name in selected_names.items() if name in by_name
    }

    projections: dict[str, object] = {}
    with operators_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["qnn_op_name"] not in PROJECTIONS:
                continue
            projections[row["qnn_op_name"].rsplit(".", 1)[-1]] = {
                "kernel_resources": row["kernel_resources"],
                "wall_span_cycles": int(row["wall_span_cycles"]),
                "active_union_cycles": int(row["active_union_cycles"]),
                "total_work_cycles": int(row["total_work_cycles"]),
                "overlapped_work_cycles": int(row["overlapped_work_cycles"]),
                "max_parallelism": int(row["max_parallelism"]),
            }
    return {"resources": resources, "kernels": kernels, "projections": projections}


def summarize(artifact_root: Path, result_root: Path) -> dict[str, object]:
    manifests: dict[str, object] = {}
    for seq in (1, 32):
        lpbq = _manifest_audit(artifact_root, "lpbq", seq)
        w8a8 = _manifest_audit(artifact_root, "w8a8", seq)
        if lpbq.pop("activation_signature") != w8a8.pop("activation_signature"):
            raise AssertionError(f"s{seq}: activation graph/qparams differ between variants")
        manifests[f"s{seq}"] = {
            "lpbq": lpbq,
            "w8a8": w8a8,
            "context_size_ratio_w8_over_lpbq": w8a8["context_bytes"] / lpbq["context_bytes"],
            "activation_contract_identical": True,
        }
    return {
        "gate": "PASS",
        "gate_scope": "experiment integrity and repeatable execution; no accuracy or speed threshold",
        "contract": {
            "layer": 14,
            "scope": "complete MLP; identical A8 qparams/layout; static-weight encoding is the only runtime variable",
            "timing": "profiling off; five fresh-process paired rounds; alternating order; 20 warmup + 500 measured",
            "optrace": "one fresh-process capture per variant and sequence length",
        },
        "manifests": manifests,
        "correctness": {f"s{seq}": _correctness(result_root, seq) for seq in (1, 32)},
        "host_reference": _host_reference(artifact_root, result_root),
        "speed": {f"s{seq}": _speed(result_root, seq) for seq in (1, 32)},
        "optrace": {
            f"{variant}_s{seq}": _optrace_presence(result_root, variant, seq)
            for variant in ("lpbq", "w8a8") for seq in (1, 32)
        },
        "optrace_analysis": {
            f"{variant}_s{seq}": _optrace_analysis(result_root, variant, seq)
            for variant in ("lpbq", "w8a8") for seq in (1, 32)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.artifact_root, args.result_root)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
