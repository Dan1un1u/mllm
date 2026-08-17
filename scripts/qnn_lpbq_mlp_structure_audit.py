#!/usr/bin/env python3
"""Audit the layer-14 Conv/MatMul LPBQ micrograph contracts and lowering."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


PROJECTIONS = {
    "gate_proj": (2048, 6144),
    "up_proj": (2048, 6144),
    "down_proj": (6144, 2048),
}
SEQUENCES = (1, 32)
PHYSICAL_MARKERS = (
    "ConvLayer",
    "expand_block_quant_to_pc_int8_weights",
    "weights_to_vtcm",
    "QNN_CastInt4ToInt8",
    "SpecialMatmul",
)
ADAPTER_MARKERS = ("ForceFormat", "Transpose", "Convert", "Copy")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def strings(path: Path) -> list[str]:
    result = subprocess.run(
        ["strings", "-a", str(path)], check=True, capture_output=True, text=True
    )
    return result.stdout.splitlines()


def one(items: list[dict], predicate, label: str) -> dict:
    matches = [item for item in items if predicate(item)]
    if len(matches) != 1:
        raise AssertionError(f"expected one {label}, found {len(matches)}")
    return matches[0]


def audit_case(root: Path, layout: str, projection: str, seq: int) -> dict:
    k, o = PROJECTIONS[projection]
    name = f"{layout}_{projection}_s{seq}"
    case = root / "contexts" / name
    context = case / f"{name}.bin"
    manifest = case / "manifests" / f"model.0.s{seq}_quant_manifest.json"
    schematic = case / "schematics" / f"model.0.s{seq}_schematic.bin"
    for path in (context, manifest, schematic):
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"missing artifact: {path}")

    data = json.loads(manifest.read_text())
    expected_op = "Conv2d" if layout == "conv" else "MatMul"
    op = one(data["operations"], lambda value: value["qnn_op_type"] == expected_op, expected_op)
    weight = one(
        data["tensors"],
        lambda value: value.get("logical_quant_dtype") == "Int4",
        "logical Int4 weight",
    )
    inputs = [value for value in data["tensors"] if value.get("tensor_type") == "APP_WRITE"]
    outputs = [value for value in data["tensors"] if value.get("tensor_type") == "APP_READ"]
    input_tensor = one(inputs, lambda _: True, "graph input")
    output_tensor = one(outputs, lambda _: True, "graph output")

    expected_weight = [1, 1, k, o] if layout == "conv" else [k, o]
    expected_axis = 3 if layout == "conv" else 1
    assert weight["dimensions"] == expected_weight, (name, weight["dimensions"])
    assert weight["tensor_type"] == "STATIC"
    assert weight["qnn_dtype"] == "SFIXED_POINT_8"
    assert weight["quant_recipe"] == {
        "block_scale_bitwidth": 4,
        "block_size": 32,
        "channel_axis": expected_axis,
        "channel_scale_dtype": "Float32",
        "quant_max": 7,
        "quant_min": -7,
        "quant_to_dtype": "Int4",
        "solved": True,
        "type": "lpbq",
    }
    quant = weight["qnn_quantization"]
    assert quant["encoding"] == "blockwise_expansion"
    assert quant["axis"] == expected_axis
    assert quant["num_blocks_per_axis"] == k // 32
    assert quant["block_scale_bitwidth"] == 4
    assert quant["block_scale_storage_bits"] == 8
    assert input_tensor["dimensions"] == [1, seq, k]
    assert output_tensor["dimensions"] == [1, seq, o]
    assert input_tensor["qnn_dtype"] == "UFIXED_POINT_8"
    assert output_tensor["qnn_dtype"] == "UFIXED_POINT_8"

    physical = strings(schematic)
    joined = "\n".join(physical)
    marker_counts = {marker: joined.count(marker) for marker in PHYSICAL_MARKERS}
    adapter_counts = {marker: joined.count(marker) for marker in ADAPTER_MARKERS}
    for required in (
        "ConvLayer",
        "expand_block_quant_to_pc_int8_weights",
        "weights_to_vtcm",
        "QNN_CastInt4ToInt8",
    ):
        assert marker_counts[required] > 0, (name, required)

    return {
        "case": name,
        "layout": layout,
        "projection": projection,
        "seq": seq,
        "qnn_op": op["qnn_op_type"],
        "weight_shape": weight["dimensions"],
        "channel_axis": expected_axis,
        "blocks_per_channel": k // 32,
        "input_qparam": input_tensor["qnn_quantization"],
        "output_qparam": output_tensor["qnn_quantization"],
        "physical_markers": marker_counts,
        "adapter_markers": adapter_counts,
        "context_sha256": digest(context),
        "schematic_sha256": digest(schematic),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = []
    failures = []
    for projection in PROJECTIONS:
        for seq in SEQUENCES:
            pair = {}
            for layout in ("conv", "matmul"):
                try:
                    result = audit_case(args.artifact_root, layout, projection, seq)
                    pair[layout] = result
                    cases.append(result)
                except Exception as error:  # retain every gate failure in one report
                    failures.append(f"{layout}_{projection}_s{seq}: {error}")
            if pair.keys() == {"conv", "matmul"}:
                for field in ("input_qparam", "output_qparam"):
                    if pair["conv"][field] != pair["matmul"][field]:
                        failures.append(f"{projection}_s{seq}: {field} differs")
                for marker in ADAPTER_MARKERS:
                    conv_count = pair["conv"]["adapter_markers"][marker]
                    candidate_count = pair["matmul"]["adapter_markers"][marker]
                    if candidate_count > conv_count:
                        failures.append(
                            f"matmul_{projection}_s{seq}: added physical {marker} "
                            f"({candidate_count} vs Conv {conv_count})"
                        )

    fc_case = args.artifact_root / "contexts" / "fc_gate_proj_s1"
    fc_log = fc_case / "compile.log"
    fc_text = fc_log.read_text(errors="replace") if fc_log.is_file() else ""
    fc_failure = {
        "case": "fc_gate_proj_s1",
        "context_exists": (fc_case / "fc_gate_proj_s1.bin").is_file(),
        "manifest_exists": (fc_case / "manifests/model.0.s1_quant_manifest.json").is_file(),
        "finalize_failure": "no properties registered for q::GenPad" in fc_text,
        "prepare_failure": "Graph prepare failed" in fc_text,
    }
    if fc_failure["context_exists"] or not all(
        (fc_failure["manifest_exists"], fc_failure["finalize_failure"], fc_failure["prepare_failure"])
    ):
        failures.append("FullyConnected failure evidence is incomplete or inconsistent")

    canonicalized = all(
        case["physical_markers"]["ConvLayer"] > 0
        and case["physical_markers"]["QNN_CastInt4ToInt8"] > 0
        for case in cases
        if case["layout"] == "matmul"
    )
    report = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "scope": "pre-finalize manifest plus finalized V79 schematic string audit",
        "adapter_rule": "candidate may not add an adapter beyond the paired Conv lowering",
        "matmul_canonicalized_to_conv_physical_family": canonicalized,
        "fully_connected": fc_failure,
        "cases": cases,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(cases), "failures": failures}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
