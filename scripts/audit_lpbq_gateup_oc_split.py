#!/usr/bin/env python3
"""Audit the layer-14 gate/up output-channel split micrograph contract."""

import argparse
import json
from pathlib import Path


PROJECTIONS = ("gate_proj", "up_proj")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def tensor_map(manifest):
    return {tensor["name"]: tensor for tensor in manifest["tensors"]}


def qparams(tensor):
    encoding = tensor["qnn_quantization"]
    return {"scale": encoding["scale"], "zero_point": encoding["zero_point"]}


def audit_u8(tensor, label, dimensions):
    require(tensor["dimensions"] == dimensions, f"{label}: dimensions {tensor['dimensions']} != {dimensions}")
    require(tensor["qnn_dtype"] == "UFIXED_POINT_8", f"{label}: not physical U8")
    recipe = tensor["quant_recipe"]
    require(recipe["type"] == "asymmetric_per_tensor", f"{label}: not asymmetric per-tensor")
    require(recipe["quant_to_dtype"] == "UInt8", f"{label}: not logical UInt8")
    require((recipe["quant_min"], recipe["quant_max"]) == (0, 255), f"{label}: invalid U8 range")


def audit_weight(tensor, label, output_channels):
    require(tensor["tensor_type"] == "STATIC", f"{label}: weight is not static")
    require(tensor["dimensions"] == [1, 1, 2048, output_channels], f"{label}: invalid HWIO shape")
    require(tensor["qnn_dtype"] == "SFIXED_POINT_8", f"{label}: invalid LPBQ carrier")
    encoding = tensor["qnn_quantization"]
    require(encoding["encoding"] == "blockwise_expansion", f"{label}: LPBQ encoding is missing")
    require(encoding["axis"] == 3, f"{label}: invalid LPBQ channel axis")
    require(encoding["axis_size"] == output_channels, f"{label}: invalid LPBQ axis size")
    require(encoding["num_blocks_per_axis"] == 64, f"{label}: invalid G32 block count")
    require(encoding["block_scale_bitwidth"] == 4, f"{label}: invalid block-scale bit width")
    recipe = tensor["quant_recipe"]
    expected = {
        "type": "lpbq",
        "block_size": 32,
        "block_scale_bitwidth": 4,
        "channel_axis": 3,
        "channel_scale_dtype": "Float32",
        "quant_to_dtype": "Int4",
        "quant_min": -7,
        "quant_max": 7,
    }
    for key, value in expected.items():
        require(recipe.get(key) == value, f"{label}: {key}={recipe.get(key)!r}, expected {value!r}")


def audit_case(root, projection, layout):
    case = f"{layout}_{projection}"
    path = root / "contexts" / case / "manifests" / "model.0.s1_quant_manifest.json"
    manifest = load_json(path)
    require(manifest["graph"] == "model.0.s1", f"{case}: unexpected graph")
    tensors = tensor_map(manifest)
    prefix = f"model.layers.14.mlp.{projection}"
    convs = [op for op in manifest["operations"] if op["name"].startswith(prefix)]
    expected_names = [prefix] if layout == "full" else [f"{prefix}.oc0", f"{prefix}.oc1"]
    require(sorted(op["name"] for op in convs) == expected_names, f"{case}: projection op set mismatch")
    concat_count = sum(op["qnn_op_type"] == "Concat" for op in manifest["operations"])
    require(concat_count == (1 if layout == "split2" else 0), f"{case}: unexpected Concat count")
    output_channels = 6144 if layout == "full" else 3072
    records = []
    for op in sorted(convs, key=lambda item: item["name"]):
        require(op["package"] == "qti.aisw" and op["qnn_op_type"] == "Conv2d", f"{op['name']}: not native QNN Conv2d")
        require(len(op["inputs"]) == 2 and len(op["outputs"]) == 1, f"{op['name']}: invalid arity")
        activation = tensors[op["inputs"][0]]
        weight = tensors[op["inputs"][1]]
        output = tensors[op["outputs"][0]]
        audit_u8(activation, f"{op['name']} input", [1, 1, 1, 2048])
        audit_weight(weight, f"{op['name']} weight", output_channels)
        audit_u8(output, f"{op['name']} output", [1, 1, 1, output_channels])
        records.append({
            "name": op["name"],
            "input_qparams": qparams(activation),
            "output_qparams": qparams(output),
            "weight_shape": weight["dimensions"],
        })
    return {"case": case, "manifest": str(path), "projection_ops": records, "concat_count": concat_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    artifact = load_json(args.model_root / "artifact_audit.json")
    require(artifact["status"] == "pass", "artifact audit did not pass")
    for projection in PROJECTIONS:
        item = artifact["projections"][projection]
        require(item["split_reconstructs_full_exactly"], f"{projection}: split does not reconstruct full tensors")
        require((item["signed_code_min"], item["signed_code_max"]) == (-7, 7), f"{projection}: signed W4 range mismatch")

    cases = {}
    for projection in PROJECTIONS:
        full = audit_case(args.model_root, projection, "full")
        split = audit_case(args.model_root, projection, "split2")
        full_input = full["projection_ops"][0]["input_qparams"]
        full_output = full["projection_ops"][0]["output_qparams"]
        for record in split["projection_ops"]:
            require(record["input_qparams"] == full_input, f"{record['name']}: input qparams changed")
            require(record["output_qparams"] == full_output, f"{record['name']}: output qparams changed")
        cases[full["case"]] = full
        cases[split["case"]] = split

    result = {
        "status": "pass",
        "contract": "same W4G32/A8 qparams; full 2048x6144 vs two 2048x3072 projections",
        "artifact_audit": str(args.model_root / "artifact_audit.json"),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"manifest_audit=pass cases={len(cases)} output={args.output}")


if __name__ == "__main__":
    main()
