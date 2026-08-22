#!/usr/bin/env python3
"""Generate deterministic real-shape inputs for the native U8 masked Softmax."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


HEADS = 16
CONTEXT = 1024
LAYER = 14


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qparam(tensor: dict) -> dict:
    quant = tensor["qnn_quantization"]
    recipe = tensor["quant_recipe"]
    if (
        tensor["qnn_dtype"] != "UFIXED_POINT_8"
        or quant.get("encoding") != "scale_offset"
        or recipe.get("type") != "asymmetric_per_tensor"
        or recipe.get("quant_to_dtype") != "UInt8"
        or recipe.get("quant_max") != 255
    ):
        raise AssertionError(f"not an asymmetric-U8 tensor: {tensor['name']}")
    return {"scale": float(quant["scale"]), "zero_point": int(quant["zero_point"])}


def _quantize(value: float, qparam: dict) -> int:
    code = math.floor(value / qparam["scale"] + qparam["zero_point"] + 0.5)
    return min(255, max(0, code))


def _dequantize(code: int, qparam: dict) -> float:
    return (code - qparam["zero_point"]) * qparam["scale"]


def _contract(manifest: Path, seq: int) -> dict:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    operations = {item["name"]: item for item in document["operations"]}
    tensors = {item["name"]: item for item in document["tensors"]}
    heads = []
    for head in range(HEADS):
        prefix = f"model.layers.{LAYER}.self_attn"
        softmax = operations[f"{prefix}.Softmax.{head}"]
        softmax_input = tensors[softmax["inputs"][0]]
        softmax_output = tensors[softmax["outputs"][0]]
        where = operations[softmax_input["producer"]]
        condition, logits_name, masked_name = where["inputs"]
        logits = tensors[logits_name]
        masked = tensors[masked_name]
        reduce_add = operations[masked["producer"]]
        reduce_tensor = tensors[reduce_add["inputs"][0]]
        minus_twenty = tensors[reduce_add["inputs"][1]]
        equal = operations[tensors[condition]["producer"]]
        mask_candidates = [tensors[name] for name in equal["inputs"] if tensors[name]["tensor_type"] == "APP_WRITE"]
        if len(mask_candidates) != 1:
            raise AssertionError(f"head {head}: cannot identify causal-mask graph input")
        mask = mask_candidates[0]
        expected_shape = [1, 1, seq, CONTEXT]
        for tensor in (logits, softmax_input, softmax_output, mask):
            if tensor["dimensions"] != expected_shape:
                raise AssertionError(
                    f"head {head}: unexpected {tensor['name']} shape {tensor['dimensions']}"
                )
        if softmax["package"] != "qti.aisw" or softmax["qnn_op_type"] != "Softmax":
            raise AssertionError(f"head {head}: not qti.aisw::Softmax")
        heads.append(
            {
                "head": head,
                "logits": _qparam(logits),
                "reduce_min": _qparam(reduce_tensor),
                "minus_twenty": _qparam(minus_twenty),
                "masked_value": _qparam(masked),
                "softmax_input": _qparam(softmax_input),
                "softmax_output": _qparam(softmax_output),
                "mask": _qparam(mask),
                "qnn_op": softmax["name"],
            }
        )
    qparam_fields = (
        "logits", "reduce_min", "minus_twenty", "masked_value",
        "softmax_input", "softmax_output", "mask",
    )
    for field in qparam_fields:
        if any(head[field] != heads[0][field] for head in heads[1:]):
            raise AssertionError(f"heads do not share {field} qparams")
    return {
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "shape": [1, 1, seq, CONTEXT],
        "heads": heads,
        "common_qparams": {field: heads[0][field] for field in qparam_fields},
    }


def _fixture(output_dir: Path, contract: dict, seq: int) -> dict:
    qparams = contract["common_qparams"]
    rng = random.Random(20260822 + seq)
    logits = bytearray(HEADS * seq * CONTEXT)
    for index in range(len(logits)):
        logits[index] = rng.randrange(256)

    mask = bytearray(seq * CONTEXT)
    visible_prefix = CONTEXT - seq
    mask_zero_code = qparams["mask"]["zero_point"]
    for row in range(seq):
        last_visible = visible_prefix + row
        for column in range(CONTEXT):
            mask[row * CONTEXT + column] = mask_zero_code if column <= last_visible else 0

    expected = bytearray(len(logits))
    for head in range(HEADS):
        for row in range(seq):
            base = (head * seq + row) * CONTEXT
            real_logits = [
                _dequantize(logits[base + column], qparams["logits"])
                for column in range(CONTEXT)
            ]
            reduced = _dequantize(
                _quantize(min(real_logits), qparams["reduce_min"]),
                qparams["reduce_min"],
            )
            minus_twenty = _dequantize(
                _quantize(-20.0, qparams["minus_twenty"]),
                qparams["minus_twenty"],
            )
            masked_value = _dequantize(
                _quantize(reduced + minus_twenty, qparams["masked_value"]),
                qparams["masked_value"],
            )
            selected = [
                real_logits[column]
                if mask[row * CONTEXT + column] == mask_zero_code
                else masked_value
                for column in range(CONTEXT)
            ]
            selected = [
                _dequantize(_quantize(value, qparams["softmax_input"]), qparams["softmax_input"])
                for value in selected
            ]
            maximum = max(selected)
            exponentials = [math.exp(value - maximum) for value in selected]
            denominator = sum(exponentials)
            for column, value in enumerate(exponentials):
                expected[base + column] = _quantize(
                    value / denominator, qparams["softmax_output"]
                )

    attn_path = output_dir / f"attn_s{seq}_u8.raw"
    mask_path = output_dir / f"causal_mask_s{seq}_u8.raw"
    expected_path = output_dir / f"host_expected_s{seq}_u8.raw"
    attn_path.write_bytes(logits)
    mask_path.write_bytes(mask)
    expected_path.write_bytes(expected)
    return {
        "attn": {"path": str(attn_path.resolve()), "bytes": len(logits), "sha256": _sha256(attn_path)},
        "mask": {"path": str(mask_path.resolve()), "bytes": len(mask), "sha256": _sha256(mask_path)},
        "host_expected": {
            "path": str(expected_path.resolve()),
            "bytes": len(expected),
            "sha256": _sha256(expected_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-s1", type=Path, required=True)
    parser.add_argument("--manifest-s32", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contracts = {
        "s1": _contract(args.manifest_s1, 1),
        "s32": _contract(args.manifest_s32, 32),
    }
    if contracts["s1"]["common_qparams"] != contracts["s32"]["common_qparams"]:
        raise AssertionError("S1 and S32 do not share identical qparams")
    fixtures = {
        "s1": _fixture(args.output_dir, contracts["s1"], 1),
        "s32": _fixture(args.output_dir, contracts["s32"], 32),
    }
    report = {
        "contract": {
            "qnn_op": "qti.aisw::Softmax in the fused native masked-softmax pattern",
            "activation": "asymmetric U8 throughout",
            "layer": LAYER,
            "heads": HEADS,
            "context": CONTEXT,
            "sequences": [1, 32],
        },
        "source_model": {
            "path": str(args.source_model.resolve()),
            "bytes": args.source_model.stat().st_size,
            "sha256": _sha256(args.source_model),
        },
        "contracts": contracts,
        "fixtures": fixtures,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    report_path = args.report or args.output_dir / "fixture_report.json"
    report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
