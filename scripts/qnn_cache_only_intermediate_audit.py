#!/usr/bin/env python3
"""Audit the full-width all-logits control against a cache-only graph."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


SEQUENCE = 32
CONTEXT = 1024
LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = 2048
VOCABULARY = 151936


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(tensor: dict, include_shape: bool = True) -> dict:
    result = {
        "qnn_dtype": tensor["qnn_dtype"],
        "qnn_quantization": tensor["qnn_quantization"],
        "quant_recipe": tensor["quant_recipe"],
    }
    if include_shape:
        result["dimensions"] = tensor["dimensions"]
    return result


def _read(path: Path) -> tuple[dict, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    tensors = {item["name"]: item for item in document["tensors"]}
    inputs = [item for item in document["tensors"] if item["tensor_type"] == "APP_WRITE"]
    outputs = [item for item in document["tensors"] if item["tensor_type"] == "APP_READ"]
    packages = collections.Counter(item["package"] for item in document["operations"])
    if set(packages) != {"qti.aisw"}:
        raise AssertionError(f"non-native package in {path}: {packages}")
    op_types = collections.Counter(item["qnn_op_type"] for item in document["operations"])
    logits = [item for item in outputs if item["dimensions"][-1] == VOCABULARY]
    hidden = [item for item in outputs if item["dimensions"] == [1, 1, SEQUENCE, HIDDEN]]
    kv = [
        item
        for item in outputs
        if item["dimensions"]
        in ([1, KV_HEADS, HEAD_DIM, SEQUENCE], [1, KV_HEADS, SEQUENCE, HEAD_DIM])
    ]
    if len(inputs) != 3 + 2 * LAYERS or len(kv) != 2 * LAYERS:
        raise AssertionError(f"unexpected input/KV signature in {path}")
    return (
        {
            "manifest": str(path.resolve()),
            "manifest_sha256": _sha256(path),
            "operation_count": len(document["operations"]),
            "operation_types": dict(sorted(op_types.items())),
            "native_packages": dict(packages),
            "input_count": len(inputs),
            "output_count": len(outputs),
            "inputs": [_signature(item) for item in inputs],
            "logits": [_signature(item) for item in logits],
            "hidden": [_signature(item) for item in hidden],
            "kv": [_signature(item) for item in kv],
        },
        {"document": document, "tensors": tensors},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    control, control_raw = _read(args.control_manifest)
    candidate, candidate_raw = _read(args.candidate_manifest)
    if control["input_count"] != 59 or candidate["input_count"] != 59:
        raise AssertionError("full-width graphs must retain all 56 past-cache inputs")
    if control["inputs"] != candidate["inputs"]:
        raise AssertionError("graph input contracts changed")
    if len(control["logits"]) != 1 or control["hidden"]:
        raise AssertionError("control must expose logits, not final hidden")
    if candidate["logits"] or len(candidate["hidden"]) != 1:
        raise AssertionError("candidate must expose final hidden, not logits")
    if control["kv"] != candidate["kv"]:
        raise AssertionError("KV output contracts changed")

    control_lm = next(
        item
        for item in control_raw["document"]["operations"]
        if item["name"].split(".")[-1] == "lm_head"
    )
    if any(
        item["name"].split(".")[-1] == "lm_head"
        for item in candidate_raw["document"]["operations"]
    ):
        raise AssertionError("candidate still contains lm_head")
    control_lm_input = control_raw["tensors"][control_lm["inputs"][0]]
    if _signature(control_lm_input) != candidate["hidden"][0]:
        raise AssertionError("cache-only hidden carrier changed lm_head-input encoding")

    control_types = control["operation_types"]
    candidate_types = candidate["operation_types"]
    changed_counts = {
        key: candidate_types.get(key, 0) - control_types.get(key, 0)
        for key in sorted(set(control_types) | set(candidate_types))
        if candidate_types.get(key, 0) != control_types.get(key, 0)
    }
    if changed_counts != {"Conv2d": -1}:
        raise AssertionError(f"unexpected operation-count changes: {changed_counts}")

    report = {
        "contract": {
            "attention_width": CONTEXT,
            "sequence": SEQUENCE,
            "only_compute_change": "remove the unchanged LPBQ lm_head",
            "candidate_replacement_output": "U8 final hidden carrier",
            "input_contracts_identical": True,
            "kv_output_contracts_identical": True,
            "hidden_qparam_matches_control_lm_head_input": True,
            "native_backend_only": True,
            "operation_count_delta": changed_counts,
        },
        "control_all_logits": {key: value for key, value in control.items() if key not in {"inputs", "kv"}},
        "candidate_cache_only": {key: value for key, value in candidate.items() if key not in {"inputs", "kv"}},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
