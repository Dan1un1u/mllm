#!/usr/bin/env python3
"""Audit all-token versus last-token-only lm_head manifests."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


SEQ = 32
LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = 2048
VOCAB = 151936


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_signature(tensor: dict, include_shape: bool = True) -> dict:
    result = {
        "qnn_dtype": tensor["qnn_dtype"],
        "qnn_quantization": tensor["qnn_quantization"],
        "quant_recipe": tensor["quant_recipe"],
    }
    if include_shape:
        result["dimensions"] = tensor["dimensions"]
    return result


def _audit(path: Path, logits_rows: int) -> tuple[dict, list[dict], dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    tensors = {item["name"]: item for item in document["tensors"]}
    inputs = [item for item in document["tensors"] if item["tensor_type"] == "APP_WRITE"]
    outputs = [item for item in document["tensors"] if item["tensor_type"] == "APP_READ"]
    if len(inputs) != 3 or len(outputs) != 1 + 2 * LAYERS:
        raise AssertionError(f"unexpected graph signature in {path}")
    logits = [item for item in outputs if item["dimensions"][-1] == VOCAB]
    if len(logits) != 1 or logits[0]["dimensions"] != [1, 1, logits_rows, VOCAB]:
        raise AssertionError(f"unexpected logits output shape in {path}")
    kv = [item for item in outputs if item not in logits]
    expected_k = [1, KV_HEADS, HEAD_DIM, SEQ]
    expected_v = [1, KV_HEADS, SEQ, HEAD_DIM]
    if sum(item["dimensions"] == expected_k for item in kv) != LAYERS:
        raise AssertionError("unexpected K-cache outputs")
    if sum(item["dimensions"] == expected_v for item in kv) != LAYERS:
        raise AssertionError("unexpected V-cache outputs")

    packages = collections.Counter(op["package"] for op in document["operations"])
    if set(packages) != {"qti.aisw"}:
        raise AssertionError(f"non-native package observed: {packages}")
    op_types = collections.Counter(op["qnn_op_type"] for op in document["operations"])
    lm_head = next(
        op for op in document["operations"] if op["name"].split(".")[-1] == "lm_head"
    )
    lm_input = tensors[lm_head["inputs"][0]]
    lm_weight = tensors[lm_head["inputs"][1]]
    lm_output = tensors[lm_head["outputs"][0]]
    if lm_input["dimensions"] != [1, 1, logits_rows, HIDDEN]:
        raise AssertionError(f"unexpected lm_head input: {lm_input['dimensions']}")
    if lm_output["dimensions"] != [1, 1, logits_rows, VOCAB]:
        raise AssertionError(f"unexpected lm_head output: {lm_output['dimensions']}")
    return (
        {
            "manifest": str(path.resolve()),
            "manifest_sha256": _sha256(path),
            "input_count": len(inputs),
            "output_count": len(outputs),
            "logits_rows": logits_rows,
            "operation_count": len(document["operations"]),
            "operation_types": dict(sorted(op_types.items())),
            "native_packages": dict(packages),
            "lm_head": {
                "input": _tensor_signature(lm_input),
                "weight": _tensor_signature(lm_weight),
                "output": _tensor_signature(lm_output),
            },
        },
        kv,
        logits[0],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    control, control_kv, control_logits = _audit(args.control_manifest, SEQ)
    candidate, candidate_kv, candidate_logits = _audit(args.candidate_manifest, 1)
    if [_tensor_signature(x) for x in control_kv] != [
        _tensor_signature(x) for x in candidate_kv
    ]:
        raise AssertionError("KV output contracts changed")
    if _tensor_signature(control_logits, include_shape=False) != _tensor_signature(
        candidate_logits, include_shape=False
    ):
        raise AssertionError("logits dtype/qparam changed")
    if control["lm_head"]["weight"] != candidate["lm_head"]["weight"]:
        raise AssertionError("lm_head LPBQ weight contract changed")

    control_types = control["operation_types"]
    candidate_types = candidate["operation_types"]
    changed_counts = {
        key: candidate_types.get(key, 0) - control_types.get(key, 0)
        for key in sorted(set(control_types) | set(candidate_types))
        if candidate_types.get(key, 0) != control_types.get(key, 0)
    }
    if changed_counts != {"StridedSlice": 1}:
        raise AssertionError(f"unexpected operation-count changes: {changed_counts}")

    report = {
        "contract": {
            "attention_width": 32,
            "sequence": SEQ,
            "only_change": "slice final hidden row before the unchanged LPBQ lm_head",
            "logits_dtype_qparam_identical": True,
            "kv_output_contracts_identical": True,
            "native_backend_only": True,
            "operation_count_delta": changed_counts,
        },
        "control_all_logits": control,
        "candidate_last_logits": candidate,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
