#!/usr/bin/env python3
"""Compare QNN quant manifests while canonicalizing opaque tensor IDs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def canonicalize(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tensors = payload.get("tensors")
    operations = payload.get("operations")
    if not isinstance(tensors, list) or not isinstance(operations, list):
        raise ValueError(f"{path}: missing operations/tensors arrays")

    canonical = copy.deepcopy(payload)
    id_to_signature: dict[str, str] = {}
    canonical_tensors: list[dict[str, Any]] = []
    for index, tensor in enumerate(canonical["tensors"]):
        name = tensor.get("name")
        if not isinstance(name, str) or name in id_to_signature:
            raise ValueError(f"{path}: invalid or duplicate tensor name at index {index}")
        del tensor["name"]
        encoded = json.dumps(tensor, sort_keys=True, separators=(",", ":")).encode("utf-8")
        signature = f"tensor:{hashlib.sha256(encoded).hexdigest()}"
        id_to_signature[name] = signature
        canonical_tensors.append(tensor)
    canonical["tensors"] = sorted(
        canonical_tensors,
        key=lambda tensor: json.dumps(tensor, sort_keys=True, separators=(",", ":")),
    )

    for operation in canonical["operations"]:
        for field in ("inputs", "outputs"):
            references = operation.get(field)
            if not isinstance(references, list):
                raise ValueError(f"{path}: operation {operation.get('name')} lacks {field}")
            try:
                operation[field] = [id_to_signature[ref] for ref in references]
            except KeyError as exc:
                raise ValueError(
                    f"{path}: operation {operation.get('name')} references unknown tensor {exc.args[0]}"
                ) from exc
    return canonical


def first_difference(left: Any, right: Any, path: str = "$") -> str:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: keys {sorted(left)} != {sorted(right)}"
        for key in left:
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = first_difference(left_item, right_item, f"{path}[{index}]")
            if difference:
                return difference
        return ""
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return ""


def digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    baseline = canonicalize(args.baseline)
    candidate = canonicalize(args.candidate)
    difference = first_difference(baseline, candidate)
    if difference:
        print(f"manifest contracts differ: {difference}", file=sys.stderr)
        return 1
    print(f"canonical_sha256={digest(candidate)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
