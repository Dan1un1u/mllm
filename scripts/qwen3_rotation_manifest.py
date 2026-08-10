#!/usr/bin/env python3
"""Build and independently validate Qwen3 rotation export manifests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open


SCHEMA_VERSION = 2
MANIFEST_KIND = "mllm.qwen3.rotation.export"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    if value.ndim == 0:
        value = value.reshape(1)
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finalize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result.pop("manifest_id", None)
    result["manifest_id"] = canonical_payload_sha256(result)
    return result


def file_record(path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "role": role,
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def build_input_records(
    source_model: Path,
    source_files: Iterable[Path],
    base_quant_checkpoint: Path,
) -> dict[str, Any]:
    return {
        "source_model": {
            "path": str(source_model.resolve()),
            "shards": [
                file_record(path, role="source_model_shard")
                for path in sorted(source_files)
            ],
        },
        "base_quant_checkpoint": file_record(
            base_quant_checkpoint, role="base_quant_checkpoint"
        ),
    }


def _git_value(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    return result.stdout.strip()


def build_provenance(
    repo_root: Path,
    command: Iterable[str],
    code_files: Iterable[Path],
) -> dict[str, Any]:
    status = _git_value(repo_root, "status", "--short") or ""
    dependencies: dict[str, str] = {"torch": torch.__version__}
    for package in ("safetensors", "numpy"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "missing"
    return {
        "git": {
            "commit": _git_value(repo_root, "rev-parse", "HEAD"),
            "branch": _git_value(repo_root, "branch", "--show-current"),
            "dirty": bool(status),
            "dirty_paths": status.splitlines(),
        },
        "command": list(command),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependencies": dependencies,
        },
        "code_files": [
            file_record(path, role="export_code")
            for path in sorted(path.resolve() for path in code_files)
        ],
    }


def independent_hadamard_matrix(order: int) -> torch.Tensor:
    if order <= 0 or order & (order - 1):
        raise ValueError(f"Hadamard order must be a positive power of two: {order}")
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < order:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix.div(math.sqrt(order)).contiguous()


def basis_record(name: str, order: int, *, applied: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "family": "normalized_sylvester_hadamard",
        "order": order,
        "normalization": f"1/sqrt({order})",
        "applied": applied,
    }
    if applied:
        matrix = independent_hadamard_matrix(order)
        record["float32_matrix_sha256"] = tensor_sha256(matrix)
    return record


def checkpoint_inventory(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    tensors: dict[str, Any] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for name in sorted(handle.keys()):
            tensor = handle.get_tensor(name)
            tensors[name] = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": tensor_sha256(tensor),
            }
    return metadata, tensors


def build_export_manifest_v2(
    legacy: dict[str, Any],
    checkpoint: Path,
    *,
    inputs: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    metadata, tensors = checkpoint_inventory(checkpoint)
    dimensions = dict(legacy["dimensions"])
    variant = str(legacy["variant"])
    r3_mode = str(legacy.get("r3_mode", "none"))
    rotate_r1_r2 = variant not in ("A_original", "B_identity_fold")
    uses_r3 = r3_mode != "none"

    rotation_contract = {
        "description": legacy["rotation"],
        "r1": basis_record("R1", int(dimensions["hidden_size"]), applied=rotate_r1_r2),
        "r2": basis_record("R2", int(dimensions["head_dim"]), applied=rotate_r1_r2),
        "r3": basis_record("R3", int(dimensions["head_dim"]), applied=uses_r3),
        "qk_pairing": "same post-RoPE orthogonal R3 on Q and current K",
        "hidden_boundary": legacy["contract"]["hidden_boundary"],
        "value_cache_boundary": legacy["contract"]["value_cache_boundary"],
        "key_cache_boundary": legacy.get("key_cache_boundary", "baseline"),
        "past_key_policy": (
            "past K already uses R3; rotate current K only"
            if uses_r3
            else "baseline key basis"
        ),
    }
    graph_expectations: dict[str, Any] = {
        "backend": "QNN HTP",
        "unintended_cpu_fallback_allowed": False,
    }
    if r3_mode == "dense":
        carrier = f"model.layers.{legacy['layer']}.self_attn.r3_dense"
        graph_expectations["r3_dense"] = {
            "expected_matmul_count": (
                int(dimensions["query_heads"]) + int(dimensions["kv_heads"])
            ),
            "shared_carrier": carrier + ".weight",
            "shared_scale1": carrier + ".scale1",
            "shared_scale2": carrier + ".scale2",
            "expected_lowering": "qti.aisw::MatMul with static LPBQ input1",
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": MANIFEST_KIND,
        "model": legacy["model"],
        "layer": legacy["layer"],
        "variant": variant,
        "dimensions": dimensions,
        "group_size": legacy["group_size"],
        "provenance": provenance,
        "inputs": inputs,
        "rotation_contract": rotation_contract,
        "quantization_contract": {
            **legacy["contract"],
            "activation_qdq": legacy["activation_qdq"],
            "r3_mode": r3_mode,
        },
        "transform_records": legacy.get("weights", {}),
        "graph_expectations": graph_expectations,
        "artifact": {
            "stage": "export",
            "format": "safetensors",
            "checkpoint": file_record(checkpoint, role="rotated_checkpoint"),
            "metadata": metadata,
            "tensor_count": len(tensors),
            "tensors": tensors,
        },
    }
    return finalize_manifest(payload)


def _check_file_record(
    record: dict[str, Any],
    errors: list[str],
    *,
    label: str,
) -> Path | None:
    path = Path(record.get("path", ""))
    if not path.is_file():
        errors.append(f"{label}: file is missing: {path}")
        return None
    actual_size = path.stat().st_size
    if actual_size != record.get("size_bytes"):
        errors.append(
            f"{label}: size mismatch: manifest={record.get('size_bytes')} actual={actual_size}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != record.get("sha256"):
        errors.append(
            f"{label}: sha256 mismatch: manifest={record.get('sha256')} actual={actual_hash}"
        )
    return path


def _decode_lpbq(
    weight: torch.Tensor,
    scale1: torch.Tensor,
    scale2: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    if weight.ndim == 4:
        if tuple(weight.shape[:2]) != (1, 1):
            raise ValueError(f"unsupported LPBQ carrier shape: {tuple(weight.shape)}")
        weight = weight.reshape(weight.shape[-2], weight.shape[-1])
    if weight.ndim != 2:
        raise ValueError(f"expected rank-2 LPBQ carrier, got {tuple(weight.shape)}")
    in_features, out_features = map(int, weight.shape)
    signed = weight.reshape(in_features, out_features).transpose(0, 1).to(torch.int16)
    signed = torch.where(signed >= 8, signed - 16, signed)
    groups_per_output = in_features // group_size
    return (
        signed.reshape(out_features, groups_per_output, group_size).float()
        * scale1.reshape(out_features, groups_per_output).float().unsqueeze(-1)
        * scale2.reshape(out_features, 1, 1).float()
    ).reshape(out_features, in_features)


def validate_export_manifest(
    manifest_path: Path,
    *,
    verify_inputs: bool = True,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks: dict[str, Any] = {}

    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("manifest_kind") != MANIFEST_KIND:
        errors.append(f"manifest_kind must be {MANIFEST_KIND!r}")

    claimed_id = payload.get("manifest_id")
    unsigned = copy.deepcopy(payload)
    unsigned.pop("manifest_id", None)
    actual_id = canonical_payload_sha256(unsigned)
    checks["manifest_id"] = {"claimed": claimed_id, "actual": actual_id}
    if claimed_id != actual_id:
        errors.append("manifest_id does not match canonical manifest payload")

    checkpoint_record = payload.get("artifact", {}).get("checkpoint", {})
    checkpoint = _check_file_record(
        checkpoint_record, errors, label="artifact.checkpoint"
    )
    if checkpoint is not None:
        try:
            metadata, inventory = checkpoint_inventory(checkpoint)
        except Exception as exc:
            errors.append(f"artifact.checkpoint: cannot reopen safetensors: {exc}")
        else:
            claimed_inventory = payload["artifact"].get("tensors", {})
            if inventory != claimed_inventory:
                errors.append("artifact tensor inventory does not match checkpoint")
            if metadata != payload["artifact"].get("metadata", {}):
                errors.append("artifact metadata does not match checkpoint")
            if len(inventory) != payload["artifact"].get("tensor_count"):
                errors.append("artifact tensor_count does not match checkpoint")

    if verify_inputs:
        base = payload.get("inputs", {}).get("base_quant_checkpoint", {})
        _check_file_record(base, errors, label="inputs.base_quant_checkpoint")
        for index, shard in enumerate(
            payload.get("inputs", {}).get("source_model", {}).get("shards", [])
        ):
            _check_file_record(
                shard, errors, label=f"inputs.source_model.shards[{index}]"
            )

    for name in ("r1", "r2", "r3"):
        record = payload.get("rotation_contract", {}).get(name, {})
        if not record.get("applied"):
            continue
        try:
            expected = independent_hadamard_matrix(int(record["order"]))
        except Exception as exc:
            errors.append(f"rotation_contract.{name}: invalid basis: {exc}")
            continue
        actual_hash = tensor_sha256(expected)
        if actual_hash != record.get("float32_matrix_sha256"):
            errors.append(f"rotation_contract.{name}: matrix hash mismatch")

    dense = payload.get("graph_expectations", {}).get("r3_dense")
    if dense and checkpoint is not None:
        dimensions = payload["dimensions"]
        expected_count = int(dimensions["query_heads"]) + int(dimensions["kv_heads"])
        if dense.get("expected_matmul_count") != expected_count:
            errors.append("r3_dense expected_matmul_count does not match head counts")
        names = (
            dense["shared_carrier"],
            dense["shared_scale1"],
            dense["shared_scale2"],
        )
        try:
            with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
                weight, scale1, scale2 = (handle.get_tensor(name) for name in names)
            decoded = _decode_lpbq(
                weight, scale1, scale2, group_size=int(payload["group_size"])
            )
            expected_r3 = independent_hadamard_matrix(
                int(payload["rotation_contract"]["r3"]["order"])
            )
            error = decoded - expected_r3
            max_abs = float(error.abs().max())
            checks["r3_dense_carrier"] = {
                "max_abs_error": max_abs,
                "exact": bool(torch.equal(decoded, expected_r3)),
            }
            if not torch.equal(decoded, expected_r3):
                errors.append(
                    f"r3_dense carrier does not decode exactly to R3: max_abs={max_abs}"
                )
        except Exception as exc:
            errors.append(f"r3_dense carrier validation failed: {exc}")

    checks["input_files_verified"] = verify_inputs
    return {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "manifest_id": payload.get("manifest_id"),
        "pass": not errors,
        "errors": errors,
        "checks": checks,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--skip-input-files", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = validate_export_manifest(
        args.manifest, verify_inputs=not args.skip_input_files
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
