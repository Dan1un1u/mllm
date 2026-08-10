#!/usr/bin/env python3
"""Build manifest-v2 gated full-model Qwen3 C/D-Dense checkpoints.

Layer artifacts are produced by qwen3_block_rotation.py. This driver replaces
the corresponding 28 layer tensors in the bit-exact G32 checkpoint, keeps all
duplicate global tensors strict-equal, and emits:
  * C_hadamard_r1_r2_folded
  * D_dense_r3_online (debug/reference boundary)
  * D_dense_r3_folded (production boundary)
The large safetensors files remain outside the Git worktree.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from export_qwen3_lpbq_g32 import _quantize_lpbq_g32
from qwen3_block_rotation import (
    VARIANT_C,
    VARIANT_D_DENSE,
    export_variants,
    normalized_fwht,
    normalized_hadamard_matrix,
)
from qwen3_rotation_manifest import (
    basis_record,
    build_input_records,
    build_provenance,
    canonical_payload_sha256,
    checkpoint_inventory,
    file_record,
    finalize_manifest,
    independent_hadamard_matrix,
    sha256_file,
    tensor_sha256,
    validate_export_manifest,
)

FULL_C = "C_hadamard_r1_r2_folded"
FULL_D_ONLINE = "D_dense_r3_online"
FULL_D_FOLDED = "D_dense_r3_folded"
DEFAULT_VARIANTS = (FULL_C, FULL_D_ONLINE, FULL_D_FOLDED)
LAYER_VARIANT = {FULL_C: VARIANT_C, FULL_D_ONLINE: VARIANT_D_DENSE, FULL_D_FOLDED: VARIANT_D_DENSE}
HIDDEN = 2048
HEAD_DIM = 128
Q_HEADS = 16
KV_HEADS = 8
GROUP = 32
CHUNK = 4096


def _decode_uint16(tensor: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    return (tensor.to(torch.float32) - zero.reshape(-1)[0].to(torch.float32)) * scale.reshape(-1)[0].to(torch.float32)


def _quantize_uint16(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = tensor.to(torch.float32)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        scale_value = 1.0 / 65535.0
        zero_value = 0
    else:
        scale_value = (high - low) / 65535.0
        zero_value = int(round(-low / scale_value))
        zero_value = max(0, min(65535, zero_value))
    scale = torch.tensor([scale_value], dtype=torch.float32)
    zero = torch.tensor([zero_value], dtype=torch.int32)
    codes = torch.round(values / scale_value + zero_value).clamp(0, 65535).to(torch.uint16)
    return codes.contiguous(), scale, zero


def _decode_lpbq_chunk(weight: torch.Tensor, scale1: torch.Tensor, scale2: torch.Tensor,
                       start: int, end: int, group_size: int) -> torch.Tensor:
    if tuple(weight.shape[:2]) != (1, 1):
        raise ValueError(f"unexpected Conv2D LPBQ shape {tuple(weight.shape)}")
    in_features = int(weight.shape[2])
    out_features = int(weight.shape[3])
    groups = in_features // group_size
    codes = weight[0, 0, :, start:end].to(torch.int16).transpose(0, 1)
    signed = torch.where(codes >= 8, codes - 16, codes)
    s1 = scale1.reshape(out_features, groups)[start:end].to(torch.float32)
    s2 = scale2.reshape(out_features)[start:end].to(torch.float32)
    return (signed.reshape(end - start, groups, group_size).to(torch.float32)
            * s1.unsqueeze(-1) * s2.reshape(-1, 1, 1)).reshape(end - start, in_features)


def _load_all(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    values: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            values[key] = handle.get_tensor(key).contiguous()
    return values, metadata


def _strict_merge_layers(base: dict[str, torch.Tensor], layer_paths: dict[int, Path]) -> dict[str, torch.Tensor]:
    merged = dict(base)
    for layer, path in sorted(layer_paths.items()):
        prefix = f"model.layers.{layer}."
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                value = handle.get_tensor(key).contiguous()
                if key.startswith(prefix):
                    merged[key] = value
                elif key not in merged:
                    merged[key] = value
                elif not torch.equal(merged[key], value):
                    raise ValueError(f"non-equal duplicate key while merging layer {layer}: {key}")
    return merged


def _fold_embedding(merged: dict[str, torch.Tensor]) -> dict[str, Any]:
    key = "model.embed_tokens.weight"
    weight = merged[key]
    scale = merged["model.embed_tokens.scale"]
    zero = merged["model.embed_tokens.zero_point"]
    if weight.dtype != torch.uint16 or tuple(weight.shape) != (151936, HIDDEN):
        raise ValueError(f"unexpected embedding carrier: {weight.dtype} {tuple(weight.shape)}")
    low = float("inf")
    high = float("-inf")
    for start in range(0, weight.shape[0], CHUNK):
        end = min(start + CHUNK, weight.shape[0])
        transformed = normalized_fwht(_decode_uint16(weight[start:end], scale, zero), dim=-1)
        low = min(low, float(transformed.min()))
        high = max(high, float(transformed.max()))
    scale_value = (high - low) / 65535.0 if high > low else 1.0 / 65535.0
    zero_value = max(0, min(65535, int(round(-low / scale_value))))
    output = torch.empty_like(weight)
    for start in range(0, weight.shape[0], CHUNK):
        end = min(start + CHUNK, weight.shape[0])
        transformed = normalized_fwht(_decode_uint16(weight[start:end], scale, zero), dim=-1)
        output[start:end] = torch.round(transformed / scale_value + zero_value).clamp(0, 65535).to(torch.uint16)
    merged[key] = output
    merged["model.embed_tokens.scale"] = torch.tensor([scale_value], dtype=torch.float32)
    merged["model.embed_tokens.zero_point"] = torch.tensor([zero_value], dtype=torch.int32)
    return {"scale": scale_value, "zero_point": zero_value, "rows": int(weight.shape[0])}


def _fold_final_boundary(merged: dict[str, torch.Tensor]) -> dict[str, Any]:
    norm = _decode_uint16(merged["model.norm.weight"], merged["model.norm.scale"], merged["model.norm.zero_point"])
    weight = merged["lm_head.weight"]
    scale1 = merged["lm_head.scale1"]
    scale2 = merged["lm_head.scale2"]
    if tuple(weight.shape[:2]) != (1, 1) or int(weight.shape[2]) != HIDDEN:
        raise ValueError(f"unexpected lm_head carrier shape {tuple(weight.shape)}")
    out_features = int(weight.shape[3])
    packed = torch.empty_like(weight)
    out_scale1 = torch.empty_like(scale1)
    out_scale2 = torch.empty_like(scale2)
    groups = HIDDEN // GROUP
    for start in range(0, out_features, CHUNK):
        end = min(start + CHUNK, out_features)
        canonical = _decode_lpbq_chunk(weight, scale1, scale2, start, end, GROUP)
        # OI lm_head matrix: W' = W * diag(gamma) * R1.
        canonical = normalized_fwht(canonical * norm.reshape(1, -1), dim=1)
        q_weight, q_scale1, q_scale2, _ = _quantize_lpbq_g32(canonical, GROUP)
        packed[:, :, :, start:end] = q_weight
        out_scale1[start * groups:end * groups] = q_scale1
        out_scale2[start:end] = q_scale2
    merged["lm_head.weight"] = packed
    merged["lm_head.scale1"] = out_scale1
    merged["lm_head.scale2"] = out_scale2
    # RMSNorm gamma is now included in the folded lm_head matrix.
    merged["model.norm.weight"] = torch.full_like(merged["model.norm.weight"], 65535)
    merged["model.norm.scale"] = torch.full_like(merged["model.norm.scale"], 1.0 / 65535.0)
    merged["model.norm.zero_point"] = torch.zeros_like(merged["model.norm.zero_point"])
    return {"norm_gamma_sha256": tensor_sha256(norm), "lm_head_out_features": out_features}


def _add_online_r1(merged: dict[str, torch.Tensor]) -> dict[str, Any]:
    r1 = normalized_hadamard_matrix(HIDDEN)
    packed, scale1, scale2, stats = _quantize_lpbq_g32(r1, GROUP)
    merged["model.r1_dense.weight"] = packed.reshape(HIDDEN, HIDDEN)
    merged["model.r1_dense.scale1"] = scale1
    merged["model.r1_dense.scale2"] = scale2
    return {"canonical_sha256": tensor_sha256(r1), "quantization": stats}


def _write_full_manifest(
    variant: str,
    checkpoint: Path,
    metadata: dict[str, str],
    input_records: dict[str, Any],
    provenance: dict[str, Any],
    layer_manifest_paths: list[Path],
    boundary_records: dict[str, Any],
    layers: list[int],
) -> dict[str, Any]:
    _, inventory = checkpoint_inventory(checkpoint)
    payload = {
        "schema_version": 2,
        "manifest_kind": "mllm.qwen3.rotation.full_model",
        "model": "Qwen3-1.7B",
        "scope": "full_model",
        "layers": layers,
        "variant": variant,
        "dimensions": {
            "hidden_size": HIDDEN,
            "head_dim": HEAD_DIM,
            "query_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "num_hidden_layers": len(layers),
        },
        "group_size": GROUP,
        "inputs": input_records,
        "provenance": provenance,
        "rotation_contract": {
            "r1": basis_record("R1", HIDDEN, applied=variant != "A_original"),
            "r2": basis_record("R2", HEAD_DIM, applied=variant != "A_original"),
            "r3": basis_record("R3", HEAD_DIM, applied=variant in (FULL_D_ONLINE, FULL_D_FOLDED)),
            "global_basis_policy": "one shared R1/R2/R3 family across all decoder layers",
            "hidden_boundary": "R1 inside decoder stack",
            "value_cache_boundary": "R2 per KV head",
            "key_cache_boundary": "R3 for D; baseline for A/C",
            "past_key_policy": "past K already uses R3; rotate current K only" if variant in (FULL_D_ONLINE, FULL_D_FOLDED) else "baseline key basis",
        },
        "boundary": boundary_records,
        "layer_manifests": [
            {"layer": layer, "path": str(path.resolve()), "manifest_id": json.loads(path.read_text())["manifest_id"]}
            for layer, path in zip(layers, layer_manifest_paths)
        ],
        "graph_expectations": {
            "backend": "QNN HTP",
            "unintended_cpu_fallback_allowed": False,
            "r3_dense": {
                "expected_matmul_count_per_layer": Q_HEADS + KV_HEADS,
                "carrier_pattern": "model.layers.{layer}.self_attn.r3_dense.weight",
                "layers": layers,
            } if variant in (FULL_D_ONLINE, FULL_D_FOLDED) else None,
        },
        "artifact": {
            "stage": "full_model_merge",
            "format": "safetensors",
            "checkpoint": file_record(checkpoint, role="full_model_checkpoint"),
            "metadata": metadata,
            "tensor_count": len(inventory),
            "tensors": inventory,
        },
    }
    return finalize_manifest(payload)


def validate_full_manifest(path: Path, verify_inputs: bool = True) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks: dict[str, Any] = {}
    unsigned = copy.deepcopy(payload)
    claimed_id = unsigned.pop("manifest_id", None)
    actual_id = canonical_payload_sha256(unsigned)
    if claimed_id != actual_id:
        errors.append("manifest_id mismatch")
    if payload.get("schema_version") != 2 or payload.get("manifest_kind") != "mllm.qwen3.rotation.full_model":
        errors.append("wrong full-model manifest schema")
    checkpoint_record = payload.get("artifact", {}).get("checkpoint", {})
    checkpoint = Path(checkpoint_record.get("path", ""))
    if not checkpoint.is_file():
        errors.append(f"checkpoint missing: {checkpoint}")
    else:
        if sha256_file(checkpoint) != checkpoint_record.get("sha256"):
            errors.append("checkpoint sha256 mismatch")
        if checkpoint.stat().st_size != checkpoint_record.get("size_bytes"):
            errors.append("checkpoint size mismatch")
        metadata, inventory = checkpoint_inventory(checkpoint)
        if metadata != payload["artifact"].get("metadata"):
            errors.append("checkpoint metadata mismatch")
        if inventory != payload["artifact"].get("tensors"):
            errors.append("checkpoint tensor inventory mismatch")
        checks["tensor_count"] = len(inventory)
    for name in ("r1", "r2", "r3"):
        record = payload.get("rotation_contract", {}).get(name, {})
        if record.get("applied"):
            expected = independent_hadamard_matrix(int(record["order"]))
            if record.get("float32_matrix_sha256") != tensor_sha256(expected):
                errors.append(f"{name} basis hash mismatch")
    layers = [int(x) for x in payload.get("layers", [])]
    variant = payload.get("variant")
    layer_records = payload.get("layer_manifests", [])
    if variant == "A_original":
        if layer_records:
            errors.append("A_original must not carry layer manifests")
    else:
        expected_layer_variant = LAYER_VARIANT.get(variant)
        if expected_layer_variant is None:
            errors.append(f"unknown full-model variant: {variant}")
        if len(layer_records) != len(layers):
            errors.append(
                "layer manifest count does not match full-model layer count: "
                f"{len(layer_records)} != {len(layers)}"
            )
        layer_checks: list[dict[str, Any]] = []
        for index, layer in enumerate(layers):
            record = layer_records[index] if index < len(layer_records) else {}
            manifest_path = Path(record.get("path", ""))
            layer_check: dict[str, Any] = {"layer": layer, "path": str(manifest_path)}
            if not manifest_path.is_file():
                errors.append(f"layer manifest missing for layer {layer}: {manifest_path}")
                layer_checks.append({**layer_check, "pass": False})
                continue
            try:
                child_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                child_report = validate_export_manifest(manifest_path, verify_inputs=False)
            except Exception as exc:
                errors.append(f"layer manifest unreadable for layer {layer}: {exc}")
                layer_checks.append({**layer_check, "pass": False, "error": str(exc)})
                continue
            child_ok = bool(child_report["pass"])
            if not child_ok:
                errors.extend(
                    f"layer {layer} manifest: {error}" for error in child_report["errors"]
                )
            if child_payload.get("layer") != layer:
                errors.append(f"layer manifest index mismatch: expected {layer}")
                child_ok = False
            if child_payload.get("variant") != expected_layer_variant:
                errors.append(
                    f"layer {layer} manifest variant mismatch: "
                    f"{child_payload.get('variant')} != {expected_layer_variant}"
                )
                child_ok = False
            if record.get("manifest_id") != child_payload.get("manifest_id"):
                errors.append(f"layer {layer} manifest_id link mismatch")
                child_ok = False
            layer_checks.append({
                **layer_check,
                "manifest_id": child_payload.get("manifest_id"),
                "pass": child_ok,
                "checks": child_report.get("checks", {}),
            })
        checks["layer_manifests"] = layer_checks
    if variant in (FULL_D_ONLINE, FULL_D_FOLDED):
        for layer in layers:
            prefix = f"model.layers.{layer}.self_attn.r3_dense."
            if not all(prefix + suffix in payload["artifact"]["tensors"] for suffix in ("weight", "scale1", "scale2")):
                errors.append(f"missing R3 carrier for layer {layer}")
        if variant == FULL_D_ONLINE and "model.r1_dense.weight" not in payload["artifact"]["tensors"]:
            errors.append("online variant missing model.r1_dense.weight")
        if variant == FULL_D_FOLDED and "model.r1_dense.weight" in payload["artifact"]["tensors"]:
            errors.append("folded variant still contains online R1 carrier")
    if verify_inputs:
        base = payload.get("inputs", {}).get("base_quant_checkpoint", {})
        if Path(base.get("path", "")).is_file() and sha256_file(Path(base["path"])) != base.get("sha256"):
            errors.append("base checkpoint input hash mismatch")
    checks["input_files_verified"] = verify_inputs
    return {"schema_version": 1, "manifest": str(path.resolve()), "manifest_id": claimed_id,
            "pass": not errors, "errors": errors, "checks": checks}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path)
    parser.add_argument("--base-quant-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--variants", nargs="+", choices=("A_original",) + DEFAULT_VARIANTS, default=list(DEFAULT_VARIANTS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", type=Path, help="Validate an existing full-model export manifest.")
    parser.add_argument("--skip-input-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.validate:
        report = validate_full_manifest(args.validate, verify_inputs=not args.skip_input_files)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(0 if report["pass"] else 1)
    if not args.source_model or not args.base_quant_checkpoint or not args.output_dir:
        raise SystemExit("--source-model, --base-quant-checkpoint, and --output-dir are required")
    layers = list(range(28)) if args.layers is None else sorted(set(args.layers))
    if layers != list(range(28)):
        raise SystemExit("full-model export currently requires exactly layers 0..27")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_root = output_dir / "layers"
    source_files = sorted(Path(p) for p in __import__("glob").glob(str(args.source_model / "*.safetensors")))
    input_records = build_input_records(args.source_model, source_files, args.base_quant_checkpoint)
    repo_root = Path(__file__).resolve().parents[1]
    provenance = build_provenance(repo_root, sys.argv, (Path(__file__), Path(__file__).with_name("qwen3_block_rotation.py"), Path(__file__).with_name("qwen3_rotation_manifest.py"), Path(__file__).with_name("export_qwen3_lpbq_g32.py")))
    started = time.perf_counter()

    for layer in layers:
        layer_dir = layer_root / f"layer_{layer}"
        if not (layer_dir / VARIANT_D_DENSE / "model.safetensors").is_file():
            export_variants(source_model=args.source_model, base_quant_checkpoint=args.base_quant_checkpoint,
                            output_dir=layer_dir, layer=layer, variants=(VARIANT_C, VARIANT_D_DENSE), overwrite=args.overwrite)
    base, base_metadata = _load_all(args.base_quant_checkpoint)
    full_records = []
    for variant in args.variants:
        if variant == "A_original":
            merged = dict(base)
            boundary = {"mode": "baseline", "embedding": "unchanged", "final_norm": "unchanged", "lm_head": "unchanged"}
            layer_paths = []
            layer_manifest_paths = []
        else:
            layer_variant = LAYER_VARIANT[variant]
            layer_paths = {layer: layer_root / f"layer_{layer}" / layer_variant / "model.safetensors" for layer in layers}
            for layer, p in layer_paths.items():
                if not p.is_file():
                    raise FileNotFoundError(p)
            merged = _strict_merge_layers(base, layer_paths)
            layer_manifest_paths = [layer_root / f"layer_{layer}" / layer_variant / "export_manifest.json" for layer in layers]
            if variant == FULL_D_ONLINE:
                boundary = {"mode": "online_reference", "embedding": "baseline", "final_norm": "baseline", "lm_head": "baseline"}
                boundary["r1_dense"] = _add_online_r1(merged)
            else:
                boundary = {"mode": "folded_production", "embedding": _fold_embedding(merged)}
                boundary["final"] = _fold_final_boundary(merged)
        variant_dir = output_dir / "variants" / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = variant_dir / "model.safetensors"
        if checkpoint.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {checkpoint}")
        metadata = dict(base_metadata)
        metadata.update({"mllm.full_model.variant": variant, "mllm.full_model.layers": "0..27",
                         "mllm.full_model.boundary": boundary["mode"],
                         "mllm.full_model.generated_by": "qwen3_full_model_rotation.py"})
        save_file(merged, str(checkpoint), metadata=metadata)
        manifest = _write_full_manifest(variant, checkpoint, metadata, input_records, provenance,
                                        layer_manifest_paths, boundary, layers)
        manifest_path = variant_dir / "export_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report = validate_full_manifest(manifest_path, verify_inputs=False)
        report_path = variant_dir / "manifest_validation.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not report["pass"]:
            raise SystemExit(f"full-model manifest validation failed for {variant}: {report['errors']}")
        full_records.append({"variant": variant, "manifest": str(manifest_path), "checkpoint": str(checkpoint),
                             "bytes": checkpoint.stat().st_size, "manifest_id": manifest["manifest_id"]})
    root = finalize_manifest({"schema_version": 2, "manifest_kind": "mllm.qwen3.rotation.full_model.experiment",
                              "model": "Qwen3-1.7B", "layers": layers, "inputs": input_records,
                              "provenance": provenance, "variants": full_records,
                              "elapsed_seconds": time.perf_counter() - started})
    (output_dir / "experiment_manifest.json").write_text(json.dumps(root, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "variants": full_records, "manifest_id": root["manifest_id"]}, indent=2))


if __name__ == "__main__":
    main()
