#!/usr/bin/env python3
"""Verify the software-side ``sym128_vsym`` deployment contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, action="append", required=True)
    return parser.parse_args()


def check_a8(value: dict[str, object], *, label: str) -> None:
    if value.get("recipe") != "sym128_vsym":
        raise ValueError(f"{label}: recipe is not sym128_vsym")
    if value.get("storage_dtype") != "UInt8" or value.get("dtype") != "UInt8":
        raise ValueError(f"{label}: storage dtype is not UInt8")
    if int(value.get("zero_point", -1)) != 128:
        raise ValueError(f"{label}: zero_point is not integer 128")
    scale = float(value["scale"])
    if not scale > 0.0:
        raise ValueError(f"{label}: scale is not positive")


def verify(training_dir: Path) -> dict[str, object]:
    manifest_path = training_dir / "streaming-train.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field, expected in (
        ("a8_recipe", "sym128_vsym"),
        ("storage_dtype", "UInt8"),
        ("zero_point", 128),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"{training_dir}: {field} != {expected!r}")
    if manifest.get("v_scale_tied") is not True:
        raise ValueError(f"{training_dir}: v_scale_tied is not true")

    rows = {int(row["layer"]): row for row in manifest.get("rows", [])}
    if set(rows) != set(range(28)):
        raise ValueError(f"{training_dir}: expected rows for all 28 layers")
    input_count = 0
    v_output_count = 0
    for layer, row in rows.items():
        for projection, value in row.get("learned_params_manifest", {}).items():
            if value is not None:
                check_a8(value, label=f"layer {layer} {projection} input")
                input_count += 1
        output = row.get("v_output_params", {}) or {}
        v_value = output.get("v_proj")
        identity = row.get("v_scale_identity")
        if v_value is None:
            if identity is not None:
                raise ValueError(f"layer {layer}: missing V output but has identity")
            continue
        check_a8(v_value, label=f"layer {layer} v_proj output")
        v_output_count += 1
        if not isinstance(identity, dict) or identity.get("tied") is not True:
            raise ValueError(f"layer {layer}: V scale identity is not tied")
        if int(identity.get("zero_point", -1)) != 128:
            raise ValueError(f"layer {layer}: V identity zero_point is not 128")
        if float(identity["scale"]) != float(v_value["scale"]):
            raise ValueError(f"layer {layer}: V identity scale differs from output scale")

    return {
        "training_dir": str(training_dir),
        "complete": bool(manifest.get("complete")),
        "a8_recipe": manifest["a8_recipe"],
        "zero_point": manifest["zero_point"],
        "storage_dtype": manifest["storage_dtype"],
        "input_a8_tensors": input_count,
        "v_output_a8_layers": v_output_count,
        "v_output_a16_layers": 28 - v_output_count,
    }


def main() -> None:
    args = parse_args()
    for training_dir in args.training_dir:
        print(json.dumps(verify(training_dir), sort_keys=True))


if __name__ == "__main__":
    main()
