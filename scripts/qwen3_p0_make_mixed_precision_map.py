#!/usr/bin/env python3
"""Convert tensor-local P0 results into a conservative mixed-precision map.

The local sensitivity map is intentionally not overwritten.  A layer is
marked as block-risky when either its one-layer block NMSE or its held-out
last-token logits cosine crosses the configured threshold.  For such layers,
the nonlinear MLP path (gate/up/down) falls back to A16 activation while
attention q/k/v/o remains A8.  Existing tensor-local A16 recommendations are
preserved.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


MLP_PROJECTIONS = {"gate_proj", "up_proj", "down_proj"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensitivity-map",
        type=Path,
        default=Path("artifacts/p0/static_a8/sensitivity-map.json"),
    )
    parser.add_argument(
        "--block-eval",
        type=Path,
        default=Path("artifacts/p0/static_a8/block-eval-all-layers.json"),
    )
    parser.add_argument("--block-nmse-threshold", type=float, default=0.01)
    parser.add_argument("--logits-cosine-threshold", type=float, default=0.99)
    parser.add_argument(
        "--full-a16-layers",
        default="",
        help="Comma-separated layers whose q/k/v/o and MLP inputs all fall back to A16.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/p0/static_a8/mixed-precision-map.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensitivity = json.loads(args.sensitivity_map.read_text(encoding="utf-8"))
    block_eval = json.loads(args.block_eval.read_text(encoding="utf-8"))
    block_by_layer = {int(row["layer"]): row for row in block_eval["rows"]}
    full_a16_layers = {
        int(item.strip())
        for item in args.full_a16_layers.split(",")
        if item.strip()
    }
    risky_layers: set[int] = set()
    risk_reasons: dict[int, list[str]] = {}
    for layer, row in block_by_layer.items():
        reasons: list[str] = []
        block_nmse = float(row["selected_mixed"]["block_output"]["nmse"])
        logits_cosine = float(row["selected_mixed"]["last_token_logits"]["cosine"])
        if block_nmse > args.block_nmse_threshold:
            reasons.append(f"block_nmse>{args.block_nmse_threshold:g}")
        if logits_cosine < args.logits_cosine_threshold:
            reasons.append(f"logits_cosine<{args.logits_cosine_threshold:g}")
        if reasons:
            risky_layers.add(layer)
            risk_reasons[layer] = reasons

    output = copy.deepcopy(sensitivity)
    output["schema_version"] = 2
    output["purpose"] = "P0 conservative mixed-precision activation map"
    output["policy"] = {
        "block_nmse_threshold": args.block_nmse_threshold,
        "logits_cosine_threshold": args.logits_cosine_threshold,
        "risky_layers": sorted(risky_layers),
        "fallback_projections": sorted(MLP_PROJECTIONS),
        "full_a16_layers": sorted(full_a16_layers),
        "attention_policy": "retain tensor-local A8 unless separately overridden",
        "weight_contract": "unrotated LPBQ W4 G32",
    }
    for row in output["rows"]:
        layer = int(row["layer"])
        projection = str(row["projection"])
        if row["recommended_precision"] == "A16":
            row["fallback_reason"] = "tensor_local_nmse_screen"
        if layer in full_a16_layers:
            row["recommended_precision"] = "A16"
            row["fallback_reason"] = "full_layer_ablation"
        elif layer in risky_layers and projection in MLP_PROJECTIONS:
            row["recommended_precision"] = "A16"
            row["fallback_reason"] = "block_gate:" + ",".join(risk_reasons[layer])
    output["summary"] = {
        "tensor_count": len(output["rows"]),
        "a16_tensor_count": sum(
            row["recommended_precision"] == "A16" for row in output["rows"]
        ),
        "a8_tensor_count": sum(
            row["recommended_precision"] == "A8" for row in output["rows"]
        ),
        "risky_layer_count": len(risky_layers),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    print(f"risky_layers={sorted(risky_layers)}")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
