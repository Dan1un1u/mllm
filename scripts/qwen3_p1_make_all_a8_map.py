#!/usr/bin/env python3
"""Materialize a reproducible all-A8 P1 map from the P0 sensitivity map."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=Path("artifacts/p0/static_a8/sensitivity-map.json"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/p0/static_a8/all-a8-map.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = copy.deepcopy(json.loads(args.input_json.read_text(encoding="utf-8")))
    output["schema_version"] = 2
    output["purpose"] = "P1 full W4A8 activation map (no A16 fallback)"
    output["policy"] = {
        "activation_precision": "A8 for every decoder Linear input",
        "fallback_projections": [],
        "source_sensitivity_map": str(args.input_json),
        "weight_contract": "unrotated LPBQ W4 G32",
    }
    for row in output["rows"]:
        row["recommended_precision"] = "A8"
        row["fallback_reason"] = None
    output["summary"] = {
        "tensor_count": len(output["rows"]),
        "a16_tensor_count": 0,
        "a8_tensor_count": len(output["rows"]),
        "risky_layer_count": 0,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
