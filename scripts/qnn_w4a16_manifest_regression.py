#!/usr/bin/env python3
"""Compare W4A16 manifests while isolating synthetic zero-bias scale metadata."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from qnn_quant_manifest_equivalent import canonicalize, digest, first_difference


def normalize_synthetic_rmsnorm_biases(source: Path, destination: Path) -> dict:
    document = json.loads(source.read_text(encoding="utf-8"))
    tensors = {tensor["name"]: tensor for tensor in document["tensors"]}
    scales = []
    count = 0
    for operation in document["operations"]:
        if operation["qnn_op_type"] != "RmsNorm":
            continue
        if len(operation["inputs"]) != 3:
            raise AssertionError(f"{operation['name']}: RmsNorm does not have three inputs")
        bias = tensors[operation["inputs"][2]]
        quant = bias.get("qnn_quantization", {})
        recipe = bias.get("quant_recipe", {})
        expected = (
            bias.get("qnn_dtype") == "UFIXED_POINT_16"
            and bias.get("logical_quant_dtype") == "UInt16"
            and bias.get("tensor_type") == "STATIC"
            and bias.get("producer") is None
            and bias.get("consumers") == [operation["name"]]
            and quant.get("defined") is True
            and quant.get("encoding") == "scale_offset"
            and quant.get("zero_point") == 0
            and recipe.get("type") == "asymmetric_per_tensor"
            and recipe.get("quant_to_dtype") == "UInt16"
        )
        if not expected:
            raise AssertionError(f"{operation['name']}: synthetic zero-bias contract changed")
        scales.append(float(quant["scale"]))
        # Both physical carriers store integer zero with zero_point=0.  Their
        # real value is exactly zero for any positive scale, so this is the one
        # field intentionally excluded from graph-contract equality.
        quant["scale"] = "SYNTHETIC_ZERO_BIAS_SCALE"
        count += 1
    if count != 729:
        raise AssertionError(f"expected 729 synthetic RmsNorm biases, got {count}")
    destination.write_text(json.dumps(document), encoding="utf-8")
    return {"count": count, "scale_min": min(scales), "scale_max": max(scales)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="w4a16-manifest-regression-") as directory:
        root = Path(directory)
        baseline_bias = normalize_synthetic_rmsnorm_biases(args.baseline, root / "baseline.json")
        candidate_bias = normalize_synthetic_rmsnorm_biases(args.candidate, root / "candidate.json")
        baseline = canonicalize(root / "baseline.json")
        candidate = canonicalize(root / "candidate.json")
    difference = first_difference(baseline, candidate)
    report = {
        "passed": not difference,
        "allowed_difference": (
            "QNN scale metadata of the static all-zero UInt16 RmsNorm bias; "
            "integer value and zero_point are both zero, so represented real bias is exactly zero"
        ),
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "baseline_bias": baseline_bias,
        "candidate_bias": candidate_bias,
        "first_unexpected_difference": difference or None,
        "normalized_baseline_sha256": digest(baseline),
        "normalized_candidate_sha256": digest(candidate),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not difference else 1


if __name__ == "__main__":
    sys.exit(main())
