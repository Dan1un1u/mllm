#!/usr/bin/env python3

"""Audit full-model qparams and sweep the masked-E2Softmax reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean

import validate_vtcm_masked_e2softmax as reference


OP_TYPE = "VtcmMaskedE2Softmax"


def inspect_manifest(path: Path, expected_count: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tensors = {tensor["name"]: tensor for tensor in payload["tensors"]}
    operations = [op for op in payload["operations"] if op.get("qnn_op_type") == OP_TYPE]

    score_scales: set[float] = set()
    score_dtypes: set[str] = set()
    mask_scales: set[float] = set()
    mask_zero_points: set[int] = set()
    mask_dtypes: set[str] = set()
    output_scales: set[float] = set()
    output_zero_points: set[int] = set()
    output_dtypes: set[str] = set()
    sequence_lengths: set[int] = set()

    for operation in operations:
        scores = tensors[operation["inputs"][0]]
        mask = tensors[operation["inputs"][1]]
        output = tensors[operation["outputs"][0]]
        score_quant = scores["qnn_quantization"]
        mask_quant = mask["qnn_quantization"]
        output_quant = output["qnn_quantization"]
        score_scales.add(float(score_quant["scale"]))
        score_dtypes.add(scores["qnn_dtype"])
        mask_scales.add(float(mask_quant["scale"]))
        mask_zero_points.add(int(mask_quant["zero_point"]))
        mask_dtypes.add(mask["qnn_dtype"])
        output_scales.add(float(output_quant["scale"]))
        output_zero_points.add(int(output_quant["zero_point"]))
        output_dtypes.add(output["qnn_dtype"])
        sequence_lengths.add(int(scores["dimensions"][2]))

    coefficients = {
        scale: max(1, min(255, int(scale * 1.4375 * 256.0 + 0.5)))
        for scale in score_scales
    }
    coefficient_saturated = [
        scale for scale in score_scales if scale * 1.4375 * 256.0 > 255.0
    ]
    contract_valid = (
        len(operations) == expected_count
        and score_dtypes == {"UFIXED_POINT_8"}
        and mask_dtypes == {"UFIXED_POINT_8"}
        and output_dtypes == {"UFIXED_POINT_8"}
        and mask_zero_points == {reference.MASK_VALID}
        and len(mask_scales) == 1
        and output_zero_points == {0}
        and len(output_scales) == 1
        and math.isclose(next(iter(output_scales)), 1.0 / reference.OUTPUT_LEVELS, rel_tol=1e-6)
        and not coefficient_saturated
        and len(sequence_lengths) == 1
    )

    sequence_length = next(iter(sequence_lengths)) if len(sequence_lengths) == 1 else 0
    scores, mask = reference.fixture(sequence_length) if sequence_length in (1, 32) else (b"", b"")
    sweep = []
    for scale in sorted(score_scales):
        observed = reference.integer_reference(scores, mask, sequence_length, scale)
        exact = reference.float_reference(scores, mask, sequence_length, scale)
        errors = [
            abs(raw / reference.OUTPUT_LEVELS - expected)
            for raw, expected in zip(observed, exact)
        ]
        row_l1 = []
        row_sums = []
        top1_matches = 0
        for row in range(sequence_length):
            begin = row * reference.CONTEXT
            end = begin + reference.CONTEXT
            valid = [
                depth
                for depth in range(reference.CONTEXT)
                if mask[begin + depth] == reference.MASK_VALID
            ]
            row_l1.append(sum(errors[begin:end]))
            row_sums.append(sum(observed[begin:end]))
            top1_matches += max(valid, key=lambda depth: observed[begin + depth]) == max(
                valid, key=lambda depth: scores[begin + depth]
            )
        sweep.append(
            {
                "score_scale": scale,
                "coefficient_q8": coefficients[scale],
                "mae_all_elements": fmean(errors),
                "max_abs_error": max(errors),
                "mean_row_l1": fmean(row_l1),
                "max_row_l1": max(row_l1),
                "quantized_row_sum_min": min(row_sums),
                "quantized_row_sum_max": max(row_sums),
                "top1_match_rate": top1_matches / sequence_length,
            }
        )

    return {
        "manifest": str(path),
        "expected_count": expected_count,
        "observed_count": len(operations),
        "sequence_length": sequence_length,
        "unique_score_scales": len(score_scales),
        "score_scale_min": min(score_scales, default=None),
        "score_scale_max": max(score_scales, default=None),
        "coefficient_saturated_scales": sorted(coefficient_saturated),
        "mask_scales": sorted(mask_scales),
        "mask_zero_points": sorted(mask_zero_points),
        "output_scales": sorted(output_scales),
        "output_zero_points": sorted(output_zero_points),
        "all_tensors_u8": score_dtypes == mask_dtypes == output_dtypes == {"UFIXED_POINT_8"},
        "contract_valid": contract_valid,
        "software_reference_sweep": sweep,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--expected-count", type=int, default=448)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports = [inspect_manifest(path, args.expected_count) for path in args.manifests]
    result = {
        "op_type": OP_TYPE,
        "accuracy_gate_applied": False,
        "manifests": reports,
        "implementation_contract_valid": all(report["contract_valid"] for report in reports),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["implementation_contract_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
