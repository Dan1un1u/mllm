#!/usr/bin/env python3

"""Generate and validate deterministic U8 masked-E2Softmax device fixtures.

This validator is intentionally independent of QNN tensor/layout code. It
models the logical U8 rows, the integer E2Softmax contract implemented by the
HTP custom op, and an exact floating-point masked Softmax reference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean


CONTEXT = 1024
SCORE_SCALE = 0.17946475744247437
SCORE_ZERO_POINT = 113
OUTPUT_LEVELS = 255
MASK_VALID = 255
MASK_INVALID = 0
EXPONENT_NUMERATOR = (16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0)
EXPONENT_COEFFICIENT = max(1, min(255, int(SCORE_SCALE * 1.4375 * 256.0 + 0.5)))
RECIPROCAL_FRACTION_BITS = 23


def fixture(seq: int) -> tuple[bytearray, bytearray]:
    scores = bytearray(seq * CONTEXT)
    mask = bytearray(seq * CONTEXT)
    for row in range(seq):
        valid_count = 913 if seq == 1 else 993 + row
        for depth in range(CONTEXT):
            # Reproducible non-monotonic data catches row/depth permutation
            # bugs while remaining within the accepted layer-14 U8 encoding.
            raw = SCORE_ZERO_POINT + ((depth * 17 + row * 29 + (depth * row % 31) * 3) % 96) - 48
            scores[row * CONTEXT + depth] = max(0, min(255, raw))
            mask[row * CONTEXT + depth] = MASK_VALID if depth < valid_count else MASK_INVALID

        # Unique peaks exercise normalization and make the expected top-1
        # invariant unambiguous for every row.
        peak = (37 + row * 67) % valid_count
        shoulder = (peak + 257) % valid_count
        scores[row * CONTEXT + peak] = 228 - row % 7
        scores[row * CONTEXT + shoulder] = 211 - row % 5
    return scores, mask


def integer_reference(
    scores: bytes, mask: bytes, seq: int, score_scale: float = SCORE_SCALE
) -> bytearray:
    exponent_coefficient = max(1, min(255, int(score_scale * 1.4375 * 256.0 + 0.5)))
    output = bytearray(seq * CONTEXT)
    for row in range(seq):
        begin = row * CONTEXT
        valid = [depth for depth in range(CONTEXT) if mask[begin + depth] == MASK_VALID]
        if not valid:
            continue
        row_max = max(scores[begin + depth] for depth in valid)
        codes = [15] * CONTEXT
        denominator = 0
        for depth in valid:
            difference = row_max - scores[begin + depth]
            code = min(14, (difference * exponent_coefficient + 128) >> 8)
            codes[depth] = code
            denominator += EXPONENT_NUMERATOR[code]

        reciprocal_q23 = ((OUTPUT_LEVELS << RECIPROCAL_FRACTION_BITS) + denominator // 2) // denominator
        for depth in valid:
            numerator = EXPONENT_NUMERATOR[codes[depth]]
            probability = (
                numerator * reciprocal_q23 + (1 << (RECIPROCAL_FRACTION_BITS - 1))
            ) >> RECIPROCAL_FRACTION_BITS
            output[begin + depth] = max(0, min(255, probability))
    return output


def float_reference(
    scores: bytes, mask: bytes, seq: int, score_scale: float = SCORE_SCALE
) -> list[float]:
    output = [0.0] * (seq * CONTEXT)
    for row in range(seq):
        begin = row * CONTEXT
        valid = [depth for depth in range(CONTEXT) if mask[begin + depth] == MASK_VALID]
        if not valid:
            continue
        row_max = max(scores[begin + depth] for depth in valid)
        exponentials = [math.exp((scores[begin + depth] - row_max) * score_scale) for depth in valid]
        denominator = sum(exponentials)
        for depth, value in zip(valid, exponentials):
            output[begin + depth] = value / denominator
    return output


def generate(seq: int, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores, mask = fixture(seq)
    expected = integer_reference(scores, mask, seq)
    paths = {
        "scores": output_dir / f"scores_s{seq}.bin",
        "mask": output_dir / f"mask_s{seq}.bin",
        "integer_reference": output_dir / f"integer_reference_s{seq}.bin",
    }
    paths["scores"].write_bytes(scores)
    paths["mask"].write_bytes(mask)
    paths["integer_reference"].write_bytes(expected)
    return {
        "seq": seq,
        "bytes_per_tensor": len(scores),
        "score_scale": SCORE_SCALE,
        "score_zero_point": SCORE_ZERO_POINT,
        "output_scale": 1.0 / OUTPUT_LEVELS,
        "output_zero_point": 0,
        "mask_valid_raw": MASK_VALID,
        "mask_invalid_raw": MASK_INVALID,
        "e2_exponent_coefficient_q8": EXPONENT_COEFFICIENT,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def analyze(seq: int, fixture_dir: Path, device_output_path: Path) -> dict[str, object]:
    scores = (fixture_dir / f"scores_s{seq}.bin").read_bytes()
    mask = (fixture_dir / f"mask_s{seq}.bin").read_bytes()
    expected = (fixture_dir / f"integer_reference_s{seq}.bin").read_bytes()
    observed = device_output_path.read_bytes()
    expected_bytes = seq * CONTEXT
    for label, payload in (("scores", scores), ("mask", mask), ("expected", expected), ("observed", observed)):
        if len(payload) != expected_bytes:
            raise ValueError(f"{label}: expected {expected_bytes} bytes, got {len(payload)}")

    deltas = [abs(int(actual) - int(want)) for actual, want in zip(observed, expected)]
    mismatches = sum(delta != 0 for delta in deltas)
    masked_nonzero = sum(actual != 0 for actual, raw_mask in zip(observed, mask) if raw_mask != MASK_VALID)
    float_probabilities = float_reference(scores, mask, seq)
    absolute_errors = [abs(actual / OUTPUT_LEVELS - reference) for actual, reference in zip(observed, float_probabilities)]

    row_sums: list[int] = []
    l1_by_row: list[float] = []
    top1_matches = 0
    nonzero_rows = 0
    for row in range(seq):
        begin = row * CONTEXT
        end = begin + CONTEXT
        valid = [depth for depth in range(CONTEXT) if mask[begin + depth] == MASK_VALID]
        row_sum = sum(observed[begin:end])
        row_sums.append(row_sum)
        nonzero_rows += row_sum > 0
        l1_by_row.append(sum(absolute_errors[begin:end]))
        device_top1 = max(valid, key=lambda depth: observed[begin + depth])
        score_top1 = max(valid, key=lambda depth: scores[begin + depth])
        top1_matches += device_top1 == score_top1

    exact_integer_match = mismatches == 0
    top1_match_rate = top1_matches / seq
    implementation_valid = exact_integer_match and masked_nonzero == 0 and nonzero_rows == seq and top1_matches == seq
    return {
        "seq": seq,
        "device_output": str(device_output_path),
        "device_vs_integer_contract": {
            "byte_exact": exact_integer_match,
            "mismatch_count": mismatches,
            "max_u8_delta": max(deltas, default=0),
        },
        "mathematical_invariants": {
            "masked_nonzero_count": masked_nonzero,
            "nonzero_probability_rows": nonzero_rows,
            "rows": seq,
            "top1_matches_score_max": top1_matches,
            "top1_match_rate": top1_match_rate,
            "quantized_row_sum_min": min(row_sums),
            "quantized_row_sum_max": max(row_sums),
            "quantized_row_sum_mean": fmean(row_sums),
        },
        "device_u8_vs_exact_float_softmax": {
            "mae_all_elements": fmean(absolute_errors),
            "max_abs_error": max(absolute_errors),
            "mean_row_l1": fmean(l1_by_row),
            "max_row_l1": max(l1_by_row),
        },
        "accuracy_gate_applied": False,
        "implementation_valid": implementation_valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "analyze"))
    parser.add_argument("--seq", type=int, choices=(1, 32), required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--device-output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.mode == "generate":
        report = generate(args.seq, args.fixture_dir)
    else:
        if args.device_output is None:
            parser.error("--device-output is required in analyze mode")
        report = analyze(args.seq, args.fixture_dir, args.device_output)

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.get("implementation_valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
