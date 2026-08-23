#!/usr/bin/env python3

"""Generate and validate deterministic U8 masked-E2Softmax device fixtures."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from statistics import fmean


MAXIMUM_CONTEXT = 1024
SCORE_SCALE = 0.17946475744247437
SCORE_ZERO_POINT = 113
OUTPUT_LEVELS = 255
MASK_VALID = 255
MASK_INVALID = 0
EXPONENT_NUMERATOR = (16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0)
RECIPROCAL_FRACTION_BITS = 23


def fixture(
    seq: int, context: int, score_zero_point: int, start_position: int
) -> tuple[bytearray, bytearray, list[int]]:
    scores = bytearray(seq * context)
    mask = bytearray(seq * context)
    positions = [start_position + row for row in range(seq)]
    for row in range(seq):
        valid_count = min(context, positions[row] + 1)
        for depth in range(context):
            raw = score_zero_point + ((depth * 17 + row * 29 + (depth * row % 31) * 3) % 96) - 48
            scores[row * context + depth] = max(0, min(255, raw))
            mask[row * context + depth] = MASK_VALID if depth < valid_count else MASK_INVALID
        peak = (37 + row * 67) % valid_count
        shoulder = (peak + 257) % valid_count
        scores[row * context + peak] = 228 - row % 7
        scores[row * context + shoulder] = 211 - row % 5
    return scores, mask, positions


def integer_reference(
    scores: bytes,
    mask: bytes,
    seq: int,
    context: int,
    score_scale: float = SCORE_SCALE,
    beta: float = 1.0,
) -> bytearray:
    coefficient = max(1, min(255, int(score_scale * beta * 1.4375 * 256.0 + 0.5)))
    output = bytearray(seq * context)
    for row in range(seq):
        begin = row * context
        valid = [depth for depth in range(context) if mask[begin + depth] == MASK_VALID]
        if not valid:
            continue
        row_max = max(scores[begin + depth] for depth in valid)
        codes = [15] * context
        denominator = 0
        for depth in valid:
            code = min(14, ((row_max - scores[begin + depth]) * coefficient + 128) >> 8)
            codes[depth] = code
            denominator += EXPONENT_NUMERATOR[code]
        reciprocal_q23 = ((OUTPUT_LEVELS << RECIPROCAL_FRACTION_BITS) + denominator // 2) // denominator
        for depth in valid:
            probability = (
                EXPONENT_NUMERATOR[codes[depth]] * reciprocal_q23
                + (1 << (RECIPROCAL_FRACTION_BITS - 1))
            ) >> RECIPROCAL_FRACTION_BITS
            output[begin + depth] = max(0, min(255, probability))
    return output


def float_reference(
    scores: bytes,
    mask: bytes,
    seq: int,
    context: int,
    score_scale: float = SCORE_SCALE,
    beta: float = 1.0,
) -> list[float]:
    output = [0.0] * (seq * context)
    for row in range(seq):
        begin = row * context
        valid = [depth for depth in range(context) if mask[begin + depth] == MASK_VALID]
        if not valid:
            continue
        row_max = max(scores[begin + depth] for depth in valid)
        exponentials = [math.exp((scores[begin + depth] - row_max) * score_scale * beta) for depth in valid]
        denominator = sum(exponentials)
        for depth, value in zip(valid, exponentials):
            output[begin + depth] = value / denominator
    return output


def generate(
    seq: int,
    context: int,
    heads: int,
    output_dir: Path,
    score_scale: float,
    score_zero_point: int,
    beta: float,
    start_position: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    head_scores, head_mask, positions = fixture(seq, context, score_zero_point, start_position)
    head_expected = integer_reference(
        head_scores, head_mask, seq, context, score_scale=score_scale, beta=beta
    )
    scores = head_scores * heads
    mask = head_mask * heads
    expected = head_expected * heads
    paths = {
        "scores": output_dir / f"scores_s{seq}.bin",
        "mask": output_dir / f"mask_s{seq}.bin",
        "positions": output_dir / f"positions_s{seq}.bin",
        "integer_reference": output_dir / f"integer_reference_s{seq}.bin",
    }
    paths["scores"].write_bytes(scores)
    paths["mask"].write_bytes(mask)
    paths["positions"].write_bytes(struct.pack(f"<{seq}i", *positions))
    paths["integer_reference"].write_bytes(expected)
    return {
        "seq": seq,
        "context": context,
        "heads": heads,
        "start_position": start_position,
        "valid_lengths": [position + 1 for position in positions],
        "bytes_per_tensor": len(scores),
        "score_scale": score_scale,
        "score_zero_point": score_zero_point,
        "beta": beta,
        "output_scale": 1.0 / OUTPUT_LEVELS,
        "output_zero_point": 0,
        "mask_valid_raw": MASK_VALID,
        "mask_invalid_raw": MASK_INVALID,
        "e2_exponent_coefficient_q8": max(
            1, min(255, int(score_scale * beta * 1.4375 * 256.0 + 0.5))
        ),
        "paths": {name: str(path) for name, path in paths.items()},
    }


def analyze(
    seq: int,
    context: int,
    heads: int,
    fixture_dir: Path,
    device_output_path: Path,
    score_scale: float,
    beta: float,
) -> dict[str, object]:
    scores = (fixture_dir / f"scores_s{seq}.bin").read_bytes()
    mask = (fixture_dir / f"mask_s{seq}.bin").read_bytes()
    expected = (fixture_dir / f"integer_reference_s{seq}.bin").read_bytes()
    observed = device_output_path.read_bytes()
    expected_bytes = heads * seq * context
    for label, payload in (("scores", scores), ("mask", mask), ("expected", expected), ("observed", observed)):
        if len(payload) != expected_bytes:
            raise ValueError(f"{label}: expected {expected_bytes} bytes, got {len(payload)}")

    deltas = [abs(int(actual) - int(want)) for actual, want in zip(observed, expected)]
    mismatches = sum(delta != 0 for delta in deltas)
    masked_nonzero = sum(actual != 0 for actual, raw_mask in zip(observed, mask) if raw_mask != MASK_VALID)
    elements_per_head = seq * context
    exact_one_head = float_reference(
        scores[:elements_per_head],
        mask[:elements_per_head],
        seq,
        context,
        score_scale=score_scale,
        beta=beta,
    )
    exact = exact_one_head * heads
    absolute_errors = [abs(actual / OUTPUT_LEVELS - reference) for actual, reference in zip(observed, exact)]
    row_sums: list[int] = []
    l1_by_row: list[float] = []
    top1_matches = 0
    nonzero_rows = 0
    rows = heads * seq
    for row in range(rows):
        begin = row * context
        end = begin + context
        valid = [depth for depth in range(context) if mask[begin + depth] == MASK_VALID]
        row_sum = sum(observed[begin:end])
        row_sums.append(row_sum)
        nonzero_rows += row_sum > 0
        l1_by_row.append(sum(absolute_errors[begin:end]))
        top1_matches += max(valid, key=lambda depth: observed[begin + depth]) == max(
            valid, key=lambda depth: scores[begin + depth]
        )

    implementation_valid = (
        mismatches == 0 and masked_nonzero == 0 and nonzero_rows == rows and top1_matches == rows
    )
    return {
        "seq": seq,
        "context": context,
        "heads": heads,
        "score_scale": score_scale,
        "beta": beta,
        "device_output": str(device_output_path),
        "device_vs_integer_contract": {
            "byte_exact": mismatches == 0,
            "mismatch_count": mismatches,
            "max_u8_delta": max(deltas, default=0),
        },
        "mathematical_invariants": {
            "masked_nonzero_count": masked_nonzero,
            "nonzero_probability_rows": nonzero_rows,
            "rows": rows,
            "top1_matches_score_max": top1_matches,
            "top1_match_rate": top1_matches / rows,
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
    parser.add_argument("--context", type=int, default=MAXIMUM_CONTEXT)
    parser.add_argument("--heads", type=int, choices=(1, 2, 4, 16), default=1)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--device-output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--score-scale", type=float, default=SCORE_SCALE)
    parser.add_argument("--score-zero-point", type=int, default=SCORE_ZERO_POINT)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--start-position", type=int)
    args = parser.parse_args()
    if not 0.0 < args.beta <= 1.0:
        parser.error("--beta must be in (0, 1]")
    if args.score_scale <= 0.0:
        parser.error("--score-scale must be positive")
    if not 0 <= args.score_zero_point <= 255:
        parser.error("--score-zero-point must be in [0, 255]")
    if args.context < 32 or args.context > MAXIMUM_CONTEXT or args.context % 32:
        parser.error("--context must be a multiple of 32 in [32, 1024]")
    start_position = args.start_position
    if start_position is None:
        start_position = 62 if args.seq == 1 else 0
    if start_position < 0 or start_position + args.seq > args.context:
        parser.error("--start-position must describe a causal prefix within the selected context")
    if args.mode == "generate":
        report = generate(
            args.seq,
            args.context,
            args.heads,
            args.fixture_dir,
            args.score_scale,
            args.score_zero_point,
            args.beta,
            start_position,
        )
    else:
        if args.device_output is None:
            parser.error("--device-output is required in analyze mode")
        report = analyze(
            args.seq,
            args.context,
            args.heads,
            args.fixture_dir,
            args.device_output,
            args.score_scale,
            args.beta,
        )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.get("implementation_valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
