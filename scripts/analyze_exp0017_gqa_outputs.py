#!/usr/bin/env python3
"""Compare EXP-0017 native GQA with A8 and precision-usable A16 references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


A8_SCALE = 0.5333649516105652
A8_ZERO = 229
A16_SCALE = 0.0021023168228566647
A16_ZERO = 58560
ATTENTION_ELEMENTS = 32 * 16 * 128
CACHE_ELEMENTS = 8 * 32 * 128


def load_exact(path: Path, dtype: np.dtype, elements: int) -> np.ndarray:
    data = np.fromfile(path, dtype=dtype)
    if data.size != elements:
        raise ValueError(f"{path}: expected {elements} elements, got {data.size}")
    return data


def error_stats(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float | int]:
    error = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
    return {
        "max": float(error.max(initial=0.0)),
        "mean": float(error.mean()),
        "p99": float(np.percentile(error, 99)),
        "nonzero": int(np.count_nonzero(error)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--a8-reference", type=Path, required=True)
    parser.add_argument("--a16-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidate = load_exact(Path(str(args.candidate) + ".attention.u8"), np.uint8, ATTENTION_ELEMENTS)
    a8 = load_exact(Path(str(args.a8_reference) + ".attention.u8"), np.uint8, ATTENTION_ELEMENTS)
    a16 = load_exact(Path(str(args.a16_reference) + ".attention.u16"), np.uint16, ATTENTION_ELEMENTS)
    candidate_real = (candidate.astype(np.float64) - A8_ZERO) * A8_SCALE
    a8_real = (a8.astype(np.float64) - A8_ZERO) * A8_SCALE
    a16_real = (a16.astype(np.float64) - A16_ZERO) * A16_SCALE

    candidate_key = load_exact(Path(str(args.candidate) + ".key.u8"), np.uint8, CACHE_ELEMENTS)
    a8_key = load_exact(Path(str(args.a8_reference) + ".key.u8"), np.uint8, CACHE_ELEMENTS)
    candidate_value = load_exact(Path(str(args.candidate) + ".value.u8"), np.uint8, CACHE_ELEMENTS)
    a8_value = load_exact(Path(str(args.a8_reference) + ".value.u8"), np.uint8, CACHE_ELEMENTS)

    centered_candidate = candidate_real - candidate_real.mean()
    centered_a16 = a16_real - a16_real.mean()
    denominator = float(np.linalg.norm(centered_candidate) * np.linalg.norm(centered_a16))
    correlation = float(np.dot(centered_candidate, centered_a16) / denominator) if denominator else 0.0
    report = {
        "experiment": "EXP-0017",
        "candidate_vs_a8_lsb": error_stats(candidate, a8),
        "candidate_vs_a8_real": error_stats(candidate_real, a8_real),
        "candidate_vs_a16_real": error_stats(candidate_real, a16_real),
        "a8_vs_a16_real": error_stats(a8_real, a16_real),
        "candidate_a16_centered_correlation": correlation,
        "candidate_output_unique_values": int(np.unique(candidate).size),
        "candidate_output_saturated": int(np.count_nonzero((candidate == 0) | (candidate == 255))),
        "new_key_candidate_vs_a8_mismatches": int(np.count_nonzero(candidate_key != a8_key)),
        "new_value_candidate_vs_a8_mismatches": int(np.count_nonzero(candidate_value != a8_value)),
        "cache_semantics_valid": bool(np.array_equal(candidate_key, a8_key)
                                      and np.array_equal(candidate_value, a8_value)),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
