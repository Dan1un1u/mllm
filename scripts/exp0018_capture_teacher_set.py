#!/usr/bin/env python3
"""Capture W4A16 block outputs for held-out and calibration records."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from exp0018_progressive_range_search import (
    apply_activation_qparams,
    capture_full_probe,
    read_records,
)
from pymllm.mobile.backends.qualcomm.transformers.qwen3.runner import Qwen3Quantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--teacher-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="0,1,2,3,4")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    indices = [int(value) for value in args.indices.split(",")]
    if len(indices) != len(set(indices)):
        raise ValueError("teacher indices must be unique")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    report = json.loads(args.teacher_report.read_text(encoding="utf-8"))
    if report.get("activation_bits") != 16:
        raise ValueError("teacher report must describe the W4A16 teacher")
    records = read_records(args.corpus)
    if min(indices) < 0 or max(indices) >= len(records):
        raise IndexError("teacher index is outside the corpus")

    start = time.perf_counter()
    quantizer = Qwen3Quantizer(
        str(args.model_path),
        mllm_qualcomm_max_length=args.max_length,
        activation_bits=16,
        linear_block_size=32,
    )
    quantizer.model.eval()
    quantizer.enable_fake_quant()
    apply_activation_qparams(quantizer.model, report)

    probes = []
    with torch.no_grad():
        for position, index in enumerate(indices, start=1):
            probe = capture_full_probe(quantizer, records[index])
            probes.append(probe)
            print(f"Captured W4A16 teacher {position}/{len(indices)}: corpus index {index}")

    payload = {
        "experiment": "EXP-0018",
        "activation_bits": 16,
        "indices": indices,
        # Records intentionally keep their native sequence lengths.  Padding is
        # not neutral in the current Qualcomm model because it constructs its
        # own causal mask from sequence length.
        "probes": {str(index): probe for index, probe in zip(indices, probes)},
        "elapsed_seconds": time.perf_counter() - start,
    }
    torch.save(payload, args.output)
    print(f"Wrote W4A16 teacher set to {args.output}")


if __name__ == "__main__":
    main()
