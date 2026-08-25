#!/usr/bin/env python3
"""Compare ordered W4A16 and W4A8 internal ActivationQDQ outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from exp0018_progressive_range_search import row_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tensor_summary(tensor: torch.Tensor) -> dict[str, float | int | list[int]]:
    value = tensor.detach().float()
    return {
        "shape": list(value.shape),
        "rms": float(torch.sqrt(torch.mean(value.square())).item()),
        "minimum": float(value.amin().item()),
        "maximum": float(value.amax().item()),
        "zero_fraction": float((value == 0).float().mean().item()),
        "unique_count": int(torch.unique(value).numel()),
    }


def main() -> None:
    args = parse_args()
    teacher = torch.load(args.teacher, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    if not torch.equal(teacher["input_ids"], candidate["input_ids"]):
        raise ValueError("internal probes use different inputs")
    if teacher["execution_order"] != candidate["execution_order"]:
        raise ValueError("internal QDQ execution order differs")

    comparisons = []
    for key in teacher["execution_order"]:
        name, call_text = key.rsplit("#", 1)
        call = int(call_text)
        reference = teacher["outputs"][name][call]
        value = candidate["outputs"][name][call]
        if reference.shape != value.shape:
            raise ValueError(f"shape mismatch at {key}: {reference.shape} != {value.shape}")
        comparisons.append(
            {
                "key": key,
                "metrics": row_metrics(reference, value),
                "teacher": tensor_summary(reference),
                "candidate": tensor_summary(value),
            }
        )

    result = {
        "experiment": "EXP-0018",
        "teacher": str(args.teacher),
        "candidate": str(args.candidate),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for item in comparisons:
        metrics = item["metrics"]
        print(
            f"{item['key']}: row_cos={metrics['row_cosine_median']:.6f} "
            f"row_nrmse={metrics['row_nrmse_median']:.6f} "
            f"zero={item['candidate']['zero_fraction']:.4f}"
        )


if __name__ == "__main__":
    main()
