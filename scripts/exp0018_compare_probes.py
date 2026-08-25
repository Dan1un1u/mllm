#!/usr/bin/env python3
"""Compare EXP-0018 software probes against one W4A16 teacher trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=PROBE_PT",
        help="Candidate label and probe path; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_candidate(specification: str) -> tuple[str, Path]:
    label, separator, path = specification.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"invalid candidate specification: {specification}")
    return label, Path(path)


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_flat = reference.detach().float().reshape(-1)
    candidate_flat = candidate.detach().float().reshape(-1)
    if reference_flat.shape != candidate_flat.shape:
        raise ValueError(
            f"shape mismatch: {reference_flat.shape} != {candidate_flat.shape}"
        )
    error = candidate_flat - reference_flat
    mse = float(torch.mean(error.square()).item())
    reference_rms = float(torch.sqrt(torch.mean(reference_flat.square())).item())
    candidate_rms = float(torch.sqrt(torch.mean(candidate_flat.square())).item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            reference_flat, candidate_flat, dim=0
        ).item()
    )
    metrics = {
        "cosine": cosine,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "nrmse": math.sqrt(mse) / max(reference_rms, 1e-12),
        "max_abs_error": float(error.abs().amax().item()),
        "reference_rms": reference_rms,
        "candidate_rms": candidate_rms,
    }
    if reference.ndim >= 2 and reference.shape[-1] > 1:
        reference_rows = reference.detach().float().reshape(-1, reference.shape[-1])
        candidate_rows = candidate.detach().float().reshape(-1, candidate.shape[-1])
        row_cosine = torch.nn.functional.cosine_similarity(
            reference_rows, candidate_rows, dim=-1
        )
        row_error_rms = torch.sqrt(
            torch.mean((candidate_rows - reference_rows).square(), dim=-1)
        )
        row_reference_rms = torch.sqrt(torch.mean(reference_rows.square(), dim=-1))
        row_nrmse = row_error_rms / torch.clamp(row_reference_rms, min=1e-12)
        metrics["row_cosine_mean"] = float(row_cosine.mean().item())
        metrics["row_cosine_median"] = float(row_cosine.median().item())
        metrics["row_cosine_min"] = float(row_cosine.amin().item())
        metrics["row_cosine_below_0_9_fraction"] = float(
            torch.mean((row_cosine < 0.9).float()).item()
        )
        metrics["row_nrmse_mean"] = float(row_nrmse.mean().item())
        metrics["row_nrmse_median"] = float(row_nrmse.median().item())
        metrics["row_nrmse_p95"] = float(
            torch.quantile(row_nrmse, 0.95).item()
        )
    return metrics


def compare(reference: dict, candidate: dict) -> dict:
    if not torch.equal(reference["input_ids"], candidate["input_ids"]):
        raise ValueError("probe inputs are not identical")
    reference_layers = reference["layer_outputs"]
    candidate_layers = candidate["layer_outputs"]
    if reference_layers.keys() != candidate_layers.keys():
        raise ValueError("probe layer sets are not identical")

    layers = {
        layer: tensor_metrics(reference_layers[layer], candidate_layers[layer])
        for layer in reference_layers
    }
    first_cosine_below_09 = next(
        (int(layer) for layer, metrics in layers.items() if metrics["cosine"] < 0.9),
        None,
    )
    first_nrmse_above_05 = next(
        (int(layer) for layer, metrics in layers.items() if metrics["nrmse"] > 0.5),
        None,
    )
    logit_metrics = tensor_metrics(reference["last_logits"], candidate["last_logits"])
    logit_metrics["reference_argmax"] = int(reference["last_logits"].argmax().item())
    logit_metrics["candidate_argmax"] = int(candidate["last_logits"].argmax().item())
    logit_metrics["top1_match"] = (
        logit_metrics["reference_argmax"] == logit_metrics["candidate_argmax"]
    )
    return {
        "layers": layers,
        "first_cosine_below_0_9": first_cosine_below_09,
        "first_nrmse_above_0_5": first_nrmse_above_05,
        "last_logits": logit_metrics,
    }


def main() -> None:
    args = parse_args()
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    comparisons = {}
    for specification in args.candidate:
        label, path = parse_candidate(specification)
        if label in comparisons:
            raise ValueError(f"duplicate candidate label: {label}")
        candidate = torch.load(path, map_location="cpu", weights_only=True)
        comparisons[label] = compare(reference, candidate)

    result = {
        "reference": str(args.reference),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for label, comparison in comparisons.items():
        print(
            f"{label}: first cosine<0.9="
            f"{comparison['first_cosine_below_0_9']}, first nrmse>0.5="
            f"{comparison['first_nrmse_above_0_5']}, logit cosine="
            f"{comparison['last_logits']['cosine']:.6f}, top1_match="
            f"{comparison['last_logits']['top1_match']}"
        )
        for layer, metrics in comparison["layers"].items():
            print(
                f"  layer={int(layer):02d} cosine={metrics['cosine']:.6f} "
                f"nrmse={metrics['nrmse']:.6f} "
                f"row_cos_median={metrics['row_cosine_median']:.6f} "
                f"row_nrmse_median={metrics['row_nrmse_median']:.6f} "
                f"reference_rms={metrics['reference_rms']:.6f} "
                f"candidate_rms={metrics['candidate_rms']:.6f}"
            )


if __name__ == "__main__":
    main()
