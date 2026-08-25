#!/usr/bin/env python3
"""Apply teacher-guided clipping to the collapsed residual QDQ chain."""

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
    generation_report,
    read_records,
    row_metrics,
    set_range,
)
from pymllm.mobile.backends.qualcomm.transformers.core.qdq import ActivationQDQ
from pymllm.mobile.backends.qualcomm.transformers.qwen3.runner import Qwen3Quantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--teacher-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-index", type=int, default=0)
    parser.add_argument("--first-layer", type=int, default=3)
    parser.add_argument("--last-layer", type=int, default=27)
    parser.add_argument("--lower-quantile", type=float, default=0.0001)
    parser.add_argument("--upper-quantile", type=float, default=0.9999)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def clipped_range(
    tensor: torch.Tensor, lower_quantile: float, upper_quantile: float
) -> tuple[float, float]:
    values = tensor.detach().float().reshape(-1)
    bounds = torch.quantile(
        values, torch.tensor([lower_quantile, upper_quantile])
    )
    return float(bounds[0].item()), float(bounds[1].item())


def activation_module(modules: dict[str, torch.nn.Module], name: str) -> ActivationQDQ:
    module = modules.get(name)
    if not isinstance(module, ActivationQDQ):
        raise TypeError(f"expected ActivationQDQ at {name}")
    return module


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    if not 0 <= args.lower_quantile < args.upper_quantile <= 1:
        raise ValueError("invalid quantile interval")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    teacher = torch.load(args.teacher_probe, map_location="cpu", weights_only=True)
    records = read_records(args.corpus)
    record = records[args.eval_index]
    if not torch.equal(
        teacher["input_ids"], torch.tensor(record["input_ids"], dtype=torch.int32)
    ):
        raise ValueError("teacher probe does not match the selected evaluation input")

    start = time.perf_counter()
    quantizer = Qwen3Quantizer(
        str(args.model_path),
        mllm_qualcomm_max_length=args.max_length,
        activation_bits=8,
        linear_block_size=32,
    )
    quantizer.model.eval()
    quantizer.enable_fake_quant()
    apply_activation_qparams(quantizer.model, baseline_report)
    modules = dict(quantizer.model.named_modules())

    applied = {}
    for layer_index in range(args.first_layer, args.last_layer + 1):
        teacher_input = teacher["layer_outputs"][str(layer_index - 1)]
        minimum, maximum = clipped_range(
            teacher_input, args.lower_quantile, args.upper_quantile
        )
        names = (
            f"model.layers.{layer_index}.input_layernorm_input_qdq",
            f"model.layers.{layer_index}.add_0_output_qdq",
        )
        for name in names:
            applied[name] = set_range(
                activation_module(modules, name), minimum, maximum
            )

    final_teacher_input = teacher["layer_outputs"][str(args.last_layer)]
    minimum, maximum = clipped_range(
        final_teacher_input, args.lower_quantile, args.upper_quantile
    )
    final_name = "model.norm_input_qdq"
    applied[final_name] = set_range(
        activation_module(modules, final_name), minimum, maximum
    )

    probe = capture_full_probe(quantizer, record)
    torch.save(probe, args.output_dir / "probe.pt")
    generation = generation_report(quantizer)
    layer_metrics = {
        layer: row_metrics(teacher["layer_outputs"][layer], candidate)
        for layer, candidate in probe["layer_outputs"].items()
    }
    logit_metrics = row_metrics(teacher["last_logits"], probe["last_logits"])
    logit_metrics["teacher_argmax"] = int(teacher["last_logits"].argmax().item())
    logit_metrics["candidate_argmax"] = int(probe["last_logits"].argmax().item())
    logit_metrics["top1_match"] = (
        logit_metrics["teacher_argmax"] == logit_metrics["candidate_argmax"]
    )

    result = {
        "experiment": "EXP-0018",
        "variant": "teacher_guided_residual_clip",
        "eval_index": args.eval_index,
        "first_layer": args.first_layer,
        "last_layer": args.last_layer,
        "lower_quantile": args.lower_quantile,
        "upper_quantile": args.upper_quantile,
        "applied_qparams": applied,
        "layer_metrics": layer_metrics,
        "last_logits": logit_metrics,
        "generation": generation,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(generation, ensure_ascii=False, sort_keys=True))
    print(
        f"last_logits cosine={logit_metrics['global_cosine']:.6f} "
        f"top1_match={logit_metrics['top1_match']}"
    )
    print(f"Wrote teacher-guided residual clip to {args.output_dir}")


if __name__ == "__main__":
    main()
