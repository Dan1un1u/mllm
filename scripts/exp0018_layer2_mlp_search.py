#!/usr/bin/env python3
"""Search the layer-2 MLP outlier path against W4A16 block output."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from exp0018_calibration_probe import (
    activation_qparam_report,
    runtime_contract_report,
)
from exp0018_progressive_block_calibration import (
    QUANTILE_PAIRS,
    activation_module,
    range_from_quantiles,
    record_inputs,
    run_to_layer,
    teacher_rows,
)
from exp0018_progressive_range_search import (
    apply_activation_qparams,
    capture_full_probe,
    generation_report,
    read_records,
    row_metrics,
    score,
    set_range,
)
from pymllm.mobile.backends.qualcomm.transformers.qwen3.runner import Qwen3Quantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--starting-report", type=Path, required=True)
    parser.add_argument("--teacher-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-indices", default="1,2,3,4")
    parser.add_argument("--eval-index", type=int, default=0)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def snapshot(module) -> dict[str, float | int]:
    observer = module.fake_quant.activation_post_process
    return {
        "min": float(observer.min_val.item()),
        "max": float(observer.max_val.item()),
        "scale": float(module.fake_quant.scale.item()),
        "zero_point": int(module.fake_quant.zero_point.item()),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    calibration_indices = [
        int(value) for value in args.calibration_indices.split(",")
    ]
    if args.eval_index in calibration_indices:
        raise ValueError("evaluation record must be held out")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    starting_report = json.loads(args.starting_report.read_text(encoding="utf-8"))
    teacher = torch.load(args.teacher_set, map_location="cpu", weights_only=True)
    records = read_records(args.corpus)
    start = time.perf_counter()

    quantizer = Qwen3Quantizer(
        str(args.model_path),
        mllm_qualcomm_max_length=args.max_length,
        activation_bits=8,
        linear_block_size=32,
    )
    quantizer.model.eval()
    quantizer.enable_fake_quant()
    apply_activation_qparams(quantizer.model, starting_report)
    modules = dict(quantizer.model.named_modules())
    inputs = record_inputs(records, calibration_indices, quantizer.model.device)
    teacher_output = teacher_rows(teacher, calibration_indices, str(args.layer))

    down_name = f"model.layers.{args.layer}.mlp.down_proj_input_qdq"
    add_name = f"model.layers.{args.layer}.add_1_lhs_input_qdq"
    down = activation_module(modules, down_name)
    add = activation_module(modules, add_name)
    baseline_ranges = {down_name: snapshot(down), add_name: snapshot(add)}
    baseline_output, raw = run_to_layer(
        quantizer,
        inputs,
        args.layer,
        raw_targets=[(down_name, down), (add_name, add)],
    )
    baseline_metrics = row_metrics(teacher_output, baseline_output)
    candidates: list[dict[str, Any]] = [
        {
            "kind": "current",
            "ranges": baseline_ranges,
            "metrics": baseline_metrics,
            "score": score(baseline_metrics),
        }
    ]

    for lower, upper in QUANTILE_PAIRS:
        try:
            down_range, down_saturation = range_from_quantiles(
                down, raw[down_name], lower, upper
            )
        except ValueError:
            continue
        _, candidate_raw = run_to_layer(
            quantizer,
            inputs,
            args.layer,
            raw_targets=[(add_name, add)],
        )
        add_values = candidate_raw[add_name]
        if add_values.numel() == 0 or float(add_values.amin()) == float(
            add_values.amax()
        ):
            continue
        try:
            add_range, add_saturation = range_from_quantiles(
                add, add_values, lower, upper
            )
        except ValueError:
            continue
        output, _ = run_to_layer(quantizer, inputs, args.layer)
        metrics = row_metrics(teacher_output, output)
        candidates.append(
            {
                "kind": "percentile",
                "lower_quantile": lower,
                "upper_quantile": upper,
                "ranges": {down_name: down_range, add_name: add_range},
                "saturation_fraction": {
                    down_name: down_saturation,
                    add_name: add_saturation,
                },
                "metrics": metrics,
                "score": score(metrics),
            }
        )

    best = max(candidates, key=lambda item: item["score"])
    for name, module in ((down_name, down), (add_name, add)):
        selected = best["ranges"][name]
        set_range(module, selected["min"], selected["max"])
    print(
        f"Layer-{args.layer} MLP: {candidates[0]['score']:.6f} -> "
        f"{best['score']:.6f} ({best['kind']} "
        f"q={best.get('lower_quantile')},{best.get('upper_quantile')})"
    )

    probe = capture_full_probe(quantizer, records[args.eval_index])
    torch.save(probe, args.output_dir / "probe.pt")
    heldout_teacher = teacher_rows(teacher, [args.eval_index], str(args.layer))
    heldout_layer = row_metrics(
        heldout_teacher, probe["layer_outputs"][str(args.layer)]
    )
    heldout_logits = row_metrics(
        teacher_rows(teacher, [args.eval_index], "last_logits"),
        probe["last_logits"],
    )
    generation = generation_report(quantizer)
    result = {
        "experiment": "EXP-0018",
        "variant": "layer2_mlp_outlier_search",
        "activation_bits": 8,
        "starting_report": str(args.starting_report),
        "calibration_indices": calibration_indices,
        "eval_index": args.eval_index,
        "layer": args.layer,
        "baseline": candidates[0],
        "best": best,
        "improvement": best["score"] - candidates[0]["score"],
        "heldout_layer": heldout_layer,
        "heldout_logits": heldout_logits,
        "generation": generation,
        "candidate_count": len(candidates),
        "elapsed_seconds": time.perf_counter() - start,
        "runtime_contract": runtime_contract_report(quantizer.model),
        "activation_qparams": activation_qparam_report(quantizer.model),
        "candidates": candidates,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(generation, ensure_ascii=False, sort_keys=True))
    print(
        f"Held-out layer-{args.layer} row cosine median="
        f"{heldout_layer['row_cosine_median']:.6f}; "
        f"logit cosine={heldout_logits['global_cosine']:.6f}"
    )
    print(f"Wrote layer-2 MLP search to {args.output_dir}")


if __name__ == "__main__":
    main()
