#!/usr/bin/env python3
"""Search one asymmetric-U8 range using a W4A16 block-output teacher."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from pymllm.mobile.backends.qualcomm.transformers.core.qdq import ActivationQDQ
from pymllm.mobile.backends.qualcomm.transformers.qwen3.runner import Qwen3Quantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--teacher-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target",
        default="model.layers.3.input_layernorm_input_qdq",
        help="ActivationQDQ whose range is searched",
    )
    parser.add_argument(
        "--linked-target",
        action="append",
        default=[],
        help=(
            "Additional ActivationQDQ in the same residual domain that receives "
            "the searched range; may be repeated"
        ),
    )
    parser.add_argument("--target-layer", type=int, default=3)
    parser.add_argument("--collection-start", type=int, default=1)
    parser.add_argument("--collection-count", type=int, default=16)
    parser.add_argument("--eval-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def scalar_tensor(value: float | int, *, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device).reshape_as(like)


def apply_activation_qparams(model: torch.nn.Module, report: dict[str, Any]) -> None:
    qparams = report["activation_qparams"]
    missing = []
    for name, module in model.named_modules():
        if not isinstance(module, ActivationQDQ):
            continue
        if name not in qparams:
            missing.append(name)
            continue
        values = qparams[name]
        observer = module.fake_quant.activation_post_process
        observer.min_val.copy_(scalar_tensor(values["min"], like=observer.min_val))
        observer.max_val.copy_(scalar_tensor(values["max"], like=observer.max_val))
        module.fake_quant.scale.copy_(
            scalar_tensor(values["scale"], like=module.fake_quant.scale)
        )
        module.fake_quant.zero_point.copy_(
            scalar_tensor(values["zero_point"], like=module.fake_quant.zero_point)
        )
        module.disable_observer()
        module.enable_fakequant()
    if missing:
        raise RuntimeError(f"baseline report is missing {len(missing)} ActivationQDQ entries")


def set_range(module: ActivationQDQ, minimum: float, maximum: float) -> dict[str, float | int]:
    minimum = min(float(minimum), 0.0)
    maximum = max(float(maximum), 0.0)
    if not minimum < maximum:
        raise ValueError(f"invalid range: [{minimum}, {maximum}]")
    observer = module.fake_quant.activation_post_process
    scale = max((maximum - minimum) / 255.0, float(observer.eps.item()))
    zero_point = int(round(-minimum / scale))
    zero_point = max(0, min(255, zero_point))
    observer.min_val.copy_(scalar_tensor(minimum, like=observer.min_val))
    observer.max_val.copy_(scalar_tensor(maximum, like=observer.max_val))
    module.fake_quant.scale.copy_(scalar_tensor(scale, like=module.fake_quant.scale))
    module.fake_quant.zero_point.copy_(
        scalar_tensor(zero_point, like=module.fake_quant.zero_point)
    )
    return {
        "min": minimum,
        "max": maximum,
        "scale": scale,
        "zero_point": zero_point,
        "effective_min": -zero_point * scale,
        "effective_max": (255 - zero_point) * scale,
    }


def read_records(corpus: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]


def model_inputs(record: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([record["input_ids"]], dtype=torch.long, device=device),
        "attention_mask": torch.tensor(
            [record["attention_mask"]], dtype=torch.long, device=device
        ),
    }


def row_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_rows = reference.detach().float().reshape(-1, reference.shape[-1])
    candidate_rows = candidate.detach().float().reshape(-1, candidate.shape[-1])
    cosine = torch.nn.functional.cosine_similarity(
        reference_rows, candidate_rows, dim=-1
    )
    error_rms = torch.sqrt(torch.mean((candidate_rows - reference_rows).square(), dim=-1))
    reference_rms = torch.sqrt(torch.mean(reference_rows.square(), dim=-1))
    nrmse = error_rms / torch.clamp(reference_rms, min=1e-12)
    global_error = candidate_rows - reference_rows
    global_nrmse = torch.sqrt(torch.mean(global_error.square())) / torch.clamp(
        torch.sqrt(torch.mean(reference_rows.square())), min=1e-12
    )
    return {
        "global_cosine": float(
            torch.nn.functional.cosine_similarity(
                reference_rows.reshape(-1), candidate_rows.reshape(-1), dim=0
            ).item()
        ),
        "global_nrmse": float(global_nrmse.item()),
        "row_cosine_mean": float(cosine.mean().item()),
        "row_cosine_median": float(cosine.median().item()),
        "row_cosine_p05": float(torch.quantile(cosine, 0.05).item()),
        "row_cosine_below_0_9_fraction": float((cosine < 0.9).float().mean().item()),
        "row_nrmse_mean": float(nrmse.mean().item()),
        "row_nrmse_median": float(nrmse.median().item()),
        "row_nrmse_p95": float(torch.quantile(nrmse, 0.95).item()),
    }


def score(metrics: dict[str, float]) -> float:
    return (
        metrics["row_cosine_median"]
        + 0.25 * metrics["row_cosine_mean"]
        + 0.10 * metrics["row_cosine_p05"]
        - 0.25 * metrics["row_nrmse_median"]
        - 0.10 * metrics["row_nrmse_p95"]
    )


def evaluate_layer(
    quantizer: Qwen3Quantizer,
    record: dict[str, Any],
    layer_index: int,
    teacher: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:
    captured: list[torch.Tensor] = []
    handle = quantizer.model.model.layers[layer_index].register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach().float().cpu())
    )
    with torch.no_grad():
        output = quantizer.model(
            **model_inputs(record, quantizer.model.device),
            use_cache=False,
            logits_to_keep=1,
        )
    handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one layer capture, got {len(captured)}")
    metrics = row_metrics(teacher, captured[0])
    metrics["last_logit_rms"] = float(
        torch.sqrt(torch.mean(output.logits.detach().float().square())).item()
    )
    return metrics, captured[0]


def collect_target_values(
    quantizer: Qwen3Quantizer,
    target: ActivationQDQ,
    records: list[dict[str, Any]],
) -> torch.Tensor:
    collected: list[torch.Tensor] = []
    handle = target.register_forward_pre_hook(
        lambda _module, inputs: collected.append(inputs[0].detach().float().cpu().reshape(-1))
    )
    with torch.no_grad():
        for index, record in enumerate(records, start=1):
            quantizer.model(
                **model_inputs(record, quantizer.model.device),
                use_cache=False,
                logits_to_keep=1,
            )
            print(f"Collected target distribution sample {index}/{len(records)}")
    handle.remove()
    return torch.cat(collected)


def capture_full_probe(
    quantizer: Qwen3Quantizer, record: dict[str, Any]
) -> dict[str, Any]:
    outputs: dict[str, torch.Tensor] = {}
    handles = []
    for index, layer in enumerate(quantizer.model.model.layers):
        handles.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, index=index: outputs.__setitem__(
                    str(index), output.detach().float().cpu()
                )
            )
        )
    with torch.no_grad():
        result = quantizer.model(
            **model_inputs(record, quantizer.model.device),
            use_cache=False,
            logits_to_keep=1,
        )
    for handle in handles:
        handle.remove()
    return {
        "probe_index": record["index"],
        "input_ids": torch.tensor(record["input_ids"], dtype=torch.int32),
        "layer_outputs": outputs,
        "last_logits": result.logits[:, -1, :].detach().float().cpu(),
    }


def generation_report(quantizer: Qwen3Quantizer, max_new_tokens: int = 16) -> dict[str, Any]:
    prompt = "为什么伟大不能被计划"
    inputs = quantizer._build_model_inputs(prompt)
    input_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        generated = quantizer.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    output_ids = generated[0][input_length:].detach().cpu().tolist()
    text = quantizer.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return {"prompt": prompt, "output_ids": output_ids, "text": text}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    teacher_probe = torch.load(args.teacher_probe, map_location="cpu", weights_only=True)
    records = read_records(args.corpus)
    if not 0 <= args.eval_index < len(records):
        raise IndexError("eval index is outside the calibration corpus")
    collection_end = args.collection_start + args.collection_count
    if collection_end > len(records):
        raise IndexError("collection range is outside the calibration corpus")
    if args.eval_index in range(args.collection_start, collection_end):
        raise ValueError("eval record must be held out from range collection")

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
    target = modules.get(args.target)
    if not isinstance(target, ActivationQDQ):
        raise TypeError(f"target is not an ActivationQDQ: {args.target}")
    linked_targets = []
    for name in args.linked_target:
        module = modules.get(name)
        if not isinstance(module, ActivationQDQ):
            raise TypeError(f"linked target is not an ActivationQDQ: {name}")
        linked_targets.append((name, module))
    teacher_layer = teacher_probe["layer_outputs"][str(args.target_layer)]
    eval_record = records[args.eval_index]

    baseline_values = baseline_report["activation_qparams"][args.target]
    baseline_range = set_range(target, baseline_values["min"], baseline_values["max"])
    baseline_metrics, _ = evaluate_layer(
        quantizer, eval_record, args.target_layer, teacher_layer
    )
    baseline_score = score(baseline_metrics)
    print(f"Baseline score={baseline_score:.6f} metrics={baseline_metrics}")

    values = collect_target_values(
        quantizer, target, records[args.collection_start:collection_end]
    )
    quantile_levels = sorted(
        set(
            [
                0.0,
                0.0000001,
                0.000001,
                0.00001,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.995,
                0.999,
                0.9995,
                0.9999,
                0.99999,
                0.999999,
                0.9999999,
                1.0,
            ]
        )
    )
    quantile_values = torch.quantile(values, torch.tensor(quantile_levels)).tolist()
    quantiles = dict(zip(quantile_levels, quantile_values, strict=True))
    lower_levels = [level for level in quantile_levels if level <= 0.005]
    upper_levels = [level for level in quantile_levels if level >= 0.995]

    candidates = [
        {
            "kind": "baseline_report",
            "lower_quantile": None,
            "upper_quantile": None,
            "range": baseline_range,
            "metrics": baseline_metrics,
            "score": baseline_score,
        }
    ]
    candidate_number = 0
    for lower_level in lower_levels:
        for upper_level in upper_levels:
            candidate_number += 1
            qparam = set_range(
                target, quantiles[lower_level], quantiles[upper_level]
            )
            linked_qparams = {
                name: set_range(
                    module, quantiles[lower_level], quantiles[upper_level]
                )
                for name, module in linked_targets
            }
            metrics, _ = evaluate_layer(
                quantizer, eval_record, args.target_layer, teacher_layer
            )
            saturation = float(
                (
                    (values < qparam["effective_min"])
                    | (values > qparam["effective_max"])
                )
                .float()
                .mean()
                .item()
            )
            candidate = {
                "kind": "percentile",
                "lower_quantile": lower_level,
                "upper_quantile": upper_level,
                "range": qparam,
                "linked_ranges": linked_qparams,
                "saturation_fraction": saturation,
                "metrics": metrics,
                "score": score(metrics),
            }
            candidates.append(candidate)
            if candidate_number % 8 == 0:
                print(
                    f"Evaluated {candidate_number}/{len(lower_levels) * len(upper_levels)} "
                    f"candidate score={candidate['score']:.6f}"
                )

    best = max(candidates, key=lambda candidate: candidate["score"])
    set_range(target, best["range"]["min"], best["range"]["max"])
    for name, module in linked_targets:
        linked_range = best.get("linked_ranges", {}).get(name, best["range"])
        set_range(module, linked_range["min"], linked_range["max"])
    best_probe = capture_full_probe(quantizer, eval_record)
    generation = generation_report(quantizer)
    torch.save(best_probe, args.output_dir / "best_probe.pt")

    result = {
        "experiment": "EXP-0018",
        "target": args.target,
        "linked_targets": args.linked_target,
        "target_layer": args.target_layer,
        "eval_index": args.eval_index,
        "collection_indices": list(range(args.collection_start, collection_end)),
        "collected_values": values.numel(),
        "distribution_min": float(values.amin().item()),
        "distribution_max": float(values.amax().item()),
        "quantiles": {str(level): value for level, value in quantiles.items()},
        "baseline": candidates[0],
        "best": best,
        "improvement": best["score"] - baseline_score,
        "generation": generation,
        "candidate_count": len(candidates),
        "elapsed_seconds": time.perf_counter() - start,
        "candidates": candidates,
    }
    (args.output_dir / "search.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"best": best, "generation": generation}, ensure_ascii=False))
    print(f"Wrote range search to {args.output_dir}")


if __name__ == "__main__":
    main()
