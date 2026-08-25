#!/usr/bin/env python3
"""Greedily calibrate A8 residual QDQ ranges against W4A16 block outputs."""

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
from exp0018_progressive_range_search import (
    apply_activation_qparams,
    capture_full_probe,
    generation_report,
    read_records,
    row_metrics,
    score,
    set_range,
)
from pymllm.mobile.backends.qualcomm.transformers.core.qdq import ActivationQDQ
from pymllm.mobile.backends.qualcomm.transformers.qwen3.runner import Qwen3Quantizer


QUANTILE_PAIRS = (
    (0.0, 1.0),
    (0.0000001, 0.9999999),
    (0.000001, 0.999999),
    (0.00001, 0.99999),
    (0.0001, 0.9999),
    (0.0005, 0.9995),
    (0.001, 0.999),
    (0.002, 0.998),
    (0.005, 0.995),
    (0.00001, 0.9999),
    (0.0001, 0.99999),
    (0.0001, 0.9995),
    (0.0005, 0.9999),
)

REFINED_QUANTILE_PAIRS = (
    (0.0, 1.0),
    (0.00001, 0.99999),
    (0.0001, 0.9999),
    (0.0001, 0.99999),
    (0.0005, 0.9999),
)


class StopAfterLayer(RuntimeError):
    """Internal control flow used to avoid evaluating later decoder blocks."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--teacher-set", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-indices", default="1,2,3,4")
    parser.add_argument("--eval-index", type=int, default=0)
    parser.add_argument("--first-layer", type=int, default=3)
    parser.add_argument("--last-layer", type=int, default=27)
    parser.add_argument(
        "--candidate-profile",
        choices=("full", "refined"),
        default="full",
        help="Use the full initial grid or the evidence-refined follow-up grid",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def activation_module(
    modules: dict[str, torch.nn.Module], name: str
) -> ActivationQDQ:
    module = modules.get(name)
    if not isinstance(module, ActivationQDQ):
        raise TypeError(f"expected ActivationQDQ at {name}")
    return module


def record_inputs(
    records: list[dict[str, Any]], indices: list[int], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "input_ids": torch.tensor(
                [records[index]["input_ids"]], dtype=torch.long, device=device
            ),
            "attention_mask": torch.tensor(
                [records[index]["attention_mask"]], dtype=torch.long, device=device
            ),
        }
        for index in indices
    ]


def teacher_rows(
    teacher: dict[str, Any], indices: list[int], key: str
) -> torch.Tensor:
    available = set(teacher["indices"])
    missing = [index for index in indices if index not in available]
    if missing:
        raise KeyError(f"teacher set is missing indices: {missing}")
    rows = []
    for index in indices:
        probe = teacher["probes"][str(index)]
        tensor = probe[key] if key == "last_logits" else probe["layer_outputs"][key]
        rows.append(tensor.reshape(-1, tensor.shape[-1]))
    return torch.cat(rows, dim=0)


def run_to_layer(
    quantizer: Qwen3Quantizer,
    inputs: list[dict[str, torch.Tensor]],
    layer_index: int,
    raw_targets: list[tuple[str, ActivationQDQ]] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    captured: list[torch.Tensor] = []
    raw: dict[str, list[torch.Tensor]] = {
        name: [] for name, _module in (raw_targets or [])
    }
    handles = []
    for name, module in raw_targets or []:
        handles.append(
            module.register_forward_pre_hook(
                lambda _module, values, name=name: raw[name].append(
                    values[0].detach().float().cpu().reshape(-1)
                )
            )
        )

    def stop_hook(_module, _inputs, output):
        captured.append(output.detach().float().cpu())
        raise StopAfterLayer

    handles.append(
        quantizer.model.model.layers[layer_index].register_forward_hook(stop_hook)
    )
    try:
        with torch.no_grad():
            for sample_inputs in inputs:
                try:
                    quantizer.model(
                        **sample_inputs, use_cache=False, logits_to_keep=1
                    )
                except StopAfterLayer:
                    pass
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(inputs):
        raise RuntimeError(
            f"expected {len(inputs)} layer captures, got {len(captured)}"
        )
    flattened = {
        name: torch.cat(values) if values else torch.empty(0)
        for name, values in raw.items()
    }
    rows = torch.cat(
        [tensor.reshape(-1, tensor.shape[-1]) for tensor in captured], dim=0
    )
    return rows, flattened


def qparam_snapshot(module: ActivationQDQ) -> dict[str, float | int]:
    observer = module.fake_quant.activation_post_process
    return {
        "min": float(observer.min_val.item()),
        "max": float(observer.max_val.item()),
        "scale": float(module.fake_quant.scale.item()),
        "zero_point": int(module.fake_quant.zero_point.item()),
    }


def range_from_quantiles(
    module: ActivationQDQ,
    values: torch.Tensor,
    lower: float,
    upper: float,
) -> tuple[dict[str, float | int], float]:
    bounds = torch.quantile(values, torch.tensor([lower, upper])).tolist()
    qparam = set_range(module, bounds[0], bounds[1])
    saturation = float(
        (
            (values < qparam["effective_min"])
            | (values > qparam["effective_max"])
        )
        .float()
        .mean()
        .item()
    )
    return qparam, saturation


def calibrate_layer(
    quantizer: Qwen3Quantizer,
    inputs: list[dict[str, torch.Tensor]],
    teacher_output: torch.Tensor,
    modules: dict[str, torch.nn.Module],
    layer_index: int,
    quantile_pairs: tuple[tuple[float, float], ...] = QUANTILE_PAIRS,
) -> dict[str, Any]:
    names = []
    if layer_index != 0:
        names.append(f"model.layers.{layer_index}.input_layernorm_input_qdq")
    names.append(f"model.layers.{layer_index}.add_0_output_qdq")
    targets = [(name, activation_module(modules, name)) for name in names]
    add_output_target = targets[-1]
    input_targets = targets[:-1]
    current = {name: qparam_snapshot(module) for name, module in targets}
    baseline_output, raw = run_to_layer(
        quantizer, inputs, layer_index, raw_targets=targets
    )
    baseline_metrics = row_metrics(teacher_output, baseline_output)
    candidates: list[dict[str, Any]] = [
        {
            "kind": "current",
            "ranges": current,
            "metrics": baseline_metrics,
            "score": score(baseline_metrics),
        }
    ]

    for lower, upper in quantile_pairs:
        ranges = {}
        saturation = {}
        # The post-attention residual distribution depends on the selected
        # input QDQ.  Recollect it after applying each input candidate rather
        # than deriving both ranges from the collapsed baseline execution.
        try:
            for name, module in input_targets:
                ranges[name], saturation[name] = range_from_quantiles(
                    module, raw[name], lower, upper
                )
        except ValueError:
            # Sparse tensors can have identical lower/upper quantiles.  They
            # are invalid ranges, not a reason to abort the layer search.
            continue
        add_name, add_module = add_output_target
        if input_targets:
            _, candidate_raw = run_to_layer(
                quantizer,
                inputs,
                layer_index,
                raw_targets=[add_output_target],
            )
            add_values = candidate_raw[add_name]
        else:
            add_values = raw[add_name]
        if add_values.numel() == 0 or float(add_values.amin()) == float(
            add_values.amax()
        ):
            continue
        try:
            ranges[add_name], saturation[add_name] = range_from_quantiles(
                add_module, add_values, lower, upper
            )
        except ValueError:
            continue
        candidate_output, _ = run_to_layer(quantizer, inputs, layer_index)
        metrics = row_metrics(teacher_output, candidate_output)
        candidates.append(
            {
                "kind": "percentile",
                "lower_quantile": lower,
                "upper_quantile": upper,
                "ranges": ranges,
                "saturation_fraction": saturation,
                "metrics": metrics,
                "score": score(metrics),
            }
        )

    best = max(candidates, key=lambda item: item["score"])
    for name, module in targets:
        selected = best["ranges"][name]
        set_range(module, selected["min"], selected["max"])
    print(
        f"Layer {layer_index:02d}: {candidates[0]['score']:.6f} -> "
        f"{best['score']:.6f} ({best['kind']} "
        f"q={best.get('lower_quantile')},{best.get('upper_quantile')})"
    )
    return {
        "layer": layer_index,
        "targets": names,
        "baseline": candidates[0],
        "best": best,
        "improvement": best["score"] - candidates[0]["score"],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def evaluate_logits(
    quantizer: Qwen3Quantizer,
    inputs: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for sample_inputs in inputs:
            output = quantizer.model(
                **sample_inputs,
                use_cache=False,
                logits_to_keep=1,
            )
            logits = output.logits.detach().float().cpu()
            outputs.append(logits.reshape(-1, logits.shape[-1]))
    return torch.cat(outputs, dim=0)


def calibrate_output_chain(
    quantizer: Qwen3Quantizer,
    inputs: list[dict[str, torch.Tensor]],
    teacher_logits: torch.Tensor,
    modules: dict[str, torch.nn.Module],
    quantile_pairs: tuple[tuple[float, float], ...] = QUANTILE_PAIRS,
) -> dict[str, Any]:
    names = (
        "model.norm_input_qdq",
        "lm_head_input_qdq",
        "lm_head_output_qdq",
    )
    targets = [(name, activation_module(modules, name)) for name in names]
    def collect_values(
        selected: list[tuple[str, ActivationQDQ]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw: dict[str, list[torch.Tensor]] = {name: [] for name, _module in selected}
        handles = [
            module.register_forward_pre_hook(
                lambda _module, values, name=name: raw[name].append(
                    values[0].detach().float().cpu().reshape(-1)
                )
            )
            for name, module in selected
        ]
        try:
            logits = evaluate_logits(quantizer, inputs)
        finally:
            for handle in handles:
                handle.remove()
        return logits, {name: torch.cat(items) for name, items in raw.items()}

    baseline_logits, values = collect_values(targets)
    current = {name: qparam_snapshot(module) for name, module in targets}
    baseline_metrics = row_metrics(teacher_logits, baseline_logits)
    candidates: list[dict[str, Any]] = [
        {
            "kind": "current",
            "ranges": current,
            "metrics": baseline_metrics,
            "score": score(baseline_metrics),
        }
    ]
    norm_target, lm_input_target, lm_output_target = targets
    for lower, upper in quantile_pairs:
        ranges = {}
        saturation = {}
        norm_name, norm_module = norm_target
        try:
            ranges[norm_name], saturation[norm_name] = range_from_quantiles(
                norm_module, values[norm_name], lower, upper
            )
        except ValueError:
            continue
        _, lm_input_values = collect_values([lm_input_target])
        lm_input_name, lm_input_module = lm_input_target
        selected_values = lm_input_values[lm_input_name]
        if float(selected_values.amin()) == float(selected_values.amax()):
            continue
        try:
            ranges[lm_input_name], saturation[lm_input_name] = range_from_quantiles(
                lm_input_module, selected_values, lower, upper
            )
        except ValueError:
            continue
        _, lm_output_values = collect_values([lm_output_target])
        lm_output_name, lm_output_module = lm_output_target
        selected_values = lm_output_values[lm_output_name]
        if float(selected_values.amin()) == float(selected_values.amax()):
            continue
        try:
            ranges[lm_output_name], saturation[lm_output_name] = range_from_quantiles(
                lm_output_module, selected_values, lower, upper
            )
        except ValueError:
            continue
        logits = evaluate_logits(quantizer, inputs)
        metrics = row_metrics(teacher_logits, logits)
        candidates.append(
            {
                "kind": "percentile",
                "lower_quantile": lower,
                "upper_quantile": upper,
                "ranges": ranges,
                "saturation_fraction": saturation,
                "metrics": metrics,
                "score": score(metrics),
            }
        )
    best = max(candidates, key=lambda item: item["score"])
    for name, module in targets:
        selected = best["ranges"][name]
        set_range(module, selected["min"], selected["max"])
    print(
        f"Output chain: {candidates[0]['score']:.6f} -> {best['score']:.6f} "
        f"({best['kind']} q={best.get('lower_quantile')},"
        f"{best.get('upper_quantile')})"
    )
    return {
        "targets": list(names),
        "baseline": candidates[0],
        "best": best,
        "improvement": best["score"] - candidates[0]["score"],
        "candidate_count": len(candidates),
        "candidates": candidates,
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
        raise ValueError("evaluation record must be held out from calibration")
    if not 0 <= args.first_layer <= args.last_layer < 28:
        raise ValueError("invalid decoder-layer interval")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    reported_bits = baseline_report.get("activation_bits")
    runtime_bits = (
        baseline_report.get("runtime_contract", {})
        .get("dynamic_activation_bits", {})
    )
    runtime_is_a8 = runtime_bits in ({"8": 957}, {8: 957})
    if reported_bits not in (None, 8) or (
        reported_bits is None and runtime_bits and not runtime_is_a8
    ):
        raise ValueError("baseline report must describe an A8 model")
    teacher = torch.load(args.teacher_set, map_location="cpu", weights_only=True)
    records = read_records(args.corpus)
    for index in calibration_indices + [args.eval_index]:
        if not 0 <= index < len(records):
            raise IndexError(f"corpus index is outside the corpus: {index}")

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
    inputs = record_inputs(records, calibration_indices, quantizer.model.device)
    quantile_pairs = (
        QUANTILE_PAIRS
        if args.candidate_profile == "full"
        else REFINED_QUANTILE_PAIRS
    )

    layer_results = []
    next_layer = args.first_layer
    if args.resume_checkpoint is not None:
        checkpoint = json.loads(args.resume_checkpoint.read_text(encoding="utf-8"))
        if checkpoint.get("experiment") != "EXP-0018":
            raise ValueError("resume checkpoint belongs to another experiment")
        if checkpoint.get("calibration_indices") != calibration_indices:
            raise ValueError("resume checkpoint uses different calibration indices")
        layer_results = checkpoint.get("completed_layers", [])
        completed = [int(result["layer"]) for result in layer_results]
        expected = list(range(args.first_layer, args.first_layer + len(completed)))
        if completed != expected:
            raise ValueError(
                f"resume layers are not contiguous: expected {expected}, got {completed}"
            )
        if completed and completed[-1] > args.last_layer:
            raise ValueError("resume checkpoint extends beyond requested last layer")
        apply_activation_qparams(quantizer.model, checkpoint)
        next_layer = args.first_layer + len(completed)
        print(
            f"Resumed {len(completed)} calibrated layers; next layer is {next_layer}"
        )

    for layer_index in range(next_layer, args.last_layer + 1):
        teacher_output = teacher_rows(teacher, calibration_indices, str(layer_index))
        result = calibrate_layer(
            quantizer,
            inputs,
            teacher_output,
            modules,
            layer_index,
            quantile_pairs,
        )
        layer_results.append(result)
        checkpoint = {
            "experiment": "EXP-0018",
            "variant": "progressive_block_output_calibration",
            "calibration_indices": calibration_indices,
            "completed_layers": layer_results,
            "activation_qparams": activation_qparam_report(quantizer.model),
        }
        (args.output_dir / "checkpoint.json").write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    teacher_logits = teacher_rows(teacher, calibration_indices, "last_logits")
    output_result = calibrate_output_chain(
        quantizer, inputs, teacher_logits, modules, quantile_pairs
    )
    eval_record = records[args.eval_index]
    probe = capture_full_probe(quantizer, eval_record)
    torch.save(probe, args.output_dir / "probe.pt")
    generation = generation_report(quantizer)
    heldout_teacher_layer = {
        str(layer): teacher_rows(teacher, [args.eval_index], str(layer))
        for layer in range(28)
    }
    heldout_layers = {
        layer: row_metrics(heldout_teacher_layer[layer], output)
        for layer, output in probe["layer_outputs"].items()
    }
    heldout_teacher_logits = teacher_rows(teacher, [args.eval_index], "last_logits")
    heldout_logits = row_metrics(heldout_teacher_logits, probe["last_logits"])
    heldout_logits["teacher_argmax"] = int(heldout_teacher_logits.argmax().item())
    heldout_logits["candidate_argmax"] = int(probe["last_logits"].argmax().item())
    heldout_logits["top1_match"] = (
        heldout_logits["teacher_argmax"] == heldout_logits["candidate_argmax"]
    )

    result = {
        "experiment": "EXP-0018",
        "variant": "progressive_block_output_calibration",
        "calibration_indices": calibration_indices,
        "eval_index": args.eval_index,
        "first_layer": args.first_layer,
        "last_layer": args.last_layer,
        "candidate_profile": args.candidate_profile,
        "quantile_pairs": quantile_pairs,
        "elapsed_seconds": time.perf_counter() - start,
        "runtime_contract": runtime_contract_report(quantizer.model),
        "layer_results": layer_results,
        "output_result": output_result,
        "heldout_layer_metrics": heldout_layers,
        "heldout_logits": heldout_logits,
        "generation": generation,
        "activation_qparams": activation_qparam_report(quantizer.model),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(generation, ensure_ascii=False, sort_keys=True))
    print(
        f"Held-out logits cosine={heldout_logits['global_cosine']:.6f} "
        f"top1_match={heldout_logits['top1_match']}"
    )
    print(f"Wrote progressive calibration to {args.output_dir}")


if __name__ == "__main__":
    main()
