#!/usr/bin/env python3
"""Optimize A8 and optional LPBQ scales against a single decoder block.

This is intentionally a block-only autograd experiment.  The BF16 teacher
is run once per prompt to capture the target layer input, attention/position
arguments, and block output.  Optimization then calls only that decoder block
with frozen G32 codes and trainable A8 scales.  ``--learn-weight-scale``
additionally learns LPBQ G32 scale1/scale2, while the exported scale1 is
rounded back to UInt4 and re-evaluated.  No 28-layer graph is ever retained.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pymllm.quantization.static_a8 import (
    A8Params,
    LearnableA8FakeQuant,
    LearnableLPBQScale,
    LPBQWeights,
    freeze_a8_zero_point,
    lpbq_quantize_g32,
)


PROJECTIONS = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/daniuniu/llm_exp/models/Qwen3-origin"))
    parser.add_argument("--prompt-tsv", type=Path, default=Path("scripts/qwen3_sm8750_v79_accuracy.tsv"))
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=Path("/home/daniuniu/llm_exp/calibration/qwen3-p0-all-layers-seed17-s96/manifest.json"),
    )
    parser.add_argument("--sensitivity-map", type=Path, default=Path("artifacts/p0/static_a8/sensitivity-map.json"))
    parser.add_argument("--layers", default="0,27")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--learn-weight-scale",
        action="store_true",
        help="Learn LPBQ G32 scale1/scale2 with fixed int4 codes.",
    )
    parser.add_argument(
        "--fixed-zero-point",
        choices=("map", "0", "128"),
        default="map",
        help="Override A8 zero-point; map keeps each calibrated integer zp.",
    )
    parser.add_argument("--max-seq-length", type=int, default=96)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("--layers must contain unique indices")
    return layers


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"no tensor in {type(value).__name__}")


def clone_structure(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, tuple):
        return tuple(clone_structure(item) for item in value)
    if isinstance(value, list):
        return [clone_structure(item) for item in value]
    return value


def move_structure(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        # Teacher captures are produced under inference_mode.  An ordinary
        # empty tensor plus copy is required here: ``to().clone()`` preserves
        # the inference flag in recent PyTorch builds.
        moved = value.to(device)
        return torch.empty_like(moved).copy_(moved)
    if isinstance(value, tuple):
        return tuple(move_structure(item, device) for item in value)
    if isinstance(value, list):
        return [move_structure(item, device) for item in value]
    return value


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if row and not row[0].startswith("#") and len(row) >= 4:
                prompts.append(row[-1])
    return prompts


def format_prompt(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return prompt


def nested_get(module: nn.Module, path: str) -> nn.Module:
    for component in path.split("."):
        module = getattr(module, component)
    return module


def nested_set(module: nn.Module, path: str, value: nn.Module) -> None:
    components = path.split(".")
    parent = nested_get(module, ".".join(components[:-1]))
    setattr(parent, components[-1], value)


def load_selected(
    path: Path,
    *,
    fixed_zero_point: int | None = None,
) -> dict[tuple[int, str], A8Params | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[int, str], A8Params | None] = {}
    for row in data["rows"]:
        key = int(row["layer"]), str(row["projection"])
        if row["recommended_precision"] == "A16":
            result[key] = None
        else:
            params = A8Params(
                **row["candidates"][row["best_strategy"]]["params"]
            )
            result[key] = (
                params
                if fixed_zero_point is None
                else freeze_a8_zero_point(params, fixed_zero_point)
            )
    return result


class BlockQLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Module,
        init: A8Params | None,
        device: torch.device,
        *,
        learn_weight_scale: bool = False,
        quantized_override: LPBQWeights | None = None,
    ) -> None:
        super().__init__()
        quantized = (
            lpbq_quantize_g32(linear.weight.detach())
            if quantized_override is None
            else quantized_override
        )
        self.lpbq_scale = (
            LearnableLPBQScale(quantized).to(device=device)
            if learn_weight_scale
            else None
        )
        if self.lpbq_scale is None:
            self.register_buffer(
                "weight",
                quantized.decoded.to(device=device, dtype=linear.weight.dtype),
            )
        bias = getattr(linear, "bias", None)
        self.register_buffer(
            "bias",
            None if bias is None else bias.detach().to(device=device, dtype=linear.weight.dtype),
        )
        self.quantizer = None if init is None else LearnableA8FakeQuant(init)
        if self.quantizer is not None:
            self.quantizer.to(device=device)
        # The P1 curriculum can first optimize the deployable LPBQ weight
        # scales with A16 inputs, then enable the static A8 fake-quantizer.
        # Keeping this as a module flag avoids replacing the trainable wrapper
        # (and its optimizer state) between the two stages.
        self.activation_quant_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantizer is not None and self.activation_quant_enabled:
            x = self.quantizer(x)
        weight = self.weight if self.lpbq_scale is None else self.lpbq_scale()
        return F.linear(x, weight.to(dtype=x.dtype), self.bias)

    @torch.no_grad()
    def export_lpbq(self) -> LPBQWeights | None:
        if self.lpbq_scale is None:
            return None
        return self.lpbq_scale.export()


def capture_examples(
    model: nn.Module,
    model_inputs: list[dict[str, torch.Tensor]],
    layer_index: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    layer = model.model.layers[layer_index]

    def pre_hook(_module, args, kwargs):
        current["hidden_states"] = args[0].detach().cpu().to(torch.bfloat16).contiguous()
        current["kwargs"] = clone_structure(kwargs)

    def post_hook(_module, _args, output):
        current["target"] = first_tensor(output).detach().cpu().float().contiguous()

    pre_handle = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post_handle = layer.register_forward_hook(post_hook)
    try:
        for model_input in model_inputs:
            current.clear()
            # no_grad (rather than inference_mode) keeps captured rotary
            # tensors ordinary tensors so the later block-only autograd pass
            # can save them for backward.
            with torch.no_grad():
                model(
                    **{key: value.to(device) for key, value in model_input.items()},
                    use_cache=False,
                    return_dict=True,
                )
            examples.append(dict(current))
    finally:
        pre_handle.remove()
        post_handle.remove()
    return examples


def install_wrapped_layer(
    layer: nn.Module,
    params: dict[str, A8Params | None],
    device: torch.device,
    *,
    learn_weight_scale: bool = False,
    lpbq_overrides: dict[str, LPBQWeights] | None = None,
) -> dict[str, nn.Module]:
    originals: dict[str, nn.Module] = {}
    for projection, path in PROJECTIONS.items():
        original = nested_get(layer, path)
        originals[path] = original
        nested_set(
            layer,
            path,
            BlockQLinear(
                original,
                params[projection],
                device,
                learn_weight_scale=learn_weight_scale,
                quantized_override=(
                    None if lpbq_overrides is None else lpbq_overrides.get(projection)
                ),
            ),
        )
    return originals


def restore_layer(layer: nn.Module, originals: dict[str, nn.Module]) -> None:
    for path, original in originals.items():
        nested_set(layer, path, original)


def block_loss(layer: nn.Module, examples: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for example in examples:
        hidden = move_structure(example["hidden_states"], device)
        kwargs = move_structure(example["kwargs"], device)
        target = move_structure(example["target"], device)
        output = first_tensor(layer(hidden, **kwargs)).float()
        nmse = (output - target).square().mean() / target.square().mean().clamp_min(1e-20)
        cosine = F.cosine_similarity(output.reshape(1, -1), target.reshape(1, -1), eps=1e-8).clamp(-1, 1)
        losses.append(nmse + 0.05 * (1.0 - cosine.mean()))
    return torch.stack(losses).mean()


def evaluate_block(layer: nn.Module, examples: list[dict[str, Any]], device: torch.device) -> dict[str, float]:
    references: list[torch.Tensor] = []
    candidates: list[torch.Tensor] = []
    with torch.inference_mode():
        for example in examples:
            output = first_tensor(
                layer(
                    example["hidden_states"].to(device),
                    **move_structure(example["kwargs"], device),
                )
            ).detach().cpu().float()
            target = example["target"]
            references.append(target.reshape(-1))
            candidates.append(output.reshape(-1))
    reference = torch.cat(references)
    candidate = torch.cat(candidates)
    nmse = (candidate - reference).square().sum() / reference.square().sum().clamp_min(1e-20)
    cosine = F.cosine_similarity(reference[None], candidate[None], eps=1e-8).clamp(-1, 1)
    return {"nmse": float(nmse), "cosine": float(cosine)}


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    layers = parse_layers(args.layers)
    manifest = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    train_indices = manifest["split"]["train_indices"]
    held_out_indices = manifest["split"]["held_out_indices"]
    prompts = load_prompts(args.prompt_tsv)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    all_indices = train_indices + held_out_indices
    model_inputs: list[dict[str, torch.Tensor]] = []
    for index in all_indices:
        encoded = tokenizer(
            format_prompt(tokenizer, prompts[index]),
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_length,
        )
        model_inputs.append(dict(encoded))
    train_count = len(train_indices)
    train_inputs = model_inputs[:train_count]
    held_out_inputs = model_inputs[train_count:]

    fixed_zero_point = (
        None if args.fixed_zero_point == "map" else int(args.fixed_zero_point)
    )
    selected = load_selected(
        args.sensitivity_map,
        fixed_zero_point=fixed_zero_point,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).eval().to(device)
    output: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "P0 single-block learned A8 scale optimization",
        "layers": layers,
        "train_indices": train_indices,
        "held_out_indices": held_out_indices,
        "steps": args.steps,
        "lr": args.lr,
        "learn_weight_scale": args.learn_weight_scale,
        "fixed_zero_point": fixed_zero_point,
        "rows": [],
        "complete": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    for layer_index in layers:
        layer = model.model.layers[layer_index]
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
        train_examples = capture_examples(model, train_inputs, layer_index, device)
        held_out_examples = capture_examples(model, held_out_inputs, layer_index, device)
        initial_params = {
            projection: selected[(layer_index, projection)] for projection in PROJECTIONS
        }
        baseline_params = {projection: None for projection in PROJECTIONS}

        baseline_originals = install_wrapped_layer(
            layer,
            baseline_params,
            device,
            learn_weight_scale=False,
        )
        baseline_metrics = evaluate_block(layer, held_out_examples, device)
        restore_layer(layer, baseline_originals)

        initial_originals = install_wrapped_layer(
            layer,
            initial_params,
            device,
            learn_weight_scale=args.learn_weight_scale,
        )
        initial_metrics = evaluate_block(layer, held_out_examples, device)

        trainable = [parameter for parameter in layer.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=args.lr) if trainable else None
        losses: list[float] = []
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        if optimizer is not None:
            for _ in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                loss = block_loss(layer, train_examples, device)
                loss.backward()
                if args.gradient_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
                optimizer.step()
                value = float(loss.detach())
                losses.append(value)
                if value < best_loss:
                    best_loss = value
                    best_state = {
                        name: value.detach().clone()
                        for name, value in layer.state_dict().items()
                        if (
                            "quantizer.log_scale" in name
                            or "lpbq_scale.log_scale" in name
                        )
                    }
            if best_state:
                current_state = layer.state_dict()
                for name, value in best_state.items():
                    current_state[name].copy_(value)
        learned_metrics = evaluate_block(layer, held_out_examples, device)
        learned_params = {
            projection: nested_get(layer, path).quantizer.export_params().as_dict()
            if nested_get(layer, path).quantizer is not None
            else None
            for projection, path in PROJECTIONS.items()
        }
        learned_weight_scales = {}
        exported_lpbq: dict[str, LPBQWeights] = {}
        if args.learn_weight_scale:
            for projection, path in PROJECTIONS.items():
                module = nested_get(layer, path)
                exported = module.export_lpbq()
                if exported is None:
                    raise RuntimeError(
                        f"missing learned LPBQ scale for {projection}"
                    )
                exported_lpbq[projection] = exported
                learned_weight_scales[projection] = {
                    "scale1": exported.scale1.tolist(),
                    "scale2": exported.scale2.tolist(),
                }
        # Reinstall the exported integer scale1/FP32 scale2 representation and
        # evaluate it separately.  This proves the reported learned score is
        # not relying on an unexportable continuous training-only scale.
        learned_a8_params = {
            projection: (
                None
                if params is None
                else A8Params(**params)
            )
            for projection, params in learned_params.items()
        }
        restore_layer(layer, initial_originals)
        deploy_originals = install_wrapped_layer(
            layer,
            learned_a8_params,
            device,
            learn_weight_scale=False,
            lpbq_overrides=(exported_lpbq or None),
        )
        deploy_metrics = evaluate_block(layer, held_out_examples, device)
        restore_layer(layer, deploy_originals)
        row = {
            "layer": layer_index,
            "baseline_w4a16": baseline_metrics,
            "initial_selected_mixed": initial_metrics,
            "learned_block_scales": learned_metrics,
            "exported_deploy_block_scales": deploy_metrics,
            "optimization": {
                "steps": len(losses),
                "initial_train_loss": losses[0] if losses else None,
                "best_train_loss": min(losses) if losses else None,
                "final_train_loss": losses[-1] if losses else None,
            },
            "initial_params": {
                projection: None if params is None else params.as_dict()
                for projection, params in initial_params.items()
            },
            "learned_params": learned_params,
            "learned_weight_scales": learned_weight_scales,
        }
        output["rows"].append(row)
        args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"layer={layer_index:02d} baseline={baseline_metrics['nmse']:.6f} "
            f"initial={initial_metrics['nmse']:.6f} learned={learned_metrics['nmse']:.6f}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output["complete"] = True
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"block scale optimization complete: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
