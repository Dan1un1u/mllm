#!/usr/bin/env python3
"""Evaluate selected P0 scales at block-output and logits level.

Only one decoder layer is modified per run.  Its seven Linear modules use
the real G32 LPBQ decode; the ``w4a16`` variant leaves their inputs unchanged,
while ``selected_mixed`` applies the sensitivity-map A8 parameters and keeps
screened A16 fallbacks unquantized on activation.  All other layers remain
the BF16 teacher.  This isolates whether tensor-local A8 improvements survive
the nonlinear block and residual stream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pymllm.quantization.static_a8 import (
    A8Params,
    fake_quantize_a8,
    lpbq_quantize_g32,
    LPBQWeights,
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
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/daniuniu/llm_exp/models/Qwen3-origin"),
    )
    parser.add_argument(
        "--prompt-tsv",
        type=Path,
        default=Path("scripts/qwen3_sm8750_v79_accuracy.tsv"),
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=Path(
            "/home/daniuniu/llm_exp/calibration/"
            "qwen3-p0-all-layers-seed17-s96/manifest.json"
        ),
    )
    parser.add_argument(
        "--sensitivity-map",
        type=Path,
        default=Path("artifacts/p0/static_a8/sensitivity-map.json"),
    )
    parser.add_argument("--layers", default="0,13,27")
    parser.add_argument(
        "--force-a16",
        default="",
        help="Comma-separated projections to force A16 in selected_mixed.",
    )
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
    raise TypeError(f"no tensor found in {type(value).__name__}")


def load_prompts(path: Path) -> list[str]:
    import csv

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


def cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.reshape(-1).float()
    got = candidate.reshape(-1).float()
    value = F.cosine_similarity(ref[None], got[None], eps=1e-8)
    return float(value.clamp(-1.0, 1.0))


def metrics(
    references: list[torch.Tensor],
    candidates: list[torch.Tensor],
    *,
    top1: bool = False,
) -> dict[str, float]:
    reference = torch.cat([value.float().reshape(-1) for value in references])
    candidate = torch.cat([value.float().reshape(-1) for value in candidates])
    nmse = (candidate - reference).square().sum() / reference.square().sum().clamp_min(1e-20)
    result = {
        "nmse": float(nmse),
        "cosine": cosine(reference, candidate),
    }
    abs_error = (candidate - reference).abs()
    epsilon = max(float(reference.abs().median()) * 1e-3, 1e-6)
    relative = abs_error / (reference.abs() + epsilon)
    result["relative_abs_error"] = float(relative.mean())
    result["abs_error_p99_9"] = float(torch.quantile(abs_error, 0.999))
    if top1:
        result["top1_agreement"] = float(
            (candidate.reshape(-1, references[0].shape[-1]).argmax(-1)
             == reference.reshape(-1, references[0].shape[-1]).argmax(-1))
            .float()
            .mean()
        )
    return result


class DeployableQLinear(nn.Module):
    """G32 LPBQ weight oracle with optional static A8 input QDQ."""

    def __init__(
        self,
        linear: nn.Module,
        *,
        a8_params: A8Params | None,
        device: torch.device,
        quantized_override: LPBQWeights | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(linear, "weight"):
            raise TypeError(f"expected Linear-like module, got {type(linear).__name__}")
        quantized = (
            lpbq_quantize_g32(linear.weight.detach())
            if quantized_override is None
            else quantized_override
        )
        dtype = linear.weight.dtype
        self.register_buffer("weight", quantized.decoded.to(device=device, dtype=dtype))
        bias = getattr(linear, "bias", None)
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(device=device, dtype=dtype))
        else:
            self.register_buffer("bias", None)
        self.a8_params = a8_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.a8_params is not None:
            x = fake_quantize_a8(
                x,
                self.a8_params.scale,
                self.a8_params.zero_point,
                quant_min=self.a8_params.quant_min,
                quant_max=self.a8_params.quant_max,
            )
        return F.linear(x, self.weight, self.bias)


def nested_get(module: nn.Module, path: str) -> nn.Module:
    for component in path.split("."):
        module = getattr(module, component)
    return module


def nested_set(module: nn.Module, path: str, value: nn.Module) -> None:
    components = path.split(".")
    parent = nested_get(module, ".".join(components[:-1]))
    setattr(parent, components[-1], value)


def load_selected_params(path: Path) -> dict[tuple[int, str], A8Params | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    selected: dict[tuple[int, str], A8Params | None] = {}
    for row in data["rows"]:
        key = (int(row["layer"]), str(row["projection"]))
        if row["recommended_precision"] == "A16":
            selected[key] = None
            continue
        candidate = row["candidates"][row["best_strategy"]]
        selected[key] = A8Params(**candidate["params"])
    return selected


def make_inputs(
    tokenizer: Any,
    prompts: list[str],
    indices: list[int],
    *,
    max_length: int = 96,
) -> list[dict[str, torch.Tensor]]:
    result = []
    for index in indices:
        encoded = tokenizer(
            format_prompt(tokenizer, prompts[index]),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        result.append({key: value for key, value in encoded.items()})
    return result


def collect_teacher(
    model: nn.Module,
    inputs: list[dict[str, torch.Tensor]],
    layers: list[int],
    device: torch.device,
) -> tuple[dict[int, list[torch.Tensor]], list[torch.Tensor]]:
    blocks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    logits: list[torch.Tensor] = []
    handles = []
    current: dict[int, torch.Tensor] = {}
    for layer in layers:
        def hook(_module, _arguments, output, layer=layer):
            current[layer] = first_tensor(output).detach().cpu().float()

        handles.append(model.model.layers[layer].register_forward_hook(hook))
    try:
        for model_input in inputs:
            current.clear()
            with torch.inference_mode():
                output = model(
                    **{key: value.to(device) for key, value in model_input.items()},
                    use_cache=False,
                    return_dict=True,
                )
            for layer in layers:
                blocks[layer].append(current[layer].contiguous())
            logits.append(output.logits[:, -1, :].detach().cpu().float())
    finally:
        for handle in handles:
            handle.remove()
    return blocks, logits


def run_variant(
    model: nn.Module,
    layer_index: int,
    inputs: list[dict[str, torch.Tensor]],
    teacher_blocks: list[torch.Tensor],
    teacher_logits: list[torch.Tensor],
    params_by_projection: dict[str, A8Params | None],
    *,
    device: torch.device,
) -> dict[str, Any]:
    layer = model.model.layers[layer_index]
    originals: dict[str, nn.Module] = {}
    for projection, path in PROJECTIONS.items():
        original = nested_get(layer, path)
        originals[path] = original
        nested_set(
            layer,
            path,
            DeployableQLinear(
                original,
                a8_params=params_by_projection[projection],
                device=device,
            ),
        )
    block_outputs: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    current: list[torch.Tensor] = []
    handle = layer.register_forward_hook(
        lambda _module, _arguments, output: current.append(
            first_tensor(output).detach().cpu().float()
        )
    )
    try:
        for model_input in inputs:
            current.clear()
            with torch.inference_mode():
                output = model(
                    **{key: value.to(device) for key, value in model_input.items()},
                    use_cache=False,
                    return_dict=True,
                )
            block_outputs.append(current[0])
            logits.append(output.logits[:, -1, :].detach().cpu().float())
    finally:
        handle.remove()
        for path, original in originals.items():
            nested_set(layer, path, original)
    return {
        "block_output": metrics(teacher_blocks, block_outputs),
        "last_token_logits": metrics(teacher_logits, logits, top1=True),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    layers = parse_layers(args.layers)
    manifest = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    held_out_indices = manifest["split"]["held_out_indices"]
    selected = load_selected_params(args.sensitivity_map)
    forced_a16 = {item.strip() for item in args.force_a16.split(",") if item.strip()}
    unknown = forced_a16 - set(PROJECTIONS)
    if unknown:
        raise ValueError(f"unknown --force-a16 projections: {sorted(unknown)}")
    for layer in layers:
        for projection in forced_a16:
            selected[(layer, projection)] = None
    prompts = load_prompts(args.prompt_tsv)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    inputs = make_inputs(tokenizer, prompts, held_out_indices)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).eval().to(device)

    teacher_blocks, teacher_logits = collect_teacher(model, inputs, layers, device)
    output: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "P0 block-output and logits gate",
        "model": str(args.model),
        "layers": layers,
        "held_out_indices": held_out_indices,
        "variants": {
            "w4a16": "target layer G32 LPBQ weights, A16 inputs",
            "selected_mixed": "target layer G32 LPBQ with sensitivity-map A8/A16 map",
        },
        "force_a16": sorted(forced_a16),
        "rows": [],
        "complete": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    for layer in layers:
        w4a16_params = {projection: None for projection in PROJECTIONS}
        mixed_params = {
            projection: selected[(layer, projection)] for projection in PROJECTIONS
        }
        w4a16 = run_variant(
            model,
            layer,
            inputs,
            teacher_blocks[layer],
            teacher_logits,
            w4a16_params,
            device=device,
        )
        mixed = run_variant(
            model,
            layer,
            inputs,
            teacher_blocks[layer],
            teacher_logits,
            mixed_params,
            device=device,
        )
        output["rows"].append(
            {
                "layer": layer,
                "w4a16": w4a16,
                "selected_mixed": mixed,
                "selected_a16_fallbacks": [
                    projection
                    for projection in PROJECTIONS
                    if mixed_params[projection] is None
                ],
            }
        )
        args.output_json.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"layer={layer:02d} "
            f"W4A16 block_nmse={w4a16['block_output']['nmse']:.6f} "
            f"mixed block_nmse={mixed['block_output']['nmse']:.6f} "
            f"mixed logits_cos={mixed['last_token_logits']['cosine']:.6f}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output["complete"] = True
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"block evaluation complete: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
