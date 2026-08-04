#!/usr/bin/env python3
"""Prefix-aware, layer-by-layer P1 scale training.

Each layer is calibrated with the already-exported quantized prefix, then only
that decoder block is trained.  The learned LPBQ scales are exported to
per-layer safetensors and immediately replaced by fixed wrappers before the
next layer is collected.  No 28-layer autograd graph is retained.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_p0_block_eval import (
    PROJECTIONS,
    collect_teacher,
    first_tensor,
    metrics,
    nested_get,
    nested_set,
)
from qwen3_p0_block_optimize import (
    BlockQLinear,
    block_loss,
    capture_examples,
    evaluate_block,
    format_prompt,
    load_prompts,
    load_selected,
    move_structure,
)
from pymllm.quantization.static_a8 import (
    A8Params,
    LPBQWeights,
    fake_quantize_a8,
    pack_lpbq_codes_hwio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "--teacher-model",
        dest="model",
        type=Path,
        default=Path("/home/daniuniu/llm_exp/models/Qwen3-origin"),
        help="BF16 teacher checkpoint; never used to regenerate fixed INT4 codes when --base-quant-checkpoint is set.",
    )
    parser.add_argument(
        "--base-quant-checkpoint",
        type=Path,
        default=None,
        help="QNN/QLinearLPBQ G32 model.safetensors supplying fixed INT4 codes and initial scale1/scale2.",
    )
    parser.add_argument(
        "--prompt-tsv", type=Path, default=Path("scripts/qwen3_sm8750_v79_accuracy.tsv")
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
        default=Path("artifacts/p0/static_a8/mixed-precision-map.json"),
    )
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--stage1-steps",
        type=int,
        default=100,
        help="W4A16 LPBQ-only curriculum steps before enabling A8 (0 disables).",
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--stage1-lr",
        type=float,
        default=0.005,
        help="Learning rate for the W4A16 LPBQ-only stage.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Linear warmup steps applied independently to each stage.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Cosine schedule floor as a fraction of each stage learning rate.",
    )
    parser.add_argument(
        "--deployment-eval-every",
        type=int,
        default=25,
        help="Re-export UInt4 LPBQ scales and evaluate every N optimizer steps.",
    )
    parser.add_argument(
        "--selection-split",
        choices=("train", "held_out"),
        default="train",
        help="Split used to select the best exported deployment candidate.",
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=96)
    parser.add_argument(
        "--fixed-zero-point",
        choices=("map", "0", "128"),
        default="128",
        help="Fixed integer A8 zero-point; map preserves calibrated values.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("prefix", "teacher"),
        default="teacher",
        help="Distill to the quantized-prefix BF16 block or original BF16 teacher output.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p1/streaming-full")
    )
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("--layers must contain unique indices")
    return layers


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def install_trainable_layer(
    layer: nn.Module,
    params: dict[str, A8Params | None],
    device: torch.device,
    *,
    lpbq_overrides: dict[str, LPBQWeights] | None = None,
) -> None:
    for projection, path in PROJECTIONS.items():
        original = nested_get(layer, path)
        override = None if lpbq_overrides is None else lpbq_overrides[projection]
        if override is not None and tuple(override.codes.shape) != tuple(original.weight.shape):
            raise ValueError(
                f"{projection}: base codes {tuple(override.codes.shape)} do not match "
                f"teacher weight {tuple(original.weight.shape)}"
            )
        nested_set(
            layer,
            path,
            BlockQLinear(
                original,
                params[projection],
                device,
                learn_weight_scale=True,
                quantized_override=override,
            ),
        )


def load_base_lpbq_layer(
    checkpoint: Path,
    layer_index: int,
) -> dict[str, LPBQWeights]:
    """Load one layer's HWIO carrier and scales from the authoritative G32 base."""

    from safetensors import safe_open

    result: dict[str, LPBQWeights] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for projection, module_path in PROJECTIONS.items():
            prefix = f"model.layers.{layer_index}.{module_path}"
            weight = handle.get_tensor(prefix + ".weight")
            scale1_flat = handle.get_tensor(prefix + ".scale1")
            scale2 = handle.get_tensor(prefix + ".scale2")
            if weight.ndim != 4 or tuple(weight.shape[:2]) != (1, 1):
                raise ValueError(
                    f"{prefix}: expected HWIO [1,1,K,O], got {tuple(weight.shape)}"
                )
            in_features = int(weight.shape[2])
            out_features = int(weight.shape[3])
            if in_features % 32:
                raise ValueError(f"{prefix}: K={in_features} is not divisible by G32")
            carrier = weight.reshape(in_features, out_features).to(torch.int16)
            nibble = torch.bitwise_and(carrier, 0x0F)
            signed = torch.where(nibble >= 8, nibble - 16, nibble)
            codes = signed.transpose(0, 1).contiguous().to(torch.int8)
            scale1 = scale1_flat.reshape(out_features, in_features // 32).contiguous()
            scale2 = scale2.reshape(out_features).contiguous().to(torch.float32)
            if tuple(scale1.shape) != (out_features, in_features // 32):
                raise ValueError(f"{prefix}: invalid scale1 shape {tuple(scale1.shape)}")
            if int(scale1.min()) < 1 or int(scale1.max()) > 16:
                raise ValueError(f"{prefix}: scale1 is outside UInt4 range [1,16]")
            decoded = (
                codes.reshape(out_features, in_features // 32, 32).float()
                * scale1.float().unsqueeze(-1)
                * scale2[:, None, None]
            ).reshape(out_features, in_features).contiguous()
            result[projection] = LPBQWeights(
                codes=codes,
                scale1=scale1.to(torch.uint8),
                scale2=scale2,
                decoded=decoded,
                group_size=32,
            )
    return result


class FixedQLinear(nn.Module):
    """A fixed exported LPBQ wrapper used as the next-layer prefix."""

    def __init__(
        self,
        scales: LPBQWeights,
        bias: torch.Tensor | None,
        a8_params: A8Params | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", scales.decoded.to(device=device, dtype=dtype))
        self.register_buffer(
            "bias",
            None if bias is None else bias.detach().to(device=device, dtype=dtype),
        )
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


def set_activation_quant_enabled(layer: nn.Module, enabled: bool) -> None:
    """Toggle A8 inputs without replacing the trainable block wrappers."""

    for path in PROJECTIONS.values():
        module = nested_get(layer, path)
        if not isinstance(module, BlockQLinear):
            raise TypeError(f"expected BlockQLinear at {path}, got {type(module).__name__}")
        module.activation_quant_enabled = bool(enabled)


def set_activation_scale_requires_grad(layer: nn.Module, enabled: bool) -> None:
    """Freeze/unfreeze only the static A8 scale parameters for the curriculum."""

    for path in PROJECTIONS.values():
        module = nested_get(layer, path)
        if not isinstance(module, BlockQLinear) or module.quantizer is None:
            continue
        for parameter in module.quantizer.parameters():
            parameter.requires_grad_(enabled)


def snapshot_trainable_parameters(layer: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in layer.named_parameters()
        if parameter.requires_grad
    }


def restore_parameters(layer: nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(layer.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if name not in parameters:
                raise KeyError(f"saved parameter {name!r} is missing from the layer")
            parameters[name].copy_(value)


def collect_exported_candidate(
    layer: nn.Module,
) -> tuple[dict[str, LPBQWeights], dict[str, A8Params | None]]:
    scales: dict[str, LPBQWeights] = {}
    a8_params: dict[str, A8Params | None] = {}
    for projection, path in PROJECTIONS.items():
        module = nested_get(layer, path)
        if not isinstance(module, BlockQLinear):
            raise TypeError(f"expected BlockQLinear at {path}, got {type(module).__name__}")
        exported = module.export_lpbq()
        if exported is None:
            raise RuntimeError(f"layer wrapper {projection} has no LPBQ scales")
        scales[projection] = exported
        a8_params[projection] = (
            None
            if module.quantizer is None
            else module.quantizer.export_params()
        )
    return scales, a8_params


def evaluate_exported_candidate(
    layer: nn.Module,
    examples: list[dict[str, Any]],
    device: torch.device,
    *,
    use_activation_quant: bool,
) -> float:
    """Evaluate the exact exported LPBQ/A8 representation, not continuous params."""

    scales, a8_params = collect_exported_candidate(layer)
    originals: dict[str, nn.Module] = {}
    for projection, path in PROJECTIONS.items():
        old = nested_get(layer, path)
        originals[path] = old
        nested_set(
            layer,
            path,
            FixedQLinear(
                scales[projection],
                old.bias,
                a8_params[projection] if use_activation_quant else None,
                dtype=old.bias.dtype if old.bias is not None else torch.bfloat16,
                device=device,
            ),
        )
    try:
        with torch.no_grad():
            return float(block_loss(layer, examples, device))
    finally:
        for path, original in originals.items():
            nested_set(layer, path, original)


def update_cosine_lr(
    optimizer: torch.optim.Optimizer,
    base_lrs: list[float],
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> None:
    if total_steps <= 0:
        return
    if warmup_steps > 0 and step <= warmup_steps:
        multiplier = step / float(warmup_steps)
    else:
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * multiplier


def train_scale_stage(
    layer: nn.Module,
    examples: list[dict[str, Any]],
    device: torch.device,
    *,
    stage_name: str,
    steps: int,
    lr: float,
    gradient_clip: float,
    warmup_steps: int,
    min_lr_ratio: float,
    deployment_eval_every: int,
    use_activation_quant: bool,
) -> dict[str, Any]:
    """Optimize one curriculum stage and select by exported deployment loss."""

    if steps < 0:
        raise ValueError(f"{stage_name}: steps must be non-negative")
    if steps == 0:
        return {
            "name": stage_name,
            "steps": 0,
            "enabled": False,
            "best_step": None,
            "best_deployment_loss": None,
            "deployment_evaluations": [],
        }
    if lr <= 0.0:
        raise ValueError(f"{stage_name}: learning rate must be positive")

    set_activation_quant_enabled(layer, use_activation_quant)
    set_activation_scale_requires_grad(layer, use_activation_quant)
    trainable = [parameter for parameter in layer.parameters() if parameter.requires_grad]
    if not trainable:
        return {
            "name": stage_name,
            "steps": 0,
            "enabled": False,
            "best_step": None,
            "best_deployment_loss": None,
            "deployment_evaluations": [],
        }

    optimizer = torch.optim.Adam(trainable, lr=lr)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_step: int | None = None
    losses: list[float] = []
    deployment_evaluations: list[dict[str, float | int]] = []
    eval_every = max(1, deployment_eval_every)
    for step in range(1, steps + 1):
        update_cosine_lr(
            optimizer,
            base_lrs,
            step,
            steps,
            warmup_steps,
            min_lr_ratio,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = block_loss(layer, examples, device)
        loss.backward()
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable, gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach()))

        if step % eval_every == 0 or step == steps:
            deployed_loss = evaluate_exported_candidate(
                layer,
                examples,
                device,
                use_activation_quant=use_activation_quant,
            )
            deployment_evaluations.append(
                {"step": step, "loss": deployed_loss}
            )
            if deployed_loss < best_loss:
                best_loss = deployed_loss
                best_step = step
                best_state = snapshot_trainable_parameters(layer)

    if best_state is None:
        raise RuntimeError(f"{stage_name}: no deployment candidate was evaluated")
    restore_parameters(layer, best_state)
    return {
        "name": stage_name,
        "steps": steps,
        "enabled": True,
        "best_step": best_step,
        "best_deployment_loss": best_loss,
        "initial_train_loss": losses[0],
        "best_train_loss": min(losses),
        "final_train_loss": losses[-1],
        "deployment_evaluations": deployment_evaluations,
    }


def export_and_freeze_layer(
    layer: nn.Module,
    learned_params: dict[str, A8Params | None],
    *,
    device: torch.device,
    output_path: Path,
) -> dict[str, dict[str, Any]]:
    from safetensors.torch import save_file

    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, dict[str, Any]] = {}
    exported: dict[str, LPBQWeights] = {}
    for projection, path in PROJECTIONS.items():
        module = nested_get(layer, path)
        scales = module.export_lpbq()
        if scales is None:
            raise RuntimeError(f"layer wrapper {projection} has no LPBQ scales")
        exported[projection] = scales
        tensors[f"{projection}.scale1"] = scales.scale1.detach().cpu()
        tensors[f"{projection}.scale2"] = scales.scale2.detach().cpu()
        code_bytes = scales.codes.detach().cpu().contiguous().numpy().tobytes()
        packed_code_bytes = (
            pack_lpbq_codes_hwio(scales.codes.detach().cpu())
            .contiguous()
            .numpy()
            .tobytes()
        )
        metadata[projection] = {
            "scale1_shape": list(scales.scale1.shape),
            "scale2_shape": list(scales.scale2.shape),
            "codes_sha256": hashlib.sha256(code_bytes).hexdigest(),
            "codes_layout": "OI signed int8 logical codes",
            "packed_codes_sha256": hashlib.sha256(packed_code_bytes).hexdigest(),
            "packed_codes_layout": "HWIO [1,1,K,O] low-nibble int8 carrier",
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path))

    # Replace trainable wrappers with fixed exported wrappers before the next
    # layer is collected.  This is the key prefix-aware streaming boundary.
    for projection, path in PROJECTIONS.items():
        old = nested_get(layer, path)
        nested_set(
            layer,
            path,
            FixedQLinear(
                exported[projection],
                old.bias,
                learned_params[projection],
                dtype=old.bias.dtype if old.bias is not None else torch.bfloat16,
                device=device,
            ),
        )
    return metadata


def run_full(
    model: nn.Module,
    inputs: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[list[torch.Tensor], dict[int, list[torch.Tensor]]]:
    layers = list(range(len(model.model.layers)))
    blocks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    logits: list[torch.Tensor] = []
    current: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in layers:
        def hook(_module, _arguments, output, layer_index=layer_index):
            current[layer_index] = first_tensor(output).detach().cpu().float()

        handles.append(model.model.layers[layer_index].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            for model_input in inputs:
                current.clear()
                output = model(
                    **{key: value.to(device) for key, value in model_input.items()},
                    use_cache=False,
                    return_dict=True,
                )
                logits.append(output.logits[:, -1, :].detach().cpu().float())
                for layer_index in layers:
                    blocks[layer_index].append(current[layer_index].contiguous())
    finally:
        for handle in handles:
            handle.remove()
    return logits, blocks


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.stage1_steps < 0:
        raise ValueError("--stage1-steps must be non-negative")
    if args.deployment_eval_every <= 0:
        raise ValueError("--deployment-eval-every must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if args.base_quant_checkpoint is not None and not args.base_quant_checkpoint.is_file():
        raise FileNotFoundError(args.base_quant_checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    layers = parse_layers(args.layers)
    manifest = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    train_indices = manifest["split"]["train_indices"]
    held_out_indices = manifest["split"]["held_out_indices"]
    prompts = load_prompts(args.prompt_tsv)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    def encode(indices: list[int]) -> list[dict[str, torch.Tensor]]:
        return [
            dict(
                tokenizer(
                    format_prompt(tokenizer, prompts[index]),
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_seq_length,
                )
            )
            for index in indices
        ]

    train_inputs = encode(train_indices)
    held_out_inputs = encode(held_out_indices)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    teacher_train_blocks, _teacher_train_logits = collect_teacher(
        model,
        train_inputs,
        list(range(len(model.model.layers))),
        device,
    )
    teacher_blocks, teacher_logits = collect_teacher(
        model,
        held_out_inputs,
        list(range(len(model.model.layers))),
        device,
    )
    teacher_logits = [value.float() for value in teacher_logits]
    fixed_zero_point = None if args.fixed_zero_point == "map" else int(args.fixed_zero_point)
    selected = load_selected(args.sensitivity_map, fixed_zero_point=fixed_zero_point)
    base_checkpoint_sha256 = (
        None
        if args.base_quant_checkpoint is None
        else sha256_file(args.base_quant_checkpoint)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "P1 prefix-aware streaming A8+LPBQ scale training",
        "model": str(args.model),
        "teacher_model": str(args.model),
        "base_quant_checkpoint": (
            None
            if args.base_quant_checkpoint is None
            else str(args.base_quant_checkpoint)
        ),
        "base_quant_checkpoint_sha256": base_checkpoint_sha256,
        "weight_code_source": (
            "base_quant_checkpoint"
            if args.base_quant_checkpoint is not None
            else "teacher_model_requantized"
        ),
        "layers": layers,
        "train_indices": train_indices,
        "held_out_indices": held_out_indices,
        "steps": args.steps,
        "lr": args.lr,
        "stage1_steps": args.stage1_steps,
        "stage1_lr": args.stage1_lr,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "deployment_eval_every": args.deployment_eval_every,
        "selection_split": args.selection_split,
        "fixed_zero_point": fixed_zero_point,
        "target_mode": args.target_mode,
        "rows": [],
        "complete": False,
    }
    manifest_path = args.output_dir / "streaming-train.json"
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for layer_index in layers:
        layer = model.model.layers[layer_index]
        train_examples = capture_examples(model, train_inputs, layer_index, device)
        held_out_examples = capture_examples(model, held_out_inputs, layer_index, device)
        if args.target_mode == "teacher":
            for example, target in zip(
                train_examples, teacher_train_blocks[layer_index]
            ):
                example["target"] = target
            for example, target in zip(
                held_out_examples, teacher_blocks[layer_index]
            ):
                example["target"] = target
        initial_params = {
            projection: selected[(layer_index, projection)] for projection in PROJECTIONS
        }
        base_lpbq = (
            None
            if args.base_quant_checkpoint is None
            else load_base_lpbq_layer(args.base_quant_checkpoint, layer_index)
        )
        install_trainable_layer(
            layer,
            initial_params,
            device,
            lpbq_overrides=base_lpbq,
        )
        selection_examples = (
            train_examples
            if args.selection_split == "train"
            else held_out_examples
        )
        stage1 = train_scale_stage(
            layer,
            selection_examples,
            device,
            stage_name="w4a16_lpbq",
            steps=args.stage1_steps,
            lr=args.stage1_lr,
            gradient_clip=args.gradient_clip,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            deployment_eval_every=args.deployment_eval_every,
            use_activation_quant=False,
        )
        stage2 = train_scale_stage(
            layer,
            selection_examples,
            device,
            stage_name="w4a8_joint",
            steps=args.steps,
            lr=args.lr,
            gradient_clip=args.gradient_clip,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            deployment_eval_every=args.deployment_eval_every,
            use_activation_quant=True,
        )

        learned_params = {
            projection: (
                nested_get(layer, path).quantizer.export_params().as_dict()
                if nested_get(layer, path).quantizer is not None
                else None
            )
            for projection, path in PROJECTIONS.items()
        }
        learned_a8_params = {
            projection: None if params is None else A8Params(**params)
            for projection, params in learned_params.items()
        }
        local_metrics = evaluate_block(layer, held_out_examples, device)
        scale_path = args.output_dir / f"layer{layer_index:02d}-lpbq-scales.safetensors"
        scale_metadata = export_and_freeze_layer(
            layer,
            learned_a8_params,
            device=device,
            output_path=scale_path,
        )
        exported_local_metrics = evaluate_block(layer, held_out_examples, device)
        row = {
            "layer": layer_index,
            "prefix_is_quantized": layer_index > min(layers),
            "weight_code_source": (
                "base_quant_checkpoint"
                if base_lpbq is not None
                else "teacher_model_requantized"
            ),
            "local_block_output": local_metrics,
            "optimization": {"stage1": stage1, "stage2": stage2},
            "exported_local_block_output": exported_local_metrics,
            "learned_params": learned_params,
            "lpbq_scales_file": scale_path.name,
            "lpbq_scale_metadata": scale_metadata,
        }
        result["rows"].append(row)
        manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"layer={layer_index:02d} exported_nmse={exported_local_metrics['nmse']:.6f} "
            f"stage1={stage1['best_deployment_loss']} "
            f"stage2={stage2['best_deployment_loss']}",
            flush=True,
        )
        del train_examples, held_out_examples, selection_examples, stage1, stage2
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_logits, final_blocks = run_full(model, held_out_inputs, device=device)
    aggregate_references = [tensor for layer in teacher_blocks.values() for tensor in layer]
    aggregate_candidates = [tensor for layer in final_blocks.values() for tensor in layer]
    result["final_eval"] = {
        "last_token_logits": metrics(teacher_logits, final_logits, top1=True),
        "block_output": {
            "aggregate": metrics(aggregate_references, aggregate_candidates),
            "layers": {
                str(layer): metrics(teacher_blocks[layer], final_blocks[layer])
                for layer in teacher_blocks
            },
        },
    }
    result["complete"] = True
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "final logits_cos="
        f"{result['final_eval']['last_token_logits']['cosine']:.6f} "
        "top1="
        f"{result['final_eval']['last_token_logits']['top1_agreement']:.6f}",
        flush=True,
    )
    print(f"streaming training complete: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
