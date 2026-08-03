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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("/home/daniuniu/llm_exp/models/Qwen3-origin")
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
    parser.add_argument("--lr", type=float, default=0.01)
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


def install_trainable_layer(
    layer: nn.Module,
    params: dict[str, A8Params | None],
    device: torch.device,
) -> None:
    for projection, path in PROJECTIONS.items():
        original = nested_get(layer, path)
        nested_set(
            layer,
            path,
            BlockQLinear(
                original,
                params[projection],
                device,
                learn_weight_scale=True,
            ),
        )


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
        metadata[projection] = {
            "scale1_shape": list(scales.scale1.shape),
            "scale2_shape": list(scales.scale2.shape),
            "codes_sha256": hashlib.sha256(code_bytes).hexdigest(),
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "P1 prefix-aware streaming A8+LPBQ scale training",
        "model": str(args.model),
        "layers": layers,
        "train_indices": train_indices,
        "held_out_indices": held_out_indices,
        "steps": args.steps,
        "lr": args.lr,
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
        install_trainable_layer(layer, initial_params, device)
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
        row = {
            "layer": layer_index,
            "prefix_is_quantized": layer_index > min(layers),
            "local_block_output": local_metrics,
            "optimization": {
                "steps": len(losses),
                "initial_train_loss": losses[0] if losses else None,
                "best_train_loss": min(losses) if losses else None,
                "final_train_loss": losses[-1] if losses else None,
            },
            "learned_params": learned_params,
            "lpbq_scales_file": scale_path.name,
            "lpbq_scale_metadata": scale_metadata,
        }
        result["rows"].append(row)
        manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"layer={layer_index:02d} local_nmse={local_metrics['nmse']:.6f} "
            f"learned_loss={row['optimization']['best_train_loss']:.6f}",
            flush=True,
        )
        del train_examples, held_out_examples, trainable, optimizer
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
