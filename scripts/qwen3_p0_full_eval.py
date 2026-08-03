#!/usr/bin/env python3
"""Evaluate the composed P0 W4A16 and mixed W4A8 model variants.

The block gate modifies one layer at a time.  This script composes the same
deployable G32 LPBQ wrappers across all 28 layers, runs the fixed held-out
prompts, and reports final last-token logits against the BF16 teacher.  It is
still a Python oracle: no QNN/FFI path is imported and no runtime rotation is
introduced.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_p0_block_eval import (
    PROJECTIONS,
    first_tensor,
    DeployableQLinear,
    collect_teacher,
    load_prompts,
    load_selected_params,
    make_inputs,
    metrics,
    nested_get,
    nested_set,
)
from pymllm.quantization.static_a8 import (
    A8Params,
    lpbq_rebuild_with_scales,
    lpbq_quantize_g32,
)


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
        default=Path("artifacts/p0/static_a8/mixed-precision-map.json"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument(
        "--learned-block",
        action="append",
        default=[],
        metavar="LAYER:JSON",
        help="Use exported learned A8/LPBQ scales for a layer; repeatable.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_learned_blocks(
    specs: list[str],
) -> dict[tuple[int, str], dict[str, Any]]:
    learned: dict[tuple[int, str], dict[str, Any]] = {}
    for spec in specs:
        try:
            layer_text, path_text = spec.split(":", 1)
            layer = int(layer_text)
        except ValueError as exc:
            raise ValueError(
                f"--learned-block must be LAYER:JSON, got {spec!r}"
            ) from exc
        data = json.loads(Path(path_text).read_text(encoding="utf-8"))
        rows = data.get("rows", [])
        row = next((item for item in rows if int(item["layer"]) == layer), None)
        if row is None:
            raise ValueError(f"layer {layer} not found in learned block {path_text}")
        for projection in PROJECTIONS:
            params = row.get("learned_params", {}).get(projection)
            scales = row.get("learned_weight_scales", {}).get(projection)
            learned[(layer, projection)] = {
                "a8_params": None if params is None else A8Params(**params),
                "weight_scales": scales,
            }
    return learned


def apply_variant(
    model: torch.nn.Module,
    params: dict[tuple[int, str], Any],
    *,
    device: torch.device,
    learned: dict[tuple[int, str], dict[str, Any]] | None = None,
) -> dict[tuple[int, str], tuple[torch.nn.Module, str]]:
    originals: dict[tuple[int, str], tuple[torch.nn.Module, str]] = {}
    for layer_index, layer in enumerate(model.model.layers):
        for projection, path in PROJECTIONS.items():
            original = nested_get(layer, path)
            originals[(layer_index, projection)] = (original, path)
            learned_entry = None if learned is None else learned.get(
                (layer_index, projection)
            )
            quantized_override = None
            a8_params = params[(layer_index, projection)]
            if learned_entry is not None:
                # Do not let a learned block artifact downgrade a projection
                # that the selected deployment map explicitly promoted to
                # A16.  Learned A8 params apply only when the base map keeps
                # that tensor at A8; LPBQ weight scales may still be reused.
                if (
                    a8_params is not None
                    and learned_entry["a8_params"] is not None
                ):
                    a8_params = learned_entry["a8_params"]
                scales = learned_entry["weight_scales"]
                if scales is not None:
                    base = lpbq_quantize_g32(original.weight.detach())
                    quantized_override = lpbq_rebuild_with_scales(
                        base,
                        torch.tensor(scales["scale1"], dtype=torch.float32),
                        torch.tensor(scales["scale2"], dtype=torch.float32),
                    )
            nested_set(
                layer,
                path,
                DeployableQLinear(
                    original,
                    a8_params=a8_params,
                    device=device,
                    quantized_override=quantized_override,
                ),
            )
    return originals


def restore_variant(
    model: torch.nn.Module,
    originals: dict[tuple[int, str], tuple[torch.nn.Module, str]],
) -> None:
    for (layer_index, _projection), (original, path) in originals.items():
        nested_set(model.model.layers[layer_index], path, original)


def run_logits(
    model: torch.nn.Module,
    inputs: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[list[torch.Tensor], dict[int, list[torch.Tensor]]]:
    logits: list[torch.Tensor] = []
    layers = list(range(len(model.model.layers)))
    blocks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    current: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in layers:
        def hook(_module, _arguments, output, layer_index=layer_index):
            current[layer_index] = first_tensor(output).detach().cpu().float()

        handles.append(model.model.layers[layer_index].register_forward_hook(hook))
    with torch.inference_mode():
        try:
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
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    manifest = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    held_out_indices = manifest["split"]["held_out_indices"]
    selected = load_selected_params(args.sensitivity_map)
    learned = load_learned_blocks(args.learned_block)
    prompts = load_prompts(args.prompt_tsv)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    inputs = make_inputs(
        tokenizer,
        prompts,
        held_out_indices,
        max_length=args.max_length,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).eval().to(device)

    teacher_blocks, teacher_logits = collect_teacher(
        model,
        inputs,
        list(range(len(model.model.layers))),
        device,
    )
    variants: dict[str, dict[tuple[int, str], Any]] = {
        "w4a16": {
            (layer, projection): None
            for layer in range(len(model.model.layers))
            for projection in PROJECTIONS
        },
        "mixed": {
            (layer, projection): selected[(layer, projection)]
            for layer in range(len(model.model.layers))
            for projection in PROJECTIONS
        },
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "P0 composed full-model W4A16/W4A8 oracle",
        "model": str(args.model),
        "held_out_indices": held_out_indices,
        "learned_blocks": args.learned_block,
        "a16_tensor_count": sum(value is None for value in variants["mixed"].values()),
        "a8_tensor_count": sum(value is not None for value in variants["mixed"].values()),
        "variants": {},
        "complete": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for name, params in variants.items():
        originals = apply_variant(
            model,
            params,
            device=device,
            learned=(learned if name == "mixed" else None),
        )
        try:
            logits, block_outputs = run_logits(model, inputs, device=device)
        finally:
            restore_variant(model, originals)
            del originals
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        per_layer = {
            str(layer): metrics(teacher_blocks[layer], block_outputs[layer])
            for layer in teacher_blocks
        }
        aggregate_references = [
            tensor
            for layer in teacher_blocks.values()
            for tensor in layer
        ]
        aggregate_candidates = [
            tensor
            for layer in block_outputs.values()
            for tensor in layer
        ]
        result["variants"][name] = {
            "last_token_logits": metrics(teacher_logits, logits, top1=True),
            "block_output": {
                "aggregate": metrics(aggregate_references, aggregate_candidates),
                "layers": per_layer,
            },
        }
        print(
            f"variant={name} logits_cos="
            f"{result['variants'][name]['last_token_logits']['cosine']:.6f} "
            f"top1="
            f"{result['variants'][name]['last_token_logits']['top1_agreement']:.6f}",
            flush=True,
        )
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    result["complete"] = True
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"full evaluation complete: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
