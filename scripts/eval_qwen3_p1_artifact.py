#!/usr/bin/env python3
"""Evaluate a deployable P1 scale artifact without optimizing any parameter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_p0_block_eval import (
    PROJECTIONS,
    collect_teacher,
    first_tensor,
    format_prompt,
    metrics,
    nested_get,
    nested_set,
)
from qwen3_p0_block_optimize import load_prompts
from qwen3_p1_streaming_train import (
    FixedQLinear,
    load_base_lpbq_layer,
    run_full,
)
from pymllm.quantization.static_a8 import A8Params, lpbq_rebuild_with_scales


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-quant-checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--prompt-tsv", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=96)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def a8_from_manifest(value: dict[str, object] | None) -> A8Params | None:
    if value is None:
        return None
    fields = {
        "method",
        "scale",
        "zero_point",
        "clip_min",
        "clip_max",
        "quant_min",
        "quant_max",
        "percentile",
        "warm_start_method",
        "recipe",
    }
    return A8Params(**{key: value[key] for key in fields if key in value})


def install_artifact(
    model: torch.nn.Module,
    base_checkpoint: Path,
    artifact_dir: Path,
    manifest: dict[str, object],
    device: torch.device,
) -> None:
    rows = {int(row["layer"]): row for row in manifest["rows"]}
    for layer_index in range(len(model.model.layers)):
        row = rows[layer_index]
        base_lpbq = load_base_lpbq_layer(base_checkpoint, layer_index)
        scale_path = artifact_dir / str(row["lpbq_scales_file"])
        with safe_open(str(scale_path), framework="pt", device="cpu") as handle:
            scales = {
                projection: lpbq_rebuild_with_scales(
                    base_lpbq[projection],
                    handle.get_tensor(f"{projection}.scale1"),
                    handle.get_tensor(f"{projection}.scale2"),
                )
                for projection in PROJECTIONS
            }
        input_params = {
            projection: a8_from_manifest(row["learned_params"][projection])
            for projection in PROJECTIONS
        }
        output_manifest = row.get("v_output_params", {})
        output_params = {
            projection: a8_from_manifest(output_manifest.get(projection))
            for projection in PROJECTIONS
        }
        layer = model.model.layers[layer_index]
        for projection, path in PROJECTIONS.items():
            original = nested_get(layer, path)
            nested_set(
                layer,
                path,
                FixedQLinear(
                    scales[projection],
                    original.bias,
                    input_params[projection],
                    output_a8_params=output_params[projection],
                    dtype=original.bias.dtype
                    if original.bias is not None
                    else torch.bfloat16,
                    device=device,
                ),
            )


def encode_inputs(
    model_path: Path,
    prompt_tsv: Path,
    prompts_indices: list[int],
    max_seq_length: int,
) -> list[dict[str, torch.Tensor]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompts = load_prompts(prompt_tsv)
    return [
        dict(
            tokenizer(
                format_prompt(tokenizer, prompts[index]),
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            )
        )
        for index in prompts_indices
    ]


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    manifest = json.loads(
        (args.artifact_dir / "streaming-train.json").read_text(encoding="utf-8")
    )
    split = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))["split"]
    train_indices = list(split["train_indices"])
    held_out_indices = list(split["held_out_indices"])
    indices = train_indices + held_out_indices
    inputs = encode_inputs(args.model, args.prompt_tsv, indices, args.max_seq_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).eval().to(device)
    with torch.inference_mode():
        teacher_blocks, teacher_logits = collect_teacher(
            model,
            inputs,
            list(range(len(model.model.layers))),
            device,
        )
    install_artifact(
        model,
        args.base_quant_checkpoint,
        args.artifact_dir,
        manifest,
        device,
    )
    candidate_logits, candidate_blocks = run_full(model, inputs, device=device)
    train_count = len(train_indices)
    result: dict[str, object] = {
        "artifact_dir": str(args.artifact_dir),
        "a8_recipe": manifest.get("a8_recipe"),
        "base_quant_checkpoint_sha256": manifest.get("base_quant_checkpoint_sha256"),
        "splits": {},
    }
    for name, start, end in (
        ("train", 0, train_count),
        ("held_out", train_count, len(indices)),
    ):
        block_metrics = {
            str(layer): metrics(
                teacher_blocks[layer][start:end],
                candidate_blocks[layer][start:end],
            )
            for layer in teacher_blocks
        }
        result["splits"][name] = {
            "last_token_logits": metrics(
                teacher_logits[start:end],
                candidate_logits[start:end],
                top1=True,
            ),
            "block_output": block_metrics,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
