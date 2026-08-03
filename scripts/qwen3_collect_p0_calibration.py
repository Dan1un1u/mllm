#!/usr/bin/env python3
"""Collect BF16 Qwen3 Linear-input shards for the P0 A8 sensitivity map.

This collector is deliberately independent of ``pymllm.mobile`` and the QNN
FFI extension.  It runs a BF16 Transformers teacher under inference mode,
captures only Linear *inputs* for all requested decoder layers, and writes one
safetensors shard per prompt.  The fixed train/held-out split is identical to
the earlier P1 calibration protocol (seed 17 by default).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    parser.add_argument("--layers", default=",".join(str(i) for i in range(28)))
    parser.add_argument("--train-prompts", type=int, default=4)
    parser.add_argument("--held-out-prompts", type=int, default=6)
    parser.add_argument("--max-seq-length", type=int, default=96)
    parser.add_argument("--prompt-seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/daniuniu/llm_exp/calibration/"
            "qwen3-p0-all-layers-seed17-s96"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory after an explicit check.",
    )
    return parser.parse_args()


def parse_csv_ints(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("--layers must contain unique layer indices")
    return layers


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if row and not row[0].startswith("#") and len(row) >= 4:
                prompts.append(row[-1])
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def split_prompt_indices(
    count: int,
    *,
    train_count: int,
    held_out_count: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if train_count <= 0 or held_out_count <= 0:
        raise ValueError("train and held-out prompt counts must be positive")
    if train_count + held_out_count > count:
        raise ValueError("requested more prompts than the TSV contains")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = torch.randperm(count, generator=generator)[
        : train_count + held_out_count
    ].tolist()
    return selected[:train_count], selected[train_count:]


def format_prompt(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return prompt


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"no tensor in hook value {type(value).__name__}")


class InputRecorder:
    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    def begin(self) -> None:
        self.values = {}

    def capture(self, name: str, value: Any) -> None:
        if name in self.values:
            raise RuntimeError(f"hook fired more than once: {name}")
        tensor = first_tensor(value).detach()
        if not tensor.is_floating_point():
            raise TypeError(f"expected floating activation for {name}")
        self.values[name] = tensor.to("cpu", dtype=torch.bfloat16).contiguous()

    def add_pre_hook(self, module: torch.nn.Module, name: str) -> None:
        def hook(_module: torch.nn.Module, arguments: tuple[Any, ...]) -> None:
            self.capture(name, arguments)

        self.handles.append(module.register_forward_pre_hook(hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def install_hooks(model: torch.nn.Module, layers: list[int]) -> InputRecorder:
    recorder = InputRecorder()
    decoder_layers = model.model.layers
    for layer_index in layers:
        layer = decoder_layers[layer_index]
        for short_name, module_path in PROJECTIONS.items():
            module = layer
            for component in module_path.split("."):
                module = getattr(module, component)
            recorder.add_pre_hook(
                module,
                f"layer_{layer_index:02d}.{short_name}_input",
            )
    return recorder


def expected_keys(layers: list[int]) -> set[str]:
    return {
        f"layer_{layer:02d}.{projection}_input"
        for layer in layers
        for projection in PROJECTIONS
    }


def tensor_inventory(values: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes": value.numel() * value.element_size(),
        }
        for name, value in sorted(values.items())
    }


def main() -> None:
    args = parse_args()
    layers = parse_csv_ints(args.layers)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the BF16 teacher collector")
    if not args.prompt_tsv.is_file():
        raise FileNotFoundError(args.prompt_tsv)
    if args.output_dir.exists():
        existing = list(args.output_dir.iterdir())
        if existing and not args.overwrite:
            raise FileExistsError(
                f"output directory is non-empty: {args.output_dir}; "
                "choose a new path or pass --overwrite"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    prompts = load_prompts(args.prompt_tsv)
    train_indices, held_out_indices = split_prompt_indices(
        len(prompts),
        train_count=args.train_prompts,
        held_out_count=args.held_out_prompts,
        seed=args.prompt_seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    print("Loading BF16 Qwen3 teacher", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).eval().to("cuda:0")
    layer_count = len(model.model.layers)
    invalid = [layer for layer in layers if not 0 <= layer < layer_count]
    if invalid:
        raise ValueError(f"layers outside [0, {layer_count - 1}]: {invalid}")

    recorder = install_hooks(model, layers)
    required = expected_keys(layers)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "P0 static A8 activation sensitivity map",
        "model": str(args.model),
        "prompt_tsv": {
            "path": str(args.prompt_tsv),
            "sha256": sha256_file(args.prompt_tsv),
        },
        "prompt_seed": args.prompt_seed,
        "split": {
            "train_indices": train_indices,
            "held_out_indices": held_out_indices,
        },
        "layers": layers,
        "projections": sorted(PROJECTIONS),
        "max_seq_length": args.max_seq_length,
        "storage": {
            "format": "safetensors, one prompt per shard",
            "floating_dtype": "torch.bfloat16",
            "teacher_grad_mode": "torch.inference_mode",
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "shards": [],
        "complete": False,
    }
    manifest_path = args.output_dir / "manifest.json"
    atomic_json(manifest_path, manifest)

    work = [
        *(('train', position, index) for position, index in enumerate(train_indices)),
        *((
            'held_out',
            position,
            index,
        ) for position, index in enumerate(held_out_indices)),
    ]
    try:
        for split_name, split_position, source_index in work:
            prompt = prompts[source_index]
            encoded = tokenizer(
                format_prompt(tokenizer, prompt),
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_length,
            )
            model_inputs = {key: value.to("cuda:0") for key, value in encoded.items()}
            recorder.begin()
            with torch.inference_mode():
                model(**model_inputs, use_cache=False, return_dict=True)
            captured = dict(recorder.values)
            if set(captured) != required:
                missing = sorted(required - set(captured))
                extra = sorted(set(captured) - required)
                raise RuntimeError(f"activation key mismatch: missing={missing}, extra={extra}")
            captured["input_ids"] = encoded["input_ids"].cpu().contiguous()
            if "attention_mask" in encoded:
                captured["attention_mask"] = encoded["attention_mask"].cpu().contiguous()
            filename = f"{split_name}-{split_position:03d}-source-{source_index:03d}.safetensors"
            shard_path = args.output_dir / filename
            save_file(
                captured,
                str(shard_path),
                metadata={
                    "split": split_name,
                    "source_index": str(source_index),
                    "purpose": "P0 static A8 sensitivity map",
                },
            )
            manifest["shards"].append(
                {
                    "split": split_name,
                    "split_position": split_position,
                    "source_index": source_index,
                    "file": filename,
                    "file_sha256": sha256_file(shard_path),
                    "token_count": int(encoded["input_ids"].shape[1]),
                    "tensors": tensor_inventory(captured),
                }
            )
            atomic_json(manifest_path, manifest)
            print(
                f"[{split_name} {split_position + 1}] source={source_index} "
                f"tokens={encoded['input_ids'].shape[1]} file={filename}",
                flush=True,
            )
            del model_inputs, encoded, captured
        manifest["complete"] = True
        atomic_json(manifest_path, manifest)
    finally:
        recorder.remove()
        del model

    print(
        json.dumps(
            {
                "complete": manifest["complete"],
                "shards": len(manifest["shards"]),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

