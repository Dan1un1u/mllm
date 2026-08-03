#!/usr/bin/env python3
"""Run the no-rotation P0 static-A8 scale study on one Linear.

The input directory can be a BF16 calibration shard directory produced by the
Qwen3 collector, while ``--weight`` can be either one safetensors file or the
original Qwen3 safetensors directory.  Only the requested tensor is loaded;
the whole model is never placed in an autograd graph.

Example (WSL):

    python scripts/qwen3_p0_static_a8.py \
      --inputs /home/daniuniu/llm_exp/calibration/qwen3-p1-layers-0-13-27-seed17-s96 \
      --input-key layer_00.o_proj_input \
      --weight /home/daniuniu/llm_exp/models/Qwen3-origin \
      --weight-key model.layers.0.self_attn.o_proj.weight \
      --output-json /home/daniuniu/llm_exp/p0/layer00-o-proj.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open



def _load_p0_oracle():
    """Load the repo-local oracle even with an older editable install active."""

    try:
        from pymllm.quantization.static_a8 import evaluate_a8_candidates

        return evaluate_a8_candidates
    except (ImportError, ModuleNotFoundError):
        # The WSL research venv may still have the previous mllm-wip editable
        # package installed.  Do not make users reinstall it just to run this
        # FFI-free script; load this file directly from the selected worktree.
        source = Path(__file__).resolve().parents[1] / "pymllm/quantization/static_a8.py"
        spec = importlib.util.spec_from_file_location("mllm_p0_static_a8", source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load P0 oracle from {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.evaluate_a8_candidates


evaluate_a8_candidates = _load_p0_oracle()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=False)
    parser.add_argument("--input-key", default=None)
    parser.add_argument(
        "--fit-split",
        choices=("train", "held_out", "all"),
        default="train",
        help="Shard filename prefix used to fit calibration/learnable scale.",
    )
    parser.add_argument(
        "--eval-split",
        choices=("train", "held_out", "all"),
        default="held_out",
        help="Shard filename prefix used only for final metrics.",
    )
    parser.add_argument("--weight", type=Path, required=False)
    parser.add_argument("--weight-key", default=None)
    parser.add_argument("--bias", type=Path, default=None)
    parser.add_argument("--bias-key", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--percentiles", default="99,99.9,99.99")
    parser.add_argument("--learnable-steps", type=int, default=200)
    parser.add_argument("--learnable-lr", type=float, default=0.03)
    parser.add_argument("--max-input-values", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use a deterministic small matrix instead of loading files.",
    )
    return parser.parse_args()


def _safetensor_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
        if files:
            return files
    raise FileNotFoundError(f"no safetensors file(s) found at {path}")


def load_tensor(path: Path, key: str | None) -> torch.Tensor:
    files = _safetensor_files(path)
    found: list[tuple[Path, str]] = []
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            selected = key
            if selected is None:
                if len(keys) != 1:
                    raise ValueError(
                        f"{file} contains {len(keys)} tensors; pass --*-key"
                    )
                selected = keys[0]
            if selected in keys:
                found.append((file, selected))
    if len(found) != 1:
        if not found:
            raise KeyError(f"tensor {key!r} not found below {path}")
        raise ValueError(f"tensor {key!r} appears in multiple files: {found}")
    file, selected = found[0]
    with safe_open(str(file), framework="pt", device="cpu") as handle:
        return handle.get_tensor(selected).contiguous()


def load_input_shards(
    path: Path,
    key: str,
    max_values: int = 0,
    split: str = "all",
) -> torch.Tensor:
    files = _safetensor_files(path)
    parts: list[torch.Tensor] = []
    for file in files:
        if split != "all" and not file.name.startswith(f"{split}-"):
            continue
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            if key not in handle.keys():
                continue
            value = handle.get_tensor(key).float()
        if value.ndim == 0:
            value = value.reshape(1, 1)
        else:
            value = value.reshape(-1, value.shape[-1])
        parts.append(value)
        if max_values > 0 and sum(int(part.shape[0]) for part in parts) >= max_values:
            break
    if not parts:
        raise KeyError(f"tensor {key!r} not found in any shard below {path}")
    result = torch.cat(parts, dim=0)
    if max_values > 0:
        result = result[:max_values]
    return result.contiguous()


def demo_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    generator = torch.Generator(device="cpu").manual_seed(17)
    inputs = torch.randn(128, 64, generator=generator, dtype=torch.float32)
    inputs[0, 0] = 24.0
    weight = torch.randn(96, 64, generator=generator, dtype=torch.float32)
    return inputs, weight, None


def main() -> None:
    args = parse_args()
    if args.demo:
        inputs, weight, bias = demo_tensors()
    else:
        if args.inputs is None or args.input_key is None:
            raise ValueError("--inputs and --input-key are required unless --demo")
        if args.weight is None or args.weight_key is None:
            raise ValueError("--weight and --weight-key are required unless --demo")
        inputs = load_input_shards(
            args.inputs,
            args.input_key,
            args.max_input_values,
            split=args.fit_split,
        )
        eval_inputs = load_input_shards(
            args.inputs,
            args.input_key,
            args.max_input_values,
            split=args.eval_split,
        )
        weight = load_tensor(args.weight, args.weight_key)
        bias = (
            None
            if args.bias is None
            else load_tensor(args.bias, args.bias_key)
        )
    if args.demo:
        eval_inputs = inputs.clone()

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    inputs = inputs.to(device=device, dtype=torch.float32)
    eval_inputs = eval_inputs.to(device=device, dtype=torch.float32)
    weight = weight.to(device=device, dtype=torch.float32)
    if bias is not None:
        bias = bias.to(device=device, dtype=torch.float32)
    percentiles = tuple(
        float(value) for value in args.percentiles.split(",") if value.strip()
    )
    result = evaluate_a8_candidates(
        inputs,
        weight,
        eval_inputs=eval_inputs,
        bias=bias,
        percentiles=percentiles,
        learnable_steps=args.learnable_steps,
        learnable_lr=args.learnable_lr,
    )
    result["inputs"] = {
        "fit_shape": list(inputs.shape),
        "eval_shape": list(eval_inputs.shape),
        "dtype": str(inputs.dtype),
        "source_key": args.input_key,
        "fit_split": args.fit_split if not args.demo else "demo",
        "eval_split": args.eval_split if not args.demo else "demo",
    }
    result["weight"]["source_key"] = args.weight_key  # type: ignore[index]
    result["environment"] = {
        "torch": torch.__version__,
        "device": str(device),
        "cuda": bool(torch.cuda.is_available()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
