#!/usr/bin/env python3
"""Build the full Qwen3 P0 static-A8 sensitivity map.

The script evaluates the seven Linear input tensors in each requested decoder
layer using the same train/held-out prompt split as the collector.  Each row
keeps the real unrotated Qwen3 weight, applies the deployment G32 LPBQ decode,
and compares Max-Min, mean/3-sigma, percentile clipping, and learnable scale.
The reference is W4A16 with the same decoded weight, isolating activation A8
error.  ``--a8-nmse-threshold`` is only a screening heuristic for an A16
fallback recommendation; it is not a model-quality gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from pymllm.quantization.static_a8 import evaluate_a8_candidates


PROJECTIONS = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}
FIXED_CANDIDATES = (
    "max_min",
    "mean_3sigma",
    "percentile_99",
    "percentile_99.9",
    "percentile_99.99",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path(
            "/home/daniuniu/llm_exp/calibration/"
            "qwen3-p0-all-layers-seed17-s96"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/daniuniu/llm_exp/models/Qwen3-origin"),
    )
    parser.add_argument("--layers", default=",".join(str(i) for i in range(28)))
    parser.add_argument("--learnable-steps", type=int, default=50)
    parser.add_argument("--learnable-lr", type=float, default=0.03)
    parser.add_argument("--max-input-values", type=int, default=0)
    parser.add_argument("--a8-nmse-threshold", type=float, default=0.02)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("--layers must contain unique indices")
    return layers


def safetensor_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files below {path}")
    return files


def tensor_index(files: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in index:
                    raise ValueError(f"duplicate tensor {key!r}")
                index[key] = file
    return index


def load_one(file: Path, key: str) -> torch.Tensor:
    with safe_open(str(file), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous()


def load_calibration_tensor(
    files: list[Path],
    key: str,
    split: str,
    max_values: int,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    rows = 0
    for file in files:
        if split != "all" and not file.name.startswith(f"{split}-"):
            continue
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            if key not in handle.keys():
                continue
            value = handle.get_tensor(key).float()
        value = value.reshape(-1, value.shape[-1])
        if max_values > 0:
            remaining = max_values - rows
            value = value[:remaining]
        parts.append(value)
        rows += int(value.shape[0])
        if max_values > 0 and rows >= max_values:
            break
    if not parts:
        raise KeyError(f"no {split} calibration tensor {key!r}")
    return torch.cat(parts, dim=0).contiguous()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    if args.a8_nmse_threshold <= 0.0:
        raise ValueError("--a8-nmse-threshold must be positive")
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    calibration_files = safetensor_files(args.calibration_dir)
    weight_files = safetensor_files(args.model)
    weight_keys = tensor_index(weight_files)
    output: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "P0 static A8 sensitivity map",
        "model": str(args.model),
        "calibration_dir": str(args.calibration_dir),
        "layers": layers,
        "projections": sorted(PROJECTIONS),
        "fit_split": "train",
        "eval_split": "held_out",
        "device": str(device),
        "torch": torch.__version__,
        "contract": {
            "rotation": "none",
            "weight": "LPBQ W4 G32",
            "activation": "static affine A8 per tensor",
            "zero_point": "fixed after fit; scale only for learnable",
            "reference": "W4A16 with the same decoded G32 weight",
        },
        "heuristic": {
            "a8_nmse_threshold": args.a8_nmse_threshold,
            "meaning": "recommend A16 only for screening; validate at block/model level",
        },
        "rows": [],
        "complete": False,
    }
    write_json(args.output_json, output)

    for layer in layers:
        for projection, module_path in PROJECTIONS.items():
            input_key = f"layer_{layer:02d}.{projection}_input"
            weight_key = f"model.layers.{layer}.{module_path}.weight"
            train_inputs = load_calibration_tensor(
                calibration_files,
                input_key,
                "train",
                args.max_input_values,
            ).to(device=device, dtype=torch.float32)
            eval_inputs = load_calibration_tensor(
                calibration_files,
                input_key,
                "held_out",
                args.max_input_values,
            ).to(device=device, dtype=torch.float32)
            try:
                weight = load_one(weight_keys[weight_key], weight_key).to(
                    device=device,
                    dtype=torch.float32,
                )
            except KeyError as error:
                raise KeyError(f"model has no expected weight {weight_key}") from error

            result = evaluate_a8_candidates(
                train_inputs,
                weight,
                eval_inputs=eval_inputs,
                learnable_steps=args.learnable_steps,
                learnable_lr=args.learnable_lr,
            )
            candidates = result["candidates"]
            best_fixed = min(
                FIXED_CANDIDATES,
                key=lambda name: candidates[name]["metrics"]["output_nmse"],
            )
            best_overall = min(
                candidates,
                key=lambda name: candidates[name]["metrics"]["output_nmse"],
            )
            best_nmse = float(candidates[best_overall]["metrics"]["output_nmse"])
            row = {
                "layer": layer,
                "projection": projection,
                "input_key": input_key,
                "weight_key": weight_key,
                "fit_shape": list(train_inputs.shape),
                "eval_shape": list(eval_inputs.shape),
                "weight_shape": list(weight.shape),
                "weight_nmse": result["weight"]["weight_nmse"],
                "best_fixed_strategy": best_fixed,
                "best_fixed_output_nmse": candidates[best_fixed]["metrics"]["output_nmse"],
                "best_strategy": best_overall,
                "best_output_nmse": best_nmse,
                "best_output_cosine": candidates[best_overall]["metrics"]["output_cosine"],
                "recommended_precision": "A8" if best_nmse <= args.a8_nmse_threshold else "A16",
                "candidates": candidates,
            }
            output["rows"].append(row)
            write_json(args.output_json, output)
            print(
                f"layer={layer:02d} {projection:10s} "
                f"best={best_overall:24s} nmse={best_nmse:.6f} "
                f"recommend={row['recommended_precision']}",
                flush=True,
            )
            del train_inputs, eval_inputs, weight, result
            if device.type == "cuda":
                torch.cuda.empty_cache()

    output["complete"] = True
    write_json(args.output_json, output)
    print(f"sensitivity map complete: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

