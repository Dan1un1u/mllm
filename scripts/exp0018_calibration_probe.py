#!/usr/bin/env python3
"""Run a software-only Qwen3 calibration probe for EXP-0018.

This driver deliberately stops before deploy conversion or model-file emission.
It records enough state to compare calibration policies against the archived
W4A16 teacher without changing the QNN graph contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import torch

from pymllm.mobile.backends.qualcomm.transformers.core.qdq import (
    ActivationQDQ,
    FixedActivationQDQ,
)
from pymllm.mobile.backends.qualcomm.transformers.core.qlinear import QLinearLPBQ
from pymllm.mobile.backends.qualcomm.transformers.core.rms_norm import QRMSNorm
from pymllm.mobile.backends.qualcomm.transformers.qwen3.runner import (
    CALIBRATION_MODES,
    Qwen3Quantizer,
    calibration_fake_quant_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--activation-bits", type=int, choices=(8, 16), required=True)
    parser.add_argument(
        "--calibration-mode", choices=CALIBRATION_MODES, required=True
    )
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--probe-index", type=int, default=0)
    parser.add_argument("--generation-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: Any) -> float | int:
    if isinstance(value, torch.Tensor):
        value = value.detach().reshape(-1)[0].item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return float(value)


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    finite_count = int(finite.sum().item())
    if finite_count:
        selected = value[finite]
        minimum = float(selected.amin().item())
        maximum = float(selected.amax().item())
        mean = float(selected.mean().item())
        rms = float(torch.sqrt(torch.mean(selected.square())).item())
    else:
        minimum = maximum = mean = rms = float("nan")
    return {
        "shape": list(value.shape),
        "numel": value.numel(),
        "finite": finite_count,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "rms": rms,
    }


def activation_qparam_report(model: torch.nn.Module) -> dict[str, Any]:
    report = {}
    for name, module in model.named_modules():
        if not isinstance(module, ActivationQDQ):
            continue
        observer = module.fake_quant.activation_post_process
        report[name] = {
            "bits": module.bits,
            "quant_min": module.fake_quant.quant_min,
            "quant_max": module.fake_quant.quant_max,
            "min": scalar(observer.min_val),
            "max": scalar(observer.max_val),
            "scale": scalar(module.fake_quant.scale),
            "zero_point": scalar(module.fake_quant.zero_point),
        }
    return report


def runtime_contract_report(model: torch.nn.Module) -> dict[str, Any]:
    dynamic_bits: dict[int, int] = {}
    dynamic_qschemes: dict[str, int] = {}
    fixed_bits: dict[int, int] = {}
    rmsnorm_bits: dict[int, int] = {}
    lpbq_block_sizes: dict[int, int] = {}
    lpbq_quant_ranges: dict[str, int] = {}
    for module in model.modules():
        if isinstance(module, ActivationQDQ):
            dynamic_bits[module.bits] = dynamic_bits.get(module.bits, 0) + 1
            qscheme = str(module.qscheme)
            dynamic_qschemes[qscheme] = dynamic_qschemes.get(qscheme, 0) + 1
        elif isinstance(module, FixedActivationQDQ):
            fixed_bits[module.bits] = fixed_bits.get(module.bits, 0) + 1
        elif isinstance(module, QRMSNorm):
            rmsnorm_bits[module.quant_bits] = rmsnorm_bits.get(module.quant_bits, 0) + 1
        elif isinstance(module, QLinearLPBQ):
            block_size = int(module.block_size[-1])
            lpbq_block_sizes[block_size] = lpbq_block_sizes.get(block_size, 0) + 1
            quant_range = (
                f"{module.weight_quant.quant_min}:{module.weight_quant.quant_max}"
            )
            lpbq_quant_ranges[quant_range] = lpbq_quant_ranges.get(quant_range, 0) + 1
    return {
        "dynamic_activation_bits": dynamic_bits,
        "dynamic_activation_qschemes": dynamic_qschemes,
        "fixed_activation_bits": fixed_bits,
        "rmsnorm_weight_bits": rmsnorm_bits,
        "lpbq_block_sizes": lpbq_block_sizes,
        "lpbq_quant_ranges": lpbq_quant_ranges,
        "fake_quant_state": calibration_fake_quant_state(model),
    }


def load_probe_record(corpus: Path, index: int) -> dict[str, Any]:
    records = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
    if not 0 <= index < len(records):
        raise IndexError(f"probe index {index} is outside corpus of size {len(records)}")
    record = records[index]
    if record.get("index") != index:
        raise RuntimeError("probe record index does not match its JSONL position")
    return record


def run_probe(
    quantizer: Qwen3Quantizer, record: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    layer_outputs: dict[str, torch.Tensor] = {}
    tensor_stats: dict[str, Any] = {}
    handles = []

    def capture_layer(index: int):
        def hook(_module, _inputs, output):
            layer_outputs[str(index)] = output.detach().float().cpu()
            tensor_stats[f"layer_{index}.block_output"] = tensor_summary(output)

        return hook

    def capture_stat(name: str):
        def hook(_module, _inputs, output):
            tensor_stats[name] = tensor_summary(output)

        return hook

    for index, layer in enumerate(quantizer.model.model.layers):
        handles.append(layer.register_forward_hook(capture_layer(index)))
        handles.append(
            layer.mlp.down_proj_input_qdq.register_forward_hook(
                capture_stat(f"layer_{index}.mlp_down_input")
            )
        )
        handles.append(
            layer.mlp.register_forward_hook(capture_stat(f"layer_{index}.mlp_output"))
        )

    device = quantizer.model.device
    model_inputs = {
        "input_ids": torch.tensor(
            [record["input_ids"]], dtype=torch.long, device=device
        ),
        "attention_mask": torch.tensor(
            [record["attention_mask"]], dtype=torch.long, device=device
        ),
    }
    with torch.no_grad():
        output = quantizer.model(
            **model_inputs,
            use_cache=False,
            logits_to_keep=1,
        )
        last_logits = output.logits[:, -1, :].detach().float().cpu()

    for handle in handles:
        handle.remove()

    payload = {
        "probe_index": record["index"],
        "input_ids": torch.tensor(record["input_ids"], dtype=torch.int32),
        "layer_outputs": layer_outputs,
        "last_logits": last_logits,
    }
    report = {
        "probe_index": record["index"],
        "sequence_length": len(record["input_ids"]),
        "tensor_stats": tensor_stats,
        "last_logit_summary": tensor_summary(last_logits),
        "last_token_argmax": int(last_logits.argmax(dim=-1).item()),
    }
    return report, payload


def generation_report(
    quantizer: Qwen3Quantizer, prompt: str, max_new_tokens: int
) -> dict[str, Any]:
    model_inputs = quantizer._build_model_inputs(prompt)
    input_length = model_inputs.input_ids.shape[1]
    with torch.no_grad():
        generated = quantizer.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    output_ids = generated[0][input_length:].detach().cpu().tolist()
    text = quantizer.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    non_space = re.sub(r"\s+", "", text)
    punctuation_only = bool(non_space) and not any(char.isalnum() for char in non_space)
    repeated_punctuation = punctuation_only and len(set(non_space)) <= 2
    return {
        "prompt": prompt,
        "output_ids": output_ids,
        "text": text,
        "empty": not bool(non_space),
        "repeated_punctuation": repeated_punctuation,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    start = time.perf_counter()
    quantizer = Qwen3Quantizer(
        str(args.model_path),
        mllm_qualcomm_max_length=args.max_length,
        activation_bits=args.activation_bits,
        linear_block_size=32,
    )
    quantizer.calibrate(
        str(args.corpus),
        num_samples=args.num_samples,
        max_seq_length=args.max_length,
        calibration_mode=args.calibration_mode,
    )
    quantizer.validate_concat_observer()

    record = load_probe_record(args.corpus, args.probe_index)
    probe_report, probe_payload = run_probe(quantizer, record)
    generation = generation_report(
        quantizer, "为什么伟大不能被计划", args.generation_tokens
    )

    report = {
        "experiment": "EXP-0018",
        "activation_bits": args.activation_bits,
        "calibration_mode": args.calibration_mode,
        "num_samples": args.num_samples,
        "max_length": args.max_length,
        "seed": args.seed,
        "corpus": str(args.corpus),
        "corpus_sha256": sha256_file(args.corpus),
        "elapsed_seconds": time.perf_counter() - start,
        "runtime_contract": runtime_contract_report(quantizer.model),
        "activation_qparams": activation_qparam_report(quantizer.model),
        "probe": probe_report,
        "generation": generation,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(probe_payload, args.output_dir / "probe.pt")
    print(json.dumps(generation, ensure_ascii=False, sort_keys=True))
    print(f"Wrote software-only probe to {args.output_dir}")


if __name__ == "__main__":
    main()
