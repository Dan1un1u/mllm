#!/usr/bin/env python3
"""Capture ordered internal ActivationQDQ outputs for selected decoder layers."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from exp0018_progressive_range_search import (
    apply_activation_qparams,
    model_inputs,
    read_records,
)
from pymllm.mobile.backends.qualcomm.transformers.core.qdq import ActivationQDQ
from pymllm.mobile.backends.qualcomm.transformers.qwen3.runner import Qwen3Quantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation-bits", type=int, choices=(8, 16), required=True)
    parser.add_argument("--layers", default="0,1,2")
    parser.add_argument("--eval-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    layers = [int(value) for value in args.layers.split(",")]
    prefixes = tuple(f"model.layers.{layer}." for layer in layers)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    records = read_records(args.corpus)
    record = records[args.eval_index]

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
    quantizer.model.eval()
    quantizer.enable_fake_quant()
    apply_activation_qparams(quantizer.model, report)

    outputs: dict[str, list[torch.Tensor]] = {}
    execution_order: list[str] = []
    handles = []
    for name, module in quantizer.model.named_modules():
        if not isinstance(module, ActivationQDQ) or not name.startswith(prefixes):
            continue

        def hook(_module, _inputs, output, name=name):
            call_index = len(outputs.setdefault(name, []))
            outputs[name].append(output.detach().float().cpu())
            execution_order.append(f"{name}#{call_index}")

        handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        result = quantizer.model(
            **model_inputs(record, quantizer.model.device),
            use_cache=False,
            logits_to_keep=1,
        )
    for handle in handles:
        handle.remove()

    payload: dict[str, Any] = {
        "experiment": "EXP-0018",
        "activation_bits": args.activation_bits,
        "layers": layers,
        "eval_index": args.eval_index,
        "input_ids": torch.tensor(record["input_ids"], dtype=torch.int32),
        "execution_order": execution_order,
        "outputs": outputs,
        "last_logits": result.logits.detach().float().cpu(),
        "elapsed_seconds": time.perf_counter() - start,
    }
    torch.save(payload, args.output)
    total_bytes = sum(
        tensor.numel() * tensor.element_size()
        for calls in outputs.values()
        for tensor in calls
    )
    print(
        f"Captured {len(execution_order)} QDQ calls ({total_bytes / 2**20:.1f} MiB) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
