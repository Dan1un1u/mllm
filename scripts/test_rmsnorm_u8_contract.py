#!/usr/bin/env python3
"""Host-side contract checks for the isolated native-U8 RMSNorm change.

This test intentionally does not load a model or any archived W4A8 artifact.
It checks the source contract and, when torch is available, exercises the
existing QRMSNorm exporter with a tiny parameter.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check_source_contract() -> None:
    recipe = (ROOT / "mllm/backends/qnn/aot/passes/LLMQuantRecipePass.cpp").read_text()
    ptq = (ROOT / "mllm/backends/qnn/aot/passes/PTQPass.cpp").read_text()
    visitor = (ROOT / "mllm/backends/qnn/aot/visitor/RMSNorm.cpp").read_text()
    model = (ROOT / "pymllm/mobile/backends/qualcomm/transformers/qwen3/modeling_qwen3.py").read_text()
    rms = (ROOT / "pymllm/mobile/backends/qualcomm/transformers/core/rms_norm.py").read_text()

    assert "quant_max = 255" in recipe
    assert "quant_ir_dtype = kUInt8PerTensorAsy" in recipe
    assert "checkTypeLimits<uint8_t>" in ptq
    assert "const bool native_u8" in visitor
    assert "parameter_quant_max = gamma_is_u8 ? 255 : 65535" in visitor
    assert "self.rms_norm_bits = self.activation_bits" in model
    assert "quant_eps = 0.0001 / quant_max" in rms


def check_python_exporter() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    from pymllm.mobile.backends.qualcomm.transformers.core.rms_norm import QRMSNorm

    import torch

    layer = QRMSNorm(32, quant_bits=8)
    layer.freeze_weight()
    layer.convert_to_deploy()
    assert layer.weight.dtype == torch.uint8
    assert int(layer.weight.min()) >= 0
    assert int(layer.weight.max()) <= 255
    assert layer.scale.numel() == 1
    assert layer.zero_point.numel() == 1
    return True


if __name__ == "__main__":
    check_source_contract()
    exporter_ran = check_python_exporter()
    print("native-U8 RMSNorm source contract: PASS")
    print(f"QRMSNorm exporter smoke test: {'PASS' if exporter_ran else 'SKIP (torch unavailable)'}")
