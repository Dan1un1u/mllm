import torch
from torch import nn

from pymllm.mobile.backends.qualcomm.transformers.core.embedding import QEmbedding
from pymllm.mobile.backends.qualcomm.transformers.core.qdq import (
    ActivationQDQ,
    FixedActivationQDQ,
)
from pymllm.mobile.backends.qualcomm.transformers.core.qlinear import QLinearLPBQ
from pymllm.mobile.backends.qualcomm.transformers.core.rms_norm import QRMSNorm
from pymllm.mobile.backends.qualcomm.transformers.qwen3.runner import (
    calibration_fake_quant_state,
    configure_calibration_fake_quant,
)


class TinyQuantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = ActivationQDQ(bits=8)
        self.fixed_activation = FixedActivationQDQ(
            scale=1.0 / 256.0, zero_point=0, bits=8
        )
        self.linear = QLinearLPBQ(4, 4, bias=False, block_size=2)
        self.norm = QRMSNorm(4, quant_bits=8)
        self.embedding = QEmbedding(8, 4, quant_bits=16)


def test_deployment_calibration_only_disables_dynamic_activation_qdq():
    model = TinyQuantModel()

    state = configure_calibration_fake_quant(model, "deployment_minmax")

    assert state == {
        "dynamic_activation": {"enabled": 0, "disabled": 1},
        "fixed_activation": {"enabled": 1, "disabled": 0},
        "lpbq_weight": {"enabled": 1, "disabled": 0},
        "rmsnorm_weight": {"enabled": 1, "disabled": 0},
        "embedding_weight": {"enabled": 1, "disabled": 0},
    }


def test_legacy_calibration_disables_every_quantizer():
    model = TinyQuantModel()

    state = configure_calibration_fake_quant(model, "legacy_minmax")

    assert all(counts == {"enabled": 0, "disabled": 1} for counts in state.values())
    assert calibration_fake_quant_state(model) == state


def test_calibration_mode_is_validated():
    model = TinyQuantModel()

    try:
        configure_calibration_fake_quant(model, "unknown")
    except ValueError as exc:
        assert "Unsupported calibration mode" in str(exc)
    else:
        raise AssertionError("invalid calibration mode was accepted")
