from __future__ import annotations

import torch

from pymllm.quantization.static_a8 import (
    LearnableLPBQScale,
    LearnableA8FakeQuant,
    calibrate_a8,
    fake_quantize_a8,
    freeze_a8_zero_point,
    lpbq_quantize_g32,
    lpbq_rebuild_with_scales,
    optimize_linear_input_scale,
)


def test_affine_a8_roundtrip_uses_fixed_integer_zero_point() -> None:
    values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 3.0])
    params = calibrate_a8(values, "max_min")
    assert params.quant_min == 0
    assert params.quant_max == 255
    assert isinstance(params.zero_point, int)
    decoded = fake_quantize_a8(values, params.scale, params.zero_point)
    assert torch.isfinite(decoded).all()
    assert decoded.min() >= params.clip_min - params.scale
    assert decoded.max() <= params.clip_max + params.scale


def test_lpbq_g32_shape_and_decode_contract() -> None:
    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(64, 64, generator=generator)
    quantized = lpbq_quantize_g32(weight)
    assert quantized.codes.shape == weight.shape
    assert quantized.scale1.shape == (64, 2)
    assert quantized.scale2.shape == (64,)
    assert quantized.scale1.dtype == torch.uint8
    assert torch.all((quantized.codes >= -7) & (quantized.codes <= 7))
    assert torch.isfinite(quantized.decoded).all()


def test_lpbq_matches_exporter_primitive_when_torchao_is_available() -> None:
    try:
        from torchao.quantization.quant_primitives import _quantize_affine
    except (ImportError, RuntimeError):
        return
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn(64, 64, generator=generator)
    quantized = lpbq_quantize_g32(weight)
    groups = weight.reshape(64, 2, 32).float()
    scale = torch.maximum(groups.amin(-1).abs(), groups.amax(-1).abs())
    scale = (scale / 7.0).clamp_min(0.0001 / 65535.0)
    expected = _quantize_affine(
        weight.float(),
        [1, 32],
        scale,
        torch.zeros_like(scale),
        torch.int32,
        quant_min=-7,
        quant_max=7,
    ).to(torch.int8)
    assert torch.equal(quantized.codes, expected)


def test_lpbq_rebuild_preserves_codes_and_clamps_scale1() -> None:
    generator = torch.Generator().manual_seed(23)
    weight = torch.randn(64, 64, generator=generator)
    base = lpbq_quantize_g32(weight)
    rebuilt = lpbq_rebuild_with_scales(
        base,
        base.scale1.float() + 0.49,
        base.scale2 * 1.25,
    )
    assert torch.equal(rebuilt.codes, base.codes)
    assert rebuilt.scale1.dtype == torch.uint8
    assert torch.all((rebuilt.scale1 >= 1) & (rebuilt.scale1 <= 16))
    assert torch.isfinite(rebuilt.decoded).all()


def test_learnable_scale_keeps_zero_point_fixed() -> None:
    values = torch.randn(128, 32)
    params = calibrate_a8(values, "learnable")
    module = LearnableA8FakeQuant(params)
    original_zero_point = int(module.zero_point)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.05)
    for _ in range(5):
        optimizer.zero_grad()
        loss = (module(values) - values).square().mean()
        loss.backward()
        optimizer.step()
    assert int(module.zero_point) == original_zero_point
    assert float(module.scale.detach()) > 0.0


def test_freezing_a8_zero_point_recomputes_effective_clip() -> None:
    values = torch.randn(128, 32)
    params = calibrate_a8(values, "learnable")
    frozen = freeze_a8_zero_point(params, 128)
    assert frozen.zero_point == 128
    assert frozen.scale == params.scale
    assert frozen.clip_min == (params.quant_min - 128) * params.scale
    assert frozen.clip_max == (params.quant_max - 128) * params.scale


def test_learnable_lpbq_scale_keeps_codes_and_exports_uint4_scales() -> None:
    generator = torch.Generator().manual_seed(31)
    weight = torch.randn(64, 64, generator=generator)
    initial = lpbq_quantize_g32(weight)
    module = LearnableLPBQScale(initial)
    assert torch.allclose(module().detach(), initial.decoded, rtol=1e-6, atol=1e-6)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
    target = weight.float()
    for _ in range(3):
        optimizer.zero_grad()
        loss = (module() - target).square().mean()
        loss.backward()
        optimizer.step()
    exported = module.export()
    assert torch.equal(exported.codes, initial.codes)
    assert exported.scale1.dtype == torch.uint8
    assert torch.all((exported.scale1 >= 1) & (exported.scale1 <= 16))
    assert torch.isfinite(exported.scale2).all()
    assert torch.isfinite(exported.decoded).all()
    assert torch.allclose(
        exported.decoded,
        module().detach(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_linear_optimizer_returns_deployable_scale() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(96, 32, generator=generator)
    weight = torch.randn(48, 32, generator=generator)
    params, losses = optimize_linear_input_scale(
        x,
        lpbq_quantize_g32(weight).decoded,
        steps=4,
        lr=0.01,
    )
    assert len(losses) == 4
    assert params.method == "learnable"
    assert 0 <= params.zero_point <= 255
    assert params.scale > 0.0
