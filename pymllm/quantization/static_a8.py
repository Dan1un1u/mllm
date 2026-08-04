"""Deployment-shaped static A8 calibration and fake-quantization helpers.

This module deliberately has no dependency on ``pymllm.mobile``.  It is the
P0 software oracle for the Qualcomm path:

* weights stay unrotated and are quantized as LPBQ W4 G32;
* activations use a static, per-tensor affine A8 quantizer;
* the zero-point is an integer deployment parameter and is not optimized;
* A8 zero-point is an integer deployment parameter; the A8 prototype learns
  only scale/clipping, while an optional block trainer can also learn the
  LPBQ G32 weight scales with fixed int4 codes.

The representation follows ``ActivationQDQ`` in the Qualcomm transformer
path: unsigned integer range ``[0, 255]`` and dequantization
``(q - zero_point) * scale``.  ``lpbq_quantize_g32`` mirrors the public G32
contract used by ``scripts/export_qwen3_lpbq_g32.py`` while keeping the OI
layout needed by ``torch.nn.functional.linear``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Iterable, Literal

import torch
from torch import nn
from torch.nn import functional as F


A8Method = Literal["max_min", "mean_3sigma", "percentile", "learnable"]
A8_QMIN = 0
A8_QMAX = 255
LPBQ_EPS = 0.0001 / 65535.0


@dataclass(frozen=True)
class A8Params:
    """Serializable static affine A8 parameters.

    ``clip_min`` and ``clip_max`` are the *effective* representable range
    after integer zero-point rounding.  Keeping those values in the manifest
    makes saturation and deployment comparisons auditable.
    """

    method: str
    scale: float
    zero_point: int
    clip_min: float
    clip_max: float
    quant_min: int = A8_QMIN
    quant_max: int = A8_QMAX
    percentile: float | None = None
    warm_start_method: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def freeze_a8_zero_point(params: A8Params, zero_point: int) -> A8Params:
    """Return ``params`` with a deployment-fixed integer zero-point.

    The scale is intentionally left unchanged: this helper is for comparing
    fixed-zp policies while optimizing only scale/clipping.  The effective
    representable range is recomputed so export metadata remains truthful.
    """

    if not params.quant_min <= int(zero_point) <= params.quant_max:
        raise ValueError(
            f"zero_point={zero_point} outside "
            f"[{params.quant_min}, {params.quant_max}]"
        )
    zero_point = int(zero_point)
    return replace(
        params,
        zero_point=zero_point,
        clip_min=(params.quant_min - zero_point) * params.scale,
        clip_max=(params.quant_max - zero_point) * params.scale,
    )


@dataclass
class LPBQWeights:
    """OI-layout LPBQ tensors plus a float32 decode used by the oracle."""

    codes: torch.Tensor
    scale1: torch.Tensor
    scale2: torch.Tensor
    decoded: torch.Tensor
    group_size: int


def pack_lpbq_codes_hwio(codes: torch.Tensor) -> torch.Tensor:
    """Pack logical signed OI INT4 codes into the QNN HWIO carrier."""

    if codes.ndim != 2:
        raise ValueError(f"expected OI codes [out, in], got {tuple(codes.shape)}")
    out_features, in_features = map(int, codes.shape)
    signed = codes.to(dtype=torch.int8).transpose(0, 1).contiguous()
    carrier = torch.bitwise_and(signed, 0x0F)
    return carrier.reshape(1, 1, in_features, out_features).contiguous()


def _finite_values(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected a torch.Tensor, got {type(x).__name__}")
    if not x.is_floating_point():
        x = x.float()
    values = x.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        raise ValueError("activation tensor has no finite values")
    return values


def _range_for_method(
    values: torch.Tensor,
    method: str,
    *,
    percentile: float = 99.9,
) -> tuple[float, float]:
    method = method.lower().replace("-", "_")
    observed_min = float(values.min())
    observed_max = float(values.max())
    if method in {"max_min", "min_max", "maxmin"}:
        low, high = observed_min, observed_max
    elif method in {"mean_3sigma", "mean3sigma", "3sigma"}:
        mean = float(values.mean())
        sigma = float(values.std(unbiased=False))
        low, high = mean - 3.0 * sigma, mean + 3.0 * sigma
    elif method in {"percentile", "percentile_clipping"}:
        if not 0.0 < percentile <= 100.0:
            raise ValueError("percentile must be in (0, 100]")
        tail = (100.0 - percentile) / 2.0
        quantiles = torch.tensor(
            [tail / 100.0, 1.0 - tail / 100.0],
            dtype=values.dtype,
            device=values.device,
        )
        low, high = (float(v) for v in torch.quantile(values, quantiles))
    else:
        raise ValueError(
            f"unknown A8 calibration method {method!r}; expected max_min, "
            "mean_3sigma, percentile, or learnable"
        )

    # An all-zero or constant tensor still needs a finite affine scale.  For
    # a near-constant tensor use the observed range as a safe fallback.
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"non-finite clipping range ({low}, {high})")
    if high <= low:
        center = 0.5 * (low + high)
        radius = max(abs(center) * 1e-3, 1e-5)
        low, high = center - radius, center + radius
    return low, high


def calibrate_a8(
    x: torch.Tensor,
    method: A8Method | str = "max_min",
    *,
    percentile: float = 99.9,
    fixed_zero_point: int | None = None,
    quant_min: int = A8_QMIN,
    quant_max: int = A8_QMAX,
    warm_start_method: str = "percentile",
    warm_start_percentile: float | None = None,
) -> A8Params:
    """Calibrate deployment-shaped static affine A8 parameters.

    For ``learnable`` the returned parameters are only an initialization;
    :class:`LearnableA8FakeQuant` can optimize the scale while retaining the
    returned integer zero-point.  For all fixed methods, the returned scale
    and zero-point are ready for QNN/QDQ export.
    """

    if quant_max <= quant_min:
        raise ValueError("quant_max must be greater than quant_min")
    values = _finite_values(x)
    requested_method = str(method).lower().replace("-", "_")
    if requested_method == "learnable":
        requested_method = "learnable"
        init_method = warm_start_method
        init_percentile = (
            percentile if warm_start_percentile is None else warm_start_percentile
        )
    else:
        init_method = requested_method
        init_percentile = percentile

    low, high = _range_for_method(
        values,
        init_method,
        percentile=init_percentile,
    )
    scale = max((high - low) / float(quant_max - quant_min), LPBQ_EPS)
    if fixed_zero_point is None:
        zero_point = int(round(quant_min - low / scale))
        zero_point = max(quant_min, min(quant_max, zero_point))
    else:
        zero_point = int(fixed_zero_point)
        if not quant_min <= zero_point <= quant_max:
            raise ValueError(
                f"fixed_zero_point={zero_point} outside [{quant_min}, {quant_max}]"
            )

    # Rounding zp changes the actual representable real range.  Record that
    # range rather than the pre-rounded observer range.
    clip_min = (quant_min - zero_point) * scale
    clip_max = (quant_max - zero_point) * scale
    return A8Params(
        method=str(method),
        scale=float(scale),
        zero_point=zero_point,
        clip_min=float(clip_min),
        clip_max=float(clip_max),
        quant_min=quant_min,
        quant_max=quant_max,
        percentile=(float(init_percentile) if init_method == "percentile" else None),
        warm_start_method=(init_method if str(method).lower() == "learnable" else None),
    )


def fake_quantize_a8(
    x: torch.Tensor,
    scale: torch.Tensor | float,
    zero_point: torch.Tensor | int,
    *,
    quant_min: int = A8_QMIN,
    quant_max: int = A8_QMAX,
    ste: bool = False,
) -> torch.Tensor:
    """Apply affine A8 fake quantization.

    With ``ste=True`` this uses the straight-through estimator for rounding
    and clamping.  The estimator is needed only during scale optimization;
    the default is an exact forward oracle.
    """

    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, dtype=x.dtype, device=x.device)
    else:
        scale = scale.to(device=x.device, dtype=x.dtype)
    if not isinstance(zero_point, torch.Tensor):
        zero_point = torch.tensor(zero_point, dtype=x.dtype, device=x.device)
    else:
        zero_point = zero_point.to(device=x.device, dtype=x.dtype)
    scale = scale.clamp_min(torch.finfo(x.dtype).eps)
    q = x / scale + zero_point
    if ste:
        # P0 learns the deployment scale, not the input tensor.  Detaching
        # the integer code gives the scale a useful dequantization gradient
        # (``d(code * scale) / d scale``) instead of the cancellation caused
        # by an identity STE through ``x / scale``.  The exact forward value
        # is still the same round-and-clamp operation.
        rounded = q.round().detach()
        bounded = rounded.clamp(quant_min, quant_max)
    else:
        bounded = q.round().clamp(quant_min, quant_max)
    return (bounded - zero_point) * scale


class LearnableA8FakeQuant(nn.Module):
    """Static A8 fake quantizer with trainable scale and fixed zero-point."""

    def __init__(self, params: A8Params):
        super().__init__()
        if params.scale <= 0.0:
            raise ValueError("initial A8 scale must be positive")
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(params.scale), dtype=torch.float32)
        )
        self.register_buffer(
            "zero_point",
            torch.tensor(params.zero_point, dtype=torch.int32),
        )
        self.quant_min = int(params.quant_min)
        self.quant_max = int(params.quant_max)
        self.initial_params = params

    @property
    def scale(self) -> torch.Tensor:
        # Keep the optimizer away from denormal scales while preserving a
        # smooth positive parameterization.
        return self.log_scale.exp().clamp_min(LPBQ_EPS)

    def effective_clip(self) -> tuple[float, float]:
        scale = float(self.scale.detach())
        zp = int(self.zero_point.detach())
        return (self.quant_min - zp) * scale, (self.quant_max - zp) * scale

    def export_params(self, method: str = "learnable") -> A8Params:
        clip_min, clip_max = self.effective_clip()
        return A8Params(
            method=method,
            scale=float(self.scale.detach()),
            zero_point=int(self.zero_point.detach()),
            clip_min=clip_min,
            clip_max=clip_max,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            percentile=self.initial_params.percentile,
            warm_start_method=self.initial_params.warm_start_method,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fake_quantize_a8(
            x,
            self.scale,
            self.zero_point,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            ste=self.training,
        )


def lpbq_quantize_g32(
    weight: torch.Tensor,
    *,
    group_size: int = 32,
    quant_min: int = -7,
    quant_max: int = 7,
) -> LPBQWeights:
    """Quantize an OI weight matrix with the public LPBQ G32 contract."""

    if weight.ndim != 2:
        raise ValueError(f"expected [out, in] weight, got {tuple(weight.shape)}")
    out_features, in_features = map(int, weight.shape)
    if group_size != 32:
        raise ValueError("P0 only permits the deployment G32 contract")
    if in_features % group_size:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={group_size}"
        )
    w = weight.detach().float().contiguous()
    groups = w.reshape(out_features, in_features // group_size, group_size)
    scale = torch.maximum(groups.amin(-1).abs(), groups.amax(-1).abs())
    scale = (scale / float(quant_max)).clamp_min(LPBQ_EPS)
    # Match the production exporter exactly when torchao is installed.  The
    # fallback keeps this pure-PyTorch oracle usable in minimal CPU test
    # environments, but GPU/WSL runs should take the torchao path.
    try:
        from torchao.quantization.quant_primitives import _quantize_affine

        codes = _quantize_affine(
            w,
            [1, group_size],
            scale,
            torch.zeros_like(scale),
            torch.int32,
            quant_min=quant_min,
            quant_max=quant_max,
        )
    except (ImportError, RuntimeError):
        codes = torch.round(groups / scale.unsqueeze(-1)).clamp(quant_min, quant_max)
    codes = codes.to(torch.int8).reshape(out_features, in_features)

    # QLinearLPBQ's UInt4 scale encoding: scale1 is [O, K/G] and scale2 is
    # one FP32 level-2 scale per output channel.
    scale2 = scale.amax(dim=1).div(16.0).clamp_min(LPBQ_EPS).float()
    scale1 = torch.round(scale / scale2.unsqueeze(1)).clamp(1, 16).to(torch.uint8)
    decoded_groups = (
        codes.reshape(out_features, in_features // group_size, group_size).float()
        * scale1.float().unsqueeze(-1)
        * scale2[:, None, None]
    )
    decoded = decoded_groups.reshape(out_features, in_features).contiguous()
    return LPBQWeights(
        codes=codes,
        scale1=scale1.contiguous(),
        scale2=scale2.contiguous(),
        decoded=decoded,
        group_size=group_size,
    )


def lpbq_rebuild_with_scales(
    base: LPBQWeights,
    scale1: torch.Tensor,
    scale2: torch.Tensor,
) -> LPBQWeights:
    """Rebuild a deployable LPBQ tensor with fixed codes and new scales."""

    if tuple(scale1.shape) != tuple(base.scale1.shape):
        raise ValueError(
            f"scale1 shape {tuple(scale1.shape)} does not match "
            f"{tuple(base.scale1.shape)}"
        )
    if tuple(scale2.shape) != tuple(base.scale2.shape):
        raise ValueError(
            f"scale2 shape {tuple(scale2.shape)} does not match "
            f"{tuple(base.scale2.shape)}"
        )
    scale1 = scale1.to(device=base.codes.device).round().clamp(1, 16).to(torch.uint8)
    scale2 = scale2.to(device=base.codes.device, dtype=torch.float32).clamp_min(LPBQ_EPS)
    out_features, in_features = base.codes.shape
    decoded = (
        base.codes.float().reshape(
            out_features, in_features // base.group_size, base.group_size
        )
        * scale1.float().unsqueeze(-1)
        * scale2[:, None, None]
    ).reshape(out_features, in_features).contiguous()
    return LPBQWeights(
        codes=base.codes.detach().clone(),
        scale1=scale1.contiguous(),
        scale2=scale2.contiguous(),
        decoded=decoded,
        group_size=base.group_size,
    )


class LearnableLPBQScale(nn.Module):
    """Learn deployable LPBQ G32 scales with fixed int4 weight codes.

    ``scale1`` is physically UInt4 in the QNN contract, while ``scale2`` is
    the per-output-channel level-2 scale.  The forward path uses rounded
    ``scale1`` through an STE and a positive continuous ``scale2``.  Therefore
    every training forward already has the same integer scale1 semantics as
    deployment; :meth:`export` emits UInt4 ``scale1`` and FP32 ``scale2``.
    Weight codes never receive gradients.
    """

    def __init__(self, quantized: LPBQWeights):
        super().__init__()
        self.register_buffer("codes", quantized.codes.detach().to(torch.int8))
        self.log_scale1 = nn.Parameter(
            quantized.scale1.detach().float().clamp_min(1.0).log()
        )
        self.log_scale2 = nn.Parameter(
            quantized.scale2.detach().float().clamp_min(LPBQ_EPS).log()
        )
        self.group_size = int(quantized.group_size)

    @property
    def scale1(self) -> torch.Tensor:
        return self.log_scale1.exp().clamp(1.0, 16.0)

    @property
    def scale2(self) -> torch.Tensor:
        return self.log_scale2.exp().clamp_min(LPBQ_EPS)

    def _forward_scale1(self) -> torch.Tensor:
        continuous = self.scale1
        rounded = continuous.round().detach()
        return rounded + continuous - continuous.detach()

    def forward(self) -> torch.Tensor:
        out_features, in_features = self.codes.shape
        groups = self.codes.float().reshape(
            out_features, in_features // self.group_size, self.group_size
        )
        decoded = (
            groups
            * self._forward_scale1().unsqueeze(-1)
            * self.scale2[:, None, None]
        )
        return decoded.reshape(out_features, in_features)

    @torch.no_grad()
    def export(self) -> LPBQWeights:
        scale1 = self.scale1.round().clamp(1, 16).to(torch.uint8).contiguous()
        scale2 = self.scale2.float().contiguous()
        out_features, in_features = self.codes.shape
        decoded = (
            self.codes.float().reshape(
                out_features, in_features // self.group_size, self.group_size
            )
            * scale1.float().unsqueeze(-1)
            * scale2[:, None, None]
        ).reshape(out_features, in_features).contiguous()
        return LPBQWeights(
            codes=self.codes.detach().clone(),
            scale1=scale1,
            scale2=scale2,
            decoded=decoded,
            group_size=self.group_size,
        )


def _nmse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    error = (candidate.float() - reference.float()).square().sum()
    denom = reference.float().square().sum().clamp_min(1e-20)
    return float(error / denom)


def _cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.float().reshape(-1)
    got = candidate.float().reshape(-1)
    value = F.cosine_similarity(ref.unsqueeze(0), got.unsqueeze(0), eps=1e-8)
    return float(value.clamp(-1.0, 1.0))


def linear_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    activation_reference: torch.Tensor | None = None,
    activation_candidate: torch.Tensor | None = None,
) -> dict[str, float]:
    """Return P0 metrics for a linear or block-output slice."""

    result = {
        "output_nmse": _nmse(reference, candidate),
        "output_cosine": _cosine(reference, candidate),
    }
    if activation_reference is not None and activation_candidate is not None:
        result["activation_nmse"] = _nmse(
            activation_reference, activation_candidate
        )
        result["activation_cosine"] = _cosine(
            activation_reference, activation_candidate
        )
    return result


def optimize_linear_input_scale(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    init: A8Params | None = None,
    init_method: str = "percentile",
    init_percentile: float = 99.9,
    steps: int = 200,
    lr: float = 0.03,
    reference_output: torch.Tensor | None = None,
    gradient_clip: float = 1.0,
) -> tuple[A8Params, list[float]]:
    """Learn only a deployable A8 scale for one Linear input tensor.

    The weight is treated as a frozen G32-decoded matrix.  By default the
    target is its W4A16 output, so this measures recovery from activation A8
    alone and does not accidentally optimize against an FP weight that QNN
    cannot deploy.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    x = inputs.float()
    w = weight.float()
    if x.shape[-1] != w.shape[-1]:
        raise ValueError(
            f"input K={x.shape[-1]} does not match weight K={w.shape[-1]}"
        )
    x2 = x.reshape(-1, x.shape[-1])
    if reference_output is None:
        with torch.no_grad():
            reference_output = F.linear(x2, w, bias)
    else:
        reference_output = reference_output.float().reshape_as(
            F.linear(x2, w, bias)
        )
    if init is None:
        init = calibrate_a8(
            x2,
            "learnable",
            warm_start_method=init_method,
            warm_start_percentile=init_percentile,
        )
    quantizer = LearnableA8FakeQuant(init).to(device=x2.device)
    optimizer = torch.optim.Adam(quantizer.parameters(), lr=lr)
    losses: list[float] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        quantized_x = quantizer(x2)
        quantized_output = F.linear(quantized_x, w, bias)
        error = (quantized_output - reference_output).float()
        loss = error.square().mean() / reference_output.float().square().mean().clamp_min(1e-20)
        loss.backward()
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(quantizer.parameters(), gradient_clip)
        optimizer.step()
        loss_value = float(loss.detach())
        losses.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                name: value.detach().clone()
                for name, value in quantizer.state_dict().items()
            }
    if best_state is not None:
        quantizer.load_state_dict(best_state)
    return quantizer.export_params(), losses


def evaluate_a8_candidates(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    eval_inputs: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    percentiles: Iterable[float] = (99.0, 99.9, 99.99),
    learnable_steps: int = 200,
    learnable_lr: float = 0.03,
) -> dict[str, object]:
    """Evaluate all P0 A8 strategies for one frozen G32 LPBQ weight."""

    x = inputs.float().reshape(-1, inputs.shape[-1])
    eval_x = x if eval_inputs is None else eval_inputs.float().reshape(
        -1, eval_inputs.shape[-1]
    )
    if eval_x.shape[-1] != x.shape[-1]:
        raise ValueError("fit and evaluation inputs must have the same K dimension")
    lpbq = lpbq_quantize_g32(weight)
    w4 = lpbq.decoded.to(device=x.device)
    eval_x = eval_x.to(device=x.device)
    b = None if bias is None else bias.float().to(device=x.device)
    with torch.no_grad():
        w4a16 = F.linear(x, w4, b)
        eval_w4a16 = F.linear(eval_x, w4, b)

    results: dict[str, object] = {
        "contract": {
            "rotation": "none",
            "weight": "LPBQ W4 G32",
            "weight_layout": "OI oracle; export HWIO separately",
            "activation": "static affine A8 per tensor",
            "activation_quant_min": A8_QMIN,
            "activation_quant_max": A8_QMAX,
            "zero_point": "fixed integer per candidate",
        },
        "weight": {
            "shape": list(weight.shape),
            "group_size": lpbq.group_size,
            "scale1_shape": list(lpbq.scale1.shape),
            "scale2_shape": list(lpbq.scale2.shape),
            "weight_nmse": _nmse(weight.float(), lpbq.decoded),
        },
        "w4a16": {
            "fit_shape": list(w4a16.shape),
            "eval_shape": list(eval_w4a16.shape),
        },
        "candidates": {},
    }
    candidates: dict[str, object] = results["candidates"]  # type: ignore[assignment]

    def record(name: str, params: A8Params, losses: list[float] | None = None) -> None:
        with torch.no_grad():
            qx = fake_quantize_a8(
                eval_x,
                params.scale,
                params.zero_point,
                quant_min=params.quant_min,
                quant_max=params.quant_max,
            )
            output = F.linear(qx, w4, b)
        entry: dict[str, object] = {
            "params": params.as_dict(),
            "metrics": linear_metrics(
                eval_w4a16,
                output,
                activation_reference=eval_x,
                activation_candidate=qx,
            ),
            "saturation_fraction": float(
                ((eval_x < params.clip_min) | (eval_x > params.clip_max)).float().mean()
            ),
        }
        if losses is not None:
            entry["optimization"] = {
                "steps": len(losses),
                "initial_loss": losses[0],
                "final_loss": losses[-1],
            }
        candidates[name] = entry

    record("max_min", calibrate_a8(x, "max_min"))
    record("mean_3sigma", calibrate_a8(x, "mean_3sigma"))
    for percentile in percentiles:
        name = f"percentile_{float(percentile):g}"
        record(name, calibrate_a8(x, "percentile", percentile=float(percentile)))

    learnable_init = calibrate_a8(
        x,
        "learnable",
        warm_start_method="percentile",
        warm_start_percentile=99.9,
    )
    learned, losses = optimize_linear_input_scale(
        x,
        w4,
        bias=b,
        init=learnable_init,
        steps=learnable_steps,
        lr=learnable_lr,
        reference_output=w4a16,
    )
    record("learnable_percentile_99.9", learned, losses)
    return results
