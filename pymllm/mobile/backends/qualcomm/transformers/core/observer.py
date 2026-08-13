import torch
import torch.nn as nn
from typing import Tuple


# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
class PerBlockParamObserver(nn.Module):
    def __init__(
        self,
        dtype: torch.dtype,
        block_size: torch.Size,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,  # noqa: B008
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        self.block_size = block_size
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps
        # TODO: expand this when QNN starts to support more configurations
        self.bitwidth_of_scale = 4
        self.num_steps = 2**self.bitwidth_of_scale
        self.calibrated = False

    def forward(self, input: torch.Tensor):
        if input.numel() == 0 or self.calibrated:
            return input

        input_detached = input.detach()
        self.original_dtype = input_detached.dtype
        if len(self.block_size) != 2 or self.block_size[0] != 1:
            raise ValueError(f"Unsupported LPBQ block shape: {self.block_size}")
        block = self.block_size[1]
        if input_detached.ndim != 2 or input_detached.shape[1] % block != 0:
            raise ValueError(
                f"Expected [out, in] weight divisible by block {block}, "
                f"got {tuple(input_detached.shape)}"
            )
        blocked = input_detached.reshape(input_detached.shape[0], -1, block)
        min_val = torch.amin(blocked, dim=-1)
        max_val = torch.amax(blocked, dim=-1)
        if not hasattr(self, "min_val") or not hasattr(self, "max_val"):
            self.min_val = min_val
            self.max_val = max_val
        else:
            assert self.min_val.shape == min_val.shape, (
                f"Can't update existing min_val - shape mismatch, self.min_val:{self.min_val.shape} != min_val:{min_val.shape}"
            )
            assert self.max_val.shape == max_val.shape, (
                f"Can't update existing max_val - shape mismatch, self.max_val {self.max_val.shape} != max_val:{max_val.shape}"
            )
            min_val = torch.min(self.min_val, min_val)
            max_val = torch.max(self.max_val, max_val)
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)

        self.calibrated = True
        return input

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert hasattr(self, "min_val") and hasattr(self, "max_val"), (
            "Expecting the observer has min_val and max_val, please run the observer before calling calculate_qparams"
        )
        denominator = float(max(abs(self.quant_min), abs(self.quant_max)))
        scale = torch.maximum(
            torch.maximum(self.min_val.abs(), self.max_val.abs()) / denominator,
            torch.as_tensor(self.eps, device=self.min_val.device),
        )
        return scale.to(torch.float32), torch.zeros_like(scale, dtype=torch.int32)


class PerBlockParamFakeQuantize(nn.Module):
    def __init__(
        self,
        dtype: torch.dtype = torch.int8,
        block_size: torch.Size = None,
        quant_min: int = None,
        quant_max: int = None,
        eps: float = torch.finfo(torch.float32).eps,  # noqa: B008
        **kwargs,
    ):
        super().__init__()
        assert block_size is not None, (
            "block_size must be provided for per-block quantization"
        )

        self.activation_post_process = PerBlockParamObserver(
            dtype=dtype,
            block_size=block_size,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            **kwargs,
        )
        self.dtype = dtype
        self.block_size = block_size
        self.quant_min = quant_min if quant_min is not None else torch.iinfo(dtype).min
        self.quant_max = quant_max if quant_max is not None else torch.iinfo(dtype).max
        self.eps = eps
        self.observer_enabled = True
        self.fake_quant_enabled = True

    def enable_observer(self):
        self.observer_enabled = True

    def disable_observer(self):
        self.observer_enabled = False

    def enable_fake_quant(self):
        self.fake_quant_enabled = True

    def disable_fake_quant(self):
        self.fake_quant_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x

        if self.observer_enabled:
            self.activation_post_process(x)
        scale, zero_point = self.activation_post_process.calculate_qparams()
        if not self.fake_quant_enabled:
            return x
        block = self.block_size[1]
        blocked = x.reshape(x.shape[0], x.shape[1] // block, block)
        quantized = torch.round(blocked / scale.unsqueeze(-1) + zero_point.unsqueeze(-1))
        quantized = quantized.clamp(self.quant_min, self.quant_max)
        return ((quantized - zero_point.unsqueeze(-1)) * scale.unsqueeze(-1)).reshape_as(x)

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.activation_post_process.calculate_qparams()

    def convert(self, model, observer_node):
        self.activation_post_process.convert(model, observer_node)


class ConcatObserver(nn.Module):
    """
    Fetch maximum data range of all tensors to be concatenated
    """

    def __init__(
        self,
        dtype=torch.uint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,  # noqa: B008
        is_dynamic=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = 0 if quant_min is None else quant_min
        self.quant_max = torch.iinfo(dtype).max if quant_max is None else quant_max
        self.eps = eps

        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        # get concat node and its inputs
        self.input_observers = []

    def add_observer(self, observer):
        self.input_observers.append(observer)

    def forward(self, x_orig):
        # calculate the min / max first
        self.min_val = min(self.min_val, x_orig.min())
        self.max_val = max(self.max_val, x_orig.max())

        # update min / max for all observers of input nodes
        for observers in self.input_observers:
            observers.min_val = self.min_val
            observers.max_val = self.max_val

        return x_orig

    def calculate_qparams(self):
        min_val = torch.minimum(self.min_val, torch.zeros_like(self.min_val))
        max_val = torch.maximum(self.max_val, torch.zeros_like(self.max_val))
        scale = torch.maximum(
            (max_val - min_val) / float(self.quant_max - self.quant_min),
            torch.as_tensor(self.eps, device=min_val.device),
        )
        zero_point = self.quant_min - torch.round(min_val / scale)
        zero_point = zero_point.clamp(self.quant_min, self.quant_max).to(torch.int32)
        return scale.reshape(1).to(torch.float32), zero_point.reshape(1)
