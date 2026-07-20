#!/usr/bin/env python3
"""Plug-in Attentive Convolution (ATConv) for CausalCLIPSeg decoder/neck."""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

# Vendored from https://github.com/price112/Attentive-Convolution (MIT), AttNet.py


class ATConv(nn.Module):
    """Same-channel 3x3 attentive convolution (adaptive routing + lateral inhibition)."""

    def __init__(self, dim: int, act_layer=nn.GELU, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        k2 = self.kernel_size * self.kernel_size
        self.padding = self.kernel_size // 2

        self.kernel_proj = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=bias)
        self.pool = nn.AdaptiveAvgPool1d(output_size=k2)
        self.kernel_act = act_layer()
        self.kernel_gen = nn.Linear(k2, k2, bias=bias)
        self.x_proj = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=bias)
        self.proj = nn.Conv2d(self.dim, self.dim, 1, bias=bias)
        self.difference_control = nn.Parameter(torch.zeros(self.dim))

    def _generate_kernels(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        kernels = self.kernel_proj(x).view(b, c, h * w)
        kernels = self.pool(kernels)
        kernels = self.kernel_act(kernels)
        kernels = self.kernel_gen(kernels)
        return kernels.reshape(b, c, self.kernel_size, self.kernel_size)

    def _apply_kernel_difference(self, kernels: torch.Tensor) -> torch.Tensor:
        mean_kernels = kernels.mean(dim=(2, 3), keepdim=True)
        factor = torch.sigmoid(self.difference_control).view(1, -1, 1, 1)
        return kernels - factor * mean_kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        kernels = self._apply_kernel_difference(self._generate_kernels(x))
        x = self.x_proj(x).reshape(1, b * c, h, w)
        kernels = kernels.reshape(b * c, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, kernels, padding=self.padding, groups=b * c)
        x = x.reshape(b, c, h, w)
        return self.proj(x)


class ATConvDropIn(nn.Module):
    """Drop-in replacement for same-channel Conv2d inside conv-BN-ReLU blocks."""

    def __init__(self, channels: int):
        super().__init__()
        self.atconv = ATConv(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.atconv(x)


DECODER_PREFIXES = (
    "base_model.neck_ad",
    "base_model.proj",
    "base_model.proj_ad",
)


def _is_same_channel_conv3(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Conv2d)
        and module.kernel_size == (3, 3)
        and module.in_channels == module.out_channels
        and module.groups == 1
        and module.in_channels >= 64
    )


def _in_decoder(name: str, prefixes: Iterable[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def _replacement_priority(name: str) -> tuple[int, str]:
    # Prefer projector + late FPN 3x3 blocks (U-KAN style: last few decoder convs).
    if name.startswith("base_model.proj"):
        return (0, name)
    if name.startswith("base_model.proj_ad"):
        return (1, name)
    if "coordconv" in name:
        return (2, name)
    if "f4_proj" in name or "aggr" in name:
        return (3, name)
    if name.startswith("base_model.neck_ad"):
        return (4, name)
    return (9, name)


def list_atconv_candidates(model: nn.Module, prefixes: Iterable[str] = DECODER_PREFIXES) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if _in_decoder(name, prefixes) and _is_same_channel_conv3(module):
            names.append(name)
    names.sort(key=_replacement_priority)
    return names


def apply_atconv_plugin(
    model: nn.Module,
    num_layers: int = 4,
    prefixes: Iterable[str] = DECODER_PREFIXES,
) -> list[str]:
    """Replace up to `num_layers` same-channel 3x3 Conv2d modules with ATConv."""
    if num_layers <= 0:
        return []
    candidates = list_atconv_candidates(model, prefixes=prefixes)[:num_layers]
    replaced: list[str] = []
    for name in candidates:
        module = model.get_submodule(name)
        if not _is_same_channel_conv3(module):
            continue
        drop_in = ATConvDropIn(module.in_channels)
        _set_submodule(model, name, drop_in)
        replaced.append(name)
    return replaced


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    if isinstance(parent, nn.Sequential):
        idx = int(parts[-1])
        parent[idx] = module
    else:
        setattr(parent, parts[-1], module)


def atconv_meta(replaced: list[str]) -> dict:
    return {
        "conv_plugin": "atconv" if replaced else "standard",
        "atconv_layers": len(replaced),
        "atconv_targets": "|".join(replaced),
    }
