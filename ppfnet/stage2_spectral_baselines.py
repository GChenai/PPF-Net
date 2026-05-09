from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .stage1_spectral_unet import _num_groups


def _validate_kernel_size(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    return kernel_size


class SRCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        bottleneck_channels: int = 32,
        kernel_size_large: int = 9,
        kernel_size_mid: int = 5,
        kernel_size_small: int = 5,
    ) -> None:
        super().__init__()
        kernel_size_large = _validate_kernel_size(kernel_size_large)
        kernel_size_mid = _validate_kernel_size(kernel_size_mid)
        kernel_size_small = _validate_kernel_size(kernel_size_small)

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size_large, padding=kernel_size_large // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, bottleneck_channels, kernel_size=kernel_size_mid, padding=kernel_size_mid // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck_channels, 1, kernel_size=kernel_size_small, padding=kernel_size_small // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DnCNNBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DnCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        depth: int = 8,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        kernel_size = _validate_kernel_size(kernel_size)
        padding = kernel_size // 2
        if depth < 2:
            raise ValueError("depth must be at least 2.")

        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.append(DnCNNBlock1D(hidden_channels, kernel_size))
        layers.append(nn.Conv1d(hidden_channels, 1, kernel_size=kernel_size, padding=padding))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EDSRResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int, res_scale: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.res_scale = float(res_scale)
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x) * self.res_scale


class EDSR1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 8,
        kernel_size: int = 3,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        kernel_size = _validate_kernel_size(kernel_size)
        padding = kernel_size // 2
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")

        self.head = nn.Conv1d(in_channels, base_channels, kernel_size=kernel_size, padding=padding)
        self.body = nn.Sequential(
            *[
                EDSRResidualBlock1D(base_channels, kernel_size=kernel_size, res_scale=res_scale)
                for _ in range(num_blocks)
            ]
        )
        self.body_tail = nn.Conv1d(base_channels, base_channels, kernel_size=kernel_size, padding=padding)
        self.tail = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(base_channels), base_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.head(x)
        residual = self.body_tail(self.body(features))
        return self.tail(features + residual)


def build_stage2_spectral_baseline(
    model_family: str,
    in_channels: int = 3,
    base_channels: int = 64,
    kernel_size: int = 3,
    num_blocks: int = 8,
    srcnn_bottleneck_channels: int = 32,
    edsr_res_scale: float = 0.1,
) -> nn.Module:
    family = str(model_family).lower()
    if family == "srcnn":
        return SRCNN1D(
            in_channels=in_channels,
            hidden_channels=base_channels,
            bottleneck_channels=srcnn_bottleneck_channels,
            kernel_size_large=max(kernel_size, 9) if max(kernel_size, 9) % 2 == 1 else max(kernel_size, 9) + 1,
            kernel_size_mid=kernel_size,
            kernel_size_small=kernel_size,
        )
    if family == "dncnn":
        return DnCNN1D(
            in_channels=in_channels,
            hidden_channels=base_channels,
            depth=num_blocks,
            kernel_size=kernel_size,
        )
    if family == "edsr":
        return EDSR1D(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            res_scale=edsr_res_scale,
        )
    raise ValueError("Unsupported stage2 spectral baseline family: {0}".format(model_family))


def supported_stage2_spectral_baselines() -> Sequence[str]:
    return ("srcnn", "dncnn", "edsr")
