from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .stage1_spectral_unet import _num_groups


class TCNResidualBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("TCN kernel_size must be odd to preserve sequence length cleanly.")

        padding = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(_num_groups(channels), channels)
        self.act1 = nn.SiLU(inplace=True)
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(_num_groups(channels), channels)
        self.act2 = nn.SiLU(inplace=True)
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.drop2(x)
        x = x + residual
        return self.act2(x)


class SpectralTCN1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(base_channels), base_channels),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                TCNResidualBlock1D(
                    channels=base_channels,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.head = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(base_channels), base_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(base_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)
