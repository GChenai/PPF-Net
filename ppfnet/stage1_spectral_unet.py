from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvNormAct1D(in_channels, out_channels, dropout=dropout)
        self.conv2 = ConvNormAct1D(out_channels, out_channels, dropout=dropout)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class DownBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.block = ResidualConvBlock1D(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UpBlock1D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = ResidualConvBlock1D(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class SpectralUNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        encoder_channels = [base_channels * mult for mult in channel_mults]
        self.stem = ResidualConvBlock1D(in_channels, encoder_channels[0], dropout=dropout)

        self.down_blocks = nn.ModuleList()
        for in_ch, out_ch in zip(encoder_channels[:-1], encoder_channels[1:]):
            self.down_blocks.append(DownBlock1D(in_ch, out_ch, dropout=dropout))

        bottleneck_channels = encoder_channels[-1] * 2
        self.bottleneck = nn.Sequential(
            DownBlock1D(encoder_channels[-1], bottleneck_channels, dropout=dropout),
            ResidualConvBlock1D(bottleneck_channels, bottleneck_channels, dropout=dropout),
        )

        decoder_specs = list(reversed(encoder_channels))
        current_channels = bottleneck_channels
        self.up_blocks = nn.ModuleList()
        for skip_channels in decoder_specs:
            out_channels = skip_channels
            self.up_blocks.append(UpBlock1D(current_channels, skip_channels, out_channels, dropout=dropout))
            current_channels = out_channels

        self.head = nn.Sequential(
            ResidualConvBlock1D(current_channels, current_channels, dropout=dropout),
            nn.Conv1d(current_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.stem(x)
        skips.append(x)

        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)

        x = self.bottleneck(x)

        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip)

        return self.head(x)


def _random_ratio(
    low: float,
    high: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> float:
    value = torch.empty(1, device=device)
    value.uniform_(low, high, generator=generator)
    return float(value.item())


def _random_int(
    low_inclusive: int,
    high_inclusive: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> int:
    if high_inclusive <= low_inclusive:
        return int(low_inclusive)
    return int(torch.randint(low_inclusive, high_inclusive + 1, (1,), device=device, generator=generator).item())


def sample_spectral_observation_mask(
    spectral_length: int,
    batch_size: int,
    device: torch.device,
    mode: str = "hybrid",
    min_observed_ratio: float = 0.25,
    max_observed_ratio: float = 0.7,
    max_bands: int = 3,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if mode not in {"point", "band", "hybrid"}:
        raise ValueError("Unsupported spectral mask mode: {0}".format(mode))

    observed = torch.ones((batch_size, 1, spectral_length), device=device, dtype=torch.float32)

    for batch_idx in range(batch_size):
        keep = torch.ones((spectral_length,), device=device, dtype=torch.bool)
        target_ratio = _random_ratio(min_observed_ratio, max_observed_ratio, device=device, generator=generator)
        desired_keep = max(1, int(round(spectral_length * target_ratio)))

        if mode in {"band", "hybrid"}:
            num_bands = _random_int(1, max_bands, device=device, generator=generator)
            for _ in range(num_bands):
                band_width = _random_int(
                    max(1, spectral_length // 12),
                    max(1, spectral_length // 4),
                    device=device,
                    generator=generator,
                )
                start = _random_int(0, max(0, spectral_length - band_width), device=device, generator=generator)
                keep[start:start + band_width] = False

        current_keep = torch.nonzero(keep, as_tuple=False).flatten()
        if mode == "point":
            perm = torch.randperm(spectral_length, device=device, generator=generator)
            keep.zero_()
            keep[perm[:desired_keep]] = True
        elif current_keep.numel() > desired_keep:
            perm = torch.randperm(int(current_keep.numel()), device=device, generator=generator)
            chosen = current_keep[perm[:desired_keep]]
            keep.zero_()
            keep[chosen] = True

        if int(keep.sum().item()) == spectral_length:
            keep[_random_int(0, spectral_length - 1, device=device, generator=generator)] = False

        observed[batch_idx, 0] = keep.to(dtype=torch.float32)

    return observed


def build_spectral_masked_inputs(
    batch: Dict[str, torch.Tensor],
    mask_mode: str = "hybrid",
    min_observed_ratio: float = 0.25,
    max_observed_ratio: float = 0.7,
    use_axis_channel: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    spectrum = batch["spectrum"]
    axis_values = batch["axis_values"]
    batch_size, _, spectral_length = spectrum.shape
    device = spectrum.device

    observed_mask = sample_spectral_observation_mask(
        spectral_length=spectral_length,
        batch_size=batch_size,
        device=device,
        mode=mask_mode,
        min_observed_ratio=min_observed_ratio,
        max_observed_ratio=max_observed_ratio,
        generator=generator,
    )
    missing_mask = 1.0 - observed_mask
    masked_spectrum = spectrum * observed_mask

    if axis_values.ndim == 2:
        axis_values = axis_values.unsqueeze(1)
    axis_min = axis_values.min(dim=-1, keepdim=True).values
    axis_max = axis_values.max(dim=-1, keepdim=True).values
    axis_channel = (axis_values - axis_min) / (axis_max - axis_min).clamp_min(1e-6)

    input_channels = [masked_spectrum, observed_mask]
    if use_axis_channel:
        input_channels.append(axis_channel.to(dtype=spectrum.dtype))
    model_input = torch.cat(input_channels, dim=1)

    return {
        "model_input": model_input,
        "masked_spectrum": masked_spectrum,
        "observed_mask": observed_mask,
        "missing_mask": missing_mask,
        "axis_channel": axis_channel,
    }


@dataclass
class SpectralLossOutput:
    loss: torch.Tensor
    mae: torch.Tensor
    mse: torch.Tensor
    rmse: torch.Tensor


def spectral_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    missing_mask: torch.Tensor,
    l2_weight: float = 0.1,
) -> SpectralLossOutput:
    denom = missing_mask.sum().clamp_min(1.0)
    abs_error = (prediction - target).abs() * missing_mask
    sq_error = (prediction - target).pow(2) * missing_mask

    mae = abs_error.sum() / denom
    mse = sq_error.sum() / denom
    rmse = torch.sqrt(mse.clamp_min(1e-12))
    loss = mae + l2_weight * mse
    return SpectralLossOutput(
        loss=loss,
        mae=mae.detach(),
        mse=mse.detach(),
        rmse=rmse.detach(),
    )
