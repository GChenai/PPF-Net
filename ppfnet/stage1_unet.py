from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


MODALITY_MODES = ("joint", "fs_only", "ts_only")
MASK_SHARING_MODES = ("shared", "independent")


def validate_modality_mode(modality_mode: str) -> str:
    if modality_mode not in MODALITY_MODES:
        raise ValueError("Unsupported modality_mode: {0}".format(modality_mode))
    return modality_mode


def validate_mask_sharing(mask_sharing: str) -> str:
    if mask_sharing not in MASK_SHARING_MODES:
        raise ValueError("Unsupported mask_sharing: {0}".format(mask_sharing))
    return mask_sharing


def modality_uses_fs(modality_mode: str) -> bool:
    modality_mode = validate_modality_mode(modality_mode)
    return modality_mode in {"joint", "fs_only"}


def modality_uses_ts(modality_mode: str) -> bool:
    modality_mode = validate_modality_mode(modality_mode)
    return modality_mode in {"joint", "ts_only"}


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, dropout=dropout)
        self.conv2 = ConvNormAct(out_channels, out_channels, dropout=dropout)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.block = ResidualConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = ResidualConvBlock(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        encoder_channels = [base_channels * mult for mult in channel_mults]
        self.stem = ResidualConvBlock(in_channels, encoder_channels[0], dropout=dropout)

        self.down_blocks = nn.ModuleList()
        for in_ch, out_ch in zip(encoder_channels[:-1], encoder_channels[1:]):
            self.down_blocks.append(DownBlock(in_ch, out_ch, dropout=dropout))

        bottleneck_channels = encoder_channels[-1] * 2
        self.bottleneck = nn.Sequential(
            DownBlock(encoder_channels[-1], bottleneck_channels, dropout=dropout),
            ResidualConvBlock(bottleneck_channels, bottleneck_channels, dropout=dropout),
        )

        decoder_specs = list(reversed(encoder_channels))
        current_channels = bottleneck_channels
        self.up_blocks = nn.ModuleList()
        for skip_channels in decoder_specs:
            out_ch = skip_channels
            self.up_blocks.append(
                UpBlock(current_channels, skip_channels, out_ch, dropout=dropout)
            )
            current_channels = out_ch

        self.head = nn.Sequential(
            ResidualConvBlock(current_channels, current_channels, dropout=dropout),
            nn.Conv2d(current_channels, out_channels, kernel_size=1),
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


class Stage1UNet(nn.Module):
    def __init__(
        self,
        fs_channels: int,
        ts_channels: int,
        modality_mode: str = "joint",
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fs_channels = int(fs_channels)
        self.ts_channels = int(ts_channels)
        self.modality_mode = validate_modality_mode(modality_mode)

        input_channels = 0
        output_channels = 0
        if modality_uses_fs(self.modality_mode):
            input_channels += self.fs_channels + 2
            output_channels += self.fs_channels
        if modality_uses_ts(self.modality_mode):
            input_channels += self.ts_channels + 2
            output_channels += self.ts_channels

        self.unet = UNet2D(
            in_channels=input_channels,
            out_channels=output_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            dropout=dropout,
        )

    def forward(
        self,
        masked_fs: Optional[torch.Tensor] = None,
        masked_ts: Optional[torch.Tensor] = None,
        fs_observed_mask: Optional[torch.Tensor] = None,
        ts_observed_mask: Optional[torch.Tensor] = None,
        fs_valid_mask: Optional[torch.Tensor] = None,
        ts_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = []

        if modality_uses_fs(self.modality_mode):
            if masked_fs is None or fs_observed_mask is None:
                raise ValueError("FS tensors are required for modality_mode={0}".format(self.modality_mode))
            if fs_valid_mask is None:
                fs_valid_mask = (fs_observed_mask > 0).to(dtype=masked_fs.dtype)
            inputs.extend([masked_fs, fs_observed_mask, fs_valid_mask])

        if modality_uses_ts(self.modality_mode):
            if masked_ts is None or ts_observed_mask is None:
                raise ValueError("TS tensors are required for modality_mode={0}".format(self.modality_mode))
            if ts_valid_mask is None:
                ts_valid_mask = (ts_observed_mask > 0).to(dtype=masked_ts.dtype)
            inputs.extend([masked_ts, ts_observed_mask, ts_valid_mask])

        x = torch.cat(inputs, dim=1)
        prediction = self.unet(x)

        pred_fs: Optional[torch.Tensor] = None
        pred_ts: Optional[torch.Tensor] = None
        channel_offset = 0
        if modality_uses_fs(self.modality_mode):
            pred_fs = prediction[:, channel_offset:channel_offset + self.fs_channels]
            channel_offset += self.fs_channels
        if modality_uses_ts(self.modality_mode):
            pred_ts = prediction[:, channel_offset:channel_offset + self.ts_channels]

        return pred_fs, pred_ts


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


def sample_observation_mask(
    valid_mask: torch.Tensor,
    mode: str = "hybrid",
    min_observed_ratio: float = 0.45,
    max_observed_ratio: float = 0.85,
    max_rectangles: int = 3,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if valid_mask.ndim != 4 or valid_mask.shape[1] != 1:
        raise ValueError("valid_mask must have shape [B, 1, H, W].")

    observed = valid_mask.clone().to(dtype=torch.float32)
    _, _, height, width = observed.shape
    device = observed.device

    for batch_idx in range(observed.shape[0]):
        valid = valid_mask[batch_idx, 0] > 0.5
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            observed[batch_idx, 0].zero_()
            continue

        keep = valid.clone()
        target_ratio = _random_ratio(min_observed_ratio, max_observed_ratio, device=device, generator=generator)

        if mode in {"block", "hybrid"}:
            rectangle_count = _random_int(1, max_rectangles, device=device, generator=generator)
            for _ in range(rectangle_count):
                rect_h = _random_int(max(1, height // 5), max(1, int(height * 0.6)), device=device, generator=generator)
                rect_w = _random_int(max(1, width // 5), max(1, int(width * 0.6)), device=device, generator=generator)
                top = _random_int(0, max(0, height - rect_h), device=device, generator=generator)
                left = _random_int(0, max(0, width - rect_w), device=device, generator=generator)
                keep[top:top + rect_h, left:left + rect_w] &= False

        current_keep_indices = torch.nonzero(keep & valid, as_tuple=False)
        desired_keep = max(1, int(round(valid_count * target_ratio)))

        if mode == "pixel":
            source_indices = torch.nonzero(valid, as_tuple=False)
            keep = torch.zeros_like(valid)
            perm = torch.randperm(int(source_indices.shape[0]), device=device, generator=generator)
            chosen = source_indices[perm[:desired_keep]]
            keep[chosen[:, 0], chosen[:, 1]] = True
        elif int(current_keep_indices.shape[0]) > desired_keep:
            perm = torch.randperm(int(current_keep_indices.shape[0]), device=device, generator=generator)
            chosen = current_keep_indices[perm[:desired_keep]]
            keep = torch.zeros_like(valid)
            keep[chosen[:, 0], chosen[:, 1]] = True

        if int((keep & valid).sum().item()) == valid_count:
            current_valid_indices = torch.nonzero(valid, as_tuple=False)
            random_index = current_valid_indices[
                _random_int(0, valid_count - 1, device=device, generator=generator)
            ]
            keep[random_index[0], random_index[1]] = False

        observed[batch_idx, 0] = keep.to(dtype=torch.float32)

    return observed


def _sample_single_modality_mask(
    valid_mask: Optional[torch.Tensor],
    mode: str,
    min_observed_ratio: float,
    max_observed_ratio: float,
    generator: Optional[torch.Generator],
) -> Optional[torch.Tensor]:
    if valid_mask is None:
        return None
    return sample_observation_mask(
        valid_mask.to(dtype=torch.float32),
        mode=mode,
        min_observed_ratio=min_observed_ratio,
        max_observed_ratio=max_observed_ratio,
        generator=generator,
    )


@dataclass
class ReconstructionLossOutput:
    loss: torch.Tensor
    fs_l1: torch.Tensor
    ts_l1: torch.Tensor
    fs_l2: torch.Tensor
    ts_l2: torch.Tensor


def masked_reconstruction_loss(
    pred_fs: Optional[torch.Tensor],
    pred_ts: Optional[torch.Tensor],
    target_fs: Optional[torch.Tensor],
    target_ts: Optional[torch.Tensor],
    fs_missing_mask: Optional[torch.Tensor],
    ts_missing_mask: Optional[torch.Tensor],
    l2_weight: float = 0.1,
) -> ReconstructionLossOutput:
    def infer_device() -> torch.device:
        for tensor in (pred_fs, pred_ts, target_fs, target_ts, fs_missing_mask, ts_missing_mask):
            if tensor is not None:
                return tensor.device
        return torch.device("cpu")

    def compute_pair(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        denom = mask.sum().clamp_min(1.0)
        l1 = (pred - target).abs().mul(mask).sum() / denom
        l2 = (pred - target).pow(2).mul(mask).sum() / denom
        return l1, l2

    device = infer_device()
    zero = torch.tensor(0.0, device=device)

    if pred_fs is not None and target_fs is not None and fs_missing_mask is not None:
        fs_l1, fs_l2 = compute_pair(pred_fs, target_fs, fs_missing_mask)
    else:
        fs_l1, fs_l2 = zero, zero

    if pred_ts is not None and target_ts is not None and ts_missing_mask is not None:
        ts_l1, ts_l2 = compute_pair(pred_ts, target_ts, ts_missing_mask)
    else:
        ts_l1, ts_l2 = zero, zero

    total = fs_l1 + ts_l1 + l2_weight * (fs_l2 + ts_l2)
    return ReconstructionLossOutput(
        loss=total,
        fs_l1=fs_l1.detach(),
        ts_l1=ts_l1.detach(),
        fs_l2=fs_l2.detach(),
        ts_l2=ts_l2.detach(),
    )


def build_masked_inputs(
    batch: Dict[str, torch.Tensor],
    modality_mode: str = "joint",
    mask_mode: str = "hybrid",
    mask_sharing: str = "shared",
    min_observed_ratio: float = 0.45,
    max_observed_ratio: float = 0.85,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    modality_mode = validate_modality_mode(modality_mode)
    mask_sharing = validate_mask_sharing(mask_sharing)

    fs_valid_mask = batch["fs_valid_mask"] if modality_uses_fs(modality_mode) else None
    ts_valid_mask = batch["ts_valid_mask"] if modality_uses_ts(modality_mode) else None

    fs_observed_mask: Optional[torch.Tensor]
    ts_observed_mask: Optional[torch.Tensor]

    if modality_mode == "joint" and mask_sharing == "shared":
        spatial_support = (fs_valid_mask > 0.5) | (ts_valid_mask > 0.5)
        observed = sample_observation_mask(
            spatial_support.to(dtype=torch.float32),
            mode=mask_mode,
            min_observed_ratio=min_observed_ratio,
            max_observed_ratio=max_observed_ratio,
            generator=generator,
        )
        fs_observed_mask = observed * fs_valid_mask
        ts_observed_mask = observed * ts_valid_mask
    else:
        fs_observed_mask = _sample_single_modality_mask(
            fs_valid_mask,
            mode=mask_mode,
            min_observed_ratio=min_observed_ratio,
            max_observed_ratio=max_observed_ratio,
            generator=generator,
        )
        ts_observed_mask = _sample_single_modality_mask(
            ts_valid_mask,
            mode=mask_mode,
            min_observed_ratio=min_observed_ratio,
            max_observed_ratio=max_observed_ratio,
            generator=generator,
        )

    masked_fs = batch["fs_features"] * fs_observed_mask if fs_observed_mask is not None else None
    masked_ts = batch["ts_features"] * ts_observed_mask if ts_observed_mask is not None else None

    fs_missing_mask = fs_valid_mask * (1.0 - fs_observed_mask) if fs_observed_mask is not None else None
    ts_missing_mask = ts_valid_mask * (1.0 - ts_observed_mask) if ts_observed_mask is not None else None

    return {
        "masked_fs": masked_fs,
        "masked_ts": masked_ts,
        "fs_observed_mask": fs_observed_mask,
        "ts_observed_mask": ts_observed_mask,
        "fs_missing_mask": fs_missing_mask,
        "ts_missing_mask": ts_missing_mask,
    }
