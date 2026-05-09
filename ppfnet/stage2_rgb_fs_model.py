from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .stage1_spectral_unet import DownBlock1D, ResidualConvBlock1D, SpectralUNet1D, UpBlock1D


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SmallRGBEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        global_embed_dim: int = 64,
        local_cond_channels: int = 16,
        use_local_rgb_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.use_local_rgb_conditioning = bool(use_local_rgb_conditioning)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_num_groups(32), 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_num_groups(48), 48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_num_groups(64), 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_num_groups(96), 96),
            nn.SiLU(inplace=True),
        )
        self.local_proj = (
            nn.Conv2d(96, local_cond_channels, kernel_size=1)
            if self.use_local_rgb_conditioning and local_cond_channels > 0
            else None
        )
        self.global_proj = nn.Linear(96, global_embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        feature_map = self.stem(x)
        local_feature_map = self.local_proj(feature_map) if self.local_proj is not None else None
        global_feature = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)
        global_embedding = self.global_proj(global_feature)
        return local_feature_map, global_embedding


def sample_local_rgb_features(
    feature_map: torch.Tensor,
    coords_xy_norm: torch.Tensor,
    pixel_to_sample_index: torch.Tensor,
) -> torch.Tensor:
    num_pixels = int(coords_xy_norm.shape[0])
    channels = int(feature_map.shape[1])
    sampled = torch.zeros((num_pixels, channels), device=feature_map.device, dtype=feature_map.dtype)

    if num_pixels == 0:
        return sampled

    unique_samples = torch.unique(pixel_to_sample_index, sorted=True)
    for sample_idx in unique_samples.tolist():
        selector = pixel_to_sample_index == sample_idx
        coords = coords_xy_norm[selector].to(dtype=feature_map.dtype)
        if coords.numel() == 0:
            continue

        grid = coords.clone()
        grid = grid * 2.0 - 1.0
        grid = grid.view(1, -1, 1, 2)
        sampled_map = F.grid_sample(
            feature_map[sample_idx:sample_idx + 1],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled_features = sampled_map.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()
        sampled_features = sampled_features.to(dtype=feature_map.dtype)
        sampled[selector] = sampled_features

    return sampled


class FiLM1D(nn.Module):
    def __init__(self, cond_dim: int, num_channels: int) -> None:
        super().__init__()
        hidden_dim = max(cond_dim, num_channels)
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, num_channels * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond = cond.to(dtype=x.dtype)
        scale_shift = self.net(cond)
        scale, shift = scale_shift.chunk(2, dim=1)
        return x * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)


class FiLMConditionedSpectralUNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4),
        dropout: float = 0.0,
        global_cond_dim: int = 16,
        local_cond_dim: int = 34,
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
        self.bottleneck_film = FiLM1D(global_cond_dim, bottleneck_channels)

        decoder_specs = list(reversed(encoder_channels))
        current_channels = bottleneck_channels
        self.up_blocks = nn.ModuleList()
        self.decoder_films = nn.ModuleList()
        for skip_channels in decoder_specs:
            out_channels = skip_channels
            self.up_blocks.append(UpBlock1D(current_channels, skip_channels, out_channels, dropout=dropout))
            self.decoder_films.append(FiLM1D(local_cond_dim, out_channels))
            current_channels = out_channels

        self.head = nn.Sequential(
            ResidualConvBlock1D(current_channels, current_channels, dropout=dropout),
            nn.Conv1d(current_channels, 1, kernel_size=1),
        )

    def set_backbone_frozen(self, freeze: bool = True) -> None:
        spectral_modules = [
            self.stem,
            self.down_blocks,
            self.bottleneck,
            self.up_blocks,
            self.head,
        ]
        for module in spectral_modules:
            for parameter in module.parameters():
                parameter.requires_grad = not freeze

    def forward(
        self,
        x: torch.Tensor,
        global_condition: torch.Tensor,
        local_condition: torch.Tensor,
    ) -> torch.Tensor:
        skips = []
        x = self.stem(x)
        skips.append(x)

        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)

        x = self.bottleneck(x)
        x = self.bottleneck_film(x, global_condition)

        for up_block, skip, film in zip(self.up_blocks, reversed(skips), self.decoder_films):
            x = up_block(x, skip)
            x = film(x, local_condition)

        return self.head(x)


class RGBConditionedSpectralStudent(nn.Module):
    def __init__(
        self,
        rgb_in_channels: int = 6,
        rgb_embed_dim: int = 64,
        local_cond_channels: int = 16,
        global_cond_channels: int = 16,
        base_channels: int = 32,
        dropout: float = 0.0,
        use_local_rgb_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.rgb_embed_dim = int(rgb_embed_dim)
        self.rgb_in_channels = int(rgb_in_channels)
        self.local_cond_channels = int(local_cond_channels)
        self.global_cond_channels = int(global_cond_channels)
        self.use_local_rgb_conditioning = bool(use_local_rgb_conditioning)
        self.baseline_channels = 1
        self.rgb_encoder = SmallRGBEncoder(
            in_channels=self.rgb_in_channels,
            global_embed_dim=self.rgb_embed_dim,
            local_cond_channels=self.local_cond_channels,
            use_local_rgb_conditioning=self.use_local_rgb_conditioning,
        )
        self.rgb_to_global_context = nn.Sequential(
            nn.Linear(self.rgb_embed_dim, max(self.rgb_embed_dim // 2, self.global_cond_channels)),
            nn.SiLU(inplace=True),
            nn.Linear(max(self.rgb_embed_dim // 2, self.global_cond_channels), self.global_cond_channels),
        )
        input_channels = 3 + 2 + self.global_cond_channels + self.baseline_channels
        local_condition_dim = 2
        if self.use_local_rgb_conditioning:
            input_channels += self.local_cond_channels
            local_condition_dim += self.local_cond_channels
        self.backbone = FiLMConditionedSpectralUNet1D(
            in_channels=input_channels,
            base_channels=base_channels,
            dropout=dropout,
            global_cond_dim=self.global_cond_channels,
            local_cond_dim=local_condition_dim,
        )

    def forward(
        self,
        masked_model_input: torch.Tensor,
        coords_xy_norm: torch.Tensor,
        rgb_images: torch.Tensor,
        pixel_to_sample_index: torch.Tensor,
        baseline_reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        local_feature_map, rgb_embedding = self.rgb_encoder(rgb_images)
        global_context = self.rgb_to_global_context(rgb_embedding)
        pixel_global_context = global_context[pixel_to_sample_index]
        spectral_length = masked_model_input.shape[-1]
        coords = coords_xy_norm.to(dtype=masked_model_input.dtype)
        global_channels = pixel_global_context.to(dtype=masked_model_input.dtype).unsqueeze(-1).expand(-1, -1, spectral_length)
        coord_channels = coords.unsqueeze(-1).expand(-1, -1, spectral_length)
        local_context = coords
        baseline_channels = baseline_reconstruction.to(dtype=masked_model_input.dtype)
        backbone_inputs = [masked_model_input, coord_channels, global_channels, baseline_channels]

        if self.use_local_rgb_conditioning:
            if local_feature_map is None:
                raise RuntimeError("Local RGB conditioning is enabled but no local feature map was produced.")
            pixel_local_features = sample_local_rgb_features(
                local_feature_map,
                coords_xy_norm=coords_xy_norm,
                pixel_to_sample_index=pixel_to_sample_index,
            )
            pixel_local_features = pixel_local_features.to(dtype=masked_model_input.dtype)
            local_context = torch.cat([pixel_local_features, coords], dim=1)
            local_channels = pixel_local_features.unsqueeze(-1).expand(-1, -1, spectral_length)
            backbone_inputs.insert(2, local_channels)

        backbone_input = torch.cat(backbone_inputs, dim=1)
        residual_prediction = self.backbone(
            backbone_input,
            global_condition=pixel_global_context,
            local_condition=local_context,
        )
        return residual_prediction


def initialize_student_from_teacher(
    student: RGBConditionedSpectralStudent,
    teacher_state_dict: Dict[str, torch.Tensor],
) -> None:
    student_state = student.backbone.state_dict()
    updated_state: Dict[str, torch.Tensor] = {}

    for key, student_tensor in student_state.items():
        if key not in teacher_state_dict:
            updated_state[key] = student_tensor
            continue

        teacher_tensor = teacher_state_dict[key]
        if teacher_tensor.shape == student_tensor.shape:
            updated_state[key] = teacher_tensor
            continue

        if (
            student_tensor.ndim == 3
            and teacher_tensor.ndim == 3
            and student_tensor.shape[0] == teacher_tensor.shape[0]
            and student_tensor.shape[2] == teacher_tensor.shape[2]
            and student_tensor.shape[1] > teacher_tensor.shape[1]
        ):
            merged = student_tensor.clone()
            merged[:, : teacher_tensor.shape[1], :] = teacher_tensor
            updated_state[key] = merged
            continue

        updated_state[key] = student_tensor

    student.backbone.load_state_dict(updated_state)


def freeze_teacher(teacher: SpectralUNet1D) -> None:
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False


def freeze_student_backbone(student: RGBConditionedSpectralStudent, freeze: bool = True) -> None:
    if hasattr(student.backbone, "set_backbone_frozen"):
        student.backbone.set_backbone_frozen(freeze)
        return

    for parameter in student.backbone.parameters():
        parameter.requires_grad = not freeze
