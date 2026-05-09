from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .stage1_spectral_unet import SpectralUNet1D
from .stage2_rgb_fs_model import SmallRGBEncoder


class PatchContextResidualStudent(nn.Module):
    def __init__(
        self,
        rgb_in_channels: int = 6,
        rgb_embed_dim: int = 64,
        cond_channels: int = 16,
        base_channels: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rgb_in_channels = int(rgb_in_channels)
        self.rgb_embed_dim = int(rgb_embed_dim)
        self.cond_channels = int(cond_channels)

        self.rgb_encoder = SmallRGBEncoder(
            in_channels=self.rgb_in_channels,
            global_embed_dim=self.rgb_embed_dim,
            local_cond_channels=0,
            use_local_rgb_conditioning=False,
        )
        self.rgb_to_cond = nn.Sequential(
            nn.Linear(self.rgb_embed_dim, max(self.rgb_embed_dim // 2, self.cond_channels)),
            nn.SiLU(inplace=True),
            nn.Linear(max(self.rgb_embed_dim // 2, self.cond_channels), self.cond_channels),
        )

        # 3 masked spectral channels + patch mean + patch std + baseline + coords(2) + valid_ratio + rgb cond
        in_channels = 3 + 1 + 1 + 1 + 2 + 1 + self.cond_channels
        self.backbone = SpectralUNet1D(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
        )

    def forward(
        self,
        masked_model_input: torch.Tensor,
        patch_mean: torch.Tensor,
        patch_std: torch.Tensor,
        patch_valid_ratio: torch.Tensor,
        coords_xy_norm: torch.Tensor,
        rgb_patch: torch.Tensor,
        baseline_reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        _, rgb_embedding = self.rgb_encoder(rgb_patch)
        spectral_length = masked_model_input.shape[-1]

        rgb_cond = self.rgb_to_cond(rgb_embedding).to(dtype=masked_model_input.dtype)
        rgb_cond = rgb_cond.unsqueeze(-1).expand(-1, -1, spectral_length)
        coord_channels = coords_xy_norm.to(dtype=masked_model_input.dtype).unsqueeze(-1).expand(-1, -1, spectral_length)
        valid_ratio_channel = patch_valid_ratio.to(dtype=masked_model_input.dtype).unsqueeze(-1).expand(-1, 1, spectral_length)

        x = torch.cat(
            [
                masked_model_input,
                patch_mean.to(dtype=masked_model_input.dtype),
                patch_std.to(dtype=masked_model_input.dtype),
                baseline_reconstruction.to(dtype=masked_model_input.dtype),
                coord_channels,
                valid_ratio_channel,
                rgb_cond,
            ],
            dim=1,
        )
        return self.backbone(x)


def initialize_patch_student_from_teacher(
    student: PatchContextResidualStudent,
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
            merged.zero_()
            merged[:, : teacher_tensor.shape[1], :] = teacher_tensor
            updated_state[key] = merged
            continue

        updated_state[key] = student_tensor

    student.backbone.load_state_dict(updated_state)


def freeze_patch_student_backbone(student: PatchContextResidualStudent, freeze: bool = True) -> None:
    for parameter in student.backbone.parameters():
        parameter.requires_grad = not freeze
