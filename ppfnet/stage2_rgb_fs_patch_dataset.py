from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .stage2_rgb_fs_dataset import _load_rgb_image
from .thz_csv import extract_valid_pixel_spectra, load_thz_csv, resolve_repo_relative_path


def _load_csv_rows(csv_path: Path | str) -> List[Dict[str, str]]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest CSV is empty: {0}".format(path))
    return rows


def _normalize_spectrum(spectrum: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return spectrum
    spectrum = spectrum.astype(np.float32, copy=True)
    if mode == "zscore":
        mean = float(spectrum.mean())
        std = float(spectrum.std())
        if std <= 1e-8:
            std = 1.0
        return (spectrum - mean) / std
    if mode == "minmax":
        min_value = float(spectrum.min())
        max_value = float(spectrum.max())
        if max_value <= min_value:
            max_value = min_value + 1e-6
        return (spectrum - min_value) / (max_value - min_value)
    raise ValueError("Unsupported normalization mode: {0}".format(mode))


def _extract_rgb_patch(
    rgb_tensor: torch.Tensor,
    coord_xy_norm: np.ndarray,
    patch_size: Tuple[int, int],
) -> torch.Tensor:
    channels, height, width = rgb_tensor.shape
    patch_h, patch_w = patch_size
    center_x = int(round(float(coord_xy_norm[0]) * max(width - 1, 1)))
    center_y = int(round(float(coord_xy_norm[1]) * max(height - 1, 1)))

    half_h = patch_h // 2
    half_w = patch_w // 2
    y0 = center_y - half_h
    x0 = center_x - half_w
    y1 = y0 + patch_h
    x1 = x0 + patch_w

    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, y1 - height)
    pad_right = max(0, x1 - width)

    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(height, y1)
    x1 = min(width, x1)

    patch = rgb_tensor[:, y0:y1, x0:x1]
    if pad_top or pad_bottom or pad_left or pad_right:
        patch = torch.nn.functional.pad(
            patch,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=1.0,
        )
    return patch.contiguous()


def _compute_patch_stats(
    cube: np.ndarray,
    valid_mask: np.ndarray,
    center_y: int,
    center_x: int,
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    radius = patch_size // 2
    y0 = max(0, center_y - radius)
    y1 = min(cube.shape[0], center_y + radius + 1)
    x0 = max(0, center_x - radius)
    x1 = min(cube.shape[1], center_x + radius + 1)

    patch = cube[y0:y1, x0:x1]
    patch_mask = valid_mask[y0:y1, x0:x1]
    patch_pixels = patch[patch_mask]
    if patch_pixels.size == 0:
        channels = cube.shape[-1]
        return (
            np.zeros((channels,), dtype=np.float32),
            np.zeros((channels,), dtype=np.float32),
            0.0,
        )

    mean = patch_pixels.mean(axis=0).astype(np.float32)
    std = patch_pixels.std(axis=0).astype(np.float32)
    ratio = float(patch_mask.mean())
    return mean, std, ratio


class Stage2RGBFSPatchDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path | str,
        image_size: Tuple[int, int] = (224, 224),
        rgb_patch_size: Tuple[int, int] = (64, 64),
        thz_patch_size: int = 7,
        normalization: str = "none",
        repo_root: Optional[Path | str] = None,
        max_pixels_per_sample: Optional[int] = None,
        pixel_selection_seed: int = 42,
        include_structure_channels: bool = True,
    ) -> None:
        super().__init__()
        self.manifest_csv = Path(manifest_csv)
        self.repo_root = Path(repo_root) if repo_root is not None else self.manifest_csv.resolve().parents[2]
        self.rows = _load_csv_rows(self.manifest_csv)
        self.image_size = image_size
        self.rgb_patch_size = rgb_patch_size
        self.thz_patch_size = int(thz_patch_size)
        self.normalization = normalization
        self.max_pixels_per_sample = max_pixels_per_sample
        self.pixel_selection_seed = int(pixel_selection_seed)
        self.include_structure_channels = bool(include_structure_channels)

        self.samples: List[Dict[str, object]] = []
        self.index_map: List[Tuple[int, int]] = []
        axis_values_ref: Optional[np.ndarray] = None
        rng = random.Random(self.pixel_selection_seed)

        for row in self.rows:
            rgb_path = resolve_repo_relative_path(row["rgb_path"], self.repo_root)
            fs_csv_path = resolve_repo_relative_path(row["fs_raw_csv_path"], self.repo_root)
            rgb_tensor = _load_rgb_image(
                rgb_path,
                image_size=self.image_size,
                include_structure_channels=self.include_structure_channels,
            )
            cube_data = load_thz_csv(fs_csv_path)
            coords_yx, spectra = extract_valid_pixel_spectra(cube_data)
            if coords_yx.shape[0] == 0:
                continue

            if axis_values_ref is None:
                axis_values_ref = cube_data.axis_values.astype(np.float32)
            elif not np.allclose(axis_values_ref, cube_data.axis_values, atol=1e-6):
                raise ValueError("Axis values are inconsistent across paired samples.")

            if self.max_pixels_per_sample is not None and coords_yx.shape[0] > self.max_pixels_per_sample:
                chosen = rng.sample(range(coords_yx.shape[0]), self.max_pixels_per_sample)
                chosen = np.asarray(chosen, dtype=np.int64)
                coords_yx = coords_yx[chosen]
                spectra = spectra[chosen]

            coords_xy_norm = np.stack([coords_yx[:, 1], coords_yx[:, 0]], axis=1).astype(np.float32)
            if cube_data.cube.shape[1] > 1:
                coords_xy_norm[:, 0] /= float(cube_data.cube.shape[1] - 1)
            if cube_data.cube.shape[0] > 1:
                coords_xy_norm[:, 1] /= float(cube_data.cube.shape[0] - 1)

            normalized_spectra = np.stack(
                [_normalize_spectrum(spectrum, mode=self.normalization) for spectrum in spectra],
                axis=0,
            ).astype(np.float32)

            patch_mean_list = []
            patch_std_list = []
            patch_ratio_list = []
            for center_y, center_x in coords_yx.tolist():
                patch_mean, patch_std, patch_ratio = _compute_patch_stats(
                    cube=cube_data.cube,
                    valid_mask=cube_data.valid_mask,
                    center_y=int(center_y),
                    center_x=int(center_x),
                    patch_size=self.thz_patch_size,
                )
                patch_mean_list.append(patch_mean)
                patch_std_list.append(patch_std)
                patch_ratio_list.append(patch_ratio)

            patch_mean = np.stack(patch_mean_list, axis=0).astype(np.float32)
            patch_std = np.stack(patch_std_list, axis=0).astype(np.float32)
            patch_ratio = np.asarray(patch_ratio_list, dtype=np.float32)

            sample_info = {
                "sample_id": row["sample_id"],
                "group_id": row.get("group_id", row["sample_id"]),
                "class_name": row.get("class_name", ""),
                "sample_name": row.get("sample_name", ""),
                "rgb_path": rgb_path,
                "rgb_tensor": rgb_tensor,
                "fs_raw_csv_path": fs_csv_path,
                "coords_yx": coords_yx.astype(np.int32, copy=False),
                "coords_xy_norm": coords_xy_norm.astype(np.float32, copy=False),
                "center_spectra": normalized_spectra,
                "patch_mean": patch_mean,
                "patch_std": patch_std,
                "patch_ratio": patch_ratio,
            }
            sample_index = len(self.samples)
            self.samples.append(sample_info)
            self.index_map.extend((sample_index, pixel_index) for pixel_index in range(coords_yx.shape[0]))

        if axis_values_ref is None:
            raise ValueError("No valid paired patch samples were found.")

        self.axis_values = axis_values_ref
        self.spectral_length = int(self.axis_values.shape[0])
        self.rgb_channels = 6 if self.include_structure_channels else 3

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_index, pixel_index = self.index_map[index]
        sample = self.samples[sample_index]
        coord_xy_norm = sample["coords_xy_norm"][pixel_index]
        rgb_patch = _extract_rgb_patch(sample["rgb_tensor"], coord_xy_norm, self.rgb_patch_size)

        return {
            "sample_id": sample["sample_id"],
            "group_id": sample["group_id"],
            "class_name": sample["class_name"],
            "sample_name": sample["sample_name"],
            "rgb_path": str(sample["rgb_path"]),
            "fs_raw_csv_path": str(sample["fs_raw_csv_path"]),
            "coord_y": int(sample["coords_yx"][pixel_index, 0]),
            "coord_x": int(sample["coords_yx"][pixel_index, 1]),
            "coord_xy_norm": torch.from_numpy(coord_xy_norm).to(dtype=torch.float32),
            "center_spectrum": torch.from_numpy(sample["center_spectra"][pixel_index]).unsqueeze(0).to(dtype=torch.float32),
            "patch_mean": torch.from_numpy(sample["patch_mean"][pixel_index]).unsqueeze(0).to(dtype=torch.float32),
            "patch_std": torch.from_numpy(sample["patch_std"][pixel_index]).unsqueeze(0).to(dtype=torch.float32),
            "patch_valid_ratio": torch.tensor([sample["patch_ratio"][pixel_index]], dtype=torch.float32),
            "axis_values": torch.from_numpy(self.axis_values).to(dtype=torch.float32),
            "rgb_patch": rgb_patch.to(dtype=torch.float32),
        }


def create_stage2_rgb_fs_patch_dataloader(
    manifest_csv: Path | str,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    dataset_kwargs: Optional[Dict[str, object]] = None,
) -> DataLoader:
    dataset = Stage2RGBFSPatchDataset(
        manifest_csv=manifest_csv,
        **(dataset_kwargs or {}),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
