from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch.utils.data import DataLoader, Dataset

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


def _border_mask(height: int, width: int, fraction: float) -> np.ndarray:
    border = max(4, int(round(min(height, width) * fraction)))
    mask = np.zeros((height, width), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    return mask


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, num_labels = ndimage.label(mask)
    if num_labels <= 0:
        return mask
    component_ids = np.arange(1, num_labels + 1)
    areas = ndimage.sum(mask, labels=labels, index=component_ids)
    largest_id = int(component_ids[int(np.argmax(areas))])
    return labels == largest_id


def _estimate_seed_mask_from_rgb(rgb: np.ndarray, border_fraction: float = 0.08) -> np.ndarray:
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)

    h, w = gray.shape
    bmask = _border_mask(h, w, border_fraction)
    border_gray = gray[bmask]
    border_chroma = chroma[bmask]

    bg_median = float(np.median(border_gray))
    bg_std = float(np.std(border_gray))
    gray_threshold = bg_median - max(0.05, 1.5 * bg_std)
    chroma_threshold = max(
        float(np.quantile(border_chroma, 0.995)) + 0.015,
        float(np.mean(border_chroma) + 3.0 * np.std(border_chroma)),
    )

    mask = gray < gray_threshold
    mask |= ((gray < (bg_median - max(0.025, 0.75 * bg_std))) & (chroma > chroma_threshold))

    kernel = max(3, int(round(min(h, w) * 0.004)))
    if kernel % 2 == 0:
        kernel += 1
    close_structure = np.ones((kernel, kernel), dtype=bool)
    open_kernel = max(3, kernel // 2)
    if open_kernel % 2 == 0:
        open_kernel += 1
    open_structure = np.ones((open_kernel, open_kernel), dtype=bool)

    mask = ndimage.binary_closing(mask, structure=close_structure)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, structure=open_structure)
    mask = ndimage.binary_fill_holes(mask)
    mask = _largest_component(mask)
    return mask.astype(np.float32)


def _build_structure_channels(rgb: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    grad_x = ndimage.sobel(gray, axis=1)
    grad_y = ndimage.sobel(gray, axis=0)
    edge = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edge *= seed_mask
    edge_max = float(edge.max()) if edge.size else 0.0
    if edge_max > 1e-8:
        edge /= edge_max

    distance = ndimage.distance_transform_edt(seed_mask > 0.5).astype(np.float32)
    distance_max = float(distance.max()) if distance.size else 0.0
    if distance_max > 1e-8:
        distance /= distance_max

    return np.stack(
        [
            seed_mask.astype(np.float32),
            edge.astype(np.float32),
            distance.astype(np.float32),
        ],
        axis=0,
    )


def _load_rgb_image(
    image_path: Path,
    image_size: Tuple[int, int],
    include_structure_channels: bool,
) -> torch.Tensor:
    with Image.open(image_path) as image:
        alpha_mask = None
        if "A" in image.getbands():
            image = image.convert("RGBA")
            alpha_mask = np.asarray(image, dtype=np.float32)[..., 3] / 255.0
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        else:
            image = image.convert("RGB")
        rgb = np.asarray(image, dtype=np.float32) / 255.0

    if alpha_mask is not None:
        seed_mask = (alpha_mask > 0.5).astype(np.float32)
    else:
        seed_mask = _estimate_seed_mask_from_rgb(rgb)

    channels = [rgb]
    if include_structure_channels:
        structure_channels = _build_structure_channels(rgb, seed_mask)
        channels.append(np.transpose(structure_channels, (1, 2, 0)))

    array = np.concatenate(channels, axis=-1)
    if image_size is not None:
        pil_mode = "RGB" if array.shape[-1] == 3 else "RGBA"
        if array.shape[-1] > 4:
            # Resize channels independently to avoid PIL channel count limits.
            resized_channels = []
            for channel_idx in range(array.shape[-1]):
                channel = Image.fromarray(np.clip(np.round(array[..., channel_idx] * 255.0), 0, 255).astype(np.uint8), mode="L")
                channel = channel.resize((image_size[1], image_size[0]), Image.BILINEAR)
                resized_channels.append(np.asarray(channel, dtype=np.float32) / 255.0)
            array = np.stack(resized_channels, axis=0)
            return torch.from_numpy(array).contiguous()
        pil_image = Image.fromarray(np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8), mode=pil_mode)
        pil_image = pil_image.resize((image_size[1], image_size[0]), Image.BILINEAR)
        array = np.asarray(pil_image, dtype=np.float32) / 255.0

    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class Stage2RGBFSPixelSampleDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path | str,
        image_size: Tuple[int, int] = (224, 224),
        normalization: str = "none",
        repo_root: Optional[Path | str] = None,
        max_pixels_per_sample: Optional[int] = None,
        pixel_selection_seed: int = 42,
        include_structure_channels: bool = True,
        resample_pixels_each_epoch: bool = False,
    ) -> None:
        super().__init__()
        self.manifest_csv = Path(manifest_csv)
        self.repo_root = Path(repo_root) if repo_root is not None else self.manifest_csv.resolve().parents[2]
        self.rows = _load_csv_rows(self.manifest_csv)
        self.image_size = image_size
        self.normalization = normalization
        self.max_pixels_per_sample = max_pixels_per_sample
        self.pixel_selection_seed = int(pixel_selection_seed)
        self.include_structure_channels = bool(include_structure_channels)
        self.resample_pixels_each_epoch = bool(resample_pixels_each_epoch)
        self.rgb_channels = 6 if self.include_structure_channels else 3
        self.current_epoch = 0

        self.samples: List[Dict[str, object]] = []
        axis_values_ref: Optional[np.ndarray] = None

        for row in self.rows:
            rgb_path = resolve_repo_relative_path(row["rgb_path"], self.repo_root)
            fs_csv_path = resolve_repo_relative_path(row["fs_raw_csv_path"], self.repo_root)
            cube_data = load_thz_csv(fs_csv_path)
            coords_yx, spectra = extract_valid_pixel_spectra(cube_data)
            if coords_yx.shape[0] == 0:
                continue

            if axis_values_ref is None:
                axis_values_ref = cube_data.axis_values.astype(np.float32)
            elif not np.allclose(axis_values_ref, cube_data.axis_values, atol=1e-6):
                raise ValueError("Axis values are inconsistent across paired samples.")

            normalized_spectra = np.stack(
                [_normalize_spectrum(spectrum, mode=self.normalization) for spectrum in spectra],
                axis=0,
            ).astype(np.float32)

            coords_xy = np.stack([coords_yx[:, 1], coords_yx[:, 0]], axis=1).astype(np.float32)
            if cube_data.cube.shape[1] > 1:
                coords_xy[:, 0] /= float(cube_data.cube.shape[1] - 1)
            if cube_data.cube.shape[0] > 1:
                coords_xy[:, 1] /= float(cube_data.cube.shape[0] - 1)

            self.samples.append(
                {
                    "sample_id": row["sample_id"],
                    "group_id": row.get("group_id", row["sample_id"]),
                    "class_name": row.get("class_name", ""),
                    "sample_name": row.get("sample_name", ""),
                    "rgb_path": rgb_path,
                    "fs_raw_csv_path": fs_csv_path,
                    "height": int(cube_data.cube.shape[0]),
                    "width": int(cube_data.cube.shape[1]),
                    "coords_yx_all": coords_yx.astype(np.int32, copy=False),
                    "coords_xy_norm_all": coords_xy.astype(np.float32, copy=False),
                    "spectra_all": normalized_spectra,
                }
            )

        if axis_values_ref is None:
            raise ValueError("No valid RGB+FS paired samples were found.")

        self.axis_values = axis_values_ref
        self.spectral_length = int(self.axis_values.shape[0])

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _select_pixel_indices(self, sample_index: int, total_pixels: int) -> np.ndarray | slice:
        if self.max_pixels_per_sample is None or total_pixels <= self.max_pixels_per_sample:
            return slice(None)

        epoch_component = self.current_epoch if self.resample_pixels_each_epoch else 0
        seed = (
            self.pixel_selection_seed
            + sample_index * 100_003
            + epoch_component * 1_000_003
        )
        rng = random.Random(seed)
        chosen = rng.sample(range(total_pixels), self.max_pixels_per_sample)
        return np.asarray(chosen, dtype=np.int64)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        chosen = self._select_pixel_indices(index, int(sample["spectra_all"].shape[0]))
        spectra = sample["spectra_all"][chosen]
        coords_xy_norm = sample["coords_xy_norm_all"][chosen]
        coords_yx = sample["coords_yx_all"][chosen]
        rgb_tensor = _load_rgb_image(
            sample["rgb_path"],
            self.image_size,
            include_structure_channels=self.include_structure_channels,
        )
        spectra_tensor = torch.from_numpy(spectra).unsqueeze(1).to(dtype=torch.float32)
        coords_xy_norm = torch.from_numpy(coords_xy_norm).to(dtype=torch.float32)
        coords_yx = torch.from_numpy(coords_yx).to(dtype=torch.long)
        axis_values = torch.from_numpy(self.axis_values).to(dtype=torch.float32)

        return {
            "sample_id": sample["sample_id"],
            "group_id": sample["group_id"],
            "class_name": sample["class_name"],
            "sample_name": sample["sample_name"],
            "rgb_path": str(sample["rgb_path"]),
            "fs_raw_csv_path": str(sample["fs_raw_csv_path"]),
            "rgb_image": rgb_tensor,
            "spectra": spectra_tensor,
            "coords_xy_norm": coords_xy_norm,
            "coords_yx": coords_yx,
            "axis_values": axis_values,
            "height": int(sample["height"]),
            "width": int(sample["width"]),
            "num_pixels": int(spectra.shape[0]),
        }


def stage2_rgb_fs_collate_fn(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not batch:
        raise ValueError("Empty batch received.")

    batch_size = len(batch)
    max_pixels = max(int(item["num_pixels"]) for item in batch)
    spectral_length = int(batch[0]["spectra"].shape[-1])

    rgb_images = torch.stack([item["rgb_image"] for item in batch], dim=0)
    axis_values = torch.stack([item["axis_values"] for item in batch], dim=0)

    spectra = torch.zeros((batch_size, max_pixels, 1, spectral_length), dtype=torch.float32)
    coords_xy_norm = torch.zeros((batch_size, max_pixels, 2), dtype=torch.float32)
    coords_yx = torch.full((batch_size, max_pixels, 2), -1, dtype=torch.long)
    pixel_mask = torch.zeros((batch_size, max_pixels), dtype=torch.float32)

    for batch_idx, item in enumerate(batch):
        count = int(item["num_pixels"])
        spectra[batch_idx, :count] = item["spectra"]
        coords_xy_norm[batch_idx, :count] = item["coords_xy_norm"]
        coords_yx[batch_idx, :count] = item["coords_yx"]
        pixel_mask[batch_idx, :count] = 1.0

    return {
        "sample_id": [str(item["sample_id"]) for item in batch],
        "group_id": [str(item["group_id"]) for item in batch],
        "class_name": [str(item["class_name"]) for item in batch],
        "sample_name": [str(item["sample_name"]) for item in batch],
        "rgb_path": [str(item["rgb_path"]) for item in batch],
        "fs_raw_csv_path": [str(item["fs_raw_csv_path"]) for item in batch],
        "rgb_image": rgb_images,
        "spectra": spectra,
        "coords_xy_norm": coords_xy_norm,
        "coords_yx": coords_yx,
        "axis_values": axis_values,
        "pixel_mask": pixel_mask,
        "height": torch.tensor([int(item["height"]) for item in batch], dtype=torch.long),
        "width": torch.tensor([int(item["width"]) for item in batch], dtype=torch.long),
        "num_pixels": torch.tensor([int(item["num_pixels"]) for item in batch], dtype=torch.long),
    }


def create_stage2_rgb_fs_dataloader(
    manifest_csv: Path | str,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    dataset_kwargs: Optional[Dict[str, object]] = None,
) -> DataLoader:
    dataset = Stage2RGBFSPixelSampleDataset(
        manifest_csv=manifest_csv,
        **(dataset_kwargs or {}),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=stage2_rgb_fs_collate_fn,
    )
