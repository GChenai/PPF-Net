from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
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


def _infer_raw_column(rows: Sequence[Dict[str, str]], modality: str) -> str:
    preferred = "fs_raw_csv_path" if modality == "fs" else "ts_raw_csv_path"
    if preferred in rows[0]:
        return preferred
    if "raw_csv_path" in rows[0]:
        return "raw_csv_path"
    raise KeyError("Could not infer raw CSV column for modality={0}".format(modality))


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


class Stage1PixelSpectralDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path | str,
        modality: str = "fs",
        normalization: str = "none",
        repo_root: Optional[Path | str] = None,
        raw_csv_column: Optional[str] = None,
        max_pixels_per_sample: Optional[int] = None,
        pixel_selection_seed: int = 42,
    ) -> None:
        super().__init__()
        self.manifest_csv = Path(manifest_csv)
        self.repo_root = Path(repo_root) if repo_root is not None else self.manifest_csv.resolve().parents[2]
        self.modality = modality.lower()
        if self.modality not in {"fs", "ts"}:
            raise ValueError("modality must be 'fs' or 'ts'")
        self.normalization = normalization
        self.rows = _load_csv_rows(self.manifest_csv)
        self.raw_csv_column = raw_csv_column or _infer_raw_column(self.rows, self.modality)
        self.max_pixels_per_sample = max_pixels_per_sample
        self.pixel_selection_seed = int(pixel_selection_seed)

        self.samples: List[Dict[str, object]] = []
        self.index_map: List[Tuple[int, int]] = []
        axis_values_ref: Optional[np.ndarray] = None
        rng = random.Random(self.pixel_selection_seed)

        for row in self.rows:
            raw_csv_path = resolve_repo_relative_path(row[self.raw_csv_column], self.repo_root)
            cube_data = load_thz_csv(raw_csv_path)
            coords, spectra = extract_valid_pixel_spectra(cube_data)

            if coords.shape[0] == 0:
                continue

            if axis_values_ref is None:
                axis_values_ref = cube_data.axis_values.astype(np.float32)
            elif not np.allclose(axis_values_ref, cube_data.axis_values, atol=1e-6):
                raise ValueError("Axis values are inconsistent across samples.")

            if self.max_pixels_per_sample is not None and coords.shape[0] > self.max_pixels_per_sample:
                chosen = rng.sample(range(coords.shape[0]), self.max_pixels_per_sample)
                chosen = np.asarray(chosen, dtype=np.int64)
                coords = coords[chosen]
                spectra = spectra[chosen]

            normalized_spectra = np.stack(
                [_normalize_spectrum(spectrum, mode=self.normalization) for spectrum in spectra],
                axis=0,
            ).astype(np.float32)

            sample_info = {
                "sample_id": row.get("pair_id") or row.get("sample_id") or row.get("sample_name", ""),
                "group_id": row.get("group_id", ""),
                "class_name": row.get("class_name", ""),
                "sample_name": row.get("sample_name", ""),
                "split": row.get("split", ""),
                "raw_csv_path": str(raw_csv_path),
                "height": int(cube_data.cube.shape[0]),
                "width": int(cube_data.cube.shape[1]),
                "coords": coords.astype(np.int32, copy=False),
                "spectra": normalized_spectra,
            }
            sample_index = len(self.samples)
            self.samples.append(sample_info)
            self.index_map.extend((sample_index, pixel_index) for pixel_index in range(coords.shape[0]))

        if axis_values_ref is None:
            raise ValueError("No valid pixel spectra found in {0}".format(self.manifest_csv))

        self.axis_values = axis_values_ref
        self.spectral_length = int(self.axis_values.shape[0])

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_index, pixel_index = self.index_map[index]
        sample = self.samples[sample_index]
        coord = sample["coords"][pixel_index]
        spectrum = sample["spectra"][pixel_index]

        return {
            "sample_id": sample["sample_id"],
            "group_id": sample["group_id"],
            "class_name": sample["class_name"],
            "sample_name": sample["sample_name"],
            "split": sample["split"],
            "raw_csv_path": sample["raw_csv_path"],
            "coord_y": int(coord[0]),
            "coord_x": int(coord[1]),
            "height": int(sample["height"]),
            "width": int(sample["width"]),
            "spectrum": torch.from_numpy(spectrum).unsqueeze(0).to(dtype=torch.float32),
            "axis_values": torch.from_numpy(self.axis_values).to(dtype=torch.float32),
        }


def create_stage1_pixel_spectral_dataloader(
    manifest_csv: Path | str,
    modality: str = "fs",
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    dataset_kwargs: Optional[Dict[str, object]] = None,
) -> DataLoader:
    dataset = Stage1PixelSpectralDataset(
        manifest_csv=manifest_csv,
        modality=modality,
        **(dataset_kwargs or {}),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_stage1_pixel_spectral_dataloaders(
    train_csv: Path | str = "outputs/ppfnet_stage1/splits/train_pairs.csv",
    val_csv: Path | str = "outputs/ppfnet_stage1/splits/val_pairs.csv",
    test_csv: Path | str = "outputs/ppfnet_stage1/splits/test_pairs.csv",
    modality: str = "fs",
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
    dataset_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, DataLoader]:
    common_kwargs = dict(
        modality=modality,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        dataset_kwargs=dataset_kwargs,
    )
    return {
        "train": create_stage1_pixel_spectral_dataloader(train_csv, shuffle=True, **common_kwargs),
        "val": create_stage1_pixel_spectral_dataloader(val_csv, shuffle=False, **common_kwargs),
        "test": create_stage1_pixel_spectral_dataloader(test_csv, shuffle=False, **common_kwargs),
    }

