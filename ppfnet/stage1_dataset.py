from __future__ import annotations

import csv
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_EXCLUDED_FEATURES = ("valid_mask",)


def infer_feature_names(
    npz_path: Path | str,
    excluded: Sequence[str] = DEFAULT_EXCLUDED_FEATURES,
) -> List[str]:
    path = Path(npz_path)
    with np.load(path) as data:
        return [name for name in data.files if name not in excluded]


def _load_csv_rows(csv_path: Path | str) -> List[Dict[str, str]]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("Manifest CSV is empty: {0}".format(path))
    return rows


def _per_channel_normalize(
    stack: np.ndarray,
    valid_mask: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "none":
        return stack

    if mode not in {"zscore", "minmax"}:
        raise ValueError("Unsupported normalization mode: {0}".format(mode))

    normalized = stack.copy()
    valid = valid_mask > 0

    for channel_idx in range(normalized.shape[0]):
        values = normalized[channel_idx][valid]
        if values.size == 0:
            continue

        if mode == "zscore":
            mean = float(values.mean())
            std = float(values.std())
            if std <= 1e-8:
                std = 1.0
            normalized[channel_idx] = (normalized[channel_idx] - mean) / std
        else:
            min_value = float(values.min())
            max_value = float(values.max())
            if max_value <= min_value:
                max_value = min_value + 1e-6
            normalized[channel_idx] = (normalized[channel_idx] - min_value) / (max_value - min_value)

        normalized[channel_idx][~valid] = 0.0

    return normalized


def _load_feature_stack(
    npz_path: Path | str,
    feature_names: Sequence[str],
    normalization: str = "none",
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(npz_path)
    with np.load(path) as data:
        missing = [name for name in feature_names if name not in data.files]
        if missing:
            raise KeyError(
                "Missing feature(s) {0} in {1}".format(", ".join(missing), path)
            )

        if "valid_mask" in data.files:
            valid_mask = data["valid_mask"].astype(np.float32)
        else:
            probe = data[feature_names[0]]
            valid_mask = np.isfinite(probe).astype(np.float32)

        channels: List[np.ndarray] = []
        for name in feature_names:
            array = data[name].astype(np.float32)
            array = np.nan_to_num(array, nan=fill_value, posinf=fill_value, neginf=fill_value)
            channels.append(array)

    stack = np.stack(channels, axis=0)
    stack = _per_channel_normalize(stack, valid_mask=valid_mask, mode=normalization)
    return stack.astype(np.float32), valid_mask.astype(np.float32)


def _resize_tensor(
    tensor: torch.Tensor,
    spatial_size: Optional[Tuple[int, int]],
    mode: str,
) -> torch.Tensor:
    if spatial_size is None:
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),
        size=spatial_size,
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    ).squeeze(0)


class Stage1FeaturePairDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path | str,
        fs_feature_names: Optional[Sequence[str]] = None,
        ts_feature_names: Optional[Sequence[str]] = None,
        normalization: str = "none",
        fill_value: float = 0.0,
        spatial_size: Optional[Tuple[int, int]] = None,
        include_valid_mask_channel: bool = False,
    ) -> None:
        super().__init__()
        self.manifest_csv = Path(manifest_csv)
        self.rows = _load_csv_rows(self.manifest_csv)

        probe_row = self.rows[0]
        self.fs_feature_names = list(
            fs_feature_names or infer_feature_names(probe_row["fs_feature_npz_path"])
        )
        self.ts_feature_names = list(
            ts_feature_names or infer_feature_names(probe_row["ts_feature_npz_path"])
        )
        self.normalization = normalization
        self.fill_value = float(fill_value)
        self.spatial_size = spatial_size
        self.include_valid_mask_channel = include_valid_mask_channel

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]

        fs_stack, fs_valid_mask = _load_feature_stack(
            row["fs_feature_npz_path"],
            feature_names=self.fs_feature_names,
            normalization=self.normalization,
            fill_value=self.fill_value,
        )
        ts_stack, ts_valid_mask = _load_feature_stack(
            row["ts_feature_npz_path"],
            feature_names=self.ts_feature_names,
            normalization=self.normalization,
            fill_value=self.fill_value,
        )

        fs_tensor = torch.from_numpy(fs_stack)
        ts_tensor = torch.from_numpy(ts_stack)
        fs_valid_tensor = torch.from_numpy(fs_valid_mask).unsqueeze(0)
        ts_valid_tensor = torch.from_numpy(ts_valid_mask).unsqueeze(0)

        fs_tensor = _resize_tensor(fs_tensor, self.spatial_size, mode="bilinear")
        ts_tensor = _resize_tensor(ts_tensor, self.spatial_size, mode="bilinear")
        fs_valid_tensor = _resize_tensor(fs_valid_tensor, self.spatial_size, mode="nearest")
        ts_valid_tensor = _resize_tensor(ts_valid_tensor, self.spatial_size, mode="nearest")

        fs_valid_tensor = (fs_valid_tensor > 0.5).to(dtype=torch.float32)
        ts_valid_tensor = (ts_valid_tensor > 0.5).to(dtype=torch.float32)

        if self.include_valid_mask_channel:
            fs_tensor = torch.cat([fs_tensor, fs_valid_tensor], dim=0)
            ts_tensor = torch.cat([ts_tensor, ts_valid_tensor], dim=0)

        return {
            "pair_id": row.get("pair_id", ""),
            "group_id": row.get("group_id", ""),
            "class_name": row.get("class_name", ""),
            "sample_name": row.get("sample_name", ""),
            "split": row.get("split", ""),
            "fs_features": fs_tensor.to(dtype=torch.float32),
            "ts_features": ts_tensor.to(dtype=torch.float32),
            "fs_valid_mask": fs_valid_tensor,
            "ts_valid_mask": ts_valid_tensor,
            "fs_feature_names": list(self.fs_feature_names),
            "ts_feature_names": list(self.ts_feature_names),
            "original_fs_shape": tuple(fs_stack.shape[-2:]),
            "original_ts_shape": tuple(ts_stack.shape[-2:]),
        }


def _pad_to_shape(
    tensor: torch.Tensor,
    target_h: int,
    target_w: int,
    value: float,
) -> torch.Tensor:
    _, height, width = tensor.shape
    pad_h = target_h - height
    pad_w = target_w - width
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Target size smaller than tensor size.")
    return F.pad(tensor, (0, pad_w, 0, pad_h), value=value)


def stage1_collate_fn(
    batch: Sequence[Dict[str, object]],
    fill_value: float = 0.0,
) -> Dict[str, object]:
    if not batch:
        raise ValueError("Empty batch received.")

    target_h = max(
        max(int(item["fs_features"].shape[-2]), int(item["ts_features"].shape[-2]))
        for item in batch
    )
    target_w = max(
        max(int(item["fs_features"].shape[-1]), int(item["ts_features"].shape[-1]))
        for item in batch
    )

    fs_features = torch.stack(
        [_pad_to_shape(item["fs_features"], target_h, target_w, value=fill_value) for item in batch],
        dim=0,
    )
    ts_features = torch.stack(
        [_pad_to_shape(item["ts_features"], target_h, target_w, value=fill_value) for item in batch],
        dim=0,
    )
    fs_valid_mask = torch.stack(
        [_pad_to_shape(item["fs_valid_mask"], target_h, target_w, value=0.0) for item in batch],
        dim=0,
    )
    ts_valid_mask = torch.stack(
        [_pad_to_shape(item["ts_valid_mask"], target_h, target_w, value=0.0) for item in batch],
        dim=0,
    )

    original_sizes = torch.tensor(
        [[item["fs_features"].shape[-2], item["fs_features"].shape[-1]] for item in batch],
        dtype=torch.long,
    )

    spatial_mask = torch.zeros((len(batch), 1, target_h, target_w), dtype=torch.float32)
    for idx, item in enumerate(batch):
        height = max(int(item["fs_features"].shape[-2]), int(item["ts_features"].shape[-2]))
        width = max(int(item["fs_features"].shape[-1]), int(item["ts_features"].shape[-1]))
        spatial_mask[idx, :, :height, :width] = 1.0

    return {
        "pair_id": [str(item["pair_id"]) for item in batch],
        "group_id": [str(item["group_id"]) for item in batch],
        "class_name": [str(item["class_name"]) for item in batch],
        "sample_name": [str(item["sample_name"]) for item in batch],
        "split": [str(item["split"]) for item in batch],
        "fs_features": fs_features,
        "ts_features": ts_features,
        "fs_valid_mask": fs_valid_mask,
        "ts_valid_mask": ts_valid_mask,
        "spatial_mask": spatial_mask,
        "original_sizes": original_sizes,
        "fs_feature_names": list(batch[0]["fs_feature_names"]),
        "ts_feature_names": list(batch[0]["ts_feature_names"]),
    }


def create_stage1_dataloader(
    manifest_csv: Path | str,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    dataset_kwargs: Optional[Dict[str, object]] = None,
) -> DataLoader:
    dataset = Stage1FeaturePairDataset(manifest_csv=manifest_csv, **(dataset_kwargs or {}))
    collate = partial(stage1_collate_fn, fill_value=float(dataset.fill_value))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )


def create_stage1_dataloaders(
    train_csv: Path | str = "outputs/ppfnet_stage1/splits/train_pairs.csv",
    val_csv: Path | str = "outputs/ppfnet_stage1/splits/val_pairs.csv",
    test_csv: Path | str = "outputs/ppfnet_stage1/splits/test_pairs.csv",
    batch_size: int = 4,
    num_workers: int = 0,
    pin_memory: bool = False,
    dataset_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, DataLoader]:
    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        dataset_kwargs=dataset_kwargs,
    )
    return {
        "train": create_stage1_dataloader(train_csv, shuffle=True, **common_kwargs),
        "val": create_stage1_dataloader(val_csv, shuffle=False, **common_kwargs),
        "test": create_stage1_dataloader(test_csv, shuffle=False, **common_kwargs),
    }

