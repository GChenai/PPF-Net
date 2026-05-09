from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


AXIS_UNIT_RE = re.compile(r"([A-Za-z]+)$")
X_RE = re.compile(r"X(\d+)$", re.IGNORECASE)
Y_RE = re.compile(r"Y(\d+)$", re.IGNORECASE)


@dataclass
class THzCube:
    source_path: Path
    cube: np.ndarray
    axis_values: np.ndarray
    valid_mask: np.ndarray
    signal_label: str
    axis_unit: str
    axis_domain: str
    header_metadata: Dict[str, str]


def strip_trailing_empty(items: Sequence[str]) -> List[str]:
    cleaned = [item.strip() for item in items]
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return cleaned


def safe_float(text: str) -> float:
    if text == "" or text.lower() == "nan":
        return float("nan")
    return float(text)


def infer_axis_unit(*texts: str) -> str:
    for text in texts:
        match = AXIS_UNIT_RE.search(text.strip())
        if match:
            return match.group(1)
    return ""


def infer_axis_domain(axis_unit: str) -> str:
    unit = axis_unit.lower()
    if unit in {"thz", "ghz", "mhz", "khz", "hz"}:
        return "frequency"
    if unit in {"ps", "ns", "us", "ms", "s", "fs"}:
        return "time"
    return "unknown"


def parse_coord(token: str, pattern: re.Pattern[str], axis_name: str) -> int:
    match = pattern.match(token.strip())
    if not match:
        raise ValueError("Could not parse {0} coordinate from token: {1}".format(axis_name, token))
    return int(match.group(1))


def resolve_repo_relative_path(path_text: str | Path, repo_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_thz_csv(path: str | Path) -> THzCube:
    source_path = Path(path)
    rows: List[List[str]] = []
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            row = strip_trailing_empty(row)
            if row:
                rows.append(row)

    if len(rows) < 6:
        raise ValueError("CSV looks too short to be a THz cube: {0}".format(source_path))

    header_metadata: Dict[str, str] = {}
    for row in rows[:4]:
        if len(row) >= 2:
            header_metadata[row[0]] = ",".join(row[1:])

    axis_row = rows[4]
    if len(axis_row) < 4:
        raise ValueError("Could not find a valid axis row in {0}".format(source_path))

    signal_label = axis_row[0]
    axis_values = np.asarray([safe_float(item) for item in axis_row[3:]], dtype=np.float32)
    axis_unit = infer_axis_unit(axis_row[1], axis_row[2])
    axis_domain = infer_axis_domain(axis_unit)
    axis_count = int(axis_values.shape[0])

    parsed_rows: List[Tuple[int, int, np.ndarray]] = []
    max_x = -1
    max_y = -1

    for row in rows[5:]:
        if len(row) < 3:
            continue
        try:
            x = parse_coord(row[1], X_RE, "X")
            y = parse_coord(row[2], Y_RE, "Y")
        except ValueError:
            continue

        values = [safe_float(item) for item in row[3:3 + axis_count]]
        if len(values) < axis_count:
            values.extend([float("nan")] * (axis_count - len(values)))

        spectra = np.asarray(values, dtype=np.float32)
        parsed_rows.append((x, y, spectra))
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    if not parsed_rows:
        raise ValueError("No pixel rows were parsed from: {0}".format(source_path))

    cube = np.full((max_y + 1, max_x + 1, axis_count), np.nan, dtype=np.float32)
    for x, y, spectra in parsed_rows:
        cube[y, x, :] = spectra

    valid_mask = np.isfinite(cube).any(axis=-1)
    return THzCube(
        source_path=source_path,
        cube=cube,
        axis_values=axis_values,
        valid_mask=valid_mask,
        signal_label=signal_label,
        axis_unit=axis_unit,
        axis_domain=axis_domain,
        header_metadata=header_metadata,
    )


def extract_valid_spectra(cube_data: THzCube) -> np.ndarray:
    spectra = cube_data.cube[cube_data.valid_mask]
    spectra = spectra[np.isfinite(spectra).all(axis=1)]
    return spectra.astype(np.float32, copy=False)


def extract_valid_pixel_spectra(cube_data: THzCube) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(cube_data.valid_mask).astype(np.int32, copy=False)
    spectra = cube_data.cube[cube_data.valid_mask]
    finite_rows = np.isfinite(spectra).all(axis=1)
    coords = coords[finite_rows]
    spectra = spectra[finite_rows]
    return coords.astype(np.int32, copy=False), spectra.astype(np.float32, copy=False)


def extract_mean_spectrum(cube_data: THzCube) -> np.ndarray:
    spectra = extract_valid_spectra(cube_data)
    if spectra.size == 0:
        return np.zeros((cube_data.axis_values.shape[0],), dtype=np.float32)
    return spectra.mean(axis=0).astype(np.float32)


def extract_median_spectrum(cube_data: THzCube) -> np.ndarray:
    spectra = extract_valid_spectra(cube_data)
    if spectra.size == 0:
        return np.zeros((cube_data.axis_values.shape[0],), dtype=np.float32)
    return np.median(spectra, axis=0).astype(np.float32)


def assemble_cube_from_pixel_spectra(
    coords: np.ndarray,
    spectra: np.ndarray,
    height: int,
    width: int,
    fill_value: float = float("nan"),
) -> np.ndarray:
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape [N, 2] as (y, x).")
    if spectra.ndim != 2:
        raise ValueError("spectra must have shape [N, C].")
    if coords.shape[0] != spectra.shape[0]:
        raise ValueError("coords and spectra must contain the same number of rows.")

    cube = np.full((height, width, spectra.shape[1]), fill_value, dtype=np.float32)
    cube[coords[:, 0], coords[:, 1], :] = spectra.astype(np.float32, copy=False)
    return cube


def save_thz_csv(
    path: Path | str,
    cube: np.ndarray,
    axis_values: np.ndarray,
    signal_label: str = "abs",
    axis_unit: str = "THz",
    header_metadata: Dict[str, str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header_metadata = dict(header_metadata or {})

    version = header_metadata.get("版本号", "V1.1")
    file_name = header_metadata.get("文件名称", str(path.with_suffix(".imgtds")))
    imaging_mode = header_metadata.get("成像模式", "重建结果")
    position_info = header_metadata.get(
        "位置信息",
        "[0，0]-[{0}，{1}]".format(cube.shape[1] - 1, cube.shape[0] - 1),
    )

    rows: List[List[str]] = [
        ["版本号", version],
        ["文件名称", file_name],
        ["成像模式", imaging_mode],
        ["位置信息", position_info],
        [
            signal_label,
            "min: {0}{1}".format(float(axis_values[0]), axis_unit),
            "max: {0}{1}".format(float(axis_values[-1]), axis_unit),
            *[repr(float(v)) for v in axis_values.tolist()],
        ],
    ]

    for y in range(cube.shape[0]):
        for x in range(cube.shape[1]):
            spectrum = cube[y, x]
            values = []
            for value in spectrum:
                if np.isnan(value):
                    values.append("nan")
                else:
                    values.append(repr(float(value)))
            rows.append([signal_label, "X{0}".format(x), "Y{0}".format(y), *values])

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
