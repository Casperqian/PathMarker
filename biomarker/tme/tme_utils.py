from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


TISSUE_LABELS: dict[int, str] = {
    0: "Background",
    1: "Tumor",
    2: "Epithelial Tissue",
    3: "Stroma",
    4: "Blood",
    5: "Necrosis",
}
TISSUE_NAME_TO_LABEL = {name: label for label, name in TISSUE_LABELS.items()}

CELL_TYPES: tuple[str, ...] = (
    "Epithelial cell",
    "Lymphocyte cell",
    "Neoplastic cell",
    "connective cell",
    "plasma cell",
    "Neutrophil cell",
    "Eosinophil cell",
)

CELL_PALETTE: dict[str, tuple[int, int, int]] = {
    "Epithelial cell": (255, 205, 86),
    "Lymphocyte cell": (74, 144, 226),
    "Neoplastic cell": (220, 72, 72),
    "connective cell": (125, 94, 166),
    "plasma cell": (255, 140, 66),
    "Neutrophil cell": (75, 181, 67),
    "Eosinophil cell": (230, 90, 170),
}

CELL_TABLE_FIELDNAMES: list[str] = [
    "slide_id",
    "cell_id",
    "cell_type",
    "score",
    "tumor_status",
    "tumor_score",
    "x_level0",
    "y_level0",
    "area_um2",
    "perimeter_um",
    "roundness",
    "eccentricity",
    "major_axis_um",
    "minor_axis_um",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "tissue_label",
    "tissue_type",
    "parent_patch",
    "tile_x0",
    "tile_y0",
]

TISSUE_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (220, 72, 72),
    2: (255, 205, 86),
    3: (96, 181, 123),
    4: (74, 144, 226),
    5: (125, 94, 166),
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)


def read_cell_table(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in (
            "x_level0",
            "y_level0",
            "score",
            "tumor_score",
            "area_um2",
            "perimeter_um",
            "roundness",
            "eccentricity",
            "major_axis_um",
            "minor_axis_um",
            "bbox_x0",
            "bbox_y0",
            "bbox_x1",
            "bbox_y1",
            "tissue_label",
        ):
            if row.get(key) not in (None, ""):
                try:
                    row[key] = float(row[key])
                except ValueError:
                    pass
        if row.get("tumor_status") not in (None, ""):
            try:
                row["tumor_status"] = int(float(row["tumor_status"]))
            except (TypeError, ValueError):
                pass
    return rows


def is_tumor_cell(cell: Mapping) -> bool:
    status = cell.get("tumor_status")
    if status not in (None, ""):
        if isinstance(status, str):
            return status.strip().lower() in {"1", "true", "yes"}
        try:
            return bool(int(float(status)))
        except (TypeError, ValueError):
            return bool(status)
    return cell.get("cell_type") == "Neoplastic cell"


def write_cell_table(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({key for row in rows for key in row.keys()} - set(CELL_TABLE_FIELDNAMES))
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary_path = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=CELL_TABLE_FIELDNAMES + extras)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def append_cell_table_rows(path: Path, rows: Sequence[Mapping], reset: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = reset or not path.exists() or path.stat().st_size == 0
    mode = "w" if reset else "a"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CELL_TABLE_FIELDNAMES)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CELL_TABLE_FIELDNAMES})


def read_tissue_mask(path: Path) -> np.ndarray:
    old_max_image_pixels = Image.MAX_IMAGE_PIXELS
    try:
        # These masks are trusted pipeline outputs and can exceed Pillow's default safety threshold.
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as image:
            return np.array(image.convert("L"), dtype=np.uint8)
    finally:
        Image.MAX_IMAGE_PIXELS = old_max_image_pixels


def colorize_tissue_mask(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in TISSUE_PALETTE.items():
        rgb[mask == label] = color
    return rgb


def filter_instances_to_core(
    instances: Iterable[Mapping],
    tile_x0: int,
    tile_y0: int,
    core_x0: int,
    core_y0: int,
    core_width: int,
    core_height: int,
) -> list[dict]:
    kept: list[dict] = []
    x_min = float(core_x0)
    y_min = float(core_y0)
    x_max = float(core_x0 + core_width)
    y_max = float(core_y0 + core_height)
    for row in instances:
        cx = float(row["centroid_x"])
        cy = float(row["centroid_y"])
        if not (x_min <= cx < x_max and y_min <= cy < y_max):
            continue
        out = dict(row)
        out["x_level0"] = float(tile_x0) + cx
        out["y_level0"] = float(tile_y0) + cy
        kept.append(out)
    return kept


def drop_duplicate_cells(rows: Sequence[Mapping], distance_um: float, mpp: float) -> list[dict]:
    if not rows:
        return []
    radius_px = abs(float(distance_um) / float(mpp))
    radius_sq = radius_px * radius_px
    bucket_size = radius_px if radius_px > 0 else 1.0
    ordered = sorted(
        [dict(row) for row in rows],
        key=lambda row: float(row.get("score") or 0.0),
        reverse=True,
    )
    kept: list[dict] = []
    grids: dict[object, dict[tuple[int, int], list[int]]] = {}
    for row in ordered:
        x = float(row["x_level0"])
        y = float(row["y_level0"])
        cell_type = row.get("cell_type")
        bucket_x = math.floor(x / bucket_size)
        bucket_y = math.floor(y / bucket_size)
        compatible_types = (cell_type, None) if cell_type else tuple(grids)
        nearby_indices: set[int] = set()
        for compatible_type in compatible_types:
            grid = grids.get(compatible_type)
            if grid is None:
                continue
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    nearby_indices.update(grid.get((bucket_x + offset_x, bucket_y + offset_y), ()))

        duplicate_index: int | None = None
        for old_index in sorted(nearby_indices):
            old = kept[old_index]
            dx = x - float(old["x_level0"])
            dy = y - float(old["y_level0"])
            if dx * dx + dy * dy <= radius_sq:
                duplicate_index = old_index
                break
        if duplicate_index is not None:
            old = kept[duplicate_index]
            if is_tumor_cell(row):
                old["tumor_status"] = 1
                scores = []
                for candidate in (old.get("tumor_score"), row.get("tumor_score")):
                    if candidate in (None, ""):
                        continue
                    try:
                        scores.append(float(candidate))
                    except (TypeError, ValueError):
                        continue
                if scores:
                    old["tumor_score"] = max(scores)
            continue

        kept_index = len(kept)
        kept.append(row)
        grid_key = cell_type if cell_type else None
        grid = grids.setdefault(grid_key, {})
        grid.setdefault((bucket_x, bucket_y), []).append(kept_index)
    return sorted(kept, key=lambda row: str(row.get("cell_id", "")))


def assign_tissue_to_cells(
    rows: Sequence[Mapping],
    tissue_mask: np.ndarray,
    downsample: float,
) -> list[dict]:
    assigned: list[dict] = []
    h, w = tissue_mask.shape[:2]
    for row in rows:
        out = dict(row)
        mx = int(math.floor(float(out["x_level0"]) / float(downsample)))
        my = int(math.floor(float(out["y_level0"]) / float(downsample)))
        if 0 <= mx < w and 0 <= my < h:
            label = int(tissue_mask[my, mx])
        else:
            label = 0
        out["tissue_label"] = label
        out["tissue_type"] = TISSUE_LABELS.get(label, "Background")
        assigned.append(out)
    return assigned


def tissue_area_um2(tissue_mask: np.ndarray, label: int, downsample: float, mpp: float) -> float:
    pixel_area = (float(downsample) * float(mpp)) ** 2
    return float(np.count_nonzero(tissue_mask == label) * pixel_area)


def valid_tissue_area_um2(tissue_mask: np.ndarray, downsample: float, mpp: float) -> float:
    pixel_area = (float(downsample) * float(mpp)) ** 2
    return float(np.count_nonzero(tissue_mask > 0) * pixel_area)


def cell_distance_um(a: Mapping, b: Mapping, mpp: float) -> float:
    dx = float(a["x_level0"]) - float(b["x_level0"])
    dy = float(a["y_level0"]) - float(b["y_level0"])
    return math.sqrt(dx * dx + dy * dy) * float(mpp)


def cells_within_radius(
    center_cells: Sequence[Mapping],
    neighbor_cells: Sequence[Mapping],
    radius_um: float,
    mpp: float,
    include_self: bool = False,
) -> list[list[dict]]:
    radius_px = float(radius_um) / float(mpp)
    if not center_cells or not neighbor_cells:
        return [[] for _center in center_cells]

    center_coords = np.asarray(
        [[float(cell["x_level0"]), float(cell["y_level0"])] for cell in center_cells],
        dtype=np.float64,
    )
    neighbor_coords = np.asarray(
        [[float(cell["x_level0"]), float(cell["y_level0"])] for cell in neighbor_cells],
        dtype=np.float64,
    )
    neighbor_ids = [cell.get("cell_id") for cell in neighbor_cells]
    tree = cKDTree(neighbor_coords)
    index_groups = tree.query_ball_point(center_coords, radius_px, return_sorted=True)
    out: list[list[dict]] = []
    for center, indices in zip(center_cells, index_groups):
        center_id = center.get("cell_id")
        near: list[dict] = []
        for idx in indices:
            if not include_self and center_id == neighbor_ids[int(idx)]:
                continue
            near.append(dict(neighbor_cells[int(idx)]))
        out.append(near)
    return out


def load_format_metrics(path: Path) -> list[str]:
    from .schema_reduction import reduce_metrics

    with path.open(newline="", encoding="utf-8") as f:
        metrics = [row["metric"].strip() for row in csv.DictReader(f) if row["metric"].strip()]
    return reduce_metrics(metrics).metrics


def write_readout_csv(path: Path, metrics: Sequence[str], values: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "results"])
        writer.writeheader()
        for metric in metrics:
            value = values.get(metric, "")
            writer.writerow({"metric": metric, "results": value})


def safe_float_mean(values: Sequence[float]) -> float | str:
    if not values:
        return ""
    return float(np.mean(values))


def safe_float_std(values: Sequence[float]) -> float | str:
    if not values:
        return ""
    return float(np.std(values, ddof=0))
