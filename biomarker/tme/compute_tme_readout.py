from __future__ import annotations

import argparse
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError, cKDTree

try:
    from . import spatial_readout, tme_utils
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from biomarker.tme import spatial_readout, tme_utils


AREA_RE = re.compile(r"^Area of (.+)$")
TISSUE_RATIO_VALID_RE = re.compile(r"^Ratio of (.+) area to valid tissue area$")
TISSUE_RATIO_RE = re.compile(r"^Ratio of (.+) area to (.+) area$")
CELL_NUM_RE = re.compile(r"^Number of (.+)$")
CELL_RATIO_ALL_RE = re.compile(r"^Ratio of (.+) to all cells$")
CELL_DENSITY_VALID_RE = re.compile(r"^Density of (.+) in valid tissue$")
CELL_NUM_TISSUE_RE = re.compile(r"^Number of (.+) in (.+)$")
CELL_DENSITY_TISSUE_RE = re.compile(r"^Density of (.+) in (.+)$")
CELL_RATIO_TISSUE_RE = re.compile(r"^Ratio of (.+) to all cells in (.+)$")
CELL_PAIR_RATIO_TISSUE_RE = re.compile(r"^Ratio of (.+) to (.+) in (.+)$")
CELL_MORPH_AVG_RE = re.compile(r"^Average (area|roundness|eccentricity|major axis|minor axis|perimeter) of (.+)$")
CELL_MORPH_STD_AREA_RE = re.compile(r"^Standard deviation of area of (.+)$")
NEIGHBOR_COUNT_RE = re.compile(r"^Number of (.+) around (.+) within ([0-9.]+) um$")
NEIGHBOR_COUNT_TISSUE_RE = re.compile(r"^Number of (.+) around (.+) in (.+) within ([0-9.]+) um$")
NEIGHBOR_RATIO_TISSUE_RE = re.compile(
    r"^Ratio of (.+) to all neighboring cells around (.+) in (.+) within ([0-9.]+) um$"
)
MIN_DISTANCE_RE = re.compile(r"^Average minimum distance from (.+) to (.+) in (.+) within ([0-9.]+) um$")
WITH_NEIGHBOR_COUNT_RE = re.compile(r"^Number of (.+) with at least one (.+) neighbor in (.+) within ([0-9.]+) um$")
WITH_NEIGHBOR_RATIO_RE = re.compile(r"^Ratio of (.+) with at least one (.+) neighbor in (.+) within ([0-9.]+) um$")
TISSUE_CONNECTED_COUNT_RE = re.compile(r"^Number of connected regions of (.+)$")
TISSUE_LARGEST_CONNECTED_AREA_RE = re.compile(r"^Area of largest connected region of (.+)$")
TISSUE_LARGEST_CONNECTED_RATIO_RE = re.compile(r"^Ratio of largest connected region area to total (.+) area$")
TISSUE_AVG_MORPH_RE = re.compile(r"^Average (eccentricity|extent|solidity) of (.+)$")
TISSUE_BOUNDARY_RE = re.compile(r"^Boundary irregularity index of (.+)$")
TISSUE_BOUNDARY_LARGEST_RE = re.compile(r"^Boundary irregularity index of largest connected region of (.+)$")
TISSUE_EULER_RE = re.compile(r"^Euler number of (.+)$")
TISSUE_FRACTAL_RE = re.compile(r"^Fractal dimension of (.+)$")
TISSUE_LACUNARITY_RE = re.compile(r"^Lacunarity of (.+)$")
TISSUE_LARGEST_MORPH_RE = re.compile(
    r"^Largest (convex area|filled area|roundness|major axis|minor axis|perimeter|eccentricity|solidity|extent|Euler number) of (.+)$"
)
CELL_DENSITY_RATIO_TISSUE_RE = re.compile(r"^Ratio of (.+) density in (.+) to (.+)$")
TISSUE_AREA_AROUND_CELL_RE = re.compile(r"^Area of (.+) around (.+) within ([0-9.]+) um$")
NEIGHBOR_DENSITY_TISSUE_RE = re.compile(r"^Density of (.+) around (.+) in (.+) within ([0-9.]+) um$")
BALL_HALL_RE = re.compile(r"^Ball-Hall index of (.+) in (.+)$")
CALINSKI_RE = re.compile(r"^Calinski-Harabasz index of (.+) in (.+)$")
CLUSTER_NUM_RE = re.compile(r"^Number of clusters of (.+) in (.+)$")
CLUSTER_AVG_EXTENT_RE = re.compile(r"^Average cluster extent of (.+) in (.+)$")
CLUSTER_STD_EXTENT_RE = re.compile(r"^Standard deviation of cluster extent of (.+) in (.+)$")
CLUSTER_AVG_SIZE_RE = re.compile(r"^Average cluster size of (.+) in (.+)$")
CLUSTER_STD_SIZE_RE = re.compile(r"^Standard deviation of cluster size of (.+) in (.+)$")
CLUSTER_LARGEST_SIZE_RE = re.compile(r"^Largest cluster size of (.+) in (.+)$")
CLUSTER_AVG_DISPERSION_RE = re.compile(r"^Average within-cluster dispersion of (.+) in (.+)$")
CLUSTER_STD_DISPERSION_RE = re.compile(r"^Standard deviation of within-cluster dispersion of (.+) in (.+)$")


MORPH_FIELD = {
    "area": "area_um2",
    "roundness": "roundness",
    "eccentricity": "eccentricity",
    "major axis": "major_axis_um",
    "minor axis": "minor_axis_um",
    "perimeter": "perimeter_um",
}

LARGEST_TISSUE_FIELD = {
    "convex area": "convex_area",
    "filled area": "filled_area",
    "roundness": "roundness",
    "major axis": "major_axis",
    "minor axis": "minor_axis",
    "perimeter": "perimeter",
    "eccentricity": "eccentricity",
    "solidity": "solidity",
    "extent": "extent",
    "Euler number": "euler_number",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-slide TME readouts from a tissue map and cell table.")
    parser.add_argument(
        "--format-csv",
        type=Path,
        default=Path(__file__).with_name("TME_Readout_Format.csv"),
    )
    parser.add_argument("--tissue-mask", type=Path, required=True)
    parser.add_argument("--tissue-metadata", type=Path, required=True)
    parser.add_argument("--cell-table", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--radii-um", default="40,80")
    parser.add_argument("--spatial-working-mb", type=float, default=256.0)
    return parser.parse_args()


def _cells_of_type(cells: Sequence[Mapping], cell_type: str) -> list[Mapping]:
    if cell_type == "Neoplastic cell":
        return [cell for cell in cells if tme_utils.is_tumor_cell(cell)]
    return [cell for cell in cells if cell.get("cell_type") == cell_type]


def filter_cells_by_min_area_px(
    cells: Sequence[Mapping],
    min_area_px: int | float,
    target_mpp: float,
) -> list[Mapping]:
    min_area_px = float(min_area_px)
    if min_area_px <= 0:
        return list(cells)
    cutoff_area_um2 = min_area_px * float(target_mpp) * float(target_mpp)
    kept: list[Mapping] = []
    for cell in cells:
        area_um2 = cell.get("area_um2")
        if area_um2 in (None, ""):
            kept.append(cell)
            continue
        try:
            if float(area_um2) >= cutoff_area_um2:
                kept.append(cell)
        except (TypeError, ValueError):
            kept.append(cell)
    return kept


def filter_nested_cells_by_bbox(
    cells: Sequence[Mapping],
    target_mpp: float = 0.25,
    min_inner_area_px: int = 144,
    min_inner_count: int = 2,
    min_inner_area_ratio: float = 0.5,
    min_inner_bbox_coverage: float = 0.9,
) -> list[Mapping]:
    """Resolve same-class nested CSV bboxes without double-counting inner instances."""
    groups: dict[tuple[object, object, object], list[tuple[int, Mapping]]] = {}
    for index, cell in enumerate(cells):
        key = (cell.get("cell_type"), cell.get("tile_x0"), cell.get("tile_y0"))
        if None in key:
            continue
        groups.setdefault(key, []).append((index, cell))

    dropped: set[int] = set()
    for group in groups.values():
        ordered = []
        for index, cell in group:
            bbox = tuple(cell.get(key) for key in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"))
            if any(value in (None, "") for value in bbox):
                continue
            ordered.append((index, float(cell.get("area_um2") or 0.0), tuple(float(value) for value in bbox)))
        ordered.sort(key=lambda item: item[1], reverse=True)
        bucket_size = 256.0
        outer_by_bucket: dict[tuple[int, int], list[tuple[int, float, tuple[float, float, float, float]]]] = {}
        children_by_outer: dict[int, list[tuple[int, float]]] = {}
        areas_by_index = {index: area for index, area, _bbox in ordered}
        for candidate_index, candidate_area, candidate_bbox in ordered:
            cx = (candidate_bbox[0] + candidate_bbox[2]) / 2.0
            cy = (candidate_bbox[1] + candidate_bbox[3]) / 2.0
            bucket = (math.floor(cx / bucket_size), math.floor(cy / bucket_size))
            candidate_bbox_area = max(1.0, (candidate_bbox[2] - candidate_bbox[0]) * (candidate_bbox[3] - candidate_bbox[1]))
            containing = []
            for outer_index, outer_area, outer_bbox in outer_by_bucket.get(bucket, ()):
                inter_x0 = max(outer_bbox[0], candidate_bbox[0])
                inter_y0 = max(outer_bbox[1], candidate_bbox[1])
                inter_x1 = min(outer_bbox[2], candidate_bbox[2])
                inter_y1 = min(outer_bbox[3], candidate_bbox[3])
                overlap = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
                if outer_area > candidate_area and overlap / candidate_bbox_area >= min_inner_bbox_coverage:
                    containing.append((outer_index, outer_area))
            if containing:
                outer_index, _outer_area = min(containing, key=lambda item: item[1])
                children_by_outer.setdefault(outer_index, []).append((candidate_index, candidate_area))

            x0_bucket = math.floor(float(candidate_bbox[0]) / bucket_size)
            y0_bucket = math.floor(float(candidate_bbox[1]) / bucket_size)
            x1_bucket = math.floor((float(candidate_bbox[2]) - 1.0) / bucket_size)
            y1_bucket = math.floor((float(candidate_bbox[3]) - 1.0) / bucket_size)
            bbox = candidate_bbox
            for bucket_x in range(x0_bucket, x1_bucket + 1):
                for bucket_y in range(y0_bucket, y1_bucket + 1):
                    outer_by_bucket.setdefault((bucket_x, bucket_y), []).append((candidate_index, candidate_area, bbox))

        min_inner_area_um2 = float(min_inner_area_px) * float(target_mpp) * float(target_mpp)
        for outer_index, children in children_by_outer.items():
            qualifying_children = [child for child in children if child[1] >= min_inner_area_um2]
            if (
                len(qualifying_children) >= min_inner_count
                and sum(child[1] for child in qualifying_children) >= min_inner_area_ratio * areas_by_index[outer_index]
            ):
                dropped.add(outer_index)
                qualifying_indices = {child_index for child_index, _child_area in qualifying_children}
                dropped.update(child_index for child_index, _child_area in children if child_index not in qualifying_indices)
            else:
                dropped.update(child_index for child_index, _child_area in children)

    return [cell for index, cell in enumerate(cells) if index not in dropped]


def _cells_in_tissue(cells: Sequence[Mapping], tissue_type: str) -> list[Mapping]:
    return [cell for cell in cells if cell.get("tissue_type") == tissue_type]


def _numeric_values(cells: Sequence[Mapping], field: str) -> list[float]:
    values: list[float] = []
    for cell in cells:
        value = cell.get(field)
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _count_neighbors(
    center_cells: Sequence[Mapping],
    neighbor_cells: Sequence[Mapping],
    radius_um: float,
    mpp: float,
) -> int:
    return int(
        sum(
            len(group)
            for group in tme_utils.cells_within_radius(
                center_cells,
                neighbor_cells,
                radius_um=radius_um,
                mpp=mpp,
            )
        )
    )


def _minimum_distances(
    center_cells: Sequence[Mapping],
    neighbor_cells: Sequence[Mapping],
    radius_um: float,
    mpp: float,
) -> list[float]:
    distances: list[float] = []
    groups = tme_utils.cells_within_radius(center_cells, neighbor_cells, radius_um=radius_um, mpp=mpp)
    for center, group in zip(center_cells, groups):
        if not group:
            continue
        distances.append(min(tme_utils.cell_distance_um(center, neighbor, mpp) for neighbor in group))
    return distances


def _component_boundary(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    eroded = ndimage.binary_erosion(arr, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return arr & ~eroded


def _convex_area_pixels(mask: np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    if not arr.any():
        return 0.0

    ys, xs = np.nonzero(_component_boundary(arr))
    corners = np.concatenate(
        [
            np.column_stack([xs, ys]),
            np.column_stack([xs + 1, ys]),
            np.column_stack([xs, ys + 1]),
            np.column_stack([xs + 1, ys + 1]),
        ],
        axis=0,
    ).astype(np.float64, copy=False)
    corners = np.unique(corners, axis=0)
    try:
        return float(ConvexHull(corners).volume)
    except QhullError:
        return float(np.count_nonzero(arr))


def _euler_number(mask: np.ndarray) -> int:
    arr = np.asarray(mask, dtype=bool)
    if not arr.any():
        return 0
    _, obj_count = ndimage.label(arr, structure=np.ones((3, 3), dtype=int))
    padded_inv = np.pad(~arr, 1, constant_values=True)
    labels, hole_count = ndimage.label(padded_inv, structure=np.ones((3, 3), dtype=int))
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    holes = [label for label in range(1, hole_count + 1) if label not in border_labels]
    return int(obj_count - len(holes))


def _region_stats(mask: np.ndarray, pixel_area_um2: float, pixel_size_um: float) -> dict[str, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {}
    area_px = float(xs.size)
    filled = ndimage.binary_fill_holes(mask)
    filled_area_px = float(np.count_nonzero(filled))
    bbox_area_px = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    perimeter_px = float(np.count_nonzero(_component_boundary(mask)))
    if xs.size > 1:
        coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        cov = np.cov(coords, rowvar=False, bias=True)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1].clip(min=0.0)
    else:
        eigvals = np.array([0.0, 0.0], dtype=np.float64)
    major_px = float(4.0 * math.sqrt(float(eigvals[0]))) if eigvals[0] > 0 else 0.0
    minor_px = float(4.0 * math.sqrt(float(eigvals[1]))) if eigvals[1] > 0 else 0.0
    eccentricity = (
        float(math.sqrt(max(0.0, 1.0 - float(eigvals[1]) / float(eigvals[0]))))
        if eigvals[0] > 0
        else 0.0
    )
    convex_area_px = max(_convex_area_pixels(mask), 1.0)
    perimeter_um = perimeter_px * pixel_size_um
    area_um2 = area_px * pixel_area_um2
    return {
        "area": area_um2,
        "filled_area": filled_area_px * pixel_area_um2,
        "convex_area": convex_area_px * pixel_area_um2,
        "perimeter": perimeter_um,
        "roundness": float(4.0 * math.pi * area_um2 / (perimeter_um * perimeter_um)) if perimeter_um else "",
        "major_axis": major_px * pixel_size_um,
        "minor_axis": minor_px * pixel_size_um,
        "eccentricity": eccentricity,
        "extent": area_px / bbox_area_px if bbox_area_px else "",
        "solidity": area_px / convex_area_px if convex_area_px else "",
        "euler_number": float(_euler_number(mask)),
        "boundary_irregularity": float(perimeter_um / math.sqrt(area_um2)) if area_um2 else "",
    }


def _box_count(mask: np.ndarray, box_size: int) -> int:
    h, w = mask.shape
    pad_h = (-h) % box_size
    pad_w = (-w) % box_size
    padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    blocks = padded.reshape(
        padded.shape[0] // box_size,
        box_size,
        padded.shape[1] // box_size,
        box_size,
    )
    return int(np.count_nonzero(blocks.any(axis=(1, 3))))


def _fractal_dimension(mask: np.ndarray) -> float | str:
    arr = np.asarray(mask, dtype=bool)
    if np.count_nonzero(arr) < 2:
        return ""
    min_dim = min(arr.shape)
    sizes = []
    size = 1
    while size <= min_dim:
        sizes.append(size)
        size *= 2
    counts = [max(_box_count(arr, size), 1) for size in sizes if _box_count(arr, size) > 0]
    sizes = sizes[: len(counts)]
    if len(counts) < 2:
        return ""
    coeff = np.polyfit(np.log(1.0 / np.asarray(sizes, dtype=float)), np.log(np.asarray(counts, dtype=float)), 1)
    return float(coeff[0])


def _lacunarity(mask: np.ndarray) -> float | str:
    arr = np.asarray(mask, dtype=np.float64)
    if arr.sum() == 0:
        return ""
    box = max(2, min(arr.shape) // 16)
    if box <= 1:
        box = 2
    sums = ndimage.uniform_filter(arr, size=box, mode="constant") * (box * box)
    values = sums[sums > 0]
    if values.size == 0:
        return ""
    mean = float(values.mean())
    return float(values.var() / (mean * mean) + 1.0) if mean else ""


def _compute_tissue_region_features(tissue_mask: np.ndarray, downsample: float, mpp: float) -> dict[str, dict]:
    pixel_size_um = float(downsample) * float(mpp)
    pixel_area_um2 = pixel_size_um * pixel_size_um
    features: dict[str, dict] = {}
    for label, name in tme_utils.TISSUE_LABELS.items():
        if label <= 0:
            continue
        mask = tissue_mask == label
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        objects = ndimage.find_objects(labels)
        regions = []
        for idx, slices in enumerate(objects, start=1):
            if slices is None:
                continue
            region = _region_stats(labels[slices] == idx, pixel_area_um2, pixel_size_um)
            if region:
                regions.append(region)
        largest = max(regions, key=lambda item: float(item.get("area") or 0.0), default={})
        features[name] = {
            "count": float(count),
            "regions": regions,
            "largest": largest,
            "fractal_dimension": _fractal_dimension(mask),
            "lacunarity": _lacunarity(mask),
            "euler_number": float(_euler_number(mask)),
        }
    return features


def _cell_coords(cells: Sequence[Mapping]) -> np.ndarray:
    return np.asarray([[float(cell["x_level0"]), float(cell["y_level0"])] for cell in cells], dtype=np.float64)


class _CellSpatialIndex:
    def __init__(self, cells: Sequence[Mapping]):
        self.cells = cells
        self.ids = [cell.get("cell_id") for cell in cells]
        self.coords = _cell_coords(cells) if cells else np.empty((0, 2), dtype=np.float64)
        self.tree = cKDTree(self.coords) if len(cells) else None


class _RadiusQueryCache:
    def __init__(self, mpp: float):
        self.mpp = float(mpp)
        self._indices: dict[int, _CellSpatialIndex] = {}
        self._summaries: dict[tuple[int, int, float], dict[str, object]] = {}

    def _index(self, cells: Sequence[Mapping]) -> _CellSpatialIndex:
        key = id(cells)
        if key not in self._indices:
            self._indices[key] = _CellSpatialIndex(cells)
        return self._indices[key]

    def summarize(
        self,
        center_cells: Sequence[Mapping],
        neighbor_cells: Sequence[Mapping],
        radius_um: float,
    ) -> dict[str, object]:
        radius = float(radius_um)
        key = (id(center_cells), id(neighbor_cells), radius)
        if key in self._summaries:
            return self._summaries[key]
        if not center_cells or not neighbor_cells:
            summary = {"count": 0, "with_count": 0, "min_mean": ""}
            self._summaries[key] = summary
            return summary

        center_index = self._index(center_cells)
        neighbor_index = self._index(neighbor_cells)
        radius_px = radius / self.mpp
        total = 0
        with_count = 0
        min_distances: list[float] = []
        chunk_size = 8192
        for start in range(0, len(center_cells), chunk_size):
            stop = min(start + chunk_size, len(center_cells))
            groups = neighbor_index.tree.query_ball_point(
                center_index.coords[start:stop],
                radius_px,
                return_sorted=False,
            )
            for rel_idx, neighbor_indices in enumerate(groups):
                center_idx = start + rel_idx
                center_id = center_index.ids[center_idx]
                valid = [int(idx) for idx in neighbor_indices if center_id != neighbor_index.ids[int(idx)]]
                if not valid:
                    continue
                total += len(valid)
                with_count += 1
                delta = neighbor_index.coords[valid] - center_index.coords[center_idx]
                min_distances.append(float(np.sqrt(np.sum(delta * delta, axis=1)).min() * self.mpp))

        summary = {
            "count": int(total),
            "with_count": int(with_count),
            "min_mean": tme_utils.safe_float_mean(min_distances),
        }
        self._summaries[key] = summary
        return summary

    def count(self, center_cells: Sequence[Mapping], neighbor_cells: Sequence[Mapping], radius_um: float) -> int:
        return int(self.summarize(center_cells, neighbor_cells, radius_um)["count"])

    def with_count(self, center_cells: Sequence[Mapping], neighbor_cells: Sequence[Mapping], radius_um: float) -> int:
        return int(self.summarize(center_cells, neighbor_cells, radius_um)["with_count"])

    def min_mean(self, center_cells: Sequence[Mapping], neighbor_cells: Sequence[Mapping], radius_um: float) -> float | str:
        return self.summarize(center_cells, neighbor_cells, radius_um)["min_mean"]


def _connected_components_by_radius(coords: np.ndarray, radius_px: float) -> list[list[int]]:
    n = int(coords.shape[0])
    if n < 2:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    tree = cKDTree(coords)
    chunk_size = 8192
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        groups_within_radius = tree.query_ball_point(coords[start:stop], float(radius_px), return_sorted=False)
        for rel_i, close in enumerate(groups_within_radius):
            i = start + rel_i
            for j in close:
                j = int(j)
                if j > i:
                    union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [group for group in groups.values() if len(group) >= 2]


def _pairwise_extent_um(pts: np.ndarray, mpp: float) -> float:
    if len(pts) < 2:
        return 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    return float(np.sqrt(np.sum(diff * diff, axis=2)).max() * float(mpp))


def _diameter_sq_degenerate(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    spread = np.ptp(pts, axis=0)
    axis = int(np.argmax(spread))
    a = pts[int(np.argmin(pts[:, axis]))]
    b = pts[int(np.argmax(pts[:, axis]))]
    delta = a - b
    return float(np.sum(delta * delta))


def _convex_polygon_diameter_sq(hull_pts: np.ndarray) -> float:
    n = int(len(hull_pts))
    if n < 2:
        return 0.0
    if n == 2:
        delta = hull_pts[0] - hull_pts[1]
        return float(np.sum(delta * delta))

    def area2(i: int, j: int, k: int) -> float:
        a = hull_pts[i]
        b = hull_pts[j]
        c = hull_pts[k]
        ab = b - a
        ac = c - a
        return abs(float(ab[0] * ac[1] - ab[1] * ac[0]))

    def dist_sq(i: int, j: int) -> float:
        delta = hull_pts[i] - hull_pts[j]
        return float(np.sum(delta * delta))

    max_sq = 0.0
    j = 1
    for i in range(n):
        ni = (i + 1) % n
        while area2(i, ni, (j + 1) % n) > area2(i, ni, j):
            j = (j + 1) % n
        max_sq = max(max_sq, dist_sq(i, j), dist_sq(ni, j))
    return max_sq


def _cluster_extent_um(pts: np.ndarray, mpp: float) -> float:
    if len(pts) <= 2048:
        return _pairwise_extent_um(pts, mpp)
    try:
        hull = ConvexHull(pts)
        diameter_sq = _convex_polygon_diameter_sq(pts[hull.vertices])
    except QhullError:
        diameter_sq = _diameter_sq_degenerate(pts)
    return float(math.sqrt(diameter_sq) * float(mpp))


def _cluster_stats_for_cells(cells: Sequence[Mapping], radius_um: float, mpp: float) -> dict[str, object]:
    if len(cells) < 2:
        return {
            "num": 0.0,
            "sizes": [],
            "extents": [],
            "dispersions": [],
            "ball_hall": "",
            "calinski": "",
        }
    coords = _cell_coords(cells)
    groups = _connected_components_by_radius(coords, radius_px=radius_um / float(mpp))
    sizes: list[float] = []
    extents: list[float] = []
    dispersions: list[float] = []
    for group in groups:
        pts = coords[group]
        centroid = pts.mean(axis=0)
        dists_um = np.sqrt(np.sum((pts - centroid) ** 2, axis=1)) * float(mpp)
        pair_extent = _cluster_extent_um(pts, mpp)
        sizes.append(float(len(group)))
        extents.append(pair_extent)
        dispersions.append(float(dists_um.mean()))
    if not groups:
        return {
            "num": 0.0,
            "sizes": [],
            "extents": [],
            "dispersions": [],
            "ball_hall": "",
            "calinski": "",
        }
    within_sq = 0.0
    labeled_count = 0
    cluster_centroids = []
    cluster_sizes = []
    for group in groups:
        pts = coords[group]
        centroid = pts.mean(axis=0)
        cluster_centroids.append(centroid)
        cluster_sizes.append(len(group))
        within_sq += float(np.sum((pts - centroid) ** 2)) * float(mpp) * float(mpp)
        labeled_count += len(group)
    ball_hall = within_sq / labeled_count if labeled_count else ""
    if len(groups) > 1 and labeled_count > len(groups):
        all_centroid = coords.mean(axis=0)
        between = 0.0
        for centroid, size in zip(cluster_centroids, cluster_sizes):
            between += float(size) * float(np.sum((centroid - all_centroid) ** 2)) * float(mpp) * float(mpp)
        calinski = (between / (len(groups) - 1)) / (within_sq / (labeled_count - len(groups))) if within_sq else ""
    else:
        calinski = ""
    return {
        "num": float(len(groups)),
        "sizes": sizes,
        "extents": extents,
        "dispersions": dispersions,
        "ball_hall": ball_hall,
        "calinski": calinski,
    }


def _disk_offsets(radius_px: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    mask = (xx * xx + yy * yy) <= radius_px * radius_px
    return np.nonzero(mask)


def _disk_kernel(radius_px: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return ((xx * xx + yy * yy) <= radius_px * radius_px).astype(np.uint16)


def _tissue_area_around_cells(
    tissue_mask: np.ndarray,
    cells: Sequence[Mapping],
    label: int,
    radius_um: float,
    downsample: float,
    mpp: float,
    cache: dict[tuple[int, int], np.ndarray],
    max_cache_items: int = 32,
) -> float:
    if not cells:
        return 0.0
    pixel_size_um = float(downsample) * float(mpp)
    pixel_area_um2 = pixel_size_um * pixel_size_um
    radius_mask_px = int(math.ceil(float(radius_um) / pixel_size_um))
    key = (id(cells), int(radius_mask_px))
    if key not in cache:
        xs = np.asarray([math.floor(float(cell["x_level0"]) / float(downsample)) for cell in cells], dtype=np.int64)
        ys = np.asarray([math.floor(float(cell["y_level0"]) / float(downsample)) for cell in cells], dtype=np.int64)
        y_offsets, x_offsets = _disk_offsets(radius_mask_px)
        label_counts = np.zeros(max(tme_utils.TISSUE_LABELS) + 1, dtype=np.float64)
        for y_offset, x_offset in zip(y_offsets, x_offsets):
            yy = ys + int(y_offset) - radius_mask_px
            xx = xs + int(x_offset) - radius_mask_px
            keep = (yy >= 0) & (yy < tissue_mask.shape[0]) & (xx >= 0) & (xx < tissue_mask.shape[1])
            if not np.any(keep):
                continue
            labels = tissue_mask[yy[keep], xx[keep]]
            label_counts += np.bincount(labels, minlength=label_counts.size)[: label_counts.size]
        cache[key] = label_counts
        if isinstance(cache, OrderedDict):
            while len(cache) > max_cache_items:
                cache.popitem(last=False)
    elif isinstance(cache, OrderedDict):
        cache.move_to_end(key)

    counts = cache[key]
    return float(counts[int(label)] * pixel_area_um2)


def compute_values(
    metrics: Sequence[str],
    tissue_mask: np.ndarray,
    tissue_metadata: Mapping,
    cells: Sequence[Mapping],
    radii_um: Sequence[float],
    spatial_working_memory_mb: float = 256.0,
) -> dict[str, object]:
    mpp = float(tissue_metadata.get("source_mpp") or tissue_metadata.get("mpp") or 1.0)
    downsample = float(tissue_metadata["downsample"])
    slide_id = str(tissue_metadata.get("slide_id") or tissue_metadata.get("file_name") or "")
    valid_area = tme_utils.valid_tissue_area_um2(tissue_mask, downsample=downsample, mpp=mpp)
    tissue_areas = {
        tissue_name: tme_utils.tissue_area_um2(tissue_mask, label, downsample=downsample, mpp=mpp)
        for label, tissue_name in tme_utils.TISSUE_LABELS.items()
        if label > 0
    }
    tissue_features = _compute_tissue_region_features(tissue_mask, downsample=downsample, mpp=mpp)
    total_cells = len(cells)

    values: dict[str, object] = {
        "File name": slide_id,
        "Slide name": Path(slide_id).stem if slide_id else "",
        "Image resolution (magnification)": tissue_metadata.get("magnification", ""),
        "Valid tissue area": valid_area,
    }

    # Pre-index common groups. The dictionaries keep loops readable and avoid repeated filtering.
    cells_by_type = {cell_type: _cells_of_type(cells, cell_type) for cell_type in tme_utils.CELL_TYPES}
    spatial_metrics = [metric for metric in metrics if spatial_readout.is_spatial_metric(metric)]
    if spatial_metrics:
        values.update(
            spatial_readout.compute_spatial_values(
                spatial_metrics,
                tissue_mask,
                cells_by_type,
                downsample=downsample,
                mpp=mpp,
                working_memory_mb=spatial_working_memory_mb,
            )
        )
    cells_by_tissue = {
        tissue_name: _cells_in_tissue(cells, tissue_name)
        for label, tissue_name in tme_utils.TISSUE_LABELS.items()
        if label > 0
    }
    cells_by_type_tissue = {
        (cell_type, tissue_name): [cell for cell in cells_by_type[cell_type] if cell.get("tissue_type") == tissue_name]
        for cell_type in tme_utils.CELL_TYPES
        for tissue_name in tissue_areas
    }
    cluster_radius_um = max(tuple(float(radius) for radius in radii_um), default=40.0)
    cluster_stats_cache: dict[tuple[str, str], dict[str, object]] = {}
    radius_query_cache = _RadiusQueryCache(mpp=mpp)
    tissue_area_around_cache: dict[tuple[str, str, float], float] = {}
    tissue_area_count_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def get_cluster_stats(cell_type: str, tissue_name: str) -> dict[str, object]:
        key = (cell_type, tissue_name)
        if key not in cluster_stats_cache:
            cluster_stats_cache[key] = _cluster_stats_for_cells(
                cells_by_type_tissue.get(key, []),
                radius_um=cluster_radius_um,
                mpp=mpp,
            )
        return cluster_stats_cache[key]

    for metric in metrics:
        if metric in values:
            continue

        m = AREA_RE.match(metric)
        if m and m.group(1) in tissue_areas:
            values[metric] = tissue_areas[m.group(1)]
            continue

        m = TISSUE_CONNECTED_COUNT_RE.match(metric)
        if m and m.group(1) in tissue_features:
            values[metric] = tissue_features[m.group(1)]["count"]
            continue

        m = TISSUE_LARGEST_CONNECTED_AREA_RE.match(metric)
        if m and m.group(1) in tissue_features:
            values[metric] = tissue_features[m.group(1)]["largest"].get("area", "")
            continue

        m = TISSUE_LARGEST_CONNECTED_RATIO_RE.match(metric)
        if m and m.group(1) in tissue_features:
            denom = tissue_areas[m.group(1)]
            area = tissue_features[m.group(1)]["largest"].get("area", 0.0)
            values[metric] = area / denom if denom else ""
            continue

        m = TISSUE_AVG_MORPH_RE.match(metric)
        if m and m.group(2) in tissue_features:
            field = m.group(1)
            values[metric] = tme_utils.safe_float_mean(
                [float(region[field]) for region in tissue_features[m.group(2)]["regions"] if region.get(field) != ""]
            )
            continue

        m = TISSUE_BOUNDARY_LARGEST_RE.match(metric)
        if m and m.group(1) in tissue_features:
            values[metric] = tissue_features[m.group(1)]["largest"].get("boundary_irregularity", "")
            continue

        m = TISSUE_BOUNDARY_RE.match(metric)
        if m and m.group(1) in tissue_features:
            values[metric] = tme_utils.safe_float_mean(
                [
                    float(region["boundary_irregularity"])
                    for region in tissue_features[m.group(1)]["regions"]
                    if region.get("boundary_irregularity") != ""
                ]
            )
            continue

        m = TISSUE_EULER_RE.match(metric)
        if m and m.group(1) in tissue_features:
            values[metric] = tissue_features[m.group(1)]["euler_number"]
            continue

        m = TISSUE_FRACTAL_RE.match(metric)
        if m and m.group(1) in tissue_features:
            values[metric] = tissue_features[m.group(1)]["fractal_dimension"]
            continue

        m = TISSUE_LACUNARITY_RE.match(metric)
        if m and m.group(1) in tissue_features:
            values[metric] = tissue_features[m.group(1)]["lacunarity"]
            continue

        m = TISSUE_LARGEST_MORPH_RE.match(metric)
        if m and m.group(2) in tissue_features:
            field = LARGEST_TISSUE_FIELD[m.group(1)]
            values[metric] = tissue_features[m.group(2)]["largest"].get(field, "")
            continue

        m = TISSUE_RATIO_VALID_RE.match(metric)
        if m and m.group(1) in tissue_areas:
            values[metric] = tissue_areas[m.group(1)] / valid_area if valid_area else ""
            continue

        m = TISSUE_RATIO_RE.match(metric)
        if m and m.group(1) in tissue_areas and m.group(2) in tissue_areas:
            denom = tissue_areas[m.group(2)]
            values[metric] = tissue_areas[m.group(1)] / denom if denom else ""
            continue

        m = CELL_NUM_RE.match(metric)
        if m and m.group(1) in cells_by_type:
            values[metric] = float(len(cells_by_type[m.group(1)]))
            continue

        m = CELL_RATIO_ALL_RE.match(metric)
        if m and m.group(1) in cells_by_type:
            values[metric] = len(cells_by_type[m.group(1)]) / total_cells if total_cells else ""
            continue

        m = CELL_DENSITY_VALID_RE.match(metric)
        if m and m.group(1) in cells_by_type:
            values[metric] = len(cells_by_type[m.group(1)]) / valid_area if valid_area else ""
            continue

        m = CELL_NUM_TISSUE_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in tissue_areas:
            values[metric] = float(
                sum(1 for cell in cells_by_type[m.group(1)] if cell.get("tissue_type") == m.group(2))
            )
            continue

        m = CELL_DENSITY_TISSUE_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in tissue_areas:
            area = tissue_areas[m.group(2)]
            count = sum(1 for cell in cells_by_type[m.group(1)] if cell.get("tissue_type") == m.group(2))
            values[metric] = count / area if area else ""
            continue

        m = CELL_DENSITY_RATIO_TISSUE_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in tissue_areas and m.group(3) in tissue_areas:
            source_area = tissue_areas[m.group(2)]
            target_area = tissue_areas[m.group(3)]
            source_count = sum(1 for cell in cells_by_type[m.group(1)] if cell.get("tissue_type") == m.group(2))
            target_count = sum(1 for cell in cells_by_type[m.group(1)] if cell.get("tissue_type") == m.group(3))
            source_density = source_count / source_area if source_area else None
            target_density = target_count / target_area if target_area else None
            values[metric] = source_density / target_density if source_density is not None and target_density else ""
            continue

        m = CELL_RATIO_TISSUE_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in cells_by_tissue:
            tissue_cells = cells_by_tissue[m.group(2)]
            count = sum(
                1 for cell in cells_by_type[m.group(1)] if cell.get("tissue_type") == m.group(2)
            )
            values[metric] = count / len(tissue_cells) if tissue_cells else ""
            continue

        m = CELL_PAIR_RATIO_TISSUE_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in cells_by_type and m.group(3) in tissue_areas:
            numerator = sum(
                1 for cell in cells_by_type[m.group(1)] if cell.get("tissue_type") == m.group(3)
            )
            denominator = sum(
                1 for cell in cells_by_type[m.group(2)] if cell.get("tissue_type") == m.group(3)
            )
            values[metric] = numerator / denominator if denominator else ""
            continue

        m = CELL_MORPH_AVG_RE.match(metric)
        if m and m.group(2) in cells_by_type:
            field = MORPH_FIELD[m.group(1)]
            values[metric] = tme_utils.safe_float_mean(_numeric_values(cells_by_type[m.group(2)], field))
            continue

        m = CELL_MORPH_STD_AREA_RE.match(metric)
        if m and m.group(1) in cells_by_type:
            values[metric] = tme_utils.safe_float_std(_numeric_values(cells_by_type[m.group(1)], "area_um2"))
            continue

        m = NEIGHBOR_COUNT_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in cells_by_type:
            radius = float(m.group(3))
            if radius in radii_um:
                values[metric] = float(radius_query_cache.count(cells_by_type[m.group(2)], cells_by_type[m.group(1)], radius))
            continue

        m = NEIGHBOR_COUNT_TISSUE_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in cells_by_type and m.group(3) in tissue_areas:
            radius = float(m.group(4))
            if radius in radii_um:
                center = cells_by_type_tissue[(m.group(2), m.group(3))]
                neighbors = cells_by_type_tissue[(m.group(1), m.group(3))]
                values[metric] = float(radius_query_cache.count(center, neighbors, radius))
            continue

        m = NEIGHBOR_RATIO_TISSUE_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in cells_by_type and m.group(3) in tissue_areas:
            radius = float(m.group(4))
            if radius in radii_um:
                center = cells_by_type_tissue[(m.group(2), m.group(3))]
                target_neighbors = cells_by_type_tissue[(m.group(1), m.group(3))]
                all_neighbors = cells_by_tissue[m.group(3)]
                numerator = radius_query_cache.count(center, target_neighbors, radius)
                denominator = radius_query_cache.count(center, all_neighbors, radius)
                values[metric] = numerator / denominator if denominator else ""
            continue

        m = NEIGHBOR_DENSITY_TISSUE_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in cells_by_type and m.group(3) in tissue_areas:
            radius = float(m.group(4))
            if radius in radii_um:
                center = cells_by_type_tissue[(m.group(2), m.group(3))]
                neighbors = cells_by_type_tissue[(m.group(1), m.group(3))]
                count = radius_query_cache.count(center, neighbors, radius)
                area = len(center) * math.pi * radius * radius
                values[metric] = count / area if area else ""
            continue

        m = TISSUE_AREA_AROUND_CELL_RE.match(metric)
        if m and m.group(1) in tissue_areas and m.group(2) in cells_by_type:
            radius = float(m.group(3))
            if radius in radii_um:
                key = (m.group(1), m.group(2), radius)
                if key not in tissue_area_around_cache:
                    label = tme_utils.TISSUE_NAME_TO_LABEL[m.group(1)]
                    tissue_area_around_cache[key] = _tissue_area_around_cells(
                        tissue_mask,
                        cells_by_type[m.group(2)],
                        label=label,
                        radius_um=radius,
                        downsample=downsample,
                        mpp=mpp,
                        cache=tissue_area_count_cache,
                    )
                values[metric] = tissue_area_around_cache[key]
            continue

        m = MIN_DISTANCE_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in cells_by_type and m.group(3) in tissue_areas:
            radius = float(m.group(4))
            if radius in radii_um:
                center = cells_by_type_tissue[(m.group(1), m.group(3))]
                neighbors = cells_by_type_tissue[(m.group(2), m.group(3))]
                values[metric] = radius_query_cache.min_mean(center, neighbors, radius)
            continue

        m = WITH_NEIGHBOR_COUNT_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in cells_by_type and m.group(3) in tissue_areas:
            radius = float(m.group(4))
            if radius in radii_um:
                center = cells_by_type_tissue[(m.group(1), m.group(3))]
                neighbors = cells_by_type_tissue[(m.group(2), m.group(3))]
                values[metric] = float(radius_query_cache.with_count(center, neighbors, radius))
            continue

        m = WITH_NEIGHBOR_RATIO_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in cells_by_type and m.group(3) in tissue_areas:
            radius = float(m.group(4))
            if radius in radii_um:
                center = cells_by_type_tissue[(m.group(1), m.group(3))]
                neighbors = cells_by_type_tissue[(m.group(2), m.group(3))]
                with_count = radius_query_cache.with_count(center, neighbors, radius)
                values[metric] = with_count / len(center) if center else ""
            continue

        m = BALL_HALL_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in tissue_areas:
            values[metric] = get_cluster_stats(m.group(1), m.group(2))["ball_hall"]
            continue

        m = CALINSKI_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in tissue_areas:
            values[metric] = get_cluster_stats(m.group(1), m.group(2))["calinski"]
            continue

        m = CLUSTER_NUM_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in tissue_areas:
            values[metric] = get_cluster_stats(m.group(1), m.group(2))["num"]
            continue

        m = CLUSTER_AVG_EXTENT_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in tissue_areas:
            values[metric] = tme_utils.safe_float_mean(get_cluster_stats(m.group(1), m.group(2))["extents"])
            continue

        m = CLUSTER_STD_EXTENT_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in tissue_areas:
            values[metric] = tme_utils.safe_float_std(get_cluster_stats(m.group(1), m.group(2))["extents"])
            continue

        m = CLUSTER_AVG_SIZE_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in tissue_areas:
            values[metric] = tme_utils.safe_float_mean(get_cluster_stats(m.group(1), m.group(2))["sizes"])
            continue

        m = CLUSTER_STD_SIZE_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in tissue_areas:
            values[metric] = tme_utils.safe_float_std(get_cluster_stats(m.group(1), m.group(2))["sizes"])
            continue

        m = CLUSTER_LARGEST_SIZE_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in tissue_areas:
            sizes = get_cluster_stats(m.group(1), m.group(2))["sizes"]
            values[metric] = max(sizes) if sizes else ""
            continue

        m = CLUSTER_AVG_DISPERSION_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in tissue_areas:
            values[metric] = tme_utils.safe_float_mean(get_cluster_stats(m.group(1), m.group(2))["dispersions"])
            continue

        m = CLUSTER_STD_DISPERSION_RE.match(metric)
        if m and m.group(1) in cells_by_type and m.group(2) in tissue_areas:
            values[metric] = tme_utils.safe_float_std(get_cluster_stats(m.group(1), m.group(2))["dispersions"])
            continue

    return values


def compute_slide_readout(
    format_csv: Path,
    tissue_mask_path: Path,
    tissue_metadata_path: Path,
    cell_table_path: Path,
    output_csv: Path,
    radii_um: Sequence[float],
    min_area_px: int | float = 0,
    cell_target_mpp: float = 0.25,
    spatial_working_memory_mb: float = 256.0,
    reassign_tissue: bool = False,
) -> dict[str, object]:
    metrics = tme_utils.load_format_metrics(format_csv)
    tissue_mask = tme_utils.read_tissue_mask(tissue_mask_path)
    metadata = tme_utils.read_json(tissue_metadata_path)
    cells = tme_utils.read_cell_table(cell_table_path)
    downsample = float(metadata["downsample"])
    if reassign_tissue:
        cells = tme_utils.assign_tissue_to_cells(cells, tissue_mask, downsample=downsample)
    cells = filter_cells_by_min_area_px(cells, min_area_px=min_area_px, target_mpp=cell_target_mpp)
    cells = filter_nested_cells_by_bbox(cells, target_mpp=cell_target_mpp)
    if cells and not all(cell.get("tissue_type") for cell in cells):
        cells = tme_utils.assign_tissue_to_cells(cells, tissue_mask, downsample=downsample)
    values = compute_values(
        metrics,
        tissue_mask,
        metadata,
        cells,
        radii_um=tuple(float(r) for r in radii_um),
        spatial_working_memory_mb=spatial_working_memory_mb,
    )
    tme_utils.write_readout_csv(output_csv, metrics, values)
    return values


def main() -> None:
    args = parse_args()
    radii = tuple(float(item) for item in args.radii_um.split(",") if item.strip())
    compute_slide_readout(
        format_csv=args.format_csv,
        tissue_mask_path=args.tissue_mask,
        tissue_metadata_path=args.tissue_metadata,
        cell_table_path=args.cell_table,
        output_csv=args.output_csv,
        radii_um=radii,
        spatial_working_memory_mb=args.spatial_working_mb,
    )


if __name__ == "__main__":
    main()
