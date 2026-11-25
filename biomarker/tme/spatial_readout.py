from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from . import tme_utils


BLOCK_UM = 500.0
MARGIN_UM = 80.0
ISOLATION_UM = 500.0
LABEL_COUNT = max(tme_utils.TISSUE_LABELS) + 1
TISSUE_NAMES = tuple(name for label, name in tme_utils.TISSUE_LABELS.items() if label > 0)
CELL_TYPES = tme_utils.CELL_TYPES

TISSUE_MORAN_RE = re.compile(r"^Tissue occupancy Moran's I at 500 um for (.+)$")
TISSUE_INTERFACE_RE = re.compile(r"^Interface length per source area from (.+) to (.+)$")
TISSUE_ISOLATION_RE = re.compile(r"^Boundary isolation beyond 500 um from (.+) to (.+)$")
CELL_MORAN_RE = re.compile(r"^Cell-density Moran's I at 500 um for (.+)$")
INTRATUMORAL_MARGIN_RE = re.compile(r"^Intratumoral 80 um margin fraction of (.+)$")
EXTRATUMORAL_MARGIN_RE = re.compile(r"^Extratumoral 80 um margin fraction of (.+)$")
BOUNDARY_COVERAGE_RE = re.compile(
    r"^Tumor-boundary 500 um block coverage by (.+) within 80 um$"
)

SPATIAL_PATTERNS = (
    TISSUE_MORAN_RE,
    TISSUE_INTERFACE_RE,
    TISSUE_ISOLATION_RE,
    CELL_MORAN_RE,
    INTRATUMORAL_MARGIN_RE,
    EXTRATUMORAL_MARGIN_RE,
    BOUNDARY_COVERAGE_RE,
)


def is_spatial_metric(metric: str) -> bool:
    return any(pattern.match(metric) for pattern in SPATIAL_PATTERNS)


def _chunk_rows(width: int, working_memory_mb: float, arrays_per_element: int = 3) -> int:
    if working_memory_mb <= 0:
        raise ValueError("working_memory_mb must be positive")
    bytes_per_row = max(1, int(width)) * 8 * max(1, int(arrays_per_element))
    return max(1, int(float(working_memory_mb) * 1024 * 1024 / bytes_per_row))


def _block_geometry(shape: tuple[int, int], pixel_size_um: float) -> tuple[np.ndarray, np.ndarray, int, int]:
    height, width = shape
    row_blocks = np.floor(np.arange(height, dtype=np.float64) * pixel_size_um / BLOCK_UM).astype(np.int64)
    col_blocks = np.floor(np.arange(width, dtype=np.float64) * pixel_size_um / BLOCK_UM).astype(np.int64)
    n_rows = int(row_blocks[-1]) + 1 if height else 0
    n_cols = int(col_blocks[-1]) + 1 if width else 0
    return row_blocks, col_blocks, n_rows, n_cols


def _block_label_histogram(
    tissue_mask: np.ndarray,
    row_blocks: np.ndarray,
    col_blocks: np.ndarray,
    n_block_rows: int,
    n_block_cols: int,
    working_memory_mb: float,
) -> np.ndarray:
    n_blocks = n_block_rows * n_block_cols
    counts = np.zeros(n_blocks * LABEL_COUNT, dtype=np.int64)
    rows_per_chunk = _chunk_rows(tissue_mask.shape[1], working_memory_mb, arrays_per_element=3)
    for start in range(0, tissue_mask.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, tissue_mask.shape[0])
        block_ids = row_blocks[start:stop, None] * n_block_cols + col_blocks[None, :]
        codes = block_ids * LABEL_COUNT + tissue_mask[start:stop].astype(np.int64, copy=False)
        counts += np.bincount(codes.ravel(), minlength=counts.size)
    return counts.reshape(n_block_rows, n_block_cols, LABEL_COUNT)


def _global_moran(values: np.ndarray, valid: np.ndarray) -> float | str:
    valid_values = np.asarray(values, dtype=np.float64)[valid]
    if valid_values.size < 2:
        return ""
    mean = float(valid_values.mean())
    deviations = np.zeros_like(values, dtype=np.float64)
    deviations[valid] = np.asarray(values, dtype=np.float64)[valid] - mean
    denominator = float(np.sum(deviations[valid] ** 2))
    if denominator <= 0:
        return ""

    horizontal = valid[:, :-1] & valid[:, 1:]
    vertical = valid[:-1, :] & valid[1:, :]
    edge_count = int(np.count_nonzero(horizontal) + np.count_nonzero(vertical))
    if edge_count == 0:
        return ""
    numerator = float(
        np.sum((deviations[:, :-1] * deviations[:, 1:])[horizontal])
        + np.sum((deviations[:-1, :] * deviations[1:, :])[vertical])
    )
    return float(valid_values.size / edge_count * numerator / denominator)


def _valid_cell_block_ids(
    cells: Sequence[Mapping],
    tissue_mask: np.ndarray,
    downsample: float,
    mpp: float,
    n_block_rows: int,
    n_block_cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not cells:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty

    x_level0 = np.asarray([float(cell["x_level0"]) for cell in cells], dtype=np.float64)
    y_level0 = np.asarray([float(cell["y_level0"]) for cell in cells], dtype=np.float64)
    mask_x = np.floor(x_level0 / downsample).astype(np.int64)
    mask_y = np.floor(y_level0 / downsample).astype(np.int64)
    in_bounds = (
        (mask_x >= 0)
        & (mask_x < tissue_mask.shape[1])
        & (mask_y >= 0)
        & (mask_y < tissue_mask.shape[0])
    )
    labels = np.zeros(len(cells), dtype=np.uint8)
    labels[in_bounds] = tissue_mask[mask_y[in_bounds], mask_x[in_bounds]]
    valid = in_bounds & (labels > 0)
    block_x = np.floor(x_level0 * mpp / BLOCK_UM).astype(np.int64)
    block_y = np.floor(y_level0 * mpp / BLOCK_UM).astype(np.int64)
    valid &= (block_x >= 0) & (block_x < n_block_cols) & (block_y >= 0) & (block_y < n_block_rows)
    block_ids = block_y * n_block_cols + block_x
    return mask_y[valid], mask_x[valid], labels[valid].astype(np.int64), block_ids[valid]


def _transition_counts(tissue_mask: np.ndarray, working_memory_mb: float) -> np.ndarray:
    counts = np.zeros(LABEL_COUNT * LABEL_COUNT, dtype=np.int64)
    height, width = tissue_mask.shape
    rows_per_chunk = _chunk_rows(width, working_memory_mb, arrays_per_element=2)
    for start in range(0, height, rows_per_chunk):
        stop = min(start + rows_per_chunk, height)
        if width > 1:
            left = tissue_mask[start:stop, :-1]
            right = tissue_mask[start:stop, 1:]
            valid = (left > 0) & (right > 0) & (left != right)
            if np.any(valid):
                codes = left[valid].astype(np.int64) * LABEL_COUNT + right[valid].astype(np.int64)
                counts += np.bincount(codes, minlength=counts.size)

        vertical_stop = min(stop, height - 1)
        if start < vertical_stop:
            upper = tissue_mask[start:vertical_stop, :]
            lower = tissue_mask[start + 1 : vertical_stop + 1, :]
            valid = (upper > 0) & (lower > 0) & (upper != lower)
            if np.any(valid):
                codes = upper[valid].astype(np.int64) * LABEL_COUNT + lower[valid].astype(np.int64)
                counts += np.bincount(codes, minlength=counts.size)
    return counts.reshape(LABEL_COUNT, LABEL_COUNT)


def _source_boundary_indices(tissue_mask: np.ndarray, source_labels: set[int]) -> dict[int, np.ndarray]:
    height, width = tissue_mask.shape
    pieces: dict[int, list[np.ndarray]] = defaultdict(list)

    if width > 1:
        left = tissue_mask[:, :-1]
        right = tissue_mask[:, 1:]
        valid = (left > 0) & (right > 0) & (left != right)
        ys, xs = np.nonzero(valid)
        if ys.size:
            left_labels = left[ys, xs]
            right_labels = right[ys, xs]
            left_flat = ys.astype(np.int64) * width + xs
            right_flat = left_flat + 1
            for label in source_labels:
                pieces[label].append(left_flat[left_labels == label])
                pieces[label].append(right_flat[right_labels == label])

    if height > 1:
        upper = tissue_mask[:-1, :]
        lower = tissue_mask[1:, :]
        valid = (upper > 0) & (lower > 0) & (upper != lower)
        ys, xs = np.nonzero(valid)
        if ys.size:
            upper_labels = upper[ys, xs]
            lower_labels = lower[ys, xs]
            upper_flat = ys.astype(np.int64) * width + xs
            lower_flat = upper_flat + width
            for label in source_labels:
                pieces[label].append(upper_flat[upper_labels == label])
                pieces[label].append(lower_flat[lower_labels == label])

    return {
        label: np.concatenate([piece for piece in pieces.get(label, []) if piece.size])
        if any(piece.size for piece in pieces.get(label, []))
        else np.empty(0, dtype=np.int64)
        for label in source_labels
    }


def _tumor_boundary_mask(tissue_mask: np.ndarray) -> np.ndarray:
    tumor_label = tme_utils.TISSUE_NAME_TO_LABEL["Tumor"]
    boundary = np.zeros_like(tissue_mask, dtype=bool)
    if tissue_mask.shape[1] > 1:
        left = tissue_mask[:, :-1]
        right = tissue_mask[:, 1:]
        interface = (left > 0) & (right > 0) & ((left == tumor_label) ^ (right == tumor_label))
        boundary[:, :-1][interface] = True
        boundary[:, 1:][interface] = True
    if tissue_mask.shape[0] > 1:
        upper = tissue_mask[:-1, :]
        lower = tissue_mask[1:, :]
        interface = (upper > 0) & (lower > 0) & ((upper == tumor_label) ^ (lower == tumor_label))
        boundary[:-1, :][interface] = True
        boundary[1:, :][interface] = True
    return boundary


def compute_spatial_values(
    metrics: Sequence[str],
    tissue_mask: np.ndarray,
    cells_by_type: Mapping[str, Sequence[Mapping]],
    downsample: float,
    mpp: float,
    working_memory_mb: float = 256.0,
) -> dict[str, object]:
    requested = [str(metric) for metric in metrics if is_spatial_metric(str(metric))]
    if not requested:
        return {}
    if working_memory_mb <= 0:
        raise ValueError("working_memory_mb must be positive")

    tissue_mask = np.asarray(tissue_mask, dtype=np.uint8)
    pixel_size_um = float(downsample) * float(mpp)
    pixel_area_um2 = pixel_size_um * pixel_size_um
    label_counts = np.bincount(tissue_mask.ravel(), minlength=LABEL_COUNT)[:LABEL_COUNT]
    row_blocks, col_blocks, n_block_rows, n_block_cols = _block_geometry(tissue_mask.shape, pixel_size_um)
    n_blocks = n_block_rows * n_block_cols
    values: dict[str, object] = {}

    tissue_moran = [(metric, TISSUE_MORAN_RE.match(metric)) for metric in requested]
    tissue_moran = [(metric, match) for metric, match in tissue_moran if match]
    cell_moran = [(metric, CELL_MORAN_RE.match(metric)) for metric in requested]
    cell_moran = [(metric, match) for metric, match in cell_moran if match]
    block_histogram: np.ndarray | None = None
    valid_counts: np.ndarray | None = None
    valid_blocks: np.ndarray | None = None
    if tissue_moran or cell_moran:
        block_histogram = _block_label_histogram(
            tissue_mask,
            row_blocks,
            col_blocks,
            n_block_rows,
            n_block_cols,
            working_memory_mb,
        )
        valid_counts = block_histogram[:, :, 1:].sum(axis=2)
        valid_blocks = valid_counts > 0

    for metric, match in tissue_moran:
        tissue_name = match.group(1)
        label = tme_utils.TISSUE_NAME_TO_LABEL.get(tissue_name)
        if label is None:
            continue
        occupancy = np.divide(
            block_histogram[:, :, label],
            valid_counts,
            out=np.zeros((n_block_rows, n_block_cols), dtype=np.float64),
            where=valid_blocks,
        )
        values[metric] = _global_moran(occupancy, valid_blocks)

    valid_area_mm2 = (
        valid_counts.astype(np.float64) * pixel_area_um2 / 1_000_000.0
        if valid_counts is not None
        else None
    )
    for metric, match in cell_moran:
        cell_type = match.group(1)
        _ys, _xs, _labels, block_ids = _valid_cell_block_ids(
            cells_by_type.get(cell_type, ()),
            tissue_mask,
            float(downsample),
            float(mpp),
            n_block_rows,
            n_block_cols,
        )
        cell_counts = np.bincount(block_ids, minlength=n_blocks).reshape(n_block_rows, n_block_cols)
        density = np.divide(
            cell_counts,
            valid_area_mm2,
            out=np.zeros((n_block_rows, n_block_cols), dtype=np.float64),
            where=valid_blocks,
        )
        values[metric] = _global_moran(density, valid_blocks)

    interface_metrics = [(metric, TISSUE_INTERFACE_RE.match(metric)) for metric in requested]
    interface_metrics = [(metric, match) for metric, match in interface_metrics if match]
    if interface_metrics:
        transitions = _transition_counts(tissue_mask, working_memory_mb)
        for metric, match in interface_metrics:
            source_name, target_name = match.groups()
            source = tme_utils.TISSUE_NAME_TO_LABEL.get(source_name)
            target = tme_utils.TISSUE_NAME_TO_LABEL.get(target_name)
            if source is None or target is None:
                continue
            if label_counts[source] == 0:
                values[metric] = ""
                continue
            edge_count = int(transitions[source, target] + transitions[target, source])
            interface_mm = edge_count * pixel_size_um / 1000.0
            source_area_mm2 = float(label_counts[source]) * pixel_area_um2 / 1_000_000.0
            values[metric] = interface_mm / source_area_mm2

    isolation_metrics = [(metric, TISSUE_ISOLATION_RE.match(metric)) for metric in requested]
    isolation_metrics = [(metric, match) for metric, match in isolation_metrics if match]
    if isolation_metrics:
        parsed_isolation = []
        for metric, match in isolation_metrics:
            source = tme_utils.TISSUE_NAME_TO_LABEL.get(match.group(1))
            target = tme_utils.TISSUE_NAME_TO_LABEL.get(match.group(2))
            if source is not None and target is not None:
                parsed_isolation.append((metric, source, target))
        boundaries = _source_boundary_indices(tissue_mask, {source for _metric, source, _target in parsed_isolation})
        by_target: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for metric, source, target in parsed_isolation:
            by_target[target].append((metric, source))
        for target, target_metrics in by_target.items():
            if label_counts[target] == 0:
                for metric, source in target_metrics:
                    values[metric] = 1.0 if boundaries[source].size else ""
                continue
            distances = ndimage.distance_transform_edt(tissue_mask != target, sampling=pixel_size_um)
            for metric, source in target_metrics:
                source_boundary = boundaries[source]
                values[metric] = (
                    float(np.mean(distances.ravel()[source_boundary] > ISOLATION_UM))
                    if source_boundary.size
                    else ""
                )
            del distances

    margin_metrics = []
    for metric in requested:
        for family, pattern in (
            ("inside", INTRATUMORAL_MARGIN_RE),
            ("outside", EXTRATUMORAL_MARGIN_RE),
            ("coverage", BOUNDARY_COVERAGE_RE),
        ):
            match = pattern.match(metric)
            if match:
                margin_metrics.append((metric, family, match.group(1)))
                break
    if margin_metrics:
        tumor_label = tme_utils.TISSUE_NAME_TO_LABEL["Tumor"]
        boundary_mask = _tumor_boundary_mask(tissue_mask)
        if label_counts[tumor_label] == 0 or not np.any(boundary_mask):
            values.update({metric: "" for metric, _family, _cell_type in margin_metrics})
        else:
            boundary_distance = ndimage.distance_transform_edt(~boundary_mask, sampling=pixel_size_um)
            boundary_y, boundary_x = np.nonzero(boundary_mask)
            boundary_block_y = row_blocks[boundary_y]
            boundary_block_x = col_blocks[boundary_x]
            boundary_blocks = np.unique(boundary_block_y * n_block_cols + boundary_block_x)
            sample_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for metric, family, cell_type in margin_metrics:
                if cell_type not in sample_cache:
                    ys, xs, labels, block_ids = _valid_cell_block_ids(
                        cells_by_type.get(cell_type, ()),
                        tissue_mask,
                        float(downsample),
                        float(mpp),
                        n_block_rows,
                        n_block_cols,
                    )
                    distances = boundary_distance[ys, xs] if ys.size else np.empty(0, dtype=np.float64)
                    sample_cache[cell_type] = (labels, block_ids, distances)
                labels, block_ids, distances = sample_cache[cell_type]
                if family == "inside":
                    eligible = labels == tumor_label
                    values[metric] = float(np.mean(distances[eligible] <= MARGIN_UM)) if np.any(eligible) else ""
                elif family == "outside":
                    eligible = labels != tumor_label
                    values[metric] = float(np.mean(distances[eligible] <= MARGIN_UM)) if np.any(eligible) else ""
                else:
                    qualifying_blocks = np.unique(block_ids[distances <= MARGIN_UM])
                    covered = np.intersect1d(boundary_blocks, qualifying_blocks, assume_unique=True).size
                    values[metric] = float(covered / boundary_blocks.size)

    return values
