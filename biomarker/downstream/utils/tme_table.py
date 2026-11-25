from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


NON_FEATURE_COLUMNS = {
    "slide_id",
    "File name",
    "Slide name",
    "Image resolution (magnification)",
}
SLIDE_SUFFIXES = (".pt", ".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".kfb")
AREA_UM2_TO_MM2 = 1e-6
MIN_DISTANCE_RADIUS_RE = re.compile(r"^Average minimum distance from .+ within ([0-9.]+) um$")
NEIGHBOR_RATIO_RE = re.compile(r"^Ratio of .+ to all neighboring cells around .+ within ([0-9.]+) um$")
WITH_NEIGHBOR_RATIO_RE = re.compile(r"^Ratio of .+ with at least one .+ neighbor in .+ within ([0-9.]+) um$")
SEMANTIC_ZERO_MISSING_PREFIXES = (
    "Calinski-Harabasz index of ",
    "Ball-Hall index of ",
    "Average cluster extent of ",
    "Standard deviation of cluster extent of ",
    "Average cluster size of ",
    "Standard deviation of cluster size of ",
    "Largest cluster size of ",
    "Average within-cluster dispersion of ",
    "Standard deviation of within-cluster dispersion of ",
)


@dataclass
class ReadoutEntry:
    slide_id: str
    features: dict[str, float]
    path: str


@dataclass
class TaskSample:
    sample_id: str
    case_id: str
    slide_id: str
    label: str | None
    event_time: float | None
    event_observed: int | None
    splits: dict[str, str]
    features: dict[str, float]
    readout_path: str
    n_slides: int = 1


@dataclass
class MissingReadout:
    case_id: str
    slide_id: str
    label: str | None
    event_time: float | None
    event_observed: int | None
    splits: dict[str, str]
    reason: str


@dataclass
class TaskTable:
    samples: list[TaskSample]
    missing: list[MissingReadout]
    feature_names: list[str]
    sample_level: str = "case"


class ReadoutIndex:
    def __init__(self, entries: list[ReadoutEntry], feature_names: list[str]):
        self.entries = entries
        self.feature_names = feature_names
        self._by_key: dict[str, ReadoutEntry] = {}
        for entry in entries:
            for key in _slide_key_variants(entry.slide_id):
                self._by_key.setdefault(key, entry)

    def get(self, slide_id: str) -> ReadoutEntry | None:
        for key in _slide_key_variants(slide_id):
            entry = self._by_key.get(key)
            if entry is not None:
                return entry
        return None


class TabularPreprocessor:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.raw_feature_names: list[str] = []
        self.all_feature_transforms: list[str] = []
        self.valid_idx: np.ndarray | None = None
        self.keep_idx_within_valid: np.ndarray | None = None
        self.medians: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None
        self.feature_names: list[str] = []
        self.feature_transforms: list[str] = []

    def fit(self, x_train: np.ndarray, feature_names: list[str]) -> "TabularPreprocessor":
        self._fit_kept_matrix(x_train, feature_names)
        return self

    def fit_transform(self, x_train: np.ndarray, feature_names: list[str]) -> np.ndarray:
        x_kept = self._fit_kept_matrix(x_train, feature_names)
        return (x_kept - self.means) / self.stds

    def _fit_kept_matrix(self, x_train: np.ndarray, feature_names: list[str]) -> np.ndarray:
        if x_train.ndim != 2:
            raise ValueError("x_train must be a 2D array")
        if x_train.shape[1] != len(feature_names):
            raise ValueError("feature_names length does not match x_train columns")
        self.raw_feature_names = list(feature_names)
        x_train, transforms = transform_raw_feature_matrix(x_train, feature_names)
        self.all_feature_transforms = transforms
        valid = ~np.all(np.isnan(x_train), axis=0)
        if not np.any(valid):
            raise ValueError("No usable TME readout features after dropping all-NaN columns")
        valid_idx = np.where(valid)[0]
        x_valid = x_train[:, valid_idx].astype(float, copy=True)
        medians = np.nanmedian(x_valid, axis=0)
        medians[~np.isfinite(medians)] = 0.0
        x_valid = _impute_with_median(x_valid, medians)
        means = np.mean(x_valid, axis=0)
        stds = np.std(x_valid, axis=0)
        keep = np.isfinite(stds) & (stds > self.eps)
        if not np.any(keep):
            raise ValueError("No usable TME readout features after dropping zero-variance columns")
        self.valid_idx = valid_idx
        self.keep_idx_within_valid = np.where(keep)[0]
        self.medians = medians[keep]
        self.means = means[keep]
        self.stds = stds[keep]
        self.feature_names = [feature_names[int(valid_idx[i])] for i in self.keep_idx_within_valid]
        self.feature_transforms = [transforms[int(valid_idx[i])] for i in self.keep_idx_within_valid]
        return x_valid[:, keep]

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.valid_idx is None or self.keep_idx_within_valid is None:
            raise RuntimeError("TabularPreprocessor must be fit before transform")
        if self.medians is None or self.means is None or self.stds is None:
            raise RuntimeError("TabularPreprocessor parameters are missing")
        x, _ = transform_raw_feature_matrix(x, self.raw_feature_names)
        x_valid = x[:, self.valid_idx].astype(float, copy=True)
        x_valid = x_valid[:, self.keep_idx_within_valid]
        x_valid = _impute_with_median(x_valid, self.medians)
        return (x_valid - self.means) / self.stds

    def save(self, path: Path) -> None:
        if self.valid_idx is None or self.keep_idx_within_valid is None:
            raise RuntimeError("Cannot save an unfitted TabularPreprocessor")
        np.savez(
            path,
            valid_idx=self.valid_idx,
            keep_idx_within_valid=self.keep_idx_within_valid,
            medians=self.medians,
            means=self.means,
            stds=self.stds,
            raw_feature_names=np.asarray(self.raw_feature_names, dtype=object),
            all_feature_transforms=np.asarray(self.all_feature_transforms, dtype=object),
            feature_names=np.asarray(self.feature_names, dtype=object),
            feature_transforms=np.asarray(self.feature_transforms, dtype=object),
        )


def _impute_with_median(x: np.ndarray, medians: np.ndarray) -> np.ndarray:
    missing = ~np.isfinite(x)
    if np.any(missing):
        rows, cols = np.where(missing)
        x[rows, cols] = medians[cols]
    return x


def is_area_um2_feature(name: str) -> bool:
    name = name.strip()
    if name == "Valid tissue area":
        return True
    if name.startswith("Area of "):
        return True
    if name.startswith("Average area of "):
        return True
    if name.startswith("Standard deviation of area of "):
        return True
    if name.startswith("Largest convex area of "):
        return True
    if name.startswith("Largest filled area of "):
        return True
    return False


def is_count_feature(name: str) -> bool:
    name = name.strip()
    if name.startswith("Number of "):
        return True
    if name.startswith("Average cluster size of "):
        return True
    if name.startswith("Standard deviation of cluster size of "):
        return True
    if name.startswith("Largest cluster size of "):
        return True
    return False


def is_raw_area_number_feature(name: str) -> bool:
    return is_area_um2_feature(name) or (
        is_count_feature(name) and not is_radius_based_spatial_count_feature(name)
    )


def is_radius_based_spatial_count_feature(name: str) -> bool:
    name = name.strip()
    return name.startswith("Number of ") and " within " in name and name.endswith(" um")


def feature_set_policy_version(feature_set: str) -> str:
    normalized = feature_set.strip().lower()
    versions = {
        "full": "full-v1",
        "no_absolute_burden": "no_absolute_burden-v1",
        "no_raw_area_number": "no_raw_area_number-v2-keep-radius-counts",
    }
    if normalized not in versions:
        raise ValueError(f"Unsupported feature set: {feature_set}")
    return versions[normalized]


def is_absolute_burden_feature(name: str) -> bool:
    name = name.strip()
    if name == "Valid tissue area":
        return True
    if name.startswith("Area of "):
        return True
    if name.startswith("Largest convex area of "):
        return True
    if name.startswith("Largest filled area of "):
        return True
    if name.startswith("Number of "):
        return True
    return False


def select_feature_names(feature_names: list[str], feature_set: str) -> list[str]:
    normalized = feature_set.strip().lower()
    if normalized == "full":
        return list(feature_names)
    if normalized == "no_absolute_burden":
        return [name for name in feature_names if not is_absolute_burden_feature(name)]
    if normalized == "no_raw_area_number":
        return [name for name in feature_names if not is_raw_area_number_feature(name)]
    raise ValueError(f"Unsupported feature set: {feature_set}")


def is_length_um_feature(name: str) -> bool:
    name = name.strip()
    lowered = name.lower()
    if "distance" in lowered:
        return True
    if "perimeter" in lowered and "boundary irregularity" not in lowered:
        return True
    if "major axis" in lowered or "minor axis" in lowered:
        return True
    if "cluster extent" in lowered:
        return True
    if "within-cluster dispersion" in lowered:
        return True
    return False


def semantic_missing_fill_value(name: str) -> float | None:
    name = name.strip()
    if name.startswith(SEMANTIC_ZERO_MISSING_PREFIXES):
        return 0.0
    if NEIGHBOR_RATIO_RE.match(name) or WITH_NEIGHBOR_RATIO_RE.match(name):
        return 0.0
    match = MIN_DISTANCE_RADIUS_RE.match(name)
    if match:
        try:
            radius = float(match.group(1))
        except ValueError:
            return None
        return radius if math.isfinite(radius) and radius >= 0 else None
    return None


def transform_raw_feature_matrix(x: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names length does not match x columns")
    transformed = x.astype(float, copy=True)
    transforms: list[str] = []
    for col_idx, name in enumerate(feature_names):
        col = transformed[:, col_idx]
        fill_value = semantic_missing_fill_value(name)
        if fill_value is not None:
            missing = ~np.isfinite(col)
            col[missing] = fill_value
        valid = np.isfinite(col)
        nonnegative = valid & (col >= 0)
        invalid_negative = valid & (col < 0)
        if name.startswith("prior_") and name.endswith("_area_mm2"):
            col[nonnegative] = np.log1p(col[nonnegative])
            col[invalid_negative] = np.nan
            transforms.append("area_mm2_to_log1p_mm2")
        elif name.startswith("prior_") and ("_count" in name or "_density_per_mm2" in name):
            col[nonnegative] = np.log1p(col[nonnegative])
            col[invalid_negative] = np.nan
            transforms.append("log1p_count" if "_count" in name else "log1p_density")
        elif is_area_um2_feature(name):
            col[nonnegative] = np.log1p(col[nonnegative] * AREA_UM2_TO_MM2)
            col[invalid_negative] = np.nan
            transforms.append("area_um2_to_log1p_mm2")
        elif is_count_feature(name):
            col[nonnegative] = np.log1p(col[nonnegative])
            col[invalid_negative] = np.nan
            transforms.append("log1p_count")
        elif is_length_um_feature(name):
            col[nonnegative] = np.log1p(col[nonnegative])
            col[invalid_negative] = np.nan
            transforms.append("log1p_length_um")
        else:
            transforms.append("none")
    return transformed, transforms


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def normalize_slide_id(value: str) -> str:
    value = (value or "").strip().strip("\"'")
    if not value:
        return ""
    name = Path(value).name
    lowered = name.lower()
    for suffix in SLIDE_SUFFIXES:
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.strip()


def parse_slide_ids(row: dict[str, str]) -> list[str]:
    raw = row.get("slide") or row.get("Slide") or row.get("WSI") or row.get("wsi") or ""
    slide_ids: list[str] = []
    seen: set[str] = set()
    for item in raw.split(";"):
        slide_id = normalize_slide_id(item)
        if slide_id and slide_id not in seen:
            slide_ids.append(slide_id)
            seen.add(slide_id)
    return slide_ids


def _slide_key_variants(slide_id: str) -> list[str]:
    normalized = normalize_slide_id(slide_id)
    values = [normalized, normalized.lower()]
    if normalized.endswith(".svs"):
        values.append(normalized[:-4])
        values.append(normalized[:-4].lower())
    return [value for value in values if value]


def _to_float(value: str | float | int | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (float, int)):
        return float(value)
    value = str(value).strip()
    if not value:
        return float("nan")
    try:
        number = float(value)
    except ValueError:
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _append_feature_names(feature_names: list[str], seen: set[str], candidates: Iterable[str]) -> None:
    for name in candidates:
        if name in NON_FEATURE_COLUMNS or name in seen:
            continue
        feature_names.append(name)
        seen.add(name)


def _read_long_readout(path: Path) -> tuple[dict[str, float], list[str]]:
    features: dict[str, float] = {}
    names: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = (row.get("metric") or "").strip()
            if not metric or metric in NON_FEATURE_COLUMNS:
                continue
            features[metric] = _to_float(row.get("results"))
            names.append(metric)
    return features, names


def _split_readout_cohort_names(cohort: str) -> list[str]:
    names = [part.strip() for part in re.split(r"[,+;]", str(cohort)) if part.strip()]
    return names or [str(cohort).strip()]


def _candidate_readout_cohorts(readout_root: Path, cohort: str) -> list[str]:
    names = _split_readout_cohort_names(cohort)
    if len(names) == 1 and names[0].upper() == "CRC" and not (Path(readout_root) / names[0]).exists():
        return ["COAD", "READ"]
    return names


def _build_single_readout_index(
    readout_root: Path,
    cohort: str,
    readout_filename: str = "TME_Readout.csv",
    matrix_filename: str = "TME_Readout_Matrix.csv",
) -> ReadoutIndex:
    cohort_dir = Path(readout_root) / cohort
    if not cohort_dir.exists():
        raise FileNotFoundError(f"Readout cohort directory does not exist: {cohort_dir}")

    matrix_path = cohort_dir / matrix_filename
    entries: list[ReadoutEntry] = []
    feature_names: list[str] = []
    seen_features: set[str] = set()
    if matrix_path.exists():
        with matrix_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "slide_id" not in reader.fieldnames:
                raise ValueError(f"Matrix readout must contain slide_id column: {matrix_path}")
            _append_feature_names(feature_names, seen_features, reader.fieldnames)
            for row in reader:
                slide_id = normalize_slide_id(row.get("slide_id", ""))
                if not slide_id:
                    continue
                features = {name: _to_float(row.get(name)) for name in feature_names}
                entries.append(ReadoutEntry(slide_id=slide_id, features=features, path=str(matrix_path)))
        return ReadoutIndex(entries, feature_names)

    for path in sorted(cohort_dir.glob(f"*/{readout_filename}")):
        slide_id = normalize_slide_id(path.parent.name)
        if not slide_id:
            continue
        features, names = _read_long_readout(path)
        _append_feature_names(feature_names, seen_features, names)
        entries.append(ReadoutEntry(slide_id=slide_id, features=features, path=str(path)))
    if not entries:
        raise FileNotFoundError(f"No TME readout CSVs found under {cohort_dir}")
    return ReadoutIndex(entries, feature_names)


def build_readout_index(
    readout_root: Path,
    cohort: str,
    readout_filename: str = "TME_Readout.csv",
    matrix_filename: str = "TME_Readout_Matrix.csv",
) -> ReadoutIndex:
    cohort_names = _candidate_readout_cohorts(Path(readout_root), cohort)
    if len(cohort_names) == 1:
        return _build_single_readout_index(
            Path(readout_root),
            cohort_names[0],
            readout_filename=readout_filename,
            matrix_filename=matrix_filename,
        )

    entries: list[ReadoutEntry] = []
    feature_names: list[str] = []
    seen_features: set[str] = set()
    errors: list[str] = []
    for cohort_name in cohort_names:
        try:
            index = _build_single_readout_index(
                Path(readout_root),
                cohort_name,
                readout_filename=readout_filename,
                matrix_filename=matrix_filename,
            )
        except FileNotFoundError as exc:
            errors.append(str(exc))
            continue
        _append_feature_names(feature_names, seen_features, index.feature_names)
        entries.extend(index.entries)
    if not entries:
        joined = "; ".join(errors) if errors else ", ".join(cohort_names)
        raise FileNotFoundError(f"No TME readouts found for cohort specification {cohort!r}: {joined}")
    return ReadoutIndex(entries, feature_names)


def _case_id(row: dict[str, str], row_idx: int) -> str:
    return (row.get("case") or row.get("ID") or row.get("id") or f"row{row_idx}").strip()


def _split_values(row: dict[str, str], task_type: str) -> dict[str, str]:
    if task_type == "classification":
        return {f"fold{i}": (row.get(f"fold{i}") or "").strip().lower() for i in range(1, 6)}
    if task_type == "survival":
        return {f"Fold {i}": (row.get(f"Fold {i}") or "").strip().lower() for i in range(5)}
    raise ValueError(f"Unknown task type: {task_type}")


SPLIT_PRIORITY = {"": 0, "unassigned": 0, "train": 1, "val": 2, "test": 3}


def _merge_splits(split_rows: list[dict[str, str]]) -> dict[str, str]:
    merged: dict[str, str] = {}
    split_keys = sorted({key for splits in split_rows for key in splits})
    for key in split_keys:
        values = [(splits.get(key, "") or "").strip().lower() for splits in split_rows]
        merged[key] = max(values, key=lambda value: SPLIT_PRIORITY.get(value, -1))
    return merged


def _classification_duplicate_slide_policy(
    rows: list[dict[str, str]],
    task_type: str,
) -> tuple[set[tuple[int, str]], set[tuple[int, str]], dict[tuple[str, str, str], dict[str, str]]]:
    if task_type != "classification":
        return set(), set(), {}
    grouped: dict[tuple[str, str], list[tuple[int, str, str, dict[str, str]]]] = defaultdict(list)
    for row_idx, row in enumerate(rows):
        case_id = _case_id(row, row_idx)
        label = (row.get("label") or "").strip()
        splits = _split_values(row, task_type)
        for slide_id in parse_slide_ids(row):
            grouped[(case_id, normalize_slide_id(slide_id))].append((row_idx, slide_id, label, splits))

    drop_keys: set[tuple[int, str]] = set()
    for (case_id, slide_key), entries in grouped.items():
        labels = {label for _, _, label, _ in entries}
        if len(labels) <= 1:
            continue
        normal_entries = [
            (row_idx, slide_id, label, splits)
            for row_idx, slide_id, label, splits in entries
            if label.lower() == "normal"
        ]
        non_normal_labels = {label for _, _, label, _ in entries if label.lower() != "normal"}
        if normal_entries and len(non_normal_labels) == 1:
            for row_idx, slide_id, _, _ in normal_entries:
                drop_keys.add((row_idx, normalize_slide_id(slide_id)))
            continue
        raise ValueError(
            f"Case {case_id} slide {slide_key} has conflicting labels for the same slide: {sorted(labels)}"
        )

    grouped_by_label: dict[tuple[str, str, str], list[tuple[int, str, dict[str, str]]]] = defaultdict(list)
    for (case_id, slide_key), entries in grouped.items():
        for row_idx, slide_id, label, splits in entries:
            if (row_idx, normalize_slide_id(slide_id)) in drop_keys:
                continue
            grouped_by_label[(case_id, slide_key, label)].append((row_idx, slide_id, splits))

    duplicate_row_drop_keys: set[tuple[int, str]] = set()
    merged_splits: dict[tuple[str, str, str], dict[str, str]] = {}
    for key, entries in grouped_by_label.items():
        merged_splits[key] = _merge_splits([splits for _, _, splits in entries])
        for row_idx, slide_id, _ in entries[1:]:
            duplicate_row_drop_keys.add((row_idx, normalize_slide_id(slide_id)))
    return drop_keys, duplicate_row_drop_keys, merged_splits


def _event_observed(value: str | None) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError("Missing survival Status column")
    return int(float(value))


def _mean_feature_dict(samples: list[TaskSample], feature_names: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    for name in feature_names:
        values = [sample.features.get(name, float("nan")) for sample in samples]
        finite_values = [float(value) for value in values if math.isfinite(float(value))]
        features[name] = float(np.mean(finite_values)) if finite_values else float("nan")
    return features


def _check_case_consistency(first: TaskSample, sample: TaskSample) -> None:
    if first.label != sample.label:
        raise ValueError(f"Case {first.case_id} has inconsistent labels across slides: {first.label!r} vs {sample.label!r}")
    if first.event_observed != sample.event_observed:
        raise ValueError(
            f"Case {first.case_id} has inconsistent survival Status values across slides: "
            f"{first.event_observed!r} vs {sample.event_observed!r}"
        )
    if first.event_time is None or sample.event_time is None:
        if first.event_time != sample.event_time:
            raise ValueError(
                f"Case {first.case_id} has inconsistent survival Event values across slides: "
                f"{first.event_time!r} vs {sample.event_time!r}"
            )
    elif not math.isclose(float(first.event_time), float(sample.event_time), rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            f"Case {first.case_id} has inconsistent survival Event values across slides: "
            f"{first.event_time!r} vs {sample.event_time!r}"
        )
    if first.splits != sample.splits:
        raise ValueError(f"Case {first.case_id} has inconsistent fold assignments across slides")


def _aggregate_case_samples(samples: list[TaskSample], feature_names: list[str]) -> list[TaskSample]:
    grouped: dict[str, list[TaskSample]] = defaultdict(list)
    case_order: list[str] = []
    for sample in samples:
        if sample.case_id not in grouped:
            case_order.append(sample.case_id)
        grouped[sample.case_id].append(sample)

    aggregated: list[TaskSample] = []
    for case_id in case_order:
        group = grouped[case_id]
        first = group[0]
        for sample in group[1:]:
            _check_case_consistency(first, sample)
        slide_ids = [sample.slide_id for sample in group]
        readout_paths = [sample.readout_path for sample in group]
        aggregated.append(
            TaskSample(
                sample_id=case_id,
                case_id=case_id,
                slide_id=";".join(slide_ids),
                label=first.label,
                event_time=first.event_time,
                event_observed=first.event_observed,
                splits=dict(first.splits),
                features=_mean_feature_dict(group, feature_names),
                readout_path=";".join(readout_paths),
                n_slides=len(group),
            )
        )
    return aggregated


def _check_all_case_consistency(samples: list[TaskSample]) -> None:
    grouped: dict[str, list[TaskSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.case_id].append(sample)
    for group in grouped.values():
        first = group[0]
        for sample in group[1:]:
            _check_case_consistency(first, sample)


def _annotate_slide_counts(samples: list[TaskSample]) -> None:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.case_id] = counts.get(sample.case_id, 0) + 1
    for sample in samples:
        sample.n_slides = counts[sample.case_id]


def build_task_table(
    label_csv: Path,
    readout_root: Path,
    cohort: str,
    task_type: str,
    sample_level: str = "case",
    readout_filename: str = "TME_Readout.csv",
    matrix_filename: str = "TME_Readout_Matrix.csv",
) -> TaskTable:
    sample_level = sample_level.strip().lower()
    if sample_level not in {"case", "slide"}:
        raise ValueError(f"sample_level must be 'case' or 'slide', got: {sample_level!r}")
    rows = read_csv_rows(Path(label_csv))
    readouts = build_readout_index(
        Path(readout_root),
        cohort,
        readout_filename=readout_filename,
        matrix_filename=matrix_filename,
    )
    drop_duplicate_label_keys, drop_duplicate_row_keys, canonical_splits = _classification_duplicate_slide_policy(
        rows, task_type
    )
    slide_samples: list[TaskSample] = []
    missing: list[MissingReadout] = []
    for row_idx, row in enumerate(rows):
        case_id = _case_id(row, row_idx)
        slide_ids = parse_slide_ids(row)
        row_splits = _split_values(row, task_type)
        label = (row.get("label") or "").strip() if task_type == "classification" else None
        event_time = _to_float(row.get("Event")) if task_type == "survival" else None
        event_observed = _event_observed(row.get("Status")) if task_type == "survival" else None
        for slide_id in slide_ids:
            slide_key = normalize_slide_id(slide_id)
            splits = canonical_splits.get((case_id, slide_key, label), row_splits)
            if (row_idx, slide_key) in drop_duplicate_label_keys:
                missing.append(
                    MissingReadout(
                        case_id=case_id,
                        slide_id=slide_id,
                        label=label,
                        event_time=event_time,
                        event_observed=event_observed,
                        splits=splits,
                        reason="excluded_conflicting_duplicate_slide_label",
                    )
                )
                continue
            if (row_idx, slide_key) in drop_duplicate_row_keys:
                missing.append(
                    MissingReadout(
                        case_id=case_id,
                        slide_id=slide_id,
                        label=label,
                        event_time=event_time,
                        event_observed=event_observed,
                        splits=splits,
                        reason="excluded_duplicate_slide_label_row",
                    )
                )
                continue
            readout = readouts.get(slide_id)
            if readout is None:
                missing.append(
                    MissingReadout(
                        case_id=case_id,
                        slide_id=slide_id,
                        label=label,
                        event_time=event_time,
                        event_observed=event_observed,
                        splits=splits,
                        reason="missing_tme_readout",
                    )
                )
                continue
            slide_samples.append(
                TaskSample(
                    sample_id=f"{case_id}|{slide_id}",
                    case_id=case_id,
                    slide_id=slide_id,
                    label=label,
                    event_time=event_time,
                    event_observed=event_observed,
                    splits=splits,
                    features=readout.features,
                    readout_path=readout.path,
                )
            )
    _annotate_slide_counts(slide_samples)
    _check_all_case_consistency(slide_samples)
    samples = slide_samples if sample_level == "slide" else _aggregate_case_samples(slide_samples, readouts.feature_names)
    return TaskTable(samples=samples, missing=missing, feature_names=readouts.feature_names, sample_level=sample_level)


def matrix_from_samples(samples: list[TaskSample], feature_names: list[str]) -> np.ndarray:
    matrix = np.empty((len(samples), len(feature_names)), dtype=float)
    for row_idx, sample in enumerate(samples):
        for col_idx, name in enumerate(feature_names):
            matrix[row_idx, col_idx] = sample.features.get(name, float("nan"))
    return matrix


def save_jsonable_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


PRIOR_EXCLUDED_COLUMNS = {"slide_id", "status", "prior_mask_downsample"}


def _number(value: object) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


@lru_cache(maxsize=None)
def _prior_features(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        return {}
    return {
        str(name).strip(): _number(value)
        for name, value in row.items()
        if str(name).strip().startswith("prior_")
        and str(name).strip() not in PRIOR_EXCLUDED_COLUMNS
    }


def build_qc_external_table(
    *,
    label_csv: Path,
    readout_root: Path,
    cohort: str,
    task_type: str,
    required_features: list[str] | tuple[str, ...],
) -> TaskTable:
    """Build an external slide table from QC TME files and adjacent prior files."""
    table = build_task_table(
        label_csv=Path(label_csv),
        readout_root=Path(readout_root),
        cohort=cohort,
        task_type=task_type,
        sample_level="slide",
        readout_filename="TME_Readout.csv",
        matrix_filename="TME_Readout_Matrix.csv",
    )
    prior_names: list[str] = []
    prior_rows: list[dict[str, float]] = []
    for sample in table.samples:
        prior = _prior_features(Path(sample.readout_path).with_name("Prior_Readout.csv"))
        prior_rows.append(prior)
        prior_names.extend(name for name in prior if name not in prior_names)
    required_prior = [str(name) for name in required_features if str(name).startswith("prior_")]
    prior_names.extend(name for name in required_prior if name not in prior_names)
    for sample, prior in zip(table.samples, prior_rows):
        for name in prior_names:
            sample.features[name] = prior.get(name, float("nan"))
    names = list(table.feature_names)
    names.extend(name for name in prior_names if name not in names)
    return replace(table, feature_names=names)

