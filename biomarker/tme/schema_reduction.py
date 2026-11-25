from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence


TISSUE_AREA_RATIO_RE = re.compile(r"^Ratio of (.+) area to (.+) area$")
CELL_PAIR_RATIO_RE = re.compile(r"^Ratio of (.+) to (.+) in (.+)$")
CELL_DENSITY_RATIO_RE = re.compile(r"^Ratio of (.+) density in (.+) to (.+)$")
NEIGHBOR_COUNT_TISSUE_RE = re.compile(
    r"^Number of (.+) around (.+) in (.+) within ([0-9.]+) um$"
)
NEIGHBOR_COUNT_RE = re.compile(r"^Number of (.+) around (.+) within ([0-9.]+) um$")
WITH_NEIGHBOR_COUNT_RE = re.compile(
    r"^Number of (.+) with at least one (.+) neighbor in (.+) within ([0-9.]+) um$"
)
FORTY_UM_RE = re.compile(r"\bwithin 40(?:\.0+)? um$")
MIN_DISTANCE_RE = re.compile(r"^Average minimum distance .+ within [0-9.]+ um$")
AVERAGE_EXTENT_RE = re.compile(r"^Average extent of .+$")
LARGEST_BOUNDARY_RE = re.compile(
    r"^Boundary irregularity index of largest connected region of .+$"
)
LARGEST_CONVEX_AREA_RE = re.compile(r"^Largest convex area of .+$")


@dataclass(frozen=True)
class ReductionResult:
    metrics: list[str]
    audit: list[dict[str, str]]


def _reciprocal_ratio(metric: str, available: set[str]) -> str | None:
    match = TISSUE_AREA_RATIO_RE.match(metric)
    if match:
        candidate = f"Ratio of {match.group(2)} area to {match.group(1)} area"
        return candidate if candidate != metric and candidate in available else None

    match = CELL_DENSITY_RATIO_RE.match(metric)
    if match:
        candidate = f"Ratio of {match.group(1)} density in {match.group(3)} to {match.group(2)}"
        return candidate if candidate != metric and candidate in available else None

    match = CELL_PAIR_RATIO_RE.match(metric)
    if match:
        candidate = f"Ratio of {match.group(2)} to {match.group(1)} in {match.group(3)}"
        return candidate if candidate != metric and candidate in available else None

    return None


def _symmetric_neighbor_count(metric: str, available: set[str]) -> str | None:
    match = NEIGHBOR_COUNT_TISSUE_RE.match(metric)
    if match:
        candidate = (
            f"Number of {match.group(2)} around {match.group(1)} in {match.group(3)} "
            f"within {match.group(4)} um"
        )
        return candidate if candidate != metric and candidate in available else None

    match = NEIGHBOR_COUNT_RE.match(metric)
    if match:
        candidate = f"Number of {match.group(2)} around {match.group(1)} within {match.group(3)} um"
        return candidate if candidate != metric and candidate in available else None

    return None


def _radius_counterpart(metric: str, available: set[str]) -> str | None:
    candidate = FORTY_UM_RE.sub("within 80 um", metric)
    return candidate if candidate != metric and candidate in available else None


def reduce_metrics(metrics: Sequence[str]) -> ReductionResult:
    """Apply the public TME schema rules while preserving canonical input order."""
    normalized = [str(metric).strip() for metric in metrics]
    duplicates = sorted(metric for metric, count in Counter(normalized).items() if count > 1)
    if duplicates:
        preview = ", ".join(repr(metric) for metric in duplicates[:3])
        raise ValueError(f"duplicate metric names are not allowed: {preview}")

    available = set(normalized)
    retained: list[str] = []
    retained_set: set[str] = set()
    audit: list[dict[str, str]] = []

    def remove(metric: str, rule: str, counterpart: str | None = None) -> None:
        audit.append(
            {
                "removed_metric": metric,
                "rule": rule,
                "counterpart": counterpart or "",
            }
        )

    for metric in normalized:
        reciprocal = _reciprocal_ratio(metric, available)
        if reciprocal in retained_set:
            remove(metric, "reciprocal_ratio", reciprocal)
            continue

        symmetric = _symmetric_neighbor_count(metric, available)
        if symmetric in retained_set:
            remove(metric, "symmetric_neighbor_count", symmetric)
            continue

        if WITH_NEIGHBOR_COUNT_RE.match(metric):
            remove(metric, "with_neighbor_count")
            continue

        if FORTY_UM_RE.search(metric) and not MIN_DISTANCE_RE.match(metric):
            remove(metric, "single_80um_radius", _radius_counterpart(metric, available))
            continue

        if AVERAGE_EXTENT_RE.match(metric):
            remove(metric, "average_extent")
            continue

        if LARGEST_BOUNDARY_RE.match(metric):
            remove(metric, "largest_boundary_irregularity")
            continue

        if LARGEST_CONVEX_AREA_RE.match(metric):
            remove(metric, "largest_convex_area")
            continue

        retained.append(metric)
        retained_set.add(metric)

    for row in audit:
        counterpart = row["counterpart"]
        if row["rule"] != "single_80um_radius" or not counterpart or counterpart in retained_set:
            continue
        symmetric = _symmetric_neighbor_count(counterpart, available)
        if symmetric in retained_set:
            row["counterpart"] = symmetric

    return ReductionResult(metrics=retained, audit=audit)
