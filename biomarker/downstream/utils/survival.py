"""Discrete-time survival utilities compatible with MambaMILplus.

The four intervals are defined from internal uncensored event-time quartiles.
The model output is four hazards; censoring is handled by the discrete-time
NLL rather than by converting survival into an ordinary four-class label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from pathlib import Path
import pandas as pd
import numpy as np
import torch


N_INTERVALS = 4


@dataclass(frozen=True)
class SurvivalBins:
    edges: np.ndarray

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges, dtype=float)
        if edges.shape != (N_INTERVALS + 1,) or not np.all(np.isfinite(edges)):
            raise ValueError("survival bins require five finite edges")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("survival bin edges must be strictly increasing")

    def label(self, times: Iterable[float]) -> np.ndarray:
        values = np.asarray(list(times), dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("survival times must be finite")
        # The final edge is open-ended in MambaMILplus; clamp external times.
        return np.clip(np.searchsorted(self.edges[1:-1], values, side="right"), 0, N_INTERVALS - 1).astype(int)


def fit_survival_bins(times: Iterable[float], events: Iterable[int], n_intervals: int = N_INTERVALS) -> SurvivalBins:
    if int(n_intervals) != N_INTERVALS:
        raise ValueError("the downstream survival contract uses exactly four intervals")
    times_array = np.asarray(list(times), dtype=float)
    events_array = np.asarray(list(events), dtype=int)
    if times_array.shape != events_array.shape or times_array.ndim != 1:
        raise ValueError("times and events must be one-dimensional arrays of equal length")
    observed = times_array[(events_array == 1) & np.isfinite(times_array)]
    if observed.size < N_INTERVALS:
        raise ValueError(f"at least {N_INTERVALS} uncensored internal cases are required")
    quantiles = np.quantile(observed, np.linspace(0.0, 1.0, N_INTERVALS + 1), method="linear")
    # qcut in MambaMILplus cannot represent tied quartiles.  Widen tied edges
    # by the smallest representable increment while preserving ordering.
    edges = np.asarray(quantiles, dtype=float)
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = np.nextafter(edges[index - 1], math.inf)
    edges[0] = min(edges[0], float(np.nanmin(times_array)) - 1e-6)
    edges[-1] = max(edges[-1], float(np.nanmax(times_array)) + 1e-6)
    return SurvivalBins(edges)


def discrete_nll_loss(
    logits: torch.Tensor,
    interval: torch.Tensor,
    event_observed: torch.Tensor,
    alpha: float = 0.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """MambaMILplus NLLSurvLoss for four sigmoid hazard logits."""
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    interval = interval.long().view(-1, 1)
    event_observed = event_observed.float().view(-1, 1)
    padded = torch.cat([torch.ones_like(event_observed), survival], dim=1)
    event_term = torch.log(torch.gather(padded, 1, interval).clamp_min(eps))
    event_term = event_term + torch.log(torch.gather(hazards, 1, interval).clamp_min(eps))
    censored_term = torch.log(torch.gather(padded, 1, interval + 1).clamp_min(eps))
    event_loss = -event_observed * event_term
    censored_loss = -(1.0 - event_observed) * censored_term
    loss = event_loss + censored_loss
    # alpha=0 matches the MambaMILplus configuration used in main_survival.py.
    if alpha:
        loss = (1.0 - alpha) * loss + alpha * event_loss
    return loss.mean()


def cox_partial_likelihood_loss(
    risks: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Negative Breslow-style Cox partial log-likelihood.

    Inputs are one scalar risk, follow-up time, and event indicator per case.
    Sorting times descending makes the cumulative denominator contain the
    patient and all patients still at risk at that event time.
    """
    risks = risks.reshape(-1)
    times = times.reshape(-1)
    events = events.reshape(-1)
    if not (risks.numel() == times.numel() == events.numel()):
        raise ValueError("risks, times, and events must have equal lengths")
    order = torch.argsort(times, descending=True)
    sorted_risks = risks[order]
    event_mask = events[order] > 0
    if not torch.any(event_mask):
        return sorted_risks.sum() * 0.0
    log_cumulative_hazard = torch.logcumsumexp(sorted_risks, dim=0)
    return -(sorted_risks[event_mask] - log_cumulative_hazard[event_mask]).mean()


def hazards_to_survival(hazards: np.ndarray) -> np.ndarray:
    hazards = np.asarray(hazards, dtype=float)
    if hazards.ndim != 2 or hazards.shape[1] != N_INTERVALS:
        raise ValueError(f"expected hazards with shape (n, {N_INTERVALS})")
    return np.cumprod(1.0 - hazards, axis=1)


def risk_from_hazards(hazards: np.ndarray) -> np.ndarray:
    return -np.sum(hazards_to_survival(np.asarray(hazards, dtype=float)), axis=1)


def concordance_index(times: Iterable[float], events: Iterable[int], risks: Iterable[float]) -> float:
    times_array = np.asarray(list(times), dtype=float)
    events_array = np.asarray(list(events), dtype=int)
    risks_array = np.asarray(list(risks), dtype=float)
    if not (times_array.shape == events_array.shape == risks_array.shape):
        raise ValueError("times, events and risks must have equal shape")

    valid = np.isfinite(times_array) & np.isfinite(risks_array)
    times_array = times_array[valid]
    events_array = events_array[valid]
    risks_array = risks_array[valid]
    if not len(times_array):
        return float("nan")

    # Process equal-time groups from longest to shortest survival. The Fenwick
    # tree contains exactly the observations with strictly longer times, so an
    # observed event can query its comparable later observations in O(log n).
    # Delaying insertion until the whole time group is queried preserves the
    # original rule that equal event times are not comparable.
    risk_values = np.unique(risks_array)
    risk_ranks = np.searchsorted(risk_values, risks_array) + 1
    order = np.argsort(-times_array, kind="stable")
    tree = np.zeros(len(risk_values) + 1, dtype=np.int64)

    def prefix_count(rank: int) -> int:
        count = 0
        while rank > 0:
            count += int(tree[rank])
            rank -= rank & -rank
        return count

    def add(rank: int) -> None:
        while rank < len(tree):
            tree[rank] += 1
            rank += rank & -rank

    comparable = concordant = 0.0
    seen = 0
    start = 0
    while start < len(order):
        end = start + 1
        time = times_array[order[start]]
        while end < len(order) and times_array[order[end]] == time:
            end += 1
        for position in range(start, end):
            index = order[position]
            if events_array[index] != 1:
                continue
            rank = int(risk_ranks[index])
            lower = prefix_count(rank - 1)
            equal = prefix_count(rank) - lower
            comparable += float(seen)
            concordant += float(lower) + 0.5 * float(equal)
        for position in range(start, end):
            add(int(risk_ranks[order[position]]))
            seen += 1
        start = end
    return float(concordant / comparable) if comparable else float("nan")


def bootstrap_cindex(
    times: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
    n_iters: int = 1000,
    seed: int = 12345,
) -> tuple[float, float]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    risks = np.asarray(risks, dtype=float)
    rng = np.random.default_rng(int(seed))
    strata = [np.flatnonzero(events == value) for value in (0, 1) if np.any(events == value)]
    values: list[float] = []
    for _ in range(int(n_iters)):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
        value = concordance_index(times[indices], events[indices], risks[indices])
        if math.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def write_predictions(path: Path, cases, times, events, intervals, hazards, survival, risks, survival_head: str):
    rows = []
    for index, (case, time, event, interval, risk) in enumerate(zip(cases, times, events, intervals, risks)):
        row = {"case_id": str(case), "event_time": float(time), "event_observed": int(event), "risk": float(risk)}
        if survival_head == "discrete":
            row["interval"] = int(interval)
            hazard = hazards[index]
            surv = survival[index]
            row.update({f"hazard_{i}": float(v) for i, v in enumerate(hazard)})
            row.update({f"survival_{i}": float(v) for i, v in enumerate(surv)})
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

