"""Feature association screening and cross-cohort pooling."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from scipy.stats import chi2, norm, t as student_t

def bh_adjust(values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg adjustment preserving missing values."""
    p = np.asarray(list(values), dtype=float)
    result = np.full(p.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return result
    ordered = finite[np.argsort(p[finite], kind="mergesort")]
    raw = p[ordered] * finite.size / np.arange(1, finite.size + 1)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    result[ordered] = np.clip(adjusted, 0.0, 1.0)
    return result

def _empty_result(status: str, *, n: int, n_positive: int, n_negative: int, iqr: float = np.nan) -> dict[str, object]:
    return {
        "n": int(n),
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "iqr": float(iqr),
        "log_odds_ratio": np.nan,
        "standard_error": np.nan,
        "odds_ratio": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "log2_odds_ratio": np.nan,
        "likelihood_ratio": np.nan,
        "p_value": np.nan,
        "status": status,
    }

def fit_logistic_feature(
    feature: Sequence[float] | np.ndarray,
    outcome: Sequence[float] | np.ndarray,
    *,
    minimum_class_count: int = 5,
) -> dict[str, object]:
    """Fit an intercept-plus-feature logistic model on a one-IQR scale."""
    x = np.asarray(feature, dtype=float)
    y = np.asarray(outcome, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & np.isin(y, [0.0, 1.0])
    x = x[keep]
    y = y[keep]
    n_positive = int(np.sum(y == 1))
    n_negative = int(np.sum(y == 0))
    if min(n_positive, n_negative) < minimum_class_count:
        return _empty_result("insufficient_class_count", n=len(y), n_positive=n_positive, n_negative=n_negative)
    iqr = float(np.subtract(*np.quantile(x, [0.75, 0.25])))
    if not np.isfinite(iqr) or iqr <= 0:
        return _empty_result("zero_iqr", n=len(y), n_positive=n_positive, n_negative=n_negative, iqr=iqr)

    scaled = (x - np.median(x)) / iqr
    design = np.column_stack([np.ones(len(scaled)), scaled])

    def objective(beta: np.ndarray) -> float:
        eta = design @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        return design.T @ (expit(design @ beta) - y)

    prevalence = float(np.clip(y.mean(), 1e-8, 1.0 - 1e-8))
    initial = np.array([np.log(prevalence / (1.0 - prevalence)), 0.0])
    fit = minimize(objective, initial, jac=gradient, method="BFGS", options={"gtol": 1e-8, "maxiter": 400})
    beta = np.asarray(fit.x, dtype=float)
    if (not np.all(np.isfinite(beta))) or abs(beta[1]) > 25:
        return _empty_result("separation_or_divergence", n=len(y), n_positive=n_positive, n_negative=n_negative, iqr=iqr)
    fitted_probability = expit(design @ beta)
    weights = fitted_probability * (1.0 - fitted_probability)
    information = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(information)
        standard_error = float(np.sqrt(covariance[1, 1]))
    except np.linalg.LinAlgError:
        return _empty_result("singular_information", n=len(y), n_positive=n_positive, n_negative=n_negative, iqr=iqr)
    if not np.isfinite(standard_error) or standard_error <= 0:
        return _empty_result("singular_information", n=len(y), n_positive=n_positive, n_negative=n_negative, iqr=iqr)

    null_nll = float(-np.sum(y * np.log(prevalence) + (1.0 - y) * np.log(1.0 - prevalence)))
    likelihood_ratio = max(0.0, 2.0 * (null_nll - objective(beta)))
    p_value = float(chi2.sf(likelihood_ratio, 1))
    z = float(norm.ppf(0.975))
    log_effect = float(beta[1])
    return {
        "n": int(len(y)),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "iqr": iqr,
        "log_odds_ratio": log_effect,
        "standard_error": standard_error,
        "odds_ratio": float(np.exp(log_effect)),
        "ci_low": float(np.exp(log_effect - z * standard_error)),
        "ci_high": float(np.exp(log_effect + z * standard_error)),
        "log2_odds_ratio": float(log_effect / np.log(2.0)),
        "likelihood_ratio": likelihood_ratio,
        "p_value": p_value,
        "status": "ok" if fit.success or np.linalg.norm(gradient(beta), ord=np.inf) < 1e-5 else "optimizer_warning",
    }

def screen_cohort(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    outcome_column: str,
    cohort: str,
    minimum_class_count: int = 5,
) -> pd.DataFrame:
    """Fit all eligible features in one cohort and apply within-cohort BH FDR."""
    rows: list[dict[str, object]] = []
    for feature in features:
        result = fit_logistic_feature(
            pd.to_numeric(frame[feature], errors="coerce"),
            pd.to_numeric(frame[outcome_column], errors="coerce"),
            minimum_class_count=minimum_class_count,
        )
        rows.append({"cohort": cohort, "feature": feature, **result})
    output = pd.DataFrame(rows)
    output["q_value"] = bh_adjust(output["p_value"])
    return output

def _empty(status: str, n: int, events: int, iqr: float = np.nan) -> dict[str, object]:
    return {"n": int(n), "n_events": int(events), "n_censored": int(n - events), "iqr": float(iqr),
            "log_hazard_ratio": np.nan, "standard_error": np.nan, "hazard_ratio": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "log2_hazard_ratio": np.nan,
            "likelihood_ratio": np.nan, "p_value": np.nan, "status": status}

def _cox_terms(beta: float, x: np.ndarray, time: np.ndarray, event: np.ndarray) -> tuple[float, float, float]:
    """Return negative partial log likelihood, gradient and observed information."""
    unique_events = np.unique(time[event == 1])
    nll = 0.0
    gradient = 0.0
    information = 0.0
    eta = beta * x
    for t in unique_events:
        event_mask = (time == t) & (event == 1)
        risk_mask = time >= t
        xr = x[risk_mask]
        er = eta[risk_mask]
        m = int(event_mask.sum())
        if not m:
            continue
        max_eta = float(np.max(er))
        weights = np.exp(er - max_eta)
        denom = float(weights.sum())
        mean = float(np.sum(weights * xr) / denom)
        variance = float(np.sum(weights * (xr - mean) ** 2) / denom)
        nll -= float(np.sum(eta[event_mask]) - m * (max_eta + np.log(denom)))
        gradient -= float(np.sum(x[event_mask]) - m * mean)
        information += m * variance
    return nll, gradient, information

def fit_cox_feature(
    feature: Sequence[float] | np.ndarray,
    time: Sequence[float] | np.ndarray,
    event: Sequence[float] | np.ndarray,
    *,
    minimum_event_count: int = 10,
) -> dict[str, object]:
    """Fit a one-feature Cox model on a one-IQR scale using Breslow ties."""
    x = np.asarray(feature, dtype=float)
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=float)
    keep = np.isfinite(x) & np.isfinite(t) & np.isfinite(e) & (t > 0) & np.isin(e, [0.0, 1.0])
    x, t, e = x[keep], t[keep], e[keep]
    n_events = int(e.sum())
    if n_events < minimum_event_count:
        return _empty("insufficient_event_count", len(x), n_events)
    iqr = float(np.subtract(*np.quantile(x, [0.75, 0.25])))
    if not np.isfinite(iqr) or iqr <= 0:
        return _empty("zero_iqr", len(x), n_events, iqr)
    if len(x) - n_events < 1:
        return _empty("insufficient_censoring_support", len(x), n_events, iqr)
    scaled = (x - np.median(x)) / iqr
    null_nll, _, _ = _cox_terms(0.0, scaled, t, e)
    # The scalar likelihood is convex for a regular univariate Cox model. A
    # generous bounded search keeps pathological near-separation finite.
    objective = lambda beta: _cox_terms(float(beta), scaled, t, e)[0]
    fit = minimize_scalar(objective, bounds=(-30.0, 30.0), method="bounded", options={"xatol": 1e-8, "maxiter": 500})
    beta = float(fit.x)
    fitted_nll, gradient, information = _cox_terms(beta, scaled, t, e)
    if not np.isfinite(beta) or abs(beta) >= 29.9 or not np.isfinite(fitted_nll):
        return _empty("separation_or_divergence", len(x), n_events, iqr)
    if not np.isfinite(information) or information <= 1e-12:
        return _empty("singular_information", len(x), n_events, iqr)
    standard_error = float(1.0 / np.sqrt(information))
    lrt = max(0.0, float(2.0 * (null_nll - fitted_nll)))
    p_value = float(chi2.sf(lrt, 1))
    z = float(norm.ppf(0.975))
    return {"n": int(len(x)), "n_events": n_events, "n_censored": int(len(x) - n_events), "iqr": iqr,
            "log_hazard_ratio": beta, "standard_error": standard_error, "hazard_ratio": float(np.exp(beta)),
            "ci_low": float(np.exp(beta - z * standard_error)), "ci_high": float(np.exp(beta + z * standard_error)),
            "log2_hazard_ratio": float(beta / np.log(2.0)), "likelihood_ratio": lrt, "p_value": p_value,
            "status": "ok" if abs(gradient) < 1e-4 else "optimizer_warning"}

def screen_survival(frame: pd.DataFrame, features: Sequence[str], *, cohort: str,
                    time_column: str = "time", event_column: str = "event",
                    minimum_event_count: int = 10) -> pd.DataFrame:
    rows = []
    for feature in features:
        result = fit_cox_feature(frame[feature], frame[time_column], frame[event_column], minimum_event_count=minimum_event_count)
        rows.append({"cohort": cohort, "feature": feature, **result})
    output = pd.DataFrame(rows)
    output["q_value"] = bh_adjust(output["p_value"])
    return output

def meta_analyse(effects: np.ndarray, standard_errors: np.ndarray) -> dict[str, object]:
    """DerSimonian-Laird random-effects pooling for log odds ratios."""
    effects = np.asarray(effects, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)
    keep = np.isfinite(effects) & np.isfinite(standard_errors) & (standard_errors > 0)
    effects = effects[keep]
    standard_errors = standard_errors[keep]
    k = int(effects.size)
    if k == 0:
        return {
            "n_cohorts": 0,
            "pooled_log_odds_ratio": np.nan,
            "pooled_standard_error": np.nan,
            "pooled_odds_ratio": np.nan,
            "pooled_ci_low": np.nan,
            "pooled_ci_high": np.nan,
            "meta_p_value": np.nan,
            "q_statistic": np.nan,
            "tau_squared": np.nan,
            "i_squared": np.nan,
            "prediction_low": np.nan,
            "prediction_high": np.nan,
        }
    variances = standard_errors**2
    fixed_weights = 1.0 / variances
    fixed_mean = float(np.sum(fixed_weights * effects) / np.sum(fixed_weights))
    q_statistic = float(np.sum(fixed_weights * (effects - fixed_mean) ** 2))
    degrees_freedom = k - 1
    c_value = float(np.sum(fixed_weights) - np.sum(fixed_weights**2) / np.sum(fixed_weights))
    tau_squared = max(0.0, (q_statistic - degrees_freedom) / c_value) if k > 1 and c_value > 0 else 0.0
    random_weights = 1.0 / (variances + tau_squared)
    pooled = float(np.sum(random_weights * effects) / np.sum(random_weights))
    pooled_se = float(np.sqrt(1.0 / np.sum(random_weights)))
    z = float(norm.ppf(0.975))
    meta_p = float(2.0 * norm.sf(abs(pooled / pooled_se)))
    i_squared = max(0.0, (q_statistic - degrees_freedom) / q_statistic) * 100.0 if k > 1 and q_statistic > 0 else 0.0
    prediction_low = np.nan
    prediction_high = np.nan
    if k >= 3:
        multiplier = float(student_t.ppf(0.975, k - 2))
        prediction_se = float(np.sqrt(tau_squared + pooled_se**2))
        prediction_low = float(np.exp(pooled - multiplier * prediction_se))
        prediction_high = float(np.exp(pooled + multiplier * prediction_se))
    return {
        "n_cohorts": k,
        "pooled_log_odds_ratio": pooled,
        "pooled_standard_error": pooled_se,
        "pooled_odds_ratio": float(np.exp(pooled)),
        "pooled_ci_low": float(np.exp(pooled - z * pooled_se)),
        "pooled_ci_high": float(np.exp(pooled + z * pooled_se)),
        "meta_p_value": meta_p,
        "q_statistic": q_statistic,
        "tau_squared": tau_squared,
        "i_squared": float(np.clip(i_squared, 0.0, 100.0)),
        "prediction_low": prediction_low,
        "prediction_high": prediction_high,
    }

def assign_evidence_tier(
    cohort_rows: pd.DataFrame,
    *,
    meta_q_value: float,
    required_cohorts: int | None = None,
    alpha: float = 0.05,
    effect_column: str = "log_odds_ratio",
) -> str:
    """Assign the prespecified cross-cohort evidence tier."""
    usable = cohort_rows.loc[
        np.isfinite(pd.to_numeric(cohort_rows[effect_column], errors="coerce"))
        & np.isfinite(pd.to_numeric(cohort_rows["q_value"], errors="coerce"))
    ].copy()
    if len(usable) < 2:
        return "landscape_only"
    if required_cohorts is not None and len(usable) != required_cohorts:
        return "none"
    directions = np.sign(usable[effect_column].to_numpy(float))
    same_direction = bool(directions[0] != 0 and np.all(directions == directions[0]))
    if not same_direction or not np.isfinite(meta_q_value) or meta_q_value >= alpha:
        return "none"
    significant = usable["q_value"].to_numpy(float) < alpha
    if bool(np.all(significant)):
        return "S"
    if bool(np.any(significant)):
        return "A"
    return "none"

def pool_features(
    cohort_results: pd.DataFrame,
    *,
    required_cohorts: int | None = None,
    effect_column: str = "log_odds_ratio",
    effect_prefix: str = "odds",
) -> pd.DataFrame:
    """Pool every estimable feature, correct meta p values, and assign tiers."""
    rows: list[dict[str, object]] = []
    if required_cohorts is None:
        required_cohorts = int(cohort_results["cohort"].nunique())
    grouped: dict[str, pd.DataFrame] = {}
    for feature, group in cohort_results.groupby("feature", sort=True):
        usable = group.loc[
            group["status"].isin(["ok", "optimizer_warning"])
            & np.isfinite(pd.to_numeric(group[effect_column], errors="coerce"))
            & np.isfinite(pd.to_numeric(group["standard_error"], errors="coerce"))
        ].copy()
        grouped[str(feature)] = usable
        summary = meta_analyse(usable[effect_column].to_numpy(float), usable["standard_error"].to_numpy(float))
        if effect_prefix != "odds":
            summary = {
                "n_cohorts": summary["n_cohorts"],
                f"pooled_log_{effect_prefix}_ratio": summary["pooled_log_odds_ratio"],
                "pooled_standard_error": summary["pooled_standard_error"],
                f"pooled_{effect_prefix}_ratio": summary["pooled_odds_ratio"],
                f"pooled_{effect_prefix}_ci_low": summary["pooled_ci_low"],
                f"pooled_{effect_prefix}_ci_high": summary["pooled_ci_high"],
                "meta_p_value": summary["meta_p_value"], "q_statistic": summary["q_statistic"],
                "tau_squared": summary["tau_squared"], "i_squared": summary["i_squared"],
                f"prediction_{effect_prefix}_low": summary["prediction_low"],
                f"prediction_{effect_prefix}_high": summary["prediction_high"],
            }
        nominal = int(np.sum(pd.to_numeric(usable["p_value"], errors="coerce") < 0.05)) if "p_value" in usable else 0
        directions = np.sign(usable[effect_column].to_numpy(float))
        rows.append(
            {
                "feature": feature,
                **summary,
                "n_nominal_cohorts": nominal,
                "direction_consistent": bool(len(directions) > 0 and directions[0] != 0 and np.all(directions == directions[0])),
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        output["meta_q_value"] = pd.Series(dtype=float)
        output["tier"] = pd.Series(dtype=object)
        return output
    output["meta_q_value"] = bh_adjust(output["meta_p_value"])
    output["tier"] = [
        assign_evidence_tier(
            grouped[str(row.feature)],
            meta_q_value=float(row.meta_q_value),
            required_cohorts=required_cohorts,
            effect_column=effect_column,
        )
        for row in output.itertuples(index=False)
    ]
    return output.sort_values(["meta_q_value", "feature"], na_position="last").reset_index(drop=True)
