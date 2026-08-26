"""Accuracy metrics: MAE, RMSSE, and CRPS for probabilistic forecasts.

Metrics are computed per series and aggregated (mean and median) across the
sample, so that results can also be broken down by intermittency class.
"""

import numpy as np
import pandas as pd


def naive_scale(train: pd.DataFrame) -> pd.Series:
    """Per-series RMSSE denominator: mean squared one-step naive error.

    This is the M5 scaling term, measured on training data. Series whose
    training history is constant have a zero denominator and yield NaN.

    The same scale must be used for every condition, otherwise an RMSSE from
    H-6 is not comparable with one from H-full — each would be divided by a
    different number. Runners therefore pass the H-full training set here
    regardless of which condition they are scoring (see RMSSE_SCALE_CONDITION).
    """
    ordered = train.sort_values(["unique_id", "ds"])
    squared_diff = ordered.groupby("unique_id")["y"].diff() ** 2
    scale = squared_diff.groupby(ordered["unique_id"]).mean()
    return scale.replace(0.0, np.nan)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error, in units sold."""
    return float(np.mean(np.abs(actual - predicted)))


def rmsse(actual: np.ndarray, predicted: np.ndarray, scale: float) -> float:
    """Root mean squared scaled error for one series."""
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual - predicted) ** 2) / scale))


def crps(actual: np.ndarray, samples: np.ndarray) -> float:
    """Continuous ranked probability score from sample paths, averaged over the
    horizon.

    `samples` has shape (horizon, n_samples). Uses the energy form
    CRPS = E|X - y| - 0.5 * E|X - X'|, with the pairwise term evaluated in
    O(n log n) per step via the sorted-sample identity
    sum_ij |x_i - x_j| = 2 * sum_i (2i - n - 1) * x_(i).

    That identity already carries the factor of 2, so `spread` below is exactly
    0.5 * E|X - X'| and is subtracted as-is.
    """
    actual = np.asarray(actual, dtype="float64")
    samples = np.asarray(samples, dtype="float64")
    if samples.ndim != 2 or samples.shape[0] != actual.shape[0]:
        raise ValueError(
            f"samples must be (horizon, n_samples); got {samples.shape} "
            f"for horizon {actual.shape[0]}"
        )

    n = samples.shape[1]
    absolute_error = np.abs(samples - actual[:, None]).mean(axis=1)

    ordered = np.sort(samples, axis=1)
    weights = 2 * np.arange(1, n + 1) - n - 1
    spread = (ordered * weights).sum(axis=1) / n**2

    return float(np.mean(absolute_error - spread))


def per_series_metrics(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    scale: pd.Series,
) -> pd.DataFrame:
    """Score one model's forecasts against the test window, series by series.

    `actual` and `forecast` are long frames keyed on (unique_id, ds), carrying
    columns `y` and `yhat`. Every test observation must have exactly one
    matching forecast — a mismatch means a runner dropped or duplicated series,
    which would silently bias the aggregates, so it raises instead.
    """
    merged = actual.merge(
        forecast, on=["unique_id", "ds"], how="left", validate="one_to_one"
    )
    missing = int(merged["yhat"].isna().sum())
    if missing:
        raise ValueError(f"{missing} test rows have no forecast")

    merged = merged.sort_values(["unique_id", "ds"])
    rows = []
    for unique_id, group in merged.groupby("unique_id", sort=True):
        y = group["y"].to_numpy(dtype="float64")
        yhat = group["yhat"].to_numpy(dtype="float64")
        rows.append(
            {
                "unique_id": unique_id,
                "mae": mae(y, yhat),
                "rmsse": rmsse(y, yhat, scale.get(unique_id, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def summarise(scores: pd.DataFrame) -> dict[str, float]:
    """Collapse per-series scores into the mean and median reported per model."""
    summary: dict[str, float] = {}
    for metric in ("mae", "rmsse", "crps"):
        if metric in scores.columns:
            summary[f"{metric}_mean"] = float(scores[metric].mean())
            summary[f"{metric}_median"] = float(scores[metric].median())
    summary["n_series"] = int(len(scores))
    summary["n_rmsse_undefined"] = int(scores["rmsse"].isna().sum())
    return summary
