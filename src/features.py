"""Feature construction for the global LightGBM model.

Features are lags, rolling means of past sales, calendar attributes and sell
price, as specified in experimental_design.md.

Sales are held in a right-aligned (n_series, n_days) array so that one function,
`sales_features`, produces the predictors for a given day. Training stacks it
over every day; recursive forecasting calls it once per step. Sharing that one
function is what keeps the training matrix and the forecast-time inputs
identical — the usual source of silent error in recursive multi-step models.
"""

import numpy as np
import pandas as pd

from src.config import LGBM_LAG_DAYS, LGBM_ROLLING_WINDOWS, STORE_ID

SALES_FEATURES = [f"lag_{lag}" for lag in LGBM_LAG_DAYS] + [
    f"roll_mean_{window}" for window in LGBM_ROLLING_WINDOWS
]
CALENDAR_FEATURES = ["day_of_week", "month", "snap", "is_event"]
FEATURE_COLUMNS = SALES_FEATURES + CALENDAR_FEATURES + ["sell_price"]
CATEGORICAL_FEATURES = ["day_of_week", "month"]

MAX_LOOKBACK = max(max(LGBM_LAG_DAYS), max(LGBM_ROLLING_WINDOWS))


def to_wide(panel: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
    """Pivot a long panel into a (n_series, n_days) sales array.

    Series start on different dates, so days before a series begins are NaN.
    All series end on the same date, which is what makes the array right-aligned
    and the lag arithmetic below valid across every condition.
    """
    wide = panel.pivot(index="unique_id", columns="ds", values="y").sort_index(axis=1)
    return wide.to_numpy(dtype="float64"), list(wide.index), wide.columns


def sales_features(values: np.ndarray, day: int) -> np.ndarray:
    """Lag and rolling-mean predictors for every series on column `day`.

    Reads only columns strictly before `day`, so a forecast for `day` never sees
    its own target. Returns (n_series, len(SALES_FEATURES)).
    """
    if day < MAX_LOOKBACK:
        raise ValueError(f"day {day} has fewer than {MAX_LOOKBACK} prior columns")

    columns = [values[:, day - lag] for lag in LGBM_LAG_DAYS]
    for window in LGBM_ROLLING_WINDOWS:
        past = values[:, day - window : day]
        with np.errstate(invalid="ignore"):
            columns.append(np.nanmean(past, axis=1))
    return np.column_stack(columns)


def calendar_table(dates: pd.DatetimeIndex, calendar: pd.DataFrame) -> pd.DataFrame:
    """Per-date calendar predictors, indexed by date."""
    snap_column = f"snap_{STORE_ID.split('_')[0]}"
    table = calendar.set_index("date").reindex(dates)
    return pd.DataFrame(
        {
            "day_of_week": table.index.dayofweek,
            "month": table.index.month,
            "snap": table[snap_column].to_numpy(dtype="float64"),
            # M5 records up to two events on a date; either one marks the day.
            "is_event": (
                table["event_name_1"].notna() | table["event_name_2"].notna()
            ).to_numpy(dtype="float64"),
        },
        index=dates,
    )


def price_matrix(
    series_ids: list[str],
    dates: pd.DatetimeIndex,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> np.ndarray:
    """Sell price per series per day, as (n_series, n_days).

    Prices are weekly in M5, so they are joined through the calendar's wm_yr_wk.
    Weeks in which an item was not offered have no price row and stay NaN, which
    LightGBM handles natively.
    """
    items = pd.Series(series_ids, name="unique_id").str.rsplit("_", n=3).str[0]
    week_of_date = calendar.set_index("date")["wm_yr_wk"].reindex(dates)

    lookup = (
        prices.pivot_table(
            index="item_id", columns="wm_yr_wk", values="sell_price", aggfunc="first"
        )
        .reindex(index=items.to_numpy())
        .reindex(columns=week_of_date.to_numpy())
    )
    return lookup.to_numpy(dtype="float64")


def day_features(
    values: np.ndarray,
    day: int,
    calendar_values: np.ndarray,
    prices: np.ndarray,
) -> np.ndarray:
    """Every series' predictors for a single day, ordered as FEATURE_COLUMNS.

    The one place features are assembled: training stacks this over many days and
    the recursive forecast calls it per step, so the two cannot drift apart.
    """
    return np.column_stack(
        [
            sales_features(values, day),
            np.repeat(calendar_values[day][None, :], values.shape[0], axis=0),
            prices[:, day],
        ]
    )


def build_design_matrix(
    values: np.ndarray,
    calendar_values: np.ndarray,
    prices: np.ndarray,
    days: range,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-day feature blocks into a training matrix and its targets.

    Rows whose target is NaN — days before a series began — are dropped.
    """
    blocks = [day_features(values, day, calendar_values, prices) for day in days]
    targets = [values[:, day] for day in days]

    features = np.vstack(blocks)
    target = np.concatenate(targets)

    observed = ~np.isnan(target)
    return features[observed], target[observed]
