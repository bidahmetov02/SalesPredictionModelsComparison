"""M5 download, stratified sampling, and history truncation.

Run once before any model: `uv run python -m src.data`.

Produces, in data/processed/:
    series_metadata.csv   the 500 sampled series with their ADI/CV2 stratum
    test.parquet          the shared 30-day test window
    train_<condition>.parquet  one training set per history condition
    calendar.parquet      calendar rows covering the sampled window
    prices.parquet        sell prices for the sampled items

Idempotent: rerunning overwrites the outputs and reproduces them exactly.
"""

import argparse
import zipfile

import numpy as np
import pandas as pd
from requests.exceptions import HTTPError

from src.config import (
    ADI_INTERMITTENT_THRESHOLD,
    CALENDAR_FILE,
    CV2_LUMPY_THRESHOLD,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    HISTORY_CONDITIONS,
    KAGGLE_COMPETITION,
    MIN_HISTORY_DAYS,
    N_SAMPLED_SERIES,
    N_VOLUME_TERCILES,
    PRICES_FILE,
    RANDOM_SEED,
    SALES_FILE,
    STORE_ID,
    TEST_HORIZON_DAYS,
)

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


# --- Download ---------------------------------------------------------------


def download_m5() -> None:
    """Fetch the M5 CSVs from Kaggle unless they are already present.

    Credentials are read by the kaggle library from outside this repo, in order:
    $KAGGLE_API_TOKEN, ~/.kaggle/access_token, then ~/.kaggle/kaggle.json.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    expected = [SALES_FILE, CALENDAR_FILE, PRICES_FILE]
    if all((DATA_RAW_DIR / name).exists() for name in expected):
        print(f"raw data already present in {DATA_RAW_DIR}")
        return

    # Imported here because the kaggle package authenticates at import time and
    # raises when no credentials exist, which would break the other functions.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"downloading {KAGGLE_COMPETITION} to {DATA_RAW_DIR} ...")
    try:
        api.competition_download_files(KAGGLE_COMPETITION, path=str(DATA_RAW_DIR))
    except HTTPError as error:
        # Valid credentials still get a 403 until the competition rules are
        # accepted on the web, which cannot be done through the API.
        if error.response is not None and error.response.status_code == 403:
            raise PermissionError(
                f"Kaggle refused the download (403). Accept the rules at "
                f"https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/rules "
                f"while signed in, then rerun."
            ) from error
        raise

    archive = DATA_RAW_DIR / f"{KAGGLE_COMPETITION}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(DATA_RAW_DIR)
        archive.unlink()

    missing = [name for name in expected if not (DATA_RAW_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing after download: {missing}")


# --- Demand classification --------------------------------------------------


def first_nonzero_index(row: np.ndarray) -> int:
    """Index of the first non-zero sale, or -1 if the series is all zeros."""
    nonzero = np.flatnonzero(row)
    return int(nonzero[0]) if nonzero.size else -1


def series_stats(row: np.ndarray) -> tuple[int, float, float, float]:
    """Return (history_days, adi, cv2, mean_daily_sales) for one series.

    History starts at the first non-zero sale: M5 pads items with leading zeros
    for the period before they were stocked, and those are structural absences
    rather than observed zero demand.

    ADI is the average inter-demand interval (periods per demand event); CV2 is
    the squared coefficient of variation of the non-zero demand sizes. Together
    they place the series in the Syntetos-Boylan quadrants.
    """
    start = first_nonzero_index(row)
    if start < 0:
        return 0, np.nan, np.nan, np.nan

    history = row[start:]
    sizes = history[history > 0]
    n_days = int(history.size)
    if sizes.size < 2:
        return n_days, np.nan, np.nan, float(history.mean())

    adi = n_days / sizes.size
    cv2 = float(sizes.std(ddof=1) / sizes.mean()) ** 2
    return n_days, float(adi), cv2, float(history.mean())


def syntetos_boylan_class(adi: float, cv2: float) -> str:
    """Classify a series into one of the four Syntetos-Boylan quadrants."""
    intermittent = adi >= ADI_INTERMITTENT_THRESHOLD
    lumpy = cv2 >= CV2_LUMPY_THRESHOLD
    if intermittent and lumpy:
        return "lumpy"
    if intermittent:
        return "intermittent"
    if lumpy:
        return "erratic"
    return "smooth"


# --- Sampling ---------------------------------------------------------------


def allocate_by_stratum(sizes: pd.Series, total: int) -> pd.Series:
    """Split `total` draws across strata proportionally to their sizes.

    Uses largest-remainder rounding so the allocation sums to exactly `total`,
    and never asks a stratum for more series than it holds.
    """
    exact = sizes / sizes.sum() * total
    allocation = np.floor(exact).astype(int)
    remainder = total - int(allocation.sum())
    if remainder > 0:
        order = (exact - allocation).sort_values(ascending=False).index
        for stratum in order[:remainder]:
            allocation[stratum] += 1

    allocation = allocation.clip(upper=sizes)

    # Clipping can leave the allocation short; hand the shortfall to whichever
    # strata still have unsampled series, largest first.
    while int(allocation.sum()) < total:
        headroom = sizes - allocation
        if headroom.max() <= 0:
            break
        allocation[headroom.idxmax()] += 1
    return allocation


def stratified_sample(metadata: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Draw N_SAMPLED_SERIES series, stratified by SB class x volume tercile."""
    eligible = metadata[
        metadata["adi"].notna() & (metadata["history_days"] >= MIN_HISTORY_DAYS)
    ].copy()
    if len(eligible) < N_SAMPLED_SERIES:
        raise ValueError(
            f"only {len(eligible)} eligible series, need {N_SAMPLED_SERIES}"
        )

    eligible["volume_tercile"] = pd.qcut(
        eligible["mean_daily_sales"].rank(method="first"),
        N_VOLUME_TERCILES,
        labels=["low", "medium", "high"],
    )
    eligible["stratum"] = (
        eligible["sb_class"].astype(str)
        + "/"
        + eligible["volume_tercile"].astype(str)
    )

    sizes = eligible.groupby("stratum", observed=True).size()
    allocation = allocate_by_stratum(sizes, N_SAMPLED_SERIES)

    rng = np.random.default_rng(seed)
    picked = [
        group.sample(n=allocation[stratum], random_state=rng.integers(2**32))
        for stratum, group in eligible.groupby("stratum", observed=True)
        if allocation[stratum] > 0
    ]
    return pd.concat(picked).sort_values("id").reset_index(drop=True)


# --- Reshaping and truncation ------------------------------------------------


def to_long(sales: pd.DataFrame, day_to_date: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Melt the wide d_1..d_N sales matrix into unique_id / ds / y rows."""
    day_columns = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(
        id_vars=["id"],
        value_vars=day_columns,
        var_name="d",
        value_name="y",
    )
    long["ds"] = long["d"].map(day_to_date)
    long = long.rename(columns={"id": "unique_id"})
    long["y"] = long["y"].astype("float32")
    return long[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])


def truncate(train: pd.DataFrame, history_days: int | None) -> pd.DataFrame:
    """Keep the last `history_days` observations of each series.

    `None` means H-full: keep everything from the series' first non-zero sale.
    All conditions end on the same date, so only the start moves.
    """
    if history_days is None:
        return train
    return train.groupby("unique_id", group_keys=False).tail(history_days)


def drop_leading_zeros(train: pd.DataFrame) -> pd.DataFrame:
    """Drop each series' rows before its first non-zero sale."""
    started = train["y"].gt(0).groupby(train["unique_id"]).cummax()
    return train[started]


# --- Pipeline ---------------------------------------------------------------


def build() -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    calendar = pd.read_csv(DATA_RAW_DIR / CALENDAR_FILE, parse_dates=["date"])
    day_to_date = dict(zip(calendar["d"], calendar["date"]))

    sales = pd.read_csv(DATA_RAW_DIR / SALES_FILE)
    sales = sales[sales["store_id"] == STORE_ID].reset_index(drop=True)
    if sales.empty:
        raise ValueError(f"no series found for store {STORE_ID}")

    day_columns = [c for c in sales.columns if c.startswith("d_")]
    test_days = day_columns[-TEST_HORIZON_DAYS:]
    train_days = day_columns[:-TEST_HORIZON_DAYS]

    # Stratification statistics are computed on the training window only, so the
    # held-out 30 days never influence which series are sampled.
    train_values = sales[train_days].to_numpy(dtype="float64")
    stats = [series_stats(row) for row in train_values]

    metadata = sales[ID_COLUMNS].copy()
    metadata[["history_days", "adi", "cv2", "mean_daily_sales"]] = pd.DataFrame(
        stats, index=metadata.index
    )
    metadata["sb_class"] = [
        syntetos_boylan_class(adi, cv2) if pd.notna(adi) else "undefined"
        for adi, cv2 in zip(metadata["adi"], metadata["cv2"])
    ]

    sampled = stratified_sample(metadata, RANDOM_SEED)
    sampled.to_csv(DATA_PROCESSED_DIR / "series_metadata.csv", index=False)

    sales = sales[sales["id"].isin(sampled["id"])].reset_index(drop=True)

    test = to_long(sales[["id"] + test_days], day_to_date)
    test.to_parquet(DATA_PROCESSED_DIR / "test.parquet", index=False)

    full_train = drop_leading_zeros(to_long(sales[["id"] + train_days], day_to_date))
    for condition, history_days in HISTORY_CONDITIONS.items():
        subset = truncate(full_train, history_days)
        subset.to_parquet(
            DATA_PROCESSED_DIR / f"train_{condition}.parquet", index=False
        )
        per_series = subset.groupby("unique_id").size()
        print(
            f"{condition:>7}: {len(subset):>9,} rows  "
            f"median {int(per_series.median())} days/series"
        )

    calendar_window = calendar[calendar["d"].isin(day_columns)]
    calendar_window.to_parquet(DATA_PROCESSED_DIR / "calendar.parquet", index=False)

    prices = pd.read_csv(DATA_RAW_DIR / PRICES_FILE)
    prices = prices[
        (prices["store_id"] == STORE_ID)
        & (prices["item_id"].isin(sampled["item_id"]))
    ]
    prices.to_parquet(DATA_PROCESSED_DIR / "prices.parquet", index=False)

    print(f"\nsampled {len(sampled)} series from store {STORE_ID}")
    print(sampled.groupby(["sb_class", "volume_tercile"], observed=True).size())
    print(f"test window: {test['ds'].min().date()} .. {test['ds'].max().date()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="assume the M5 CSVs are already in data/raw/",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_m5()
    build()


if __name__ == "__main__":
    main()
