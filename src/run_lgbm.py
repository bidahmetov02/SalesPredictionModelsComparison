"""Run the global LightGBM model for one history condition.

    uv run python -m src.run_lgbm --condition H-6

One model is trained across all 500 series per condition, then the 30-day
horizon is produced recursively: each predicted day becomes the lag input for
the next, which is why inference cost is reported separately from training.
"""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    CLIP_NEGATIVE_FORECASTS,
    DATA_PROCESSED_DIR,
    HISTORY_CONDITIONS,
    LGBM_NUM_BOOST_ROUND,
    LGBM_PARAMS,
    RESULTS_DIR,
    RMSSE_SCALE_CONDITION,
    TEST_HORIZON_DAYS,
)
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MAX_LOOKBACK,
    build_design_matrix,
    calendar_table,
    day_features,
    price_matrix,
    to_wide,
)
from src.harness import measure
from src.metrics import naive_scale, per_series_metrics, summarise

MODEL_NAME = "LightGBM"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=list(HISTORY_CONDITIONS))
    parser.add_argument(
        "--limit",
        type=int,
        help="run on only the first N series, for smoke tests; results are "
        "written under a separate filename so they cannot overwrite a real run",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(DATA_PROCESSED_DIR / f"train_{args.condition}.parquet")
    test = pd.read_parquet(DATA_PROCESSED_DIR / "test.parquet")
    calendar = pd.read_parquet(DATA_PROCESSED_DIR / "calendar.parquet")
    prices_raw = pd.read_parquet(DATA_PROCESSED_DIR / "prices.parquet")
    scale = naive_scale(
        pd.read_parquet(DATA_PROCESSED_DIR / f"train_{RMSSE_SCALE_CONDITION}.parquet")
    )

    if args.limit:
        kept = sorted(train["unique_id"].unique())[: args.limit]
        train = train[train["unique_id"].isin(kept)]
        test = test[test["unique_id"].isin(kept)]
        print(f"SMOKE TEST: limited to {len(kept)} series")

    train_values, series_ids, train_dates = to_wide(train)
    test_dates = pd.DatetimeIndex(sorted(test["ds"].unique()))
    all_dates = train_dates.append(test_dates)

    # Test columns start as NaN and are filled in by the recursive loop.
    values = np.hstack(
        [train_values, np.full((len(series_ids), len(test_dates)), np.nan)]
    )
    calendar_values = calendar_table(all_dates, calendar).to_numpy(dtype="float64")
    prices = price_matrix(series_ids, all_dates, calendar, prices_raw)

    n_train_days = len(train_dates)
    features, target = build_design_matrix(
        values, calendar_values, prices, range(MAX_LOOKBACK, n_train_days)
    )
    print(
        f"training rows {len(target):,} over {n_train_days - MAX_LOOKBACK} days "
        f"x {len(series_ids)} series"
    )

    def fit() -> lgb.Booster:
        dataset = lgb.Dataset(
            features,
            label=target,
            feature_name=FEATURE_COLUMNS,
            categorical_feature=CATEGORICAL_FEATURES,
            free_raw_data=False,
        )
        return lgb.train(LGBM_PARAMS, dataset, num_boost_round=LGBM_NUM_BOOST_ROUND)

    booster, train_cost = measure(fit)

    def forecast() -> np.ndarray:
        # Works on a copy: measure() reruns this, and the recursion writes its
        # own predictions back into the array it reads from.
        work = values.copy()
        for day in range(n_train_days, len(all_dates)):
            predicted = booster.predict(day_features(work, day, calendar_values, prices))
            if CLIP_NEGATIVE_FORECASTS:
                predicted = np.maximum(predicted, 0.0)
            work[:, day] = predicted
        return work[:, n_train_days:]

    predictions, inference_cost = measure(forecast)

    forecast_long = pd.DataFrame(
        {
            "unique_id": np.repeat(series_ids, TEST_HORIZON_DAYS),
            "ds": np.tile(test_dates.to_numpy(), len(series_ids)),
            "yhat": predictions.ravel(),
        }
    )

    scores = per_series_metrics(test, forecast_long, scale)
    scores.insert(0, "condition", args.condition)
    scores.insert(0, "model", MODEL_NAME)

    cost = {f"train_{k}": v for k, v in train_cost.items()}
    cost.update({f"inference_{k}": v for k, v in inference_cost.items()})
    summary = {
        "model": MODEL_NAME,
        "condition": args.condition,
        **summarise(scores),
        **cost,
    }

    print(
        f"{MODEL_NAME:>14}  RMSSE {summary['rmsse_mean']:.4f}  MAE {summary['mae_mean']:.4f}"
        f"  train {summary['train_seconds_median']:.2f}s"
        f"  infer {summary['inference_seconds_median']:.2f}s"
        f"  peak {summary['train_peak_rss_mb']:.0f}MB"
    )

    stem = f"lgbm_{args.condition}"
    if args.limit:
        stem += f"_limit{args.limit}"
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{stem}_summary.csv", index=False)
    scores.to_csv(RESULTS_DIR / f"{stem}_per_series.csv", index=False)
    print(f"wrote {RESULTS_DIR / stem}_summary.csv and _per_series.csv")


if __name__ == "__main__":
    main()
