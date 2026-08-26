"""Run the five statistical baselines for one history condition.

    uv run python -m src.run_stats --condition H-6

Each model is fitted in its own StatsForecast instance so that training and
inference cost is attributed per model rather than to the batch as a whole.
"""

import argparse

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    CrostonClassic,
    Naive,
    SeasonalNaive,
)

from src.config import (
    CLIP_NEGATIVE_FORECASTS,
    DATA_PROCESSED_DIR,
    FREQ,
    HISTORY_CONDITIONS,
    RESULTS_DIR,
    RMSSE_SCALE_CONDITION,
    SEASONAL_PERIOD,
    STATSFORECAST_N_JOBS,
    TEST_HORIZON_DAYS,
)
from src.harness import measure, time_median
from src.metrics import naive_scale, per_series_metrics, summarise

MODELS = {
    "Naive": lambda: Naive(),
    "SeasonalNaive": lambda: SeasonalNaive(season_length=SEASONAL_PERIOD),
    "AutoETS": lambda: AutoETS(season_length=SEASONAL_PERIOD),
    "AutoARIMA": lambda: AutoARIMA(season_length=SEASONAL_PERIOD),
    "Croston": lambda: CrostonClassic(),
}


def pool_overhead_seconds(train: pd.DataFrame) -> float:
    """Fixed cost of spawning statsforecast's worker pool, in seconds.

    Every model here is fitted with n_jobs=STATSFORECAST_N_JOBS, which starts a
    process per core; each worker is a fresh interpreter that must import numpy
    and statsforecast before doing any work. That cost is the same regardless of
    which model runs, so for the cheap models it dominates the reported time.

    Measured as the gap between parallel and serial fits of Naive — the model
    whose own work is closest to zero — and recorded alongside every result so
    the compute table can be read correctly rather than taken at face value.
    """
    if STATSFORECAST_N_JOBS == 1:
        return 0.0

    serial = StatsForecast(models=[Naive()], freq=FREQ, n_jobs=1)
    parallel = StatsForecast(models=[Naive()], freq=FREQ, n_jobs=STATSFORECAST_N_JOBS)
    _, serial_cost = time_median(lambda: serial.fit(df=train))
    _, parallel_cost = time_median(lambda: parallel.fit(df=train))
    return parallel_cost["seconds_median"] - serial_cost["seconds_median"]


def run_model(name: str, train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit one model on the 500-series batch and forecast the test horizon.

    Fitting and prediction are measured separately: the paper reports training
    and inference cost as distinct quantities.
    """
    engine = StatsForecast(
        models=[MODELS[name]()], freq=FREQ, n_jobs=STATSFORECAST_N_JOBS
    )

    _, train_cost = measure(lambda: engine.fit(df=train))
    forecast, inference_cost = measure(lambda: engine.predict(h=TEST_HORIZON_DAYS))

    forecast = forecast.reset_index() if "unique_id" not in forecast.columns else forecast
    column = [c for c in forecast.columns if c not in ("unique_id", "ds")][0]
    forecast = forecast.rename(columns={column: "yhat"})[["unique_id", "ds", "yhat"]]
    if CLIP_NEGATIVE_FORECASTS:
        forecast["yhat"] = forecast["yhat"].clip(lower=0.0)

    cost = {f"train_{k}": v for k, v in train_cost.items()}
    cost.update({f"inference_{k}": v for k, v in inference_cost.items()})
    return forecast, cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=list(HISTORY_CONDITIONS))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        default=list(MODELS),
        help="subset of models to run (default: all five)",
    )
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
    scale = naive_scale(
        pd.read_parquet(DATA_PROCESSED_DIR / f"train_{RMSSE_SCALE_CONDITION}.parquet")
    )

    if args.limit:
        kept = sorted(train["unique_id"].unique())[: args.limit]
        train = train[train["unique_id"].isin(kept)]
        test = test[test["unique_id"].isin(kept)]
        print(f"SMOKE TEST: limited to {len(kept)} series")

    overhead = pool_overhead_seconds(train)
    print(f"statsforecast worker-pool overhead: {overhead:.2f}s (fixed, per fit)\n")

    summaries, per_series = [], []
    for name in args.models:
        forecast, cost = run_model(name, train)
        scores = per_series_metrics(test, forecast, scale)

        scores.insert(0, "condition", args.condition)
        scores.insert(0, "model", name)
        per_series.append(scores)

        summary = {
            "model": name,
            "condition": args.condition,
            **summarise(scores),
            **cost,
            "pool_overhead_seconds": overhead,
        }
        summaries.append(summary)
        print(
            f"{name:>14}  RMSSE {summary['rmsse_mean']:.4f}  MAE {summary['mae_mean']:.4f}"
            f"  train {summary['train_seconds_median']:.2f}s"
            f"  infer {summary['inference_seconds_median']:.2f}s"
            f"  peak {summary['train_peak_rss_mb']:.0f}MB"
        )

    stem = f"stats_{args.condition}"
    if args.limit:
        stem += f"_limit{args.limit}"
    pd.DataFrame(summaries).to_csv(RESULTS_DIR / f"{stem}_summary.csv", index=False)
    pd.concat(per_series).to_csv(RESULTS_DIR / f"{stem}_per_series.csv", index=False)
    print(f"\nwrote {RESULTS_DIR / stem}_summary.csv and _per_series.csv")


if __name__ == "__main__":
    main()
