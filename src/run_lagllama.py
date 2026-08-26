"""Run Lag-Llama zero-shot for one history condition.

    uv run python -m src.run_lagllama --condition H-6

Zero-shot means no fitting happens. The cost reported in the training slot is
therefore checkpoint loading and model construction — the setup a practitioner
pays before any forecast exists — and it is labelled as such in the results.

Runs on CPU by design; MPS/GPU stays disabled so that timings are comparable
with the statistical and gradient-boosted families.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood
from huggingface_hub import hf_hub_download
from lag_llama.gluon.estimator import LagLlamaEstimator

from src.config import (
    CLIP_NEGATIVE_FORECASTS,
    DATA_PROCESSED_DIR,
    FREQ,
    HISTORY_CONDITIONS,
    LAGLLAMA_BATCH_SIZE,
    LAGLLAMA_CHECKPOINT_FILE,
    LAGLLAMA_CHECKPOINT_REPO,
    LAGLLAMA_CONTEXT_LENGTH,
    LAGLLAMA_DEVICE,
    LAGLLAMA_NUM_SAMPLES,
    RANDOM_SEED,
    RESULTS_DIR,
    RMSSE_SCALE_CONDITION,
    TEST_HORIZON_DAYS,
)
from src.harness import measure
from src.metrics import crps, naive_scale, per_series_metrics, summarise

MODEL_NAME = "Lag-Llama (zero-shot)"


def allow_checkpoint_globals() -> None:
    """Permit the non-tensor classes stored in the Lag-Llama checkpoint.

    Torch 2.6+ unpickles with weights_only=True, and PyTorch Lightning loads the
    checkpoint through its own torch.load call that we cannot pass arguments to.
    Allowlisting exactly the two classes the official checkpoint carries is
    narrower than disabling the safety check globally.
    """
    torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])


def build_predictor(checkpoint_path: str):
    """Construct the zero-shot predictor from the pretrained checkpoint.

    The architecture arguments must match the checkpoint, so they are read from
    it rather than restated here.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=LAGLLAMA_DEVICE, weights_only=False
    )
    model_args = checkpoint["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path=checkpoint_path,
        prediction_length=TEST_HORIZON_DAYS,
        context_length=LAGLLAMA_CONTEXT_LENGTH,
        input_size=model_args["input_size"],
        n_layer=model_args["n_layer"],
        n_embd_per_head=model_args["n_embd_per_head"],
        n_head=model_args["n_head"],
        scaling=model_args["scaling"],
        time_feat=model_args["time_feat"],
        batch_size=LAGLLAMA_BATCH_SIZE,
        num_parallel_samples=LAGLLAMA_NUM_SAMPLES,
        device=torch.device(LAGLLAMA_DEVICE),
    )
    module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    return estimator.create_predictor(transformation, module)


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

    allow_checkpoint_globals()
    torch.manual_seed(RANDOM_SEED)
    torch.set_default_device(LAGLLAMA_DEVICE)

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

    dataset = PandasDataset.from_long_dataframe(
        train.assign(y=train["y"].astype("float32")),
        item_id="unique_id",
        timestamp="ds",
        target="y",
        freq=FREQ,
    )

    checkpoint_path = hf_hub_download(
        repo_id=LAGLLAMA_CHECKPOINT_REPO, filename=LAGLLAMA_CHECKPOINT_FILE
    )
    predictor, setup_cost = measure(lambda: build_predictor(checkpoint_path))

    def forecast() -> list:
        torch.manual_seed(RANDOM_SEED)
        return list(predictor.predict(dataset, num_samples=LAGLLAMA_NUM_SAMPLES))

    forecasts, inference_cost = measure(forecast)

    test_dates = pd.DatetimeIndex(sorted(test["ds"].unique()))
    actual_by_id = {
        unique_id: group.sort_values("ds")["y"].to_numpy(dtype="float64")
        for unique_id, group in test.groupby("unique_id")
    }

    rows, crps_scores = [], {}
    for entry in forecasts:
        unique_id = str(entry.item_id)
        # (num_samples, horizon) -> per-step median is the point forecast.
        samples = np.asarray(entry.samples, dtype="float64")
        if CLIP_NEGATIVE_FORECASTS:
            samples = np.maximum(samples, 0.0)
        point = np.median(samples, axis=0)

        crps_scores[unique_id] = crps(actual_by_id[unique_id], samples.T)
        rows.append(
            pd.DataFrame({"unique_id": unique_id, "ds": test_dates, "yhat": point})
        )

    forecast_long = pd.concat(rows, ignore_index=True)
    scores = per_series_metrics(test, forecast_long, scale)
    scores["crps"] = scores["unique_id"].map(crps_scores)
    scores.insert(0, "condition", args.condition)
    scores.insert(0, "model", MODEL_NAME)

    cost = {f"train_{k}": v for k, v in setup_cost.items()}
    cost.update({f"inference_{k}": v for k, v in inference_cost.items()})
    summary = {
        "model": MODEL_NAME,
        "condition": args.condition,
        # Recorded so the compute table is not read as fitting cost.
        "train_is_setup_only": True,
        **summarise(scores),
        **cost,
    }

    print(
        f"{MODEL_NAME}  RMSSE {summary['rmsse_mean']:.4f}  MAE {summary['mae_mean']:.4f}"
        f"  CRPS {summary['crps_mean']:.4f}"
        f"  setup {summary['train_seconds_median']:.2f}s"
        f"  infer {summary['inference_seconds_median']:.2f}s"
        f"  peak {summary['inference_peak_rss_mb']:.0f}MB"
    )

    stem = f"lagllama_{args.condition}"
    if args.limit:
        stem += f"_limit{args.limit}"
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{stem}_summary.csv", index=False)
    scores.to_csv(RESULTS_DIR / f"{stem}_per_series.csv", index=False)
    print(f"wrote {RESULTS_DIR / stem}_summary.csv and _per_series.csv")


if __name__ == "__main__":
    main()
