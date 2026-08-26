"""Seeds, paths, experimental conditions, and model hyperparameters.

Single source of truth for every constant used across the pipeline.
See experimental_design.md for the rationale behind each value.
"""

from pathlib import Path
from typing import Final

# --- Reproducibility --------------------------------------------------------

RANDOM_SEED: Final[int] = 42

# --- Paths -------------------------------------------------------------------

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Final[Path] = REPO_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Final[Path] = REPO_ROOT / "data" / "processed"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"

# --- Data source ---------------------------------------------------------

KAGGLE_COMPETITION: Final[str] = "m5-forecasting-accuracy"

# The evaluation file carries the full 1,941 days; the validation file stops 28
# days earlier.
SALES_FILE: Final[str] = "sales_train_evaluation.csv"
CALENDAR_FILE: Final[str] = "calendar.csv"
PRICES_FILE: Final[str] = "sell_prices.csv"

# --- Sampling ------------------------------------------------------------

STORE_ID: Final[str] = "CA_1"
N_SAMPLED_SERIES: Final[int] = 500

# A series only enters the sampling frame if it has at least this many days of
# history before the test window (counted from its first non-zero sale, since
# M5 pads not-yet-introduced items with leading zeros). Set above the longest
# truncation so that H-6, H-12 and H-full are genuinely different for every
# sampled series.
MIN_HISTORY_DAYS: Final[int] = 730

# Syntetos-Boylan classification, used to stratify by demand intermittency.
ADI_INTERMITTENT_THRESHOLD: Final[float] = 1.32
CV2_LUMPY_THRESHOLD: Final[float] = 0.49

N_VOLUME_TERCILES: Final[int] = 3

# --- Experimental conditions (history truncation) -------------------------

# Training history length in days, ending at the same date in every condition.
# H-full uses the entire available history (no truncation).
HISTORY_CONDITIONS: Final[dict[str, int | None]] = {
    "H-1": 30,
    "H-6": 180,
    "H-12": 365,
    "H-full": None,
}
TEST_HORIZON_DAYS: Final[int] = 30

# --- Evaluation ------------------------------------------------------------

# Demand cannot be negative, so forecasts are floored at zero before scoring.
# Applied identically in every runner: it is a stated post-processing step, not
# a per-model advantage.
CLIP_NEGATIVE_FORECASTS: Final[bool] = True


# The RMSSE denominator is computed once, from this condition's training data,
# and reused when scoring every condition. A per-condition denominator would
# divide each condition's errors by a different number, making the H-6 vs
# H-full comparison — the point of the experiment — meaningless.
RMSSE_SCALE_CONDITION: Final[str] = "H-full"

# Rolling-origin evaluation is secondary/optional (see experimental_design.md 7).
N_ROLLING_FOLDS: Final[int] = 3
ROLLING_STEP_DAYS: Final[int] = 30

# --- Timing / measurement protocol -----------------------------------------

N_TIMING_RUNS: Final[int] = 3  # each timing is run 3x; the median is reported

# --- Statistical models (statsforecast) ------------------------------------

FREQ: Final[str] = "D"
SEASONAL_PERIOD: Final[int] = 7  # weekly seasonality (Seasonal Naive, ETS, ARIMA)

# -1 uses every core, matching the "per-series fitting, multi-core CPU" premise.
STATSFORECAST_N_JOBS: Final[int] = -1

# --- LightGBM ----------------------------------------------------------------

LGBM_LAG_DAYS: Final[list[int]] = [1, 7, 14, 28]
LGBM_ROLLING_WINDOWS: Final[list[int]] = [7, 28]

# Modest, fixed defaults — no hyperparameter search (experimental_design.md 7).
# Native lightgbm.train API (not the sklearn wrapper) to avoid a scikit-learn
# dependency; num_threads=0 means all available cores.
LGBM_PARAMS: Final[dict[str, object]] = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "seed": RANDOM_SEED,
    "num_threads": 0,
    "verbosity": -1,
}
LGBM_NUM_BOOST_ROUND: Final[int] = 100

# --- Lag-Llama ---------------------------------------------------------------

# CPU-only for every model, including Lag-Llama — MPS/GPU explicitly disabled
# so timings stay comparable across model families.
LAGLLAMA_DEVICE: Final[str] = "cpu"
LAGLLAMA_NUM_SAMPLES: Final[int] = 100  # sample paths; point forecast = median

LAGLLAMA_CHECKPOINT_REPO: Final[str] = "time-series-foundation-models/Lag-Llama"
LAGLLAMA_CHECKPOINT_FILE: Final[str] = "lag-llama.ckpt"

# The context window the model attends over. Note that this caps how much
# history Lag-Llama can use, independently of the truncation condition: if it is
# shorter than H-6, all three conditions present the model with the same input.
# Whether that happens is a reportable result, not something to tune away.
LAGLLAMA_CONTEXT_LENGTH: Final[int] = 32
LAGLLAMA_BATCH_SIZE: Final[int] = 32
