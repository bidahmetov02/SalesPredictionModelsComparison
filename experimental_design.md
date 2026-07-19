# Experimental Design

## 1. Data

**Source.** M5 competition dataset (Walmart, Kaggle): daily unit sales for 30,490 SKU–store series over ~5.4 years (2011–2016), plus calendar and price files.

**Sample.** A stratified sample of **500 SKU–store series** from a single store (or two stores from different states for robustness). Stratification by:
- demand intermittency (ADI — average demand interval: smooth / intermittent / lumpy),
- sales volume tercile (low / medium / high).

This keeps every experiment tractable on a laptop while preserving the demand-pattern diversity that motivates including Croston's method. The sampled series IDs are fixed once (seeded) and reused across all conditions and models.

## 2. Experimental conditions (history truncation)

Each series is truncated to simulate data scarcity. Three training-history lengths, all ending at the same date:

| Condition | Training history |
|---|---|
| H-6  | last 6 months (~180 days) |
| H-12 | last 12 months (~365 days) |
| H-full | full available history (~5 years, reference upper bound) |

The **test window is identical in all conditions**: the final 30 days of the dataset, never seen in training. This isolates the effect of history length — the only thing that changes between conditions is how much past data each model sees.

**Evaluation scheme.** Primary: single 30-day holdout (matches the app's use case). Secondary (if time permits): rolling-origin evaluation with 3 folds, stepping the forecast origin back 30 days per fold, to check stability of the rankings.

## 3. Models

| Paradigm | Models | Implementation |
|---|---|---|
| Statistical | Naive, Seasonal Naive (m=7), ETS (AutoETS), ARIMA (AutoARIMA), Croston | `statsforecast` (Nixtla) — per-series fitting, multi-core CPU |
| Gradient-boosted trees | LightGBM, one **global model** trained across all series per condition | `lightgbm` |
| Foundation model | Lag-Llama: (a) zero-shot, (b) fine-tuned per condition (optional) | official `lag-llama` repo, PyTorch |

**LightGBM features.** Lag features (1, 7, 14, 28 days), rolling means (7, 28), calendar features (day-of-week, month, event flags from M5 calendar), sell price. Recursive multi-step forecasting for the 30-day horizon. Modest fixed hyperparameters (no per-condition tuning — tuning cost is itself part of the compute story and is noted, not optimized away).

**Lag-Llama.** Pretrained checkpoint, probabilistic output (100 sample paths per series); point forecast = median of samples. Zero-shot is the primary mode; fine-tuning (few epochs, CPU) is a secondary experiment and may be dropped if compute cost is prohibitive — which is itself a reportable finding.

## 4. Metrics

**Accuracy** (per series, then aggregated as mean and median across the 500 series):
- MAE — interpretable in units sold, matches the app's user-facing error,
- RMSSE — scale-free, the official M5 metric, comparable across SKUs,
- (optional) CRPS for Lag-Llama's probabilistic forecasts.

**Computational cost**, measured on the reference machine:
- wall-clock training time (total for the 500-series batch),
- wall-clock inference time for one 30-day forecast batch,
- peak memory (RSS, via `psutil`).

Each timing is run 3 times; the median is reported.

## 5. Hardware & measurement protocol

- Reference machine: Apple MacBook Air (M2, 8-core, 16 GB RAM or as available), **CPU-only** for all models — including Lag-Llama (MPS/GPU disabled) — so that results reflect the commodity-hardware constraint and are comparable across model families.
- Timing runs performed with the machine plugged in, no other heavy applications running.
- Fixed random seeds everywhere (sampling, LightGBM, PyTorch).
- Environment pinned with `uv` lockfile; versions of all libraries reported in the paper.

## 6. Analysis plan

1. Accuracy tables: model × history-length condition (MAE, RMSSE), overall and broken down by intermittency class.
2. Compute tables: training and inference time, memory, per model × condition.
3. Accuracy-vs-compute scatter plot (RMSSE vs. total wall-clock time, log scale) — the paper's headline figure.
4. Key comparisons answering the research question:
   - Does Lag-Llama zero-shot beat statistical/LightGBM models at H-6 (scarce data)?
   - How much accuracy does LightGBM lose going from H-full to H-6?
   - What is the compute premium of Lag-Llama over ETS/Croston, and is it justified?
5. Significance check: paired comparison of per-series errors (Wilcoxon signed-rank or Diebold–Mariano on the top contenders).

## 7. Scope guards (to keep effort bounded)

- No hyperparameter search; fixed sensible defaults, stated openly as a design choice consistent with the low-compute premise.
- Fine-tuning Lag-Llama is optional; zero-shot alone is a valid result.
- Rolling-origin evaluation is optional; single holdout is the primary protocol.
- One store (500 series) is sufficient; a second store is a robustness appendix at most.
