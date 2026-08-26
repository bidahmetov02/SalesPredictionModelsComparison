# Experimental Design

## 1. Data

**Source.** M5 competition dataset (Walmart, Kaggle): daily unit sales for 30,490 SKU–store series over ~5.4 years (2011–2016), plus calendar and price files.

**Sample.** A stratified sample of **500 SKU–store series** from a single store (or two stores from different states for robustness). Stratification by:
- demand pattern, via the Syntetos–Boylan quadrants — **smooth / erratic / intermittent / lumpy** — from average demand interval (ADI ≥ 1.32 = intermittent) and the squared coefficient of variation of non-zero demand sizes (CV² ≥ 0.49 = lumpy),
- sales volume tercile (low / medium / high).

Four classes rather than three: the two thresholds define four quadrants, and *erratic* (frequent demand, highly variable order sizes) is a real cell that would otherwise have to be folded somewhere arbitrary. Both statistics are computed on training data only, so the held-out window never influences which series are sampled.

**Eligibility.** M5 pads a series with zeros for the period before the item was stocked. These are structural absences, not observed zero demand, so each series begins at its **first non-zero sale**. A series then enters the sampling frame only if it has at least **730 days** of history before the test window — otherwise a late-launched item could carry fewer than 365 days in total, making H-12 and H-full identical for it and quietly weakening the comparison the experiment exists to make. In store CA_1 this leaves 2,806 of 3,049 series eligible (92%); 243 are excluded.

This keeps every experiment tractable on a laptop while preserving the demand-pattern diversity that motivates including Croston's method. The sampled series IDs are fixed once (seeded) and reused across all conditions and models. The resulting sample reproduces the eligible population's class shares to within 0.1 pp: intermittent 70.6%, lumpy 18.6%, smooth 8.2%, erratic 2.6%.

## 2. Experimental conditions (history truncation)

Each series is truncated to simulate data scarcity. Three training-history lengths, all ending at the same date:

| Condition | Training history |
|---|---|
| H-1  | last month (30 days) |
| H-6  | last 6 months (180 days) |
| H-12 | last 12 months (365 days) |
| H-full | all history since the item's first sale (reference upper bound) |

**H-full is not uniformly ~5 years.** Because each series starts at its first non-zero sale, H-full spans 730–1,911 days (2.0–5.2 years), median 1,859 days (5.1 years); 233 of the 500 sampled series have less than five years. The 730-day floor guarantees H-full is strictly longer than H-12 for every series, but the upper bound varies by item, and results for H-full should be read as "all available history" rather than "five years".

The **test window is identical in all conditions**: the final 30 days of the dataset, never seen in training. This isolates the effect of history length — the only thing that changes between conditions is how much past data each model sees.

**Evaluation scheme.** Primary: single 30-day holdout (matches the app's use case). Secondary (if time permits): rolling-origin evaluation with 3 folds, stepping the forecast origin back 30 days per fold, to check stability of the rankings.

## 3. Models

| Paradigm | Models | Implementation |
|---|---|---|
| Statistical | Naive, Seasonal Naive (m=7), ETS (AutoETS), ARIMA (AutoARIMA), Croston | `statsforecast` (Nixtla) — per-series fitting, multi-core CPU |
| Gradient-boosted trees | LightGBM, one **global model** trained across all series per condition | `lightgbm` |
| Foundation model | Lag-Llama: (a) zero-shot, (b) fine-tuned per condition (optional) | official `lag-llama` repo, PyTorch |

**LightGBM features.** Lag features (1, 7, 14, 28 days), rolling means (7, 28), calendar features from the M5 calendar (day-of-week, month, an event flag set when either `event_name_1` or `event_name_2` is present, and the store's SNAP-benefit-day flag), sell price. Rolling means are computed on lagged values, so no feature for day *t* reads day *t* or later. Recursive multi-step forecasting for the 30-day horizon: each predicted day becomes the lag input for the next. Modest fixed hyperparameters (no per-condition tuning — tuning cost is itself part of the compute story and is noted, not optimized away).

**The feature window bites hardest at H-1.** A row is only usable for training once its 28-day lag and rolling mean exist, so a 30-day history yields **two usable days per series — 1,000 training rows, against 76,000 at H-6**. This is a structural property of a global tree model with a 28-day window, not a failure to be tuned away: shortening the lags for the shortest condition would be per-condition tuning and would break comparability across the grid. The feature set is therefore held fixed everywhere and the consequence reported as a result.

**Lag-Llama.** Pretrained checkpoint, probabilistic output (100 sample paths per series); point forecast = median of samples. The checkpoint's trained attention context is 32 days, but it additionally draws 84 lag features reaching back 1,092 days, giving an effective receptive field of **1,124 days** — so the truncation conditions do genuinely change what the model can see, rather than presenting it with identical inputs. Context length is left at the checkpoint's trained value; enlarging it would be both hyperparameter tuning and off-distribution for the pretrained weights. Zero-shot is the primary mode; fine-tuning (few epochs, CPU) is a secondary experiment and may be dropped if compute cost is prohibitive — which is itself a reportable finding.

## 4. Metrics

**Accuracy** (per series, then aggregated as mean and median across the 500 series):
- MAE — interpretable in units sold, matches the app's user-facing error,
- RMSSE — scale-free, the official M5 metric, comparable across SKUs,
- (optional) CRPS for Lag-Llama's probabilistic forecasts, computed from the 100 sample paths.

**RMSSE scaling.** The denominator is the mean squared one-step naive error on training data. It is computed **once, from H-full, and reused when scoring every condition**. A per-condition denominator would divide each condition's errors by a different number, which would make an H-6 RMSSE incomparable with an H-full one and destroy the central comparison; using the longest history also matches the official M5 definition.

**Post-processing.** Demand cannot be negative, so every model's forecasts are floored at zero before scoring. This is applied identically across all model families, so it is a stated protocol step rather than an advantage given to the models (AutoETS, AutoARIMA) that can otherwise emit negative values.

**Computational cost**, measured on the reference machine:
- wall-clock training time (total for the 500-series batch),
- wall-clock inference time for one 30-day forecast batch,
- peak memory (RSS, via `psutil`).

Each timing is run 3 times and the median reported. Peak memory is measured in a **separate, untimed fourth run**: sampling RSS requires a polling thread that contends for the GIL, and keeping it outside the timed runs is what allows "nothing extraneous inside a timed section" to hold literally. Each measured function therefore executes four times and must be idempotent. Memory accounting includes child processes, since `statsforecast` fits series in a worker pool and measuring only the parent would attribute almost none of that memory to the model.

**Parallelism and its fixed cost.** `statsforecast` parallelises across series with a process pool (`n_jobs=-1`), which starts one interpreter per core; each worker must import numpy and statsforecast before doing any work. That startup is roughly 2.3 s and is the same regardless of which model runs, so it dominates the cheapest models — Naive itself fits 500 series in about 0.01 s. Every statistical result therefore records `pool_overhead_seconds` alongside its timings, so the compute table can be read correctly rather than at face value. All three families are given the whole machine (LightGBM and Lag-Llama use threads and pay no equivalent startup cost), which is the fair cross-family comparison; the asymmetry is in how each library parallelises, and is reported rather than hidden.

## 5. Hardware & measurement protocol

- Reference machine: Apple MacBook Air (M2, 8-core, 16 GB RAM or as available), **CPU-only** for all models — including Lag-Llama (MPS/GPU disabled) — so that results reflect the commodity-hardware constraint and are comparable across model families.
- Timing runs performed with the machine plugged in, no other heavy applications running. The reference machine is **fanless**, so it thermally throttles under sustained load: conditions are run in separate sessions with the machine allowed to cool between them, and the run order is recorded. A multi-hour unbroken grid would make later measurements systematically slower than earlier ones for reasons that have nothing to do with the models.
- Fixed random seeds everywhere they affect an outcome: sampling, LightGBM, and PyTorch. The `statsforecast` models used here (Naive, Seasonal Naive, AutoETS, AutoARIMA, Croston) are deterministic given their inputs and expose no seed.
- CPU-only is enforced, not assumed: the Lag-Llama runner raises if its configured device is anything other than `cpu`.
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
