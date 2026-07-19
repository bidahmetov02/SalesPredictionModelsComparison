# CLAUDE.md

**Source of truth: `experimental_design.md`.** Read it before making any decision about data, models, metrics, or protocol. This file summarizes it and adds engineering conventions; if the two ever disagree, `experimental_design.md` wins.

## Project purpose

Academic benchmarking experiment (university article, supporting a SaaS demand-forecasting app): compare statistical models (Naive, Seasonal Naive, AutoETS, AutoARIMA, Croston via `statsforecast`), LightGBM (one global model per condition), and Lag-Llama (zero-shot; fine-tuning optional) for 30-day SKU-level demand forecasting on a stratified 500-series sample of the M5 dataset, under two joint constraints: short sales history and commodity CPU-only hardware. Both accuracy (MAE, RMSSE) and computational cost (wall-clock time, peak RSS memory) are experimental results — the headline figure is an accuracy-vs-compute scatter plot.

## Experimental grid

Every model runs under every truncation condition. Same 500 series (fixed, seeded sample), same 30-day test window (final 30 days) in all conditions.

| | H-6 (~180 days) | H-12 (~365 days) | H-full (~5 years) |
|---|---|---|---|
| Naive, Seasonal Naive (m=7), AutoETS, AutoARIMA, Croston (`statsforecast`) | ✓ | ✓ | ✓ |
| LightGBM (one global model per condition) | ✓ | ✓ | ✓ |
| Lag-Llama zero-shot | ✓ | ✓ | ✓ |
| Lag-Llama fine-tuned (optional, may be dropped) | (✓) | (✓) | (✓) |

Primary evaluation: single 30-day holdout. Rolling-origin (3 folds) is optional/secondary. One store; a second store is at most a robustness appendix.

## Hard constraints

- **CPU-only, everywhere.** Lag-Llama must run with MPS/GPU explicitly disabled (`device="cpu"`, never `mps`). This is a measurement-parity requirement, not a workaround — do not "helpfully" enable MPS.
- **Timing and memory are results.** Measurement protocol: wall-clock training time for the 500-series batch, wall-clock inference time for one 30-day forecast batch, peak RSS via `psutil`. Each timing run 3 times; report the median. Don't add code inside timed sections that isn't part of the model's work.
- **Fixed seeds everywhere** — series sampling, LightGBM, PyTorch/Lag-Llama. All seeds live in one config module, never inlined.
- **No hyperparameter search.** Fixed, modest, stated defaults. This is a design choice of the paper; do not tune.
- **uv for everything**: `uv add` for dependencies, `uv run` to execute. Never pip, never conda. Keep `uv.lock` committed — pinned versions are reported in the paper.
- **Minimal dependencies.** Before adding a package, check whether an existing one covers it.
- Reference machine: M2 MacBook Air. Nothing may assume more than 16 GB RAM.

## Repo structure (intended)

```
experimental_design.md      # source of truth for the experiment
CLAUDE.md
pyproject.toml / uv.lock
src/
    config.py               # seeds, paths, condition definitions, model hyperparameters
    data.py                 # M5 download/load + stratified 500-series sampling + truncation
    features.py             # LightGBM feature engineering (lags, rolling means, calendar, price)
    metrics.py              # MAE, RMSSE, optional CRPS
    harness.py              # timing (3-run median) + peak-RSS measurement wrapper
    run_stats.py            # statsforecast runner (all 5 statistical models)
    run_lgbm.py             # LightGBM global-model runner
    run_lagllama.py         # Lag-Llama runner (zero-shot; optional fine-tune)
    aggregate.py            # collect per-run results into paper tables/figures
data/
    raw/                    # M5 CSVs from Kaggle (gitignored)
    processed/              # sampled + truncated series (gitignored, regenerable)
results/                    # one CSV per model × condition run, plus aggregated tables (committed)
```

Each runner is a script executed as `uv run python -m src.run_<x> --condition H-6` (or similar), writes a results CSV, and exits. Runners share `config`, `metrics`, and `harness`; they do not import each other.

## Coding conventions

- **Boring over clever.** Plain functions and scripts; no classes unless a library demands one, no abstractions for a single use site, no premature generality. This is a reproducible pipeline for a paper, not a framework.
- All experiment parameters (sample size, condition lengths, horizon, seeds, hyperparameters) are named constants in `src/config.py` — nothing magic in runner code.
- Deterministic, idempotent steps: rerunning a stage overwrites its output and produces the same result. Intermediate data goes to `data/processed/`, results to `results/` as plain CSV.
- Raw M5 data is never modified and never committed; `data/` stays gitignored except for structure.
- Type hints on function signatures; comments only for non-obvious experimental-protocol reasons (e.g. why MPS is disabled), not narration.
- No notebooks in the pipeline. If notebooks are used at all, they are for figure polishing only and read from `results/`.
- Python ≥3.12, per `pyproject.toml`.
