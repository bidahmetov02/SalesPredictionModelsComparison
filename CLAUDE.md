# CLAUDE.md

Working instructions for this repo. The experiment itself is specified in **`experimental_design.md`** — that file is the source of truth. If anything here disagrees with it, `experimental_design.md` wins.

This repo is **public**. A README for general readers will be added later; until then, this file and `experimental_design.md` are the documentation.

## What this project is

A benchmarking experiment for a university article. We compare three families of forecasting models on 30-day, SKU-level demand forecasting:

- **Statistical**: Naive, Seasonal Naive, AutoETS, AutoARIMA, Croston — via `statsforecast`
- **Gradient-boosted trees**: LightGBM, one global model per condition
- **Foundation model**: Lag-Llama, zero-shot (fine-tuning is optional)

The data is a stratified sample of 500 SKU–store series from the M5 (Walmart) dataset. The question: how do these models compare when sales history is short **and** hardware is a commodity laptop CPU? Accuracy (MAE, RMSSE) and computational cost (wall-clock time, peak memory) are both experimental results. The headline figure of the paper is an accuracy-vs-compute scatter plot.

## Experimental grid

Every model runs under three history-truncation conditions. The 500 series are sampled once with a fixed seed. The test window is always the same: the final 30 days, never seen in training.

| | H-6 (~180 days) | H-12 (~365 days) | H-full (~5 years) |
|---|---|---|---|
| Statistical (5 models, `statsforecast`) | ✓ | ✓ | ✓ |
| LightGBM (global model per condition) | ✓ | ✓ | ✓ |
| Lag-Llama zero-shot | ✓ | ✓ | ✓ |
| Lag-Llama fine-tuned (optional) | (✓) | (✓) | (✓) |

Primary evaluation is a single 30-day holdout. Rolling-origin evaluation (3 folds) is a secondary, optional check. One store is enough; a second store is at most a robustness appendix.

## Hard constraints

- **CPU-only, for every model.** Lag-Llama must run with `device="cpu"` — MPS/GPU explicitly disabled. This keeps timings comparable across model families. Do not "helpfully" enable MPS.
- **Timing and memory are results, not incidentals.** We measure: training wall-clock for the 500-series batch, inference wall-clock for one 30-day forecast batch, and peak RSS via `psutil`. Every timing runs 3 times; we report the median. Nothing extraneous goes inside a timed section.
- **Fixed seeds everywhere** — sampling, LightGBM, PyTorch. All seeds live in `src/config.py`, never inline.
- **No hyperparameter search.** Fixed, modest defaults, stated openly in the paper. Do not tune.
- **uv for everything.** `uv add` to install, `uv run` to execute. Never pip, never conda. `uv.lock` stays committed — the paper reports the pinned versions.
- **Minimal dependencies.** Before adding a package, check whether an existing one already covers the need.
- Reference machine: M2 MacBook Air, 16 GB RAM. Nothing may assume more.

## Repo layout

```
experimental_design.md      # the experiment spec — source of truth
CLAUDE.md                   # this file
pyproject.toml / uv.lock    # dependencies, pinned
src/
    config.py               # seeds, paths, conditions, model hyperparameters
    data.py                 # M5 download, stratified sampling, truncation
    features.py             # LightGBM features (lags, rolling means, calendar, price)
    metrics.py              # MAE, RMSSE, optional CRPS
    harness.py              # timing (3-run median) and peak-memory measurement
    run_stats.py            # runs the 5 statistical models
    run_lgbm.py             # runs LightGBM
    run_lagllama.py         # runs Lag-Llama
    aggregate.py            # collects results into the paper's tables and figures
data/
    raw/                    # M5 CSVs from Kaggle — gitignored, never committed
    processed/              # sampled + truncated series — gitignored, regenerable
results/                    # result CSVs and aggregated tables — committed
```

Each runner is a standalone script: it takes a condition (e.g. `uv run python -m src.run_stats --condition H-6`), writes a results CSV, and exits. Runners share `config`, `metrics`, and `harness`, but never import each other.

## Coding conventions

- **Boring over clever.** Plain functions and scripts. No classes unless a library requires one. No abstractions with a single use site. This is a reproducible pipeline for a paper, not a framework.
- Every experiment parameter (sample size, history lengths, horizon, seeds, hyperparameters) is a named constant in `src/config.py`. No magic numbers in runner code.
- Every stage is deterministic and idempotent: rerunning it overwrites its output and produces the same result.
- Intermediate data goes to `data/processed/`; results go to `results/` as plain CSV.
- Type hints on function signatures. Comments only where the experimental protocol needs explaining (e.g. why MPS is off) — no narration.
- No notebooks in the pipeline. Notebooks, if used at all, only polish figures from `results/`.
- Python ≥ 3.12.

## Public repo — no secrets, no data

Everything committed here is published. Two standing rules:

1. **No credentials, ever.** Kaggle tokens, `.env` files, API keys — none of it enters the working tree. Code reads credentials only from their standard locations outside the repo (e.g. `~/.kaggle/kaggle.json`). Never hardcode a secret or a personal path, even temporarily. `.gitignore` blocks the common cases, but the gitignore is a safety net, not the rule.
2. **No dataset files.** The M5 data is under Kaggle competition terms and is never committed — the repo ships only the script that downloads it.

Before committing, check `git diff --staged` for anything that looks like a credential, a private path, or a data file. When in doubt, leave it out.
