"""Collect runner outputs into the paper's tables and headline figure.

    uv run python -m src.aggregate

Reads every results/*_summary.csv and results/*_per_series.csv, ignoring the
_limit* files smoke tests produce, and writes:

    table_accuracy.csv           model x condition
    table_accuracy_by_class.csv  model x condition x intermittency class
    table_compute.csv            timing and memory
    table_significance.csv       Wilcoxon signed-rank against the best model
    figure_accuracy_vs_compute.png   the headline figure
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import pandas as pd
from scipy.stats import wilcoxon

from src.config import DATA_PROCESSED_DIR, HISTORY_CONDITIONS, RESULTS_DIR

# Families drive colour in the figure. The first three categorical slots are the
# ones validated for all-pairs separation, which is what a scatter needs.
FAMILY_OF_MODEL = {
    "Naive": "Statistical",
    "SeasonalNaive": "Statistical",
    "AutoETS": "Statistical",
    "AutoARIMA": "Statistical",
    "Croston": "Statistical",
    "LightGBM": "Gradient-boosted trees",
    "Lag-Llama (zero-shot)": "Foundation model",
}
FAMILY_COLOUR = {
    "Statistical": "#2a78d6",
    "Gradient-boosted trees": "#eb6834",
    "Foundation model": "#1baf7a",
}
FAMILY_MARKER = {
    "Statistical": "o",
    "Gradient-boosted trees": "s",
    "Foundation model": "D",
}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOUR = "#d8d8d4"


def load_results(kind: str) -> pd.DataFrame:
    """Concatenate results/*_{kind}.csv, skipping smoke-test output."""
    paths = sorted(
        path
        for path in RESULTS_DIR.glob(f"*_{kind}.csv")
        if "_limit" not in path.name and not path.name.startswith("table_")
    )
    if not paths:
        raise FileNotFoundError(
            f"no *_{kind}.csv in {RESULTS_DIR}; run the runners first"
        )
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def order_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort rows into the experimental order rather than alphabetical."""
    frame = frame.copy()
    frame["condition"] = pd.Categorical(
        frame["condition"], categories=list(HISTORY_CONDITIONS), ordered=True
    )
    return frame.sort_values(["model", "condition"])


def accuracy_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = ["model", "condition", "mae_mean", "mae_median", "rmsse_mean", "rmsse_median"]
    if "crps_mean" in summary.columns:
        columns += ["crps_mean", "crps_median"]
    return order_conditions(summary)[columns + ["n_series", "n_rmsse_undefined"]]


def accuracy_by_class(per_series: pd.DataFrame) -> pd.DataFrame:
    """Break accuracy down by Syntetos-Boylan class.

    M5 at SKU level is dominated by intermittent demand, so an overall mean can
    hide that a model is strong only on the rare smooth series.
    """
    metadata = pd.read_csv(DATA_PROCESSED_DIR / "series_metadata.csv")
    joined = per_series.merge(
        metadata[["id", "sb_class"]], left_on="unique_id", right_on="id", how="left"
    )
    grouped = (
        joined.groupby(["model", "condition", "sb_class"], observed=True)
        .agg(
            rmsse_mean=("rmsse", "mean"),
            rmsse_median=("rmsse", "median"),
            mae_mean=("mae", "mean"),
            n_series=("unique_id", "size"),
        )
        .reset_index()
    )
    return order_conditions(grouped)


def compute_table(summary: pd.DataFrame) -> pd.DataFrame:
    frame = order_conditions(summary).copy()
    frame["total_seconds_median"] = (
        frame["train_seconds_median"] + frame["inference_seconds_median"]
    )
    columns = [
        "model",
        "condition",
        "train_seconds_median",
        "inference_seconds_median",
        "total_seconds_median",
        "train_peak_rss_mb",
        "inference_peak_rss_mb",
        "train_n_runs",
    ]
    if "pool_overhead_seconds" in frame.columns:
        columns.append("pool_overhead_seconds")
    if "train_is_setup_only" in frame.columns:
        columns.append("train_is_setup_only")
    return frame[columns]


def significance_tests(per_series: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank of each model against the best one, per condition.

    Paired on per-series RMSSE, which is what makes the test valid: both models
    forecast the same 500 series, so the pairs are matched by construction.
    """
    rows = []
    for condition, group in per_series.groupby("condition", observed=True):
        wide = group.pivot_table(
            index="unique_id", columns="model", values="rmsse"
        ).dropna()
        if wide.shape[1] < 2:
            continue
        best = wide.mean().idxmin()
        for model in wide.columns:
            if model == best:
                continue
            statistic, p_value = wilcoxon(wide[best], wide[model])
            rows.append(
                {
                    "condition": condition,
                    "best_model": best,
                    "compared_with": model,
                    "rmsse_mean_best": wide[best].mean(),
                    "rmsse_mean_other": wide[model].mean(),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "n_pairs": len(wide),
                }
            )
    return pd.DataFrame(rows)


def headline_figure(compute: pd.DataFrame, path) -> None:
    """Accuracy against compute, one panel per condition.

    Small multiples rather than one crowded axis: 18 points on a single scatter
    would need both colour and shape just to separate the conditions, and the
    comparison the paper makes is within a condition.
    """
    conditions = [c for c in HISTORY_CONDITIONS if c in set(compute["condition"])]
    figure, axes = plt.subplots(
        1, len(conditions), figsize=(4.2 * len(conditions), 4.4), sharey=True
    )
    axes = [axes] if len(conditions) == 1 else list(axes)

    for axis, condition in zip(axes, conditions):
        panel = compute[compute["condition"] == condition]
        for _, row in panel.iterrows():
            family = FAMILY_OF_MODEL.get(row["model"], "Statistical")
            axis.scatter(
                row["total_seconds_median"],
                row["rmsse_mean"],
                s=90,
                color=FAMILY_COLOUR[family],
                marker=FAMILY_MARKER[family],
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
            # Direct labels are also the contrast relief the palette check asks
            # for: identity never rests on colour alone.
            axis.annotate(
                row["model"].replace(" (zero-shot)", ""),
                (row["total_seconds_median"], row["rmsse_mean"]),
                textcoords="offset points",
                xytext=(7, 4),
                fontsize=8,
                color=TEXT_SECONDARY,
            )

        axis.set_xscale("log")
        # Log minor ticks label every 2x/3x/4x step and collide into mush at
        # this width; keep the gridlines, drop their labels, and render the
        # decade labels as plain seconds rather than 10^n.
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g}s" if value >= 1 else f"{value:g}s")
        )
        # Room for the direct labels, which would otherwise clip at the edge.
        axis.margins(x=0.28)
        axis.set_title(condition, fontsize=11, color=TEXT_PRIMARY)
        axis.set_xlabel("total wall-clock time (log scale)", fontsize=9, color=TEXT_SECONDARY)
        axis.grid(True, which="major", color=GRID_COLOUR, linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(labelsize=8, colors=TEXT_SECONDARY)

    axes[0].set_ylabel("RMSSE (mean over series)", fontsize=9, color=TEXT_SECONDARY)
    handles = [
        plt.Line2D(
            [], [], marker=FAMILY_MARKER[family], color=FAMILY_COLOUR[family],
            linestyle="none", markersize=8, markeredgecolor="white", label=family,
        )
        for family in FAMILY_COLOUR
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle(
        "Accuracy against compute cost, by history condition",
        fontsize=12,
        color=TEXT_PRIMARY,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    summary = load_results("summary")
    per_series = load_results("per_series")

    accuracy = accuracy_table(summary)
    by_class = accuracy_by_class(per_series)
    compute = compute_table(summary)
    significance = significance_tests(per_series)

    accuracy.to_csv(RESULTS_DIR / "table_accuracy.csv", index=False)
    by_class.to_csv(RESULTS_DIR / "table_accuracy_by_class.csv", index=False)
    compute.to_csv(RESULTS_DIR / "table_compute.csv", index=False)
    significance.to_csv(RESULTS_DIR / "table_significance.csv", index=False)

    figure_path = RESULTS_DIR / "figure_accuracy_vs_compute.png"
    headline_figure(
        compute.merge(accuracy[["model", "condition", "rmsse_mean"]], on=["model", "condition"]),
        figure_path,
    )

    print(f"models x conditions: {len(summary)} rows from {RESULTS_DIR}")
    print(f"\n{accuracy.to_string(index=False)}")
    print(f"\nwrote 4 tables and {figure_path.name}")


if __name__ == "__main__":
    main()
