from __future__ import annotations

"""
Publication-style real-data summary panel.

Inputs:
    experiments/results/realdata/real01_ret1_20080516_R1_rec0.csv
    experiments/results/realdata/real02_pvc3_area17.csv
    experiments/results/realdata/real03_pvc11_monkey2.csv

Outputs:
    experiments/results/realdata/real04_combined.csv
    experiments/results/realdata/real04_summary_by_dataset_topm.csv
    figures/realdata/real04_summary_panel.png
    figures/realdata/real04_summary_panel.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D


RESULTS_DIR = Path("experiments/results/realdata")
FIG_DIR = Path("figures/realdata")

DEFAULT_INPUTS = [
    RESULTS_DIR / "real01_ret1_20080516_R1_rec0.csv",
    RESULTS_DIR / "real02_pvc3_area17.csv",
    RESULTS_DIR / "real03_pvc11_monkey2.csv",
]

DATASET_ORDER = [
    "ret-1",
    "PVC-3 area17",
    "PVC-11 monkey2",
]

DATASET_LABELS = {
    "ret-1": "R1 · ret-1",
    "PVC-3 area17": "R2 · PVC-3 area17",
    "PVC-11 monkey2": "R3 · PVC-11 monkey2",
}

# Bright but still paper-friendly / colorblind-safe.
PALETTE = {
    "ret-1": "#3E7CB1",          # blue
    "PVC-3 area17": "#2FA889",   # teal-green
    "PVC-11 monkey2": "#E08B3E", # orange
}

TEXT = "#20242A"
MUTED_TEXT = "#667085"
GRID = "#D9DEE7"
SPINE = "#C7CED8"


def infer_dataset_name(path: Path) -> str:
    """Infer canonical dataset name from the filename.

    This intentionally overrides any inconsistent dataset labels inside CSVs.
    """
    name = path.name.lower()

    if "ret1" in name or "ret-1" in name:
        return "ret-1"
    if "pvc3" in name or "pvc-3" in name:
        return "PVC-3 area17"
    if "pvc11" in name or "pvc-11" in name:
        return "PVC-11 monkey2"

    raise ValueError(f"Cannot infer dataset name from filename: {path}")


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column by case-insensitive matching."""
    lookup = {c.lower(): c for c in df.columns}

    for cand in candidates:
        key = cand.lower()
        if key in lookup:
            return lookup[key]

    return None


def load_one_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")

    df = pd.read_csv(path)

    # Force canonical paper-facing dataset names from file path.
    df["dataset"] = infer_dataset_name(path)

    topm_col = first_existing(
        df,
        [
            "top_m",
            "topm",
            "m",
            "top_m_pairs",
            "top_pairs",
        ],
    )
    if topm_col is None:
        raise ValueError(
            f"{path} is missing a top-m column. "
            f"Available columns: {list(df.columns)}"
        )

    bic_col = first_existing(
        df,
        [
            "bicdiff_candidate_count",
            "bic_diff_candidate_count",
            "bicdiff_cand",
            "bicdiff",
            "BICdiff",
            "BICcand",
            "bic_candidate_count",
        ],
    )
    if bic_col is None:
        raise ValueError(
            f"{path} is missing a candidate-count BIC-diff column. "
            f"Available columns: {list(df.columns)}"
        )

    df["top_m"] = pd.to_numeric(df[topm_col], errors="coerce")
    df["bicdiff_candidate_count"] = pd.to_numeric(df[bic_col], errors="coerce")

    df = df.dropna(subset=["top_m", "bicdiff_candidate_count"]).copy()
    df["top_m"] = df["top_m"].astype(int)

    # Keep only the paper-facing top-m values if extra rows exist.
    df = df[df["top_m"].isin([1, 2, 3])].copy()

    return df


def build_summary(combined: pd.DataFrame) -> pd.DataFrame:
    summary = (
        combined.groupby(["dataset", "top_m"], as_index=False)
        .agg(
            n_windows=("bicdiff_candidate_count", "size"),
            positive_windows=("bicdiff_candidate_count", lambda x: int((x > 0).sum())),
            mean_bicdiff_candidate_count=("bicdiff_candidate_count", "mean"),
            median_bicdiff_candidate_count=("bicdiff_candidate_count", "median"),
            min_bicdiff_candidate_count=("bicdiff_candidate_count", "min"),
            max_bicdiff_candidate_count=("bicdiff_candidate_count", "max"),
        )
    )

    summary["positive_rate"] = (
        100.0 * summary["positive_windows"] / summary["n_windows"]
    )

    dataset_rank = {ds: i for i, ds in enumerate(DATASET_ORDER)}
    summary["dataset_rank"] = summary["dataset"].map(dataset_rank)
    summary = summary.sort_values(["dataset_rank", "top_m"]).drop(columns=["dataset_rank"])

    return summary


def get_row(summary: pd.DataFrame, dataset: str, top_m: int) -> pd.Series | None:
    row = summary[(summary["dataset"] == dataset) & (summary["top_m"] == top_m)]
    if row.empty:
        return None
    return row.iloc[0]


def setup_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GRID, linewidth=0.9, alpha=0.85)
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="both", colors="#3D4451", labelsize=9.5)


def plot_bic_stability(ax, summary: pd.DataFrame) -> None:
    setup_axis(ax)

    offsets = {
        "ret-1": -0.08,
        "PVC-3 area17": 0.00,
        "PVC-11 monkey2": 0.08,
    }

    for dataset in DATASET_ORDER:
        xs = []
        means = []
        err_low = []
        err_high = []

        for top_m in [1, 2, 3]:
            row = get_row(summary, dataset, top_m)
            if row is None:
                continue

            mean = float(row["mean_bicdiff_candidate_count"])
            low = float(row["min_bicdiff_candidate_count"])
            high = float(row["max_bicdiff_candidate_count"])

            xs.append(top_m + offsets[dataset])
            means.append(mean)
            err_low.append(mean - low)
            err_high.append(high - mean)

        if not xs:
            continue

        ax.errorbar(
            xs,
            means,
            yerr=[err_low, err_high],
            fmt="o-",
            color=PALETTE[dataset],
            linewidth=2.25,
            markersize=6.2,
            capsize=3.8,
            elinewidth=1.35,
            markeredgecolor="white",
            markeredgewidth=0.9,
            alpha=0.96,
            zorder=3,
        )

    ax.axhline(
        0.0,
        color="#667085",
        linewidth=1.15,
        linestyle=(0, (4, 2)),
        alpha=0.9,
        zorder=2,
    )

    ax.set_xlim(0.75, 3.25)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Top-m candidate pairs", fontsize=10.2, color=TEXT, labelpad=6)
    ax.set_ylabel("Mean candidate-count ΔBIC", fontsize=10.2, color=TEXT, labelpad=8)
    ax.set_title("(A) BIC stability", fontsize=12.5, fontweight="bold", color=TEXT, pad=12)


def plot_positive_rate(ax, summary: pd.DataFrame) -> None:
    setup_axis(ax)

    topm_values = [1, 2, 3]
    x = np.arange(len(topm_values), dtype=float)
    width = 0.20

    for j, dataset in enumerate(DATASET_ORDER):
        heights = []
        labels = []

        for top_m in topm_values:
            row = get_row(summary, dataset, top_m)
            if row is None:
                heights.append(np.nan)
                labels.append("")
            else:
                heights.append(float(row["positive_rate"]))
                labels.append(
                    f'{int(row["positive_windows"])}/{int(row["n_windows"])}'
                )

        xpos = x + (j - 1) * width

        bars = ax.bar(
            xpos,
            heights,
            width=width,
            color=PALETTE[dataset],
            alpha=0.93,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )

        # Put labels inside bars to avoid crowded labels above 100%.
        for bar, lab, val in zip(bars, labels, heights):
            if not lab or np.isnan(val):
                continue

            if val >= 25:
                y = val - 5.5
                va = "top"
                color = "white"
                weight = "bold"
            else:
                y = val + 3.0
                va = "bottom"
                color = "#384250"
                weight = "normal"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                lab,
                ha="center",
                va=va,
                fontsize=8.9,
                fontweight=weight,
                color=color,
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in topm_values])
    ax.set_xlim(-0.45, len(topm_values) - 0.55)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])

    ax.set_xlabel("Top-m candidate pairs", fontsize=10.2, color=TEXT, labelpad=6)
    ax.set_ylabel("Positive-window rate (%)", fontsize=10.2, color=TEXT, labelpad=8)
    ax.set_title("(B) Positive-window rate", fontsize=12.5, fontweight="bold", color=TEXT, pad=12)


def plot_median_heatmap(ax, summary: pd.DataFrame, fig) -> None:
    topm_values = [1, 2, 3]

    mat = np.full((len(DATASET_ORDER), len(topm_values)), np.nan)

    for i, dataset in enumerate(DATASET_ORDER):
        for j, top_m in enumerate(topm_values):
            row = get_row(summary, dataset, top_m)
            if row is not None:
                mat[i, j] = float(row["median_bicdiff_candidate_count"])

    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        vmax = 20.0
    else:
        vmax = max(20.0, float(np.max(np.abs(finite))))

    cmap = LinearSegmentedColormap.from_list(
        "bic_support",
        ["#D95F5F", "#F7F7F7", "#2F80C1"],
    )
    cmap.set_bad(color="white")

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    ax.imshow(
        np.ma.masked_invalid(mat),
        cmap=cmap,
        norm=norm,
        aspect="auto",
        zorder=2,
    )

    ax.set_xticks(np.arange(len(topm_values)))
    ax.set_xticklabels([str(m) for m in topm_values])

    # Use compact R1/R2/R3 row labels; full names are already in the legend.
    ax.set_yticks(np.arange(len(DATASET_ORDER)))
    ax.set_yticklabels(["R1", "R2", "R3"])

    ax.tick_params(axis="both", colors="#3D4451", labelsize=9.5)
    ax.set_xlabel("Top-m candidate pairs", fontsize=10.2, color=TEXT, labelpad=6)
    ax.set_title("(C) Median ?BIC support", fontsize=12.5, fontweight="bold", color=TEXT, pad=12)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                txt = f"{val:.1f}"
            else:
                txt = "?"

            color = "white" if np.isfinite(val) and abs(val) > 0.55 * vmax else "#1F2937"

            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=10.2,
                fontweight="bold",
                color=color,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(-0.5, len(topm_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DATASET_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)


def print_summary(summary: pd.DataFrame) -> None:
    display = summary.copy()
    display["positive"] = (
        display["positive_windows"].astype(int).astype(str)
        + "/"
        + display["n_windows"].astype(int).astype(str)
    )

    print("\nSummary by dataset and top_m")
    print("-" * 100)
    print(
        display[
            [
                "dataset",
                "top_m",
                "positive",
                "positive_rate",
                "mean_bicdiff_candidate_count",
                "median_bicdiff_candidate_count",
                "min_bicdiff_candidate_count",
                "max_bicdiff_candidate_count",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[str(p) for p in DEFAULT_INPUTS],
        help="Per-dataset real-data CSV files.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help="Directory for combined and summary CSV outputs.",
    )
    parser.add_argument(
        "--fig-dir",
        default=str(FIG_DIR),
        help="Directory for panel figure outputs.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dfs = [load_one_csv(Path(p)) for p in args.inputs]
    combined = pd.concat(dfs, ignore_index=True)

    dataset_rank = {ds: i for i, ds in enumerate(DATASET_ORDER)}
    combined["dataset_rank"] = combined["dataset"].map(dataset_rank)
    combined = (
        combined.sort_values(["dataset_rank", "top_m"])
        .drop(columns=["dataset_rank"])
        .reset_index(drop=True)
    )

    summary = build_summary(combined)

    combined_csv = results_dir / "real04_combined.csv"
    summary_csv = results_dir / "real04_summary_by_dataset_topm.csv"

    combined.to_csv(combined_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": TEXT,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.2, 5.8),
        gridspec_kw={"width_ratios": [1.05, 1.05, 1.05]},
    )
    fig.patch.set_facecolor("white")

    plot_bic_stability(axes[0], summary)
    plot_positive_rate(axes[1], summary)
    plot_median_heatmap(axes[2], summary, fig)

    handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE[ds],
            marker="o",
            linewidth=2.4,
            markersize=6.2,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=DATASET_LABELS[ds],
        )
        for ds in DATASET_ORDER
    ]

    # Manual layout. Do not use tight_layout here; it causes header overlap.
    fig.subplots_adjust(
        left=0.055,
        right=0.970,
        bottom=0.155,
        top=0.745,
        wspace=0.30,
    )

    fig.suptitle(
        "Formal real-data summary",
        fontsize=16.0,
        fontweight="bold",
        color=TEXT,
        y=0.965,
    )

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=10.5,
        bbox_to_anchor=(0.5, 0.905),
        columnspacing=2.2,
        handlelength=2.6,
        handletextpad=0.55,
    )

    fig.text(
        0.5,
        0.835,
        "Candidate-count ΔBIC is the primary held-out statistic. Positive values favor the HTH model after candidate-complexity penalty.",
        ha="center",
        va="center",
        fontsize=9.7,
        color=MUTED_TEXT,
    )

    out_png = fig_dir / "real04_summary_panel.png"
    out_pdf = fig_dir / "real04_summary_panel.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print_summary(summary)

    print("\nSaved:")
    print(f"  {combined_csv}")
    print(f"  {summary_csv}")
    print(f"  {out_png}")
    print(f"  {out_pdf}")


if __name__ == "__main__":
    main()
