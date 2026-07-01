"""
Figure 2 -- synthetic recovery across repeated datasets.

Five histograms of recovered parameters, with true value and sample mean marked,
plus a forest-style summary panel showing mean relative deviation from truth
with sampling spread.

Reads:
    experiments/results/synthetic/syn01_recovery_robustness.pkl

Writes:
    figures/synthetic/syn01_recovery_robustness.png
    figures/synthetic/syn01_recovery_robustness.pdf
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT_PATH = HERE / "results" / "synthetic" / "syn01_recovery_robustness.pkl"
FIG_DIR = ROOT / "figures" / "synthetic"
FIG_DIR.mkdir(parents=True, exist_ok=True)


PALETTE = {
    "hth": "#3E7CB1",
    "warn": "#E08B3E",
    "amber": "#E0B43E",
}


KEYS = ["mu[0]", "mu[1]", "mu[2]", "a[2->0]", "alpha_hyper(0,1)"]
DISPLAY = [
    r"$\mu_0$",
    r"$\mu_1$",
    r"$\mu_2$",
    r"$\alpha_{2\to0}$",
    r"$\alpha_{(0,1)}$",
]
IS_HYPEREDGE = [False, False, False, False, True]


def apply_style() -> None:
    """Compact paper-style plotting defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_results() -> tuple[dict, dict, int]:
    """Load and validate the synthetic recovery result file."""
    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {RESULT_PATH}")

    with open(RESULT_PATH, "rb") as f:
        data = pickle.load(f)

    required_top = {"recovered", "true_values"}
    missing_top = sorted(required_top - set(data))
    if missing_top:
        raise KeyError(f"Missing keys in {RESULT_PATH}: {missing_top}")

    recovered = data["recovered"]
    true_values = data["true_values"]

    missing_recovered = [k for k in KEYS if k not in recovered]
    missing_true = [k for k in KEYS if k not in true_values]

    if missing_recovered:
        raise KeyError(f"Missing recovered parameter arrays: {missing_recovered}")
    if missing_true:
        raise KeyError(f"Missing true parameter values: {missing_true}")

    n_seeds = int(data.get("N_SEEDS", len(np.asarray(recovered[KEYS[0]]))))

    return recovered, true_values, n_seeds


def relative_error_percent(mean_value: float, true_value: float) -> float:
    """Absolute relative error in percent, robust to very small true values."""
    denom = max(abs(float(true_value)), 1e-12)
    return abs(float(mean_value) - float(true_value)) / denom * 100.0


def relative_deviation_percent(values: np.ndarray, true_value: float) -> tuple[float, float]:
    """Mean relative deviation and relative SD in percent."""
    denom = max(abs(float(true_value)), 1e-12)
    mean_dev = (float(np.mean(values)) - float(true_value)) / denom * 100.0
    rel_sd = float(np.std(values, ddof=0)) / denom * 100.0
    return mean_dev, rel_sd


def main() -> None:
    apply_style()
    recovered, true_values, n_seeds = load_results()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.6, 5.6),
        gridspec_kw={"wspace": 0.26, "hspace": 0.42},
    )
    axes = axes.flatten()

    # ------------------------------------------------------------------
    # First five panels: histograms of recovered parameters
    # ------------------------------------------------------------------
    for i, (ax, key, label) in enumerate(zip(axes[:5], KEYS, DISPLAY)):
        values = np.asarray(recovered[key], dtype=float)
        true_value = float(true_values[key])
        mean_value = float(np.mean(values))
        rel_error = relative_error_percent(mean_value, true_value)

        lo = min(float(np.min(values)), true_value)
        hi = max(float(np.max(values)), true_value)
        pad = (hi - lo) * 0.10 + 1e-6
        bins = np.linspace(lo - pad, hi + pad, 12)

        ax.hist(
            values,
            bins=bins,
            color=PALETTE["hth"],
            alpha=0.80,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        ax.axvline(true_value, color=PALETTE["warn"], linewidth=2.2, zorder=4)
        ax.axvline(mean_value, color="0.20", linewidth=1.6, linestyle=(0, (4, 3)), zorder=5)

        ax.set_title(f"{label}   (rel. err {rel_error:.1f}%)", loc="left", fontsize=10)
        ax.set_xlabel("inferred value")

        if i % 3 == 0:
            ax.set_ylabel("count")

        ax.set_xlim(lo - pad, hi + pad)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        ax.grid(axis="y", color="0.92", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].legend(
        handles=[
            Line2D([0], [0], color=PALETTE["warn"], linewidth=2.2, label="true"),
            Line2D([0], [0], color="0.20", linewidth=1.6, linestyle=(0, (4, 3)), label="mean"),
        ],
        loc="upper left",
        fontsize=8,
        handlelength=1.6,
        labelspacing=0.3,
        frameon=True,
        framealpha=0.92,
    )

    # ------------------------------------------------------------------
    # Sixth panel: forest summary of mean relative deviation +/- relative SD
    # ------------------------------------------------------------------
    axF = axes[5]
    y_positions = np.arange(len(KEYS))[::-1]

    rel_dev = []
    rel_sd = []

    for key in KEYS:
        values = np.asarray(recovered[key], dtype=float)
        mean_dev, spread = relative_deviation_percent(values, float(true_values[key]))
        rel_dev.append(mean_dev)
        rel_sd.append(spread)

    rel_dev = np.asarray(rel_dev, dtype=float)
    rel_sd = np.asarray(rel_sd, dtype=float)

    colors = [PALETTE["amber"] if is_h else PALETTE["hth"] for is_h in IS_HYPEREDGE]

    axF.axvspan(-5, 5, color="0.90", zorder=0)
    axF.axvline(0, color=PALETTE["warn"], linewidth=1.4, zorder=1)

    for y, mean_dev, spread, color in zip(y_positions, rel_dev, rel_sd, colors):
        axF.errorbar(
            mean_dev,
            y,
            xerr=spread,
            fmt="o",
            color=color,
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=3,
            elinewidth=1.4,
            zorder=3,
        )

    axF.set_yticks(y_positions)
    axF.set_yticklabels(DISPLAY)
    axF.set_ylim(-0.6, len(KEYS) - 0.4)
    axF.set_xlabel("relative deviation from truth (%)")
    axF.set_title(r"recovery summary (mean $\pm$ SD)", loc="left", fontsize=10)

    for spine in ("top", "right"):
        axF.spines[spine].set_visible(False)

    axF.grid(axis="x", color="0.92", linewidth=0.6)
    axF.set_axisbelow(True)

    axF.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=PALETTE["hth"],
                markeredgecolor="white",
                markersize=6,
                label="baseline / pairwise",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=PALETTE["amber"],
                markeredgecolor="white",
                markersize=6,
                label="hyperedge",
            ),
        ],
        loc="upper right",
        fontsize=7.6,
        handlelength=1.2,
        labelspacing=0.3,
        frameon=True,
        framealpha=0.92,
    )

    # No top-level title. The manuscript caption explains the figure.

    out_png = FIG_DIR / "syn01_recovery_robustness.png"
    out_pdf = FIG_DIR / "syn01_recovery_robustness.pdf"

    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Loaded: {RESULT_PATH}")
    print(f"N seeds: {n_seeds}")
    print(f"Saved:  {out_png}")
    print(f"Saved:  {out_pdf}")


if __name__ == "__main__":
    main()
