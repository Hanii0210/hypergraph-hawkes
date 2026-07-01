"""
Figure 4(a) -- kernel-timescale ablation.

Two-panel design:

    (a) recovered hyperedge weight vs kernel rate:
        per-seed estimates, mean +/- SD, and the true value;
    (b) estimator dispersion vs kernel rate.

Reads:
    experiments/results/synthetic/syn08_bias_ablation.pkl

Writes:
    figures/synthetic/syn08_bias_ablation.png
    figures/synthetic/syn08_bias_ablation.pdf
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "synthetic" / "syn08_bias_ablation.pkl"
FIG_DIR = ROOT / "figures" / "synthetic"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRUE_ALPHA = 0.4

PALETTE = {
    "hth": "#3E7CB1",
    "warn": "#E08B3E",
}

BIAS_BANDS = {
    "good": "#2FA889",
    "mid": "#E0B43E",
    "bad": "#D95F5F",
}


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


def bias_band_color(relative_bias: float) -> str:
    """Colour-code relative bias by magnitude."""
    magnitude = abs(relative_bias)
    if magnitude < 5:
        return BIAS_BANDS["good"]
    if magnitude < 15:
        return BIAS_BANDS["mid"]
    return BIAS_BANDS["bad"]


def main() -> None:
    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {RESULT_PATH}")

    apply_style()

    with open(RESULT_PATH, "rb") as f:
        results = pickle.load(f)

    betas = np.array([r["beta"] for r in results], dtype=float)
    means = np.array([r["mean"] for r in results], dtype=float)
    stds = np.array([r["std"] for r in results], dtype=float)
    rel_bias = np.array([r["rel_bias"] for r in results], dtype=float)
    values = [np.asarray(r["values"], dtype=float) for r in results]

    x = np.arange(len(betas))
    beta_labels = [f"{b:g}" for b in betas]

    fig, (axA, axB) = plt.subplots(
        1,
        2,
        figsize=(9.6, 3.9),
        gridspec_kw={"wspace": 0.28},
    )

    # ------------------------------------------------------------------
    # (a) Recovery: per-seed jitter + mean +/- SD
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)

    for i, v in enumerate(values):
        jitter_x = x[i] + rng.uniform(-0.16, 0.16, size=v.size)
        axA.scatter(
            jitter_x,
            v,
            s=12,
            color=PALETTE["hth"],
            alpha=0.30,
            edgecolor="none",
            zorder=2,
        )

    axA.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o",
        color=PALETTE["hth"],
        markersize=7,
        markerfacecolor=PALETTE["hth"],
        markeredgecolor="white",
        markeredgewidth=0.9,
        capsize=4,
        elinewidth=1.5,
        zorder=4,
        label=r"mean $\pm$ SD",
    )

    axA.axhline(
        TRUE_ALPHA,
        color=PALETTE["warn"],
        linestyle="--",
        linewidth=1.6,
        zorder=3,
        label=r"true $\alpha=0.4$",
    )

    for i in range(len(x)):
        y_text = means[i] + stds[i]
        axA.annotate(
            f"{rel_bias[i]:+.0f}%",
            xy=(x[i], y_text),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=bias_band_color(rel_bias[i]),
            fontweight="bold",
        )

    axA.set_xticks(x)
    axA.set_xticklabels(beta_labels)
    axA.set_xlabel(r"kernel decay rate $\beta$")
    axA.set_ylabel(r"inferred $\alpha_{(0,1)}$")
    axA.set_title("(a) recovered weight vs kernel rate", loc="left")
    axA.set_ylim(-0.05, float(np.max(means + stds)) + 0.14)

    for spine in ("top", "right"):
        axA.spines[spine].set_visible(False)

    axA.grid(axis="y", color="0.90", linewidth=0.7)
    axA.set_axisbelow(True)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE["hth"],
            marker="o",
            markerfacecolor=PALETTE["hth"],
            markeredgecolor="white",
            markersize=7,
            label=r"mean $\pm$ SD",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["warn"],
            linestyle="--",
            linewidth=1.6,
            label=r"true $\alpha=0.4$",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=BIAS_BANDS["good"],
            markeredgecolor="none",
            markersize=7,
            label=r"$|$bias$|<5\%$",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=BIAS_BANDS["mid"],
            markeredgecolor="none",
            markersize=7,
            label=r"$5$--$15\%$",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=BIAS_BANDS["bad"],
            markeredgecolor="none",
            markersize=7,
            label=r"$>15\%$",
        ),
    ]

    axA.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=7.6,
        handlelength=1.4,
        labelspacing=0.3,
        borderpad=0.4,
        frameon=True,
        framealpha=0.92,
    )

    # ------------------------------------------------------------------
    # (b) Dispersion grows with beta
    # ------------------------------------------------------------------
    axB.bar(
        x,
        stds,
        width=0.6,
        color=PALETTE["hth"],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    axB.plot(
        x,
        stds,
        "o-",
        color=PALETTE["warn"],
        markersize=4,
        linewidth=1.4,
        zorder=4,
        label="monotone trend",
    )

    for i in range(len(x)):
        axB.annotate(
            f"{stds[i]:.3f}",
            xy=(x[i], stds[i]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="0.25",
        )

    axB.set_xticks(x)
    axB.set_xticklabels(beta_labels)
    axB.set_xlabel(r"kernel decay rate $\beta$")
    axB.set_ylabel(r"SD of $\hat{\alpha}_{(0,1)}$")
    axB.set_title("(b) estimator dispersion grows with faster kernels", loc="left")
    axB.set_ylim(0, float(np.max(stds)) * 1.22)

    axB.legend(loc="upper left", handlelength=1.6, frameon=True, framealpha=0.92)

    for spine in ("top", "right"):
        axB.spines[spine].set_visible(False)

    axB.grid(axis="y", color="0.90", linewidth=0.7)
    axB.set_axisbelow(True)

    out_png = FIG_DIR / "syn08_bias_ablation.png"
    out_pdf = FIG_DIR / "syn08_bias_ablation.pdf"

    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
