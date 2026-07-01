"""
Figure 5 -- interaction-baseline comparison.

Held-out log-likelihood gain over the pairwise model for a parameter-matched
3-way interaction baseline versus the pattern-completion HTH mechanism, across
true hyperedge strength. The margin, HTH minus baseline, is annotated above
each HTH bar. At alpha = 0, neither model gains.

Reads:
    experiments/results/synthetic/syn10_interaction_baseline.pkl

Writes:
    figures/synthetic/syn10_interaction_baseline.png
    figures/synthetic/syn10_interaction_baseline.pdf
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT_PATH = HERE / "results" / "synthetic" / "syn10_interaction_baseline.pkl"
FIG_DIR = ROOT / "figures" / "synthetic"
FIG_DIR.mkdir(parents=True, exist_ok=True)


PALETTE = {
    "pairwise": "#667085",
    "hth": "#3E7CB1",
    "amber": "#E0B43E",
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
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {RESULT_PATH}")

    apply_style()

    with open(RESULT_PATH, "rb") as f:
        data = pickle.load(f)

    rows = data["rows"]

    alphas = np.array([r["alpha"] for r in rows], dtype=float)
    dL_inter = np.array([r["dL_inter"] for r in rows], dtype=float)
    dL_HTH = np.array([r["dL_HTH"] for r in rows], dtype=float)
    margin = np.array([r["margin"] for r in rows], dtype=float)
    dLi_sem = np.array([r.get("dL_inter_sem", 0.0) for r in rows], dtype=float)
    dLh_sem = np.array([r.get("dL_HTH_sem", 0.0) for r in rows], dtype=float)

    x = np.arange(len(alphas))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    ax.bar(
        x - width / 2,
        dL_inter,
        width=width,
        color=PALETTE["pairwise"],
        edgecolor="white",
        linewidth=0.6,
        yerr=dLi_sem,
        capsize=3,
        error_kw={"elinewidth": 1.1, "ecolor": "0.35"},
        label="3-way interaction baseline",
        zorder=3,
    )

    ax.bar(
        x + width / 2,
        dL_HTH,
        width=width,
        color=PALETTE["hth"],
        edgecolor="white",
        linewidth=0.6,
        yerr=dLh_sem,
        capsize=3,
        error_kw={"elinewidth": 1.1, "ecolor": "0.35"},
        label="HTH (pattern-completion)",
        zorder=3,
    )

    ax.axhline(0, color="0.4", linewidth=1.0, zorder=1)

    for i in range(len(alphas)):
        ax.annotate(
            f"{dL_inter[i]:.2f}",
            xy=(x[i] - width / 2, dL_inter[i] + dLi_sem[i]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="0.35",
        )

        ax.annotate(
            f"{dL_HTH[i]:.2f}",
            xy=(x[i] + width / 2, dL_HTH[i] + dLh_sem[i]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="0.15",
        )

        if alphas[i] > 0:
            ax.annotate(
                f"+{margin[i]:.1f}",
                xy=(x[i] + width / 2, dL_HTH[i] + dLh_sem[i]),
                xytext=(0, 17),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=PALETTE["amber"],
                fontweight="bold",
            )

    ax.annotate(
        "null",
        xy=(x[0], 0),
        xytext=(x[0], -0.05),
        ha="center",
        va="top",
        fontsize=8,
        color="0.5",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_xlabel(r"true hyperedge strength $\alpha$")
    ax.set_ylabel(r"held-out gain over pairwise  $\Delta\ell$")

    ymax = float(np.max(dL_HTH + dLh_sem))
    ax.set_ylim(-0.3, ymax * 1.20 if ymax > 0 else 1.0)

    ax.legend(loc="upper left", handlelength=1.4, frameon=True, framealpha=0.92)

    ax.text(
        0.02,
        0.80,
        r"amber: margin (HTH $-$ baseline)",
        transform=ax.transAxes,
        fontsize=7.6,
        color=PALETTE["amber"],
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.grid(axis="y", color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)

    out_png = FIG_DIR / "syn10_interaction_baseline.png"
    out_pdf = FIG_DIR / "syn10_interaction_baseline.pdf"

    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Loaded: {RESULT_PATH}")
    print(f"Saved:  {out_png}")
    print(f"Saved:  {out_pdf}")


if __name__ == "__main__":
    main()
