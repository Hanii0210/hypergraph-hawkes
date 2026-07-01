"""
Figure 3 -- likelihood separation (falsifiability), decision-axis view.

Each scenario is placed as a dot on a single Delta-BIC evidence axis, split into
zones by the +/-6 strong-evidence thresholds. Scenario A, generated with a
hyperedge, should favour HTH; scenario B, generated under a pairwise-only null,
should favour the pairwise model. The nested-model likelihood gain Delta-loglik,
which is non-negative by construction, is annotated as support.

Because Delta-BIC for scenario A can be positive but below the +6 line, the
figure separates positive evidence from strong evidence.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT_PATH = HERE / "results" / "synthetic" / "syn05_likelihood_separation.pkl"
FIG_DIR = ROOT / "figures" / "synthetic"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))

try:
    from paper_style import apply_style, PALETTE
except ModuleNotFoundError:
    PALETTE = {
        "pairwise": "#667085",
        "hth": "#3E7CB1",
        "warn": "#E08B3E",
        "accent": "#2FA889",
        "amber": "#E0B43E",
    }

    def apply_style() -> None:
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


def get_palette(key: str, fallback: str) -> str:
    """Use shared paper palette when available, otherwise use fallback."""
    return PALETTE[key] if key in PALETTE else fallback


def main() -> None:
    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {RESULT_PATH}")

    apply_style()

    with open(RESULT_PATH, "rb") as f:
        result = pickle.load(f)

    scenarios = ["A_with_hyperedge", "B_no_hyperedge"]
    y_labels = [
        "A: data WITH hyperedge\n(expect HTH)",
        "B: pairwise-only null\n(expect pairwise)",
    ]

    bic_diff = np.array([result[s]["bic_diff"] for s in scenarios], dtype=float)
    delta_ll = np.array([result[s]["delta_L"] for s in scenarios], dtype=float)

    y_positions = np.array([1, 0], dtype=float)

    # Fixed decision-axis limits from the Claude version, with a small safety
    # extension in case regenerated values move slightly outside the range.
    xlo = min(-11.0, float(np.min(bic_diff)) - 1.5)
    xhi = max(9.0, float(np.max(bic_diff)) + 1.5)

    pairwise_color = get_palette("pairwise", "#667085")
    hth_color = get_palette("hth", "#3E7CB1")
    warn_color = get_palette("warn", "#E08B3E")

    fig, ax = plt.subplots(figsize=(7.6, 2.9))

    # ------------------------------------------------------------------
    # Evidence zones
    # ------------------------------------------------------------------
    ax.axvspan(xlo, -6, color=pairwise_color, alpha=0.14, zorder=0)
    ax.axvspan(-6, 6, color="0.965", zorder=0)
    ax.axvspan(6, xhi, color=hth_color, alpha=0.12, zorder=0)

    ax.axvline(0, color="0.35", linewidth=1.2, zorder=2)

    for threshold in (-6, 6):
        ax.axvline(
            threshold,
            color=warn_color,
            linestyle=(0, (3, 2)),
            linewidth=1.3,
            zorder=2,
        )

    ax.text(
        -8.5,
        1.72,
        "strong: pairwise",
        color=pairwise_color,
        fontsize=7.6,
        ha="center",
        va="center",
        fontweight="bold",
    )

    ax.text(
        7.5,
        1.72,
        "strong: HTH",
        color=hth_color,
        fontsize=7.6,
        ha="center",
        va="center",
        fontweight="bold",
    )

    # ------------------------------------------------------------------
    # Scenario points on the decision axis
    # ------------------------------------------------------------------
    for y, bic, dll in zip(y_positions, bic_diff, delta_ll):
        color = hth_color if bic > 0 else pairwise_color

        ax.plot(
            [0, bic],
            [y, y],
            color=color,
            linewidth=1.8,
            solid_capstyle="round",
            zorder=3,
        )

        ax.scatter(
            [bic],
            [y],
            s=80,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )

        ha = "left" if bic > 0 else "right"
        dx = 8 if bic > 0 else -8

        ax.annotate(
            rf"$\Delta$BIC = {bic:+.2f}",
            xy=(bic, y),
            xytext=(dx, 8),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="0.12",
        )

        ax.annotate(
            rf"$\Delta\ell$ = {dll:+.2f}",
            xy=(bic, y),
            xytext=(dx, -9),
            textcoords="offset points",
            ha=ha,
            va="top",
            fontsize=8,
            color="0.42",
        )

    # ------------------------------------------------------------------
    # Axes and labels
    # ------------------------------------------------------------------
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_ylim(-0.6, 1.95)

    ax.set_xlim(xlo, xhi)
    ax.set_xticks([-9, -6, -3, 0, 3, 6, 9])

    ax.set_xlabel(
        r"$\Delta$BIC  "
        r"($\leftarrow$ favours pairwise    $\cdot$    favours HTH $\rightarrow$)"
    )

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis="y", length=0)
    ax.grid(False)

    out_png = FIG_DIR / "syn05_likelihood_separation.png"
    out_pdf = FIG_DIR / "syn05_likelihood_separation.pdf"

    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
