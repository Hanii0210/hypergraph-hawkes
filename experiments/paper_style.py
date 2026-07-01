"""
Shared house style for all paper figures (Taylor & Francis / JSCS look).

Import and call apply_style() at the top of every plotting script so the whole
paper shares one typographic and colour system, consistent with Figure 1
(serif text, Computer-Modern maths, a restrained palette).

    from paper_style import apply_style, PALETTE, DATASET_COLORS, despine
    apply_style()
"""
from __future__ import annotations
import matplotlib as mpl

# --- coherent, colourblind-aware, print-safe palette (matches Figure 1) ---
PALETTE = {
    "hth":      "#2b6cb0",   # primary: HTH / favours-HTH
    "pairwise": "#8a8f99",   # neutral grey: pairwise / baseline
    "accent":   "#1f8a4c",   # green: positive / target
    "warn":     "#c0392b",   # red: negative / threshold
    "amber":    "#d08a1d",   # secondary categorical
    "muted":    "0.45",
    "band":     "#f0a868",   # Delta-window fill (Figure 1)
}

# fixed per-dataset colours so every real-data panel agrees
DATASET_COLORS = {
    "ret-1":          "#2b6cb0",
    "PVC-3 area17":   "#d08a1d",
    "PVC-11 monkey2": "#2e7d55",
}

# bias-magnitude bands (used by syn08 / syn09)
BIAS_BANDS = {
    "good": "#1f8a4c",   # |bias| < 5%
    "mid":  "#d08a1d",   # 5% <= |bias| < 15%
    "bad":  "#c0392b",   # |bias| >= 15%
}


def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "0.3",
        "axes.axisbelow": True,
        "axes.grid": False,
        "grid.color": "0.9",
        "grid.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.color": "0.3",
        "ytick.color": "0.3",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
        "patch.linewidth": 0.7,
    })


def despine(ax, keep=("left", "bottom")):
    """Hide spines not in `keep`; add a light y-grid behind the data."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.set_axisbelow(True)