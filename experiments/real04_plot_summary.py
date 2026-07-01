"""
Figure 6 -- formal real-data summary panel (real04_summary_panel).

Builds the three-panel held-out real-data summary used in the manuscript:

    (a) candidate-count BIC stability across top-m, shown as mean with min-max range;
    (b) positive held-out window rate per dataset and top-m;
    (c) median candidate-count BIC as a diverging heatmap.

Reads:
    experiments/results/realdata/real04_summary_by_dataset_topm.csv

Writes:
    figures/realdata/real04_summary_panel.png
    figures/realdata/real04_summary_panel.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DATASET_ORDER = ["ret-1", "PVC-3 area17", "PVC-11 monkey2"]
DATASET_SHORT = {
    "ret-1": "ret-1",
    "PVC-3 area17": "PVC-3",
    "PVC-11 monkey2": "PVC-11",
}
DATASET_COLORS = {
    "ret-1": "#3E7CB1",
    "PVC-3 area17": "#2FA889",
    "PVC-11 monkey2": "#E08B3E",
}
MARKERS = {
    "ret-1": "o",
    "PVC-3 area17": "s",
    "PVC-11 monkey2": "^",
}


def apply_style() -> None:
    """Compact manuscript-style plotting defaults."""
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
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_csv(cli_path: str | None) -> Path:
    """Find the real-data summary CSV robustly across old/new filenames."""
    here = Path(__file__).resolve().parent
    root = here.parent

    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path))

    candidates.extend(
        [
            here / "results" / "realdata" / "real04_summary_by_dataset_topm.csv",
            root / "experiments" / "results" / "realdata" / "real04_summary_by_dataset_topm.csv",
            Path.cwd() / "experiments" / "results" / "realdata" / "real04_summary_by_dataset_topm.csv",

            # Legacy fallback names, kept only to avoid breaking older local copies.
            here / "results" / "realdata" / "realdata_summary_by_dataset_topm.csv",
            root / "experiments" / "results" / "realdata" / "realdata_summary_by_dataset_topm.csv",
            Path.cwd() / "experiments" / "results" / "realdata" / "realdata_summary_by_dataset_topm.csv",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {c}" for c in candidates)
    raise SystemExit(f"Could not find the real-data summary CSV. Tried:\n{tried}")


def validate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardise the summary table."""
    required = {
        "dataset",
        "top_m",
        "n_windows",
        "positive_windows",
        "mean_bicdiff_candidate_count",
        "median_bicdiff_candidate_count",
        "min_bicdiff_candidate_count",
        "max_bicdiff_candidate_count",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in summary CSV: {missing}")

    df = df.copy()
    df["top_m"] = df["top_m"].astype(int)
    df["n_windows"] = df["n_windows"].astype(int)
    df["positive_windows"] = df["positive_windows"].astype(int)

    expected = {(ds, m) for ds in DATASET_ORDER for m in sorted(df["top_m"].unique())}
    observed = {(str(r.dataset), int(r.top_m)) for r in df.itertuples()}
    missing_rows = sorted(expected - observed)
    if missing_rows:
        raise ValueError(f"Missing dataset/top_m rows: {missing_rows}")

    return df


def positive_rate_percent(sub: pd.DataFrame) -> np.ndarray:
    """Compute positive-window rate as percent from counts.

    This avoids ambiguity because some CSVs store positive_rate as 100.0,
    while others may store it as 1.0.
    """
    return 100.0 * sub["positive_windows"].to_numpy(dtype=float) / sub["n_windows"].to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to real04_summary_by_dataset_topm.csv")
    parser.add_argument("--fig-dir", default=None, help="Output directory for figures")
    args = parser.parse_args()

    csv_path = resolve_csv(args.csv)
    df = validate_summary(pd.read_csv(csv_path))

    here = Path(__file__).resolve().parent
    fig_dir = Path(args.fig_dir) if args.fig_dir else here.parent / "figures" / "realdata"
    fig_dir.mkdir(parents=True, exist_ok=True)

    apply_style()

    topm = sorted(df["top_m"].unique())
    fig, (axA, axB, axC) = plt.subplots(
        1,
        3,
        figsize=(10.8, 3.55),
        gridspec_kw={"width_ratios": [1.05, 1.05, 1.14], "wspace": 0.32},
    )

    # ------------------------------------------------------------------
    # (a) BIC stability: mean with min-max whiskers
    # ------------------------------------------------------------------
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds].sort_values("top_m")
        x = sub["top_m"].to_numpy(dtype=float)
        mean = sub["mean_bicdiff_candidate_count"].to_numpy(dtype=float)
        lo = mean - sub["min_bicdiff_candidate_count"].to_numpy(dtype=float)
        hi = sub["max_bicdiff_candidate_count"].to_numpy(dtype=float) - mean
        color = DATASET_COLORS[ds]

        axA.errorbar(
            x,
            mean,
            yerr=[lo, hi],
            color=color,
            marker=MARKERS[ds],
            markersize=5.8,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.0,
            capsize=3,
            elinewidth=1.15,
            label=DATASET_SHORT[ds],
            zorder=3,
        )

    axA.axhline(0, color="0.45", linestyle="--", linewidth=0.9, zorder=1)
    axA.set_xticks(topm)
    axA.set_xlabel(r"top-$m$ candidate pairs")
    axA.set_ylabel(r"candidate-count $\Delta$BIC")
    axA.set_title("(a) held-out BIC stability", loc="left")
    axA.grid(axis="y", color="0.90", linewidth=0.7)
    axA.set_axisbelow(True)
    for spine in ("top", "right"):
        axA.spines[spine].set_visible(False)

    # ------------------------------------------------------------------
    # (b) Positive-window rate: grouped bars
    # ------------------------------------------------------------------
    n_ds = len(DATASET_ORDER)
    width = 0.26
    x0 = np.arange(len(topm))

    for i, ds in enumerate(DATASET_ORDER):
        sub = df[df["dataset"] == ds].set_index("top_m").reindex(topm)
        rate = positive_rate_percent(sub)
        pos = sub["positive_windows"].to_numpy(dtype=int)
        nwin = sub["n_windows"].to_numpy(dtype=int)
        xs = x0 + (i - (n_ds - 1) / 2) * width

        axB.bar(
            xs,
            rate,
            width=width,
            color=DATASET_COLORS[ds],
            edgecolor="white",
            linewidth=0.6,
            label=DATASET_SHORT[ds],
            zorder=3,
        )

        for xb, r, pp, nn in zip(xs, rate, pos, nwin):
            axB.text(
                xb,
                min(r + 3.0, 111.0),
                f"{pp}/{nn}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="0.25",
            )

    axB.axhline(100, color="0.75", linestyle=":", linewidth=0.9, zorder=1)
    axB.set_xticks(x0)
    axB.set_xticklabels([str(m) for m in topm])
    axB.set_ylim(0, 116)
    axB.set_yticks([0, 25, 50, 75, 100])
    axB.set_xlabel(r"top-$m$ candidate pairs")
    axB.set_ylabel("positive windows (%)")
    axB.set_title("(b) positive-window rate", loc="left")
    axB.grid(axis="y", color="0.90", linewidth=0.7)
    axB.set_axisbelow(True)
    for spine in ("top", "right"):
        axB.spines[spine].set_visible(False)

    # ------------------------------------------------------------------
    # (c) Median BIC heatmap, centred at zero
    # ------------------------------------------------------------------
    M = np.array(
        [
            [
                df[(df["dataset"] == ds) & (df["top_m"] == m)]
                ["median_bicdiff_candidate_count"]
                .iloc[0]
                for m in topm
            ]
            for ds in DATASET_ORDER
        ],
        dtype=float,
    )

    vmax = max(float(np.abs(M).max()), 1e-9)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = axC.imshow(M, cmap="RdBu", norm=norm, aspect="auto")
    axC.set_xticks(range(len(topm)))
    axC.set_xticklabels([str(m) for m in topm])
    axC.set_yticks(range(len(DATASET_ORDER)))
    axC.set_yticklabels([DATASET_SHORT[d] for d in DATASET_ORDER])
    axC.set_xlabel(r"top-$m$ candidate pairs")
    axC.set_title(r"(c) median $\Delta$BIC", loc="left")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            value = M[i, j]
            text_color = "white" if abs(value) > 0.55 * vmax else "0.15"
            axC.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    for spine in axC.spines.values():
        spine.set_visible(False)
    axC.tick_params(length=0)

    cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
    cb.set_label(r"median $\Delta$BIC", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    # ------------------------------------------------------------------
    # Shared legend and title
    # ------------------------------------------------------------------
    handles = [
        Line2D(
            [0],
            [0],
            color=DATASET_COLORS[ds],
            marker=MARKERS[ds],
            markersize=6,
            markerfacecolor=DATASET_COLORS[ds],
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.0,
            label=DATASET_SHORT[ds],
        )
        for ds in DATASET_ORDER
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.03),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=2.2,
    )


    out_png = fig_dir / "real04_summary_panel.png"
    out_pdf = fig_dir / "real04_summary_panel.pdf"

    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Read:  {csv_path}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
