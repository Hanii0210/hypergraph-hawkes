"""
Plot formal real-data HTH summary figures, v2.

Changes from the first plotting script:
    1. Dataset colors are fixed and consistent across points, lines, bars.
    2. BIC stability plot uses deterministic within-cluster jitter without
       changing dataset color.
    3. Exposure diagnostic has two versions:
         - full scale
         - clipped y-axis for main-text readability
    4. Output filenames use *_v2 so old figures are not overwritten.

Inputs expected:
    experiments/results/realdata/realdata_ret1_20080516_R1_rec0.csv
    experiments/results/realdata/realdata_pvc3_area17.csv
    experiments/results/realdata/realdata_pvc11_monkey2.csv

Outputs:
    experiments/results/realdata/realdata_combined.csv
    experiments/results/realdata/realdata_summary_by_dataset_topm.csv

    figures/realdata/realdata_bic_stability.png
    figures/realdata/realdata_bic_stability.pdf
    figures/realdata/realdata_positive_window_rate.png
    figures/realdata/realdata_positive_window_rate.pdf
    figures/realdata/realdata_exposure_diagnostic_full.png
    figures/realdata/realdata_exposure_diagnostic_full.pdf
    figures/realdata/realdata_exposure_diagnostic_clipped.png
    figures/realdata/realdata_exposure_diagnostic_clipped.pdf
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUTS = [
    "experiments/results/realdata/realdata_ret1_20080516_R1_rec0.csv",
    "experiments/results/realdata/realdata_pvc3_area17.csv",
    "experiments/results/realdata/realdata_pvc11_monkey2.csv",
]

DATASET_ORDER = ["ret-1", "PVC-3 area17", "PVC-11 monkey2"]

# Explicit project palette. Keep consistent across all real-data figures.
DATASET_COLORS = {
    "ret-1": "#1f77b4",
    "PVC-3 area17": "#ff7f0e",
    "PVC-11 monkey2": "#2ca02c",
}

DATASET_MARKERS = {
    "ret-1": "o",
    "PVC-3 area17": "s",
    "PVC-11 monkey2": "^",
}


def canonical_dataset(name: str) -> str:
    name = str(name)
    if name.startswith("ret-1"):
        return "ret-1"
    if name == "PVC-3 area17":
        return "PVC-3 area17"
    if name == "PVC-11 monkey2":
        return "PVC-11 monkey2"
    return name


def to_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def read_rows(path: Path) -> List[Dict]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row = dict(row)
            row["source_csv"] = str(path)
            row["dataset_raw"] = row.get("dataset", "")
            row["dataset"] = canonical_dataset(row.get("dataset", ""))
            row["top_m"] = to_int(row.get("top_m"))
            row["window_start"] = to_float(row.get("window_start"))
            row["window_end"] = to_float(row.get("window_end"))
            row["bicdiff_candidate_count"] = to_float(row.get("bicdiff_candidate_count"))
            row["bicdiff_active_count"] = to_float(row.get("bicdiff_active_count"))
            row["delta_L_heldout"] = to_float(row.get("delta_L_heldout"))
            row["main_alpha"] = to_float(row.get("main_alpha"))
            row["main_C_e"] = to_float(row.get("main_C_e"))
            row["main_n_completions"] = to_int(row.get("main_n_completions"))
            row["main_min_member_spikes"] = to_int(row.get("main_min_member_spikes"))
            rows.append(row)
        return rows


def write_rows(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fields = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sort_key_dataset(ds: str) -> int:
    return DATASET_ORDER.index(ds) if ds in DATASET_ORDER else 999


def group_summary(rows: Sequence[Dict]) -> List[Dict]:
    out = []
    datasets = sorted({r["dataset"] for r in rows}, key=sort_key_dataset)
    top_ms = sorted({int(r["top_m"]) for r in rows})

    for ds in datasets:
        for top_m in top_ms:
            vals = [
                float(r["bicdiff_candidate_count"])
                for r in rows
                if r["dataset"] == ds
                and int(r["top_m"]) == top_m
                and math.isfinite(float(r["bicdiff_candidate_count"]))
            ]
            if not vals:
                continue

            arr = np.asarray(vals, dtype=float)
            out.append({
                "dataset": ds,
                "top_m": int(top_m),
                "n_windows": int(arr.size),
                "positive_windows": int(np.sum(arr > 0)),
                "positive_rate": float(np.mean(arr > 0)),
                "mean_bicdiff_candidate_count": float(np.mean(arr)),
                "median_bicdiff_candidate_count": float(np.median(arr)),
                "min_bicdiff_candidate_count": float(np.min(arr)),
                "max_bicdiff_candidate_count": float(np.max(arr)),
            })
    return out


def plot_bic_stability(rows: Sequence[Dict], fig_dir: Path) -> None:
    datasets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in rows)]
    top_ms = sorted({int(r["top_m"]) for r in rows})

    # Offset datasets around each integer top_m.
    ds_offsets = {
        ds: offset
        for ds, offset in zip(datasets, np.linspace(-0.20, 0.20, max(len(datasets), 1)))
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for ds in datasets:
        color = DATASET_COLORS.get(ds, "0.3")
        marker = DATASET_MARKERS.get(ds, "o")
        mean_x = []
        mean_y = []

        for top_m in top_ms:
            sub = [
                r for r in rows
                if r["dataset"] == ds and int(r["top_m"]) == int(top_m)
            ]
            vals = [float(r["bicdiff_candidate_count"]) for r in sub]
            if not vals:
                continue

            center = float(top_m) + ds_offsets[ds]
            # Deterministic window-level jitter so repeated runs look identical.
            jitter = np.linspace(-0.035, 0.035, len(vals)) if len(vals) > 1 else np.array([0.0])
            xs = center + jitter

            ax.scatter(
                xs,
                vals,
                color=color,
                marker=marker,
                alpha=0.55,
                s=34,
                linewidths=0.8,
                edgecolors=color,
            )

            mean_x.append(center)
            mean_y.append(float(np.mean(vals)))

        ax.plot(
            mean_x,
            mean_y,
            color=color,
            marker=marker,
            linewidth=2.4,
            markersize=6,
            label=ds,
        )

    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="0.25")
    ax.set_xlabel("Number of selected candidate hyperedges (top_m)")
    ax.set_ylabel("Held-out candidate-count BIC difference")
    ax.set_title("Real-data HTH support across windows")
    ax.set_xticks(top_ms)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "realdata_bic_stability.png", dpi=300)
    fig.savefig(fig_dir / "realdata_bic_stability.pdf")
    plt.close(fig)


def plot_positive_window_rate(summary: Sequence[Dict], fig_dir: Path) -> None:
    datasets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in summary)]
    top_ms = sorted({int(r["top_m"]) for r in summary})

    width = 0.22
    offsets = {
        ds: offset
        for ds, offset in zip(datasets, np.linspace(-width, width, max(len(datasets), 1)))
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for ds in datasets:
        color = DATASET_COLORS.get(ds, "0.3")
        xs = []
        heights = []
        labels = []

        for top_m in top_ms:
            match = [
                r for r in summary
                if r["dataset"] == ds and int(r["top_m"]) == int(top_m)
            ]
            if not match:
                continue

            r = match[0]
            xs.append(float(top_m) + offsets[ds])
            heights.append(float(r["positive_rate"]))
            labels.append(f"{r['positive_windows']}/{r['n_windows']}")

        bars = ax.bar(xs, heights, width=width, color=color, label=ds, alpha=0.90)

        for bar, label in zip(bars, labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Number of selected candidate hyperedges (top_m)")
    ax.set_ylabel("Fraction of windows with BICdiff > 0")
    ax.set_title("Positive held-out BIC support rate")
    ax.set_xticks(top_ms)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "realdata_positive_window_rate.png", dpi=300)
    fig.savefig(fig_dir / "realdata_positive_window_rate.pdf")
    plt.close(fig)


def plot_exposure_diagnostic(rows: Sequence[Dict], fig_dir: Path, clipped: bool) -> None:
    datasets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in rows)]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    for ds in datasets:
        color = DATASET_COLORS.get(ds, "0.3")
        marker = DATASET_MARKERS.get(ds, "o")
        sub = [
            r for r in rows
            if r["dataset"] == ds
            and math.isfinite(float(r["main_C_e"]))
            and math.isfinite(float(r["main_alpha"]))
            and float(r["main_C_e"]) > 0
            and float(r["main_alpha"]) >= 0
        ]
        if not sub:
            continue

        x = np.asarray([float(r["main_C_e"]) for r in sub], dtype=float)
        y = np.asarray([float(r["main_alpha"]) for r in sub], dtype=float)
        ncomp = np.asarray([float(r["main_n_completions"]) for r in sub], dtype=float)
        sizes = 28 + 2.0 * np.sqrt(np.maximum(ncomp, 1.0))

        ax.scatter(
            x,
            y,
            s=sizes,
            color=color,
            marker=marker,
            alpha=0.58,
            edgecolors=color,
            linewidths=0.8,
            label=ds,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Hyperedge exposure compensator $C_e$")
    ax.set_ylabel("Estimated main hyperedge coefficient")
    if clipped:
        ax.set_ylim(-0.03, 1.25)
        ax.set_title("Exposure diagnostic for selected HTH candidates (main scale)")
    else:
        ax.set_title("Exposure diagnostic for selected HTH candidates (full scale)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()

    suffix = "clipped" if clipped else "full"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"realdata_exposure_diagnostic_v2_{suffix}.png", dpi=300)
    fig.savefig(fig_dir / f"realdata_exposure_diagnostic_v2_{suffix}.pdf")
    plt.close(fig)


def print_summary(summary: Sequence[Dict]) -> None:
    print("\nSummary by dataset and top_m")
    print("-" * 96)
    print(
        f"{'dataset':<18} {'top_m':>5} {'positive':>10} {'mean BIC':>12} "
        f"{'median BIC':>12} {'min':>10} {'max':>10}"
    )
    for r in summary:
        print(
            f"{r['dataset']:<18} {int(r['top_m']):>5} "
            f"{int(r['positive_windows'])}/{int(r['n_windows']):<7} "
            f"{float(r['mean_bicdiff_candidate_count']):>12.3f} "
            f"{float(r['median_bicdiff_candidate_count']):>12.3f} "
            f"{float(r['min_bicdiff_candidate_count']):>10.3f} "
            f"{float(r['max_bicdiff_candidate_count']):>10.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--fig-dir", default="figures/realdata")
    parser.add_argument("--results-dir", default="experiments/results/realdata")
    args = parser.parse_args()

    rows = []
    missing = []

    for p in args.inputs:
        path = Path(p)
        if not path.exists():
            missing.append(str(path))
            continue
        rows.extend(read_rows(path))

    if missing:
        print("Missing input files:")
        for m in missing:
            print(f"  {m}")

    if not rows:
        raise FileNotFoundError(
            "No input rows loaded. Run realdata_pvc3.py, realdata_pvc11.py, "
            "and realdata_ret1_fixed.py first."
        )

    summary = group_summary(rows)

    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    write_rows(results_dir / "realdata_combined.csv", rows)
    write_rows(results_dir / "realdata_summary_by_dataset_topm.csv", summary)

    plot_bic_stability(rows, fig_dir)
    plot_positive_window_rate(summary, fig_dir)
    plot_exposure_diagnostic(rows, fig_dir, clipped=False)
    plot_exposure_diagnostic(rows, fig_dir, clipped=True)

    print_summary(summary)
    print("\nSaved:")
    for p in [
        results_dir / "realdata_combined.csv",
        results_dir / "realdata_summary_by_dataset_topm.csv",
        fig_dir / "realdata_bic_stability.png",
        fig_dir / "realdata_bic_stability.pdf",
        fig_dir / "realdata_positive_window_rate.png",
        fig_dir / "realdata_positive_window_rate.pdf",
        fig_dir / "realdata_exposure_diagnostic_full.png",
        fig_dir / "realdata_exposure_diagnostic_full.pdf",
        fig_dir / "realdata_exposure_diagnostic_clipped.png",
        fig_dir / "realdata_exposure_diagnostic_clipped.pdf",
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
