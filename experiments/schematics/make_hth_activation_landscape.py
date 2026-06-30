"""
Side-by-side HTH figure driven by the actual experiment CSVs.

    (a) LEFT  -- primary 2D view: computed target intensity (lambda_pairwise vs
        lambda_with_hyperedge) over the spike raster.
    (b) RIGHT -- supplementary 3D surface: computed lambda_with_hyperedge over
        (time since anchor) x (trigger gap), with the Delta gate.

Reads (same directory by default):
    hth_activation_meta.csv, hth_activation_events.csv,
    hth_activation_left_curve.csv, hth_activation_surface.csv

Outputs: figures/hth_combo_data.png / .pdf
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "axes.linewidth": 0.8})

C_TRIG, C_HYPER, C_TARGET, C_BASE, C_BAND = \
    "#2b6cb0", "#c0392b", "#1f8a4c", "0.45", "#f0a868"


def draw_2d(ax_i, ax_r, lc, ev, P):
    mu, t_star, delta = P["mu"], P["t_star"], P["delta"]
    d_lo, d_hi = t_star - delta, t_star          # Delta window ending at the anchor
    t_max = float(lc.t.max())

    ax_i.fill_between(lc.t, lc.lambda_pairwise, lc.lambda_with_hyperedge,
                      where=(lc.lambda_gain > 1e-9), color=C_HYPER, alpha=0.13, lw=0)
    ax_i.plot(lc.t, lc.lambda_pairwise, color=C_TRIG, lw=1.6, ls=(0, (4, 2)),
              label="pairwise-only")
    ax_i.plot(lc.t, lc.lambda_with_hyperedge, color=C_HYPER, lw=2.4,
              label="with hyperedge")
    ax_i.axhline(mu, color=C_BASE, lw=1.0, ls=":", zorder=1)

    ax_i.text(0.10, mu - 0.13, r"baseline $\mu$", color=C_BASE, fontsize=9, va="top")
    # leader to the decaying tail, label in empty upper-right
    s_lab = t_star + 0.95
    y_lab = float(np.interp(s_lab, lc.t, lc.lambda_with_hyperedge))
    ax_i.annotate(r"$\alpha_e\,\phi(t-t^{*})$", color=C_HYPER, fontsize=12,
                  xy=(s_lab, y_lab), xytext=(t_star + 1.9, mu + 0.78),
                  arrowprops=dict(arrowstyle="-", color=C_HYPER, lw=0.9))

    ymax = float(lc.lambda_with_hyperedge.max())
    ax_i.set_ylabel(r"target intensity $\lambda(t)$")
    ax_i.set_ylim(0, ymax + 0.30)
    ax_i.legend(loc="upper right", frameon=False, fontsize=9, handlelength=2.4)
    for s in ("top", "right"):
        ax_i.spines[s].set_visible(False)

    lane_label = {0: "neuron 1", 1: "neuron 2", 2: "target"}
    for y in (0, 1, 2):
        ax_r.hlines(y, 0, t_max, color="0.85", lw=1.0, zorder=1)
    for _, r in ev.iterrows():
        c = C_TARGET if r["kind"] == "target" else C_TRIG
        ax_r.vlines(r["time"], r["y"] - 0.28, r["y"] + 0.28, color=c, lw=2.6, zorder=5)
        ax_r.scatter([r["time"]], [r["y"] + 0.28], s=22, color=c, zorder=6)
    tgt = ev[ev.kind == "target"].iloc[0]
    ax_r.annotate("target event", color=C_TARGET, fontsize=9,
                  xy=(tgt["time"], 2.3), xytext=(tgt["time"] + 0.55, 2.55),
                  arrowprops=dict(arrowstyle="->", color=C_TARGET, lw=0.9))

    ax_r.set_yticks([0, 1, 2]); ax_r.set_yticklabels([lane_label[i] for i in (0, 1, 2)], fontsize=9)
    ax_r.set_ylim(-0.7, 3.0); ax_r.set_xlim(0, t_max)
    ax_r.set_xlabel(r"time $t$")
    for s in ("top", "right", "left"):
        ax_r.spines[s].set_visible(False)
    ax_r.tick_params(axis="y", length=0)

    for ax in (ax_i, ax_r):
        ax.axvspan(d_lo, d_hi, color=C_BAND, alpha=0.22, lw=0, zorder=0)
        ax.axvline(t_star, color=C_HYPER, lw=1.2, ls="--", zorder=2)
    # labels off the dashed anchor line (anchor sits at the band's right edge)
    ax_i.text(t_star + 0.1, ymax + 0.25, r"anchor $t^{*}$",
              color=C_HYPER, fontsize=9.5, ha="left", va="top")
    ax_r.text((d_lo + d_hi) / 2, -0.55, r"within $\Delta$",
              color="#b9742a", fontsize=9, ha="center", va="center")


def draw_surface(fig, ax, sf, P):
    delta = P["delta"]
    piv = sf.pivot(index="g_gap", columns="s_rel", values="lambda_gain")
    X, Y = np.meshgrid(piv.columns.values, piv.index.values)
    Z = piv.values
    cmap = plt.cm.viridis
    zlo, zhi = float(np.nanmin(Z)), float(np.nanmax(Z))
    floor = zlo - 0.06 * (zhi - zlo)

    # gain surface: a clean 'waterfall' -- rises at the anchor, decays in time,
    # cut off sharply at the Delta gate. Full-res surface + faint wireframe.
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, vmin=zlo, vmax=zhi,
                           rcount=120, ccount=160, linewidth=0,
                           edgecolor="none", alpha=0.97, antialiased=True)
    ax.plot_wireframe(X, Y, Z, rstride=12, cstride=16,
                      color="0.25", linewidth=0.25, alpha=0.5)
    ax.contourf(X, Y, Z, zdir="z", offset=floor, cmap="Greys", alpha=0.40, levels=10)
    # Delta gate on the floor
    ax.plot([X.min(), X.max()], [delta, delta], [floor, floor],
            color=C_HYPER, ls="--", lw=1.6, zorder=10)
    ax.text(X.min() + 0.15, delta + 0.12, floor, r"$\Delta$", color=C_HYPER, fontsize=12)

    ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(floor, zhi)
    ax.xaxis.set_major_locator(FixedLocator([-1, 1, 3, 5]))
    ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1.0, 1.5, 2.0]))
    ax.zaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6]))
    ax.tick_params(labelsize=8, pad=1)
    ax.set_xlabel(r"$t-t^{*}$", labelpad=4, fontsize=10)
    ax.set_ylabel(r"trigger gap", labelpad=4, fontsize=10)
    ax.set_zlabel(r"hyperedge gain $\alpha_e\phi$", labelpad=4, fontsize=10)
    ax.view_init(elev=26, azim=38)
    ax.grid(False)
    try:
        ax.set_box_aspect((6, 4, 3), zoom=1.02)
    except Exception:
        pass
    cb = fig.colorbar(surf, ax=ax, orientation="horizontal",
                      fraction=0.035, pad=0.02, shrink=0.62, aspect=28)
    cb.set_label(r"hyperedge gain $\Delta\lambda$", fontsize=9)
    cb.ax.tick_params(labelsize=8)


def _resolve_data_dir(cli_dir):
    """Find the folder containing the HTH CSVs. Tries --data-dir first, then a
    set of common locations relative to the script and the working directory."""
    marker = "hth_activation_meta.csv"
    here = Path(__file__).resolve().parent
    roots, subs = [], [".", "data", "csvs", "figures", "experiments"]
    if cli_dir:
        roots.append(Path(cli_dir))
    roots += [Path.cwd(), here, here.parent]
    seen, candidates = set(), []
    for r in roots:
        for s in subs:
            c = (r / s)
            if c not in seen:
                seen.add(c); candidates.append(c)
    for c in candidates:
        if (c / marker).exists():
            return c
    listing = "\n".join(f"  - {c.resolve()}" for c in candidates)
    raise SystemExit(
        f"\nCould not find '{marker}' (and the other HTH CSVs).\n"
        "Put the four CSV files in one folder, then either run the script from\n"
        "that folder or pass  --data-dir <path-to-folder>.\n\nLooked in:\n" + listing + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None,
                   help="folder containing the 4 hth_activation_*.csv files")
    p.add_argument("--fig-dir", default=None,
                   help="output folder for the figure (default: ./figures)")
    a = p.parse_args()
    dd = _resolve_data_dir(a.data_dir)
    print(f"Reading CSVs from: {dd.resolve()}")
    fig_dir = Path(a.fig_dir) if a.fig_dir else Path.cwd() / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(dd / "hth_activation_meta.csv")
    P = dict(zip(m.key, m.value.astype(float)))
    ev = pd.read_csv(dd / "hth_activation_events.csv")
    lc = pd.read_csv(dd / "hth_activation_left_curve.csv")
    sf = pd.read_csv(dd / "hth_activation_surface.csv")

    fig = plt.figure(figsize=(12.2, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.0], height_ratios=[2.0, 1.15],
                          wspace=0.14, hspace=0.10,
                          left=0.06, right=0.98, top=0.88, bottom=0.10)
    ax_i = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[1, 0], sharex=ax_i)
    plt.setp(ax_i.get_xticklabels(), visible=False)
    ax3d = fig.add_subplot(gs[:, 1], projection="3d", computed_zorder=False)

    draw_2d(ax_i, ax_r, lc, ev, P)
    draw_surface(fig, ax3d, sf, P)

    fig.suptitle("Hyperedge-triggered Hawkes activation", fontsize=13, y=0.965)
    fig.text(0.065, 0.90, "(a)", fontsize=12, fontweight="bold")
    fig.text(0.60, 0.90, "(b)", fontsize=12, fontweight="bold")

    fig.savefig(fig_dir / "hth_combo_data.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "hth_combo_data.pdf", bbox_inches="tight")
    plt.close(fig); print("Saved hth_combo_data.{png,pdf}")


if __name__ == "__main__":
    main()