"""
Figure 2 -- synthetic recovery across repeated datasets.

Five histograms of the recovered parameters (true value and sample mean marked)
plus a forest-style summary panel showing every parameter's mean relative
deviation from truth with its sampling spread, so the reader sees at a glance
that the means sit within a few percent of truth while the interaction weights
carry the larger finite-sample spread. Interpretation is left to the caption.
"""
import sys, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_style import apply_style, PALETTE

apply_style()

with open(HERE / "results/synthetic/syn01_recovery_robustness.pkl", "rb") as f:
    data = pickle.load(f)
recovered, true_values, N = data["recovered"], data["true_values"], data["N_SEEDS"]

keys  = ["mu[0]", "mu[1]", "mu[2]", "a[2->0]", "alpha_hyper(0,1)"]
disp  = [r"$\mu_0$", r"$\mu_1$", r"$\mu_2$", r"$\alpha_{2\to0}$", r"$\alpha_{(0,1)}$"]
is_hyper = [False, False, False, False, True]

fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.6),
                         gridspec_kw={"wspace": 0.26, "hspace": 0.42})
axes = axes.flatten()

for i, (ax, k, dsp) in enumerate(zip(axes[:5], keys, disp)):
    v = np.asarray(recovered[k], float); t = true_values[k]
    mean = v.mean(); rel = abs(mean - t) / t * 100
    lo, hi = min(v.min(), t), max(v.max(), t)
    pad = (hi - lo) * 0.10 + 1e-6
    bins = np.linspace(lo - pad, hi + pad, 12)
    ax.hist(v, bins=bins, color=PALETTE["hth"], alpha=0.80,
            edgecolor="white", linewidth=0.8, zorder=2)
    ax.axvline(t, color=PALETTE["warn"], lw=2.2, zorder=4)
    ax.axvline(mean, color="0.2", lw=1.6, ls=(0, (4, 3)), zorder=5)
    ax.set_title(f"{dsp}   (rel. err {rel:.1f}%)", loc="left", fontsize=10)
    ax.set_xlabel("inferred value"); 
    if i % 3 == 0:
        ax.set_ylabel("count")
    ax.set_xlim(lo - pad, hi + pad)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="0.92", lw=0.6); ax.set_axisbelow(True)

# shared true/mean key in the first panel
axes[0].legend(handles=[
    Line2D([0], [0], color=PALETTE["warn"], lw=2.2, label="true"),
    Line2D([0], [0], color="0.2", lw=1.6, ls=(0, (4, 3)), label="mean")],
    loc="upper left", fontsize=8, handlelength=1.6, labelspacing=0.3)

# ---------- 6th panel: forest summary (mean relative deviation +/- rel SD) ----------
axF = axes[5]
y = np.arange(len(keys))[::-1]           # top-to-bottom = mu0..alpha_hyper
reldev = np.array([(np.asarray(recovered[k]).mean() - true_values[k]) / true_values[k] * 100
                   for k in keys])
relsd = np.array([np.asarray(recovered[k]).std() / true_values[k] * 100 for k in keys])
colors = [PALETTE["amber"] if h else PALETTE["hth"] for h in is_hyper]

axF.axvspan(-5, 5, color="0.90", zorder=0)          # +/-5% reference band
axF.axvline(0, color=PALETTE["warn"], lw=1.4, zorder=1)
for yi, m, s, c in zip(y, reldev, relsd, colors):
    axF.errorbar(m, yi, xerr=s, fmt="o", color=c, ms=6, mfc=c, mec="white",
                 mew=0.8, capsize=3, elinewidth=1.4, zorder=3)
axF.set_yticks(y); axF.set_yticklabels(disp)
for _tl, _h in zip(axF.get_yticklabels(), is_hyper):
    _tl.set_color(PALETTE["amber"] if _h else PALETTE["hth"])
axF.set_ylim(-0.6, len(keys) - 0.4)
axF.set_xlabel("relative deviation from truth (%)")
axF.set_title(r"recovery summary (mean $\pm$ SD)", loc="left", fontsize=10)
for s in ("top", "right"):
    axF.spines[s].set_visible(False)
axF.grid(axis="x", color="0.92", lw=0.6); axF.set_axisbelow(True)
# colour key carried by the tick-label colours (blue = baseline/pairwise,
# amber = hyperedge); stated in the caption. A compact inline reminder:
axF.annotate("amber = hyperedge", xy=(0.985, 0.04), xycoords="axes fraction",
             ha="right", va="bottom", fontsize=7.2, color=PALETTE["amber"])

fig.savefig("figures/synthetic/syn01_recovery_robustness.png")
fig.savefig("figures/synthetic/syn01_recovery_robustness.pdf")
plt.close(fig)
print("Saved figures/synthetic/syn01_recovery_robustness.{png,pdf}")