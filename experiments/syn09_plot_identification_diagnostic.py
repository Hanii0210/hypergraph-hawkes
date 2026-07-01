"""
Figure 4(b) -- identification diagnostic (true hyperedge, no pairwise excitation).

Three neutral panels across true hyperedge strength:
    (a) candidate nomination -- max member-pair footprint vs the nomination
        threshold, with the nomination rate annotated;
    (b) recovery bias once nominated -- small relative bias, large sampling
        spread (a +/-5% reference band is shown);
    (c) held-out detectability -- Delta-loglik gain vs the practical detectability
        threshold.
Interpretation is left to the caption.
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
with open(HERE / "results/synthetic/syn09_identification_diagnostic.pkl", "rb") as f:
    d = pickle.load(f)
rows = d["rows"]
alphas    = np.array([r["alpha"] for r in rows])
footprint = np.array([r["footprint_mean"] for r in rows])
foot_std  = np.array([r["footprint_std"] for r in rows])
nom_pct   = np.array([r["nominated_pct"] for r in rows])
bias_pct  = np.array([r["bias_pct"] for r in rows])
bias_sem  = np.array([r.get("bias_sem", 0.0) for r in rows])
dL        = np.array([r["dL"] for r in rows])
dL_sem    = np.array([r.get("dL_sem", 0.0) for r in rows])
thr       = d["threshold"]
x = np.arange(len(alphas))
xl = [f"{a:g}" for a in alphas]

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(10.6, 3.5),
                                    gridspec_kw={"wspace": 0.32})

# ---------- (a) nomination ----------
axA.bar(x, footprint, yerr=foot_std, width=0.6, color=PALETTE["hth"],
        edgecolor="white", linewidth=0.6, capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "0.35"}, zorder=3)
axA.axhline(thr, color=PALETTE["warn"], ls="--", lw=1.6, zorder=4)
for i in range(len(x)):
    axA.annotate(f"{nom_pct[i]:.0f}%", xy=(x[i], footprint[i] + foot_std[i]),
                 xytext=(0, 6), textcoords="offset points", ha="center",
                 fontsize=9, fontweight="bold", color=PALETTE["accent"])
axA.set_xticks(x); axA.set_xticklabels(xl)
axA.set_xlabel(r"true hyperedge strength $\alpha$")
axA.set_ylabel("max member-pair footprint")
axA.set_ylim(0, float(np.max(footprint + foot_std)) * 1.75)
axA.set_title("(a) candidate nomination", loc="left")
from matplotlib.lines import Line2D as _L2D
axA.legend(handles=[
    _L2D([0],[0], marker="s", ls="", mfc=PALETTE["accent"], mec="none", ms=7,
         label="nomination rate (%)"),
    _L2D([0],[0], color=PALETTE["warn"], ls="--", lw=1.6, label=f"threshold {thr}")],
    loc="upper right", handlelength=1.6, fontsize=8, labelspacing=0.3)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)
axA.grid(axis="y", color="0.92", lw=0.6); axA.set_axisbelow(True)

# ---------- (b) recovery bias ----------
axB.axhspan(-5, 5, color="0.90", zorder=0)
axB.bar(x, bias_pct, yerr=bias_sem, width=0.6, color=PALETTE["hth"],
        edgecolor="white", linewidth=0.6, capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "0.35"}, zorder=3)
axB.axhline(0, color=PALETTE["warn"], lw=1.4, zorder=4)
for i in range(len(x)):
    axB.annotate(f"{bias_pct[i]:+.0f}%", xy=(x[i], bias_pct[i]), xytext=(0, -12 if bias_pct[i] < 0 else 6),
                 textcoords="offset points", ha="center", fontsize=9, color="0.2")
axB.set_xticks(x); axB.set_xticklabels(xl)
axB.set_xlabel(r"true hyperedge strength $\alpha$")
axB.set_ylabel("recovery bias (%)")
lo = float(np.min(bias_pct - bias_sem)); hi = float(np.max(bias_pct + bias_sem))
axB.set_ylim(lo - 3, hi + 3)
axB.set_title("(b) recovery bias once nominated", loc="left")
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
axB.grid(axis="y", color="0.92", lw=0.6); axB.set_axisbelow(True)

# ---------- (c) detectability ----------
cc = [PALETTE["accent"] if v > 3 else PALETTE["amber"] for v in dL]
axC.bar(x, dL, yerr=dL_sem, width=0.6, color=cc, edgecolor="white",
        linewidth=0.6, capsize=3, error_kw={"elinewidth": 1.2, "ecolor": "0.35"},
        zorder=3)
axC.axhline(0, color="0.4", lw=1.0, zorder=1)
axC.axhline(3, color="0.45", ls=":", lw=1.5, zorder=4,
            label=r"detectable ($\Delta\ell\approx3$)")
for i in range(len(x)):
    axC.annotate(f"{dL[i]:+.1f}", xy=(x[i], dL[i] + dL_sem[i]), xytext=(0, 5),
                 textcoords="offset points", ha="center", fontsize=9, fontweight="bold",
                 color="0.2")
axC.set_xticks(x); axC.set_xticklabels(xl)
axC.set_xlabel(r"true hyperedge strength $\alpha$")
axC.set_ylabel(r"$\Delta\ell$ (HTH $-$ pairwise)")
axC.set_ylim(0, float(np.max(dL + dL_sem)) * 1.18)
axC.set_title("(c) held-out detectability", loc="left")
axC.legend(loc="upper left", handlelength=1.6)
for s in ("top", "right"):
    axC.spines[s].set_visible(False)
axC.grid(axis="y", color="0.92", lw=0.6); axC.set_axisbelow(True)

fig.savefig("figures/synthetic/syn09_identification_diagnostic.png")
fig.savefig("figures/synthetic/syn09_identification_diagnostic.pdf")
plt.close(fig)

print("Saved figures/synthetic/syn09_identification_diagnostic.{png,pdf}")