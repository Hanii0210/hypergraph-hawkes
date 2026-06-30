import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("experiments/exp11_bias_ablation.pkl", "rb") as f:
    results = pickle.load(f)

betas      = np.array([r["beta"]     for r in results])
means      = np.array([r["mean"]     for r in results])
stds       = np.array([r["std"]      for r in results])
rel_biases = np.array([r["rel_bias"] for r in results])
sems       = np.array([r.get("sem", 0.0) for r in results])

TRUE_ALPHA = 0.4


# =============================================================================
# Two-panel figure
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                         gridspec_kw={"top": 0.85, "bottom": 0.13,
                                      "wspace": 0.32})

# --- Panel 1: recovered alpha_hyper vs beta ---
ax = axes[0]
ax.errorbar(betas, means, yerr=sems, fmt="o-",
            color="steelblue", linewidth=2.5, markersize=10,
            capsize=6, capthick=2, elinewidth=1.5,
            label=r"inferred $\alpha_{(0,1)}$ (mean $\pm$ SEM)")
ax.axhline(TRUE_ALPHA, color="#c0392b", linestyle="--", linewidth=2,
           label=f"true $\\alpha = {TRUE_ALPHA}$")

for x, y in zip(betas, means):
    ax.annotate(f"{y:.3f}", xy=(x, y),
                xytext=(8, 10), textcoords="offset points", fontsize=9)

ax.set_xscale("log")
ax.set_xlabel(r"kernel decay rate $\beta$  (log scale)", fontsize=11)
ax.set_ylabel(r"inferred $\alpha_{(0,1)}$", fontsize=11)
ax.set_title("Recovered hyperedge weight vs kernel timescale",
             fontsize=12, pad=10)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

# --- Panel 2: relative bias vs beta (clean, no overlapping annotations) ---
ax = axes[1]
colors = ["#27ae60" if abs(b) < 5 else "#e67e22" if abs(b) < 15
          else "#c0392b" for b in rel_biases]

ax.bar(range(len(betas)), rel_biases, color=colors,
       edgecolor="black", alpha=0.85, width=0.6,
       yerr=100.0 * sems / TRUE_ALPHA, capsize=5,
       error_kw={"elinewidth": 1.3, "ecolor": "#333333"})
ax.axhline(0, color="black", linewidth=1)
ax.axhline(5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
ax.axhline(-5, color="gray", linestyle=":", linewidth=1, alpha=0.5)

ax.set_xticks(range(len(betas)))
ax.set_xticklabels([f"$\\beta$={b}" for b in betas], fontsize=10)
ax.set_ylabel("relative bias (%)", fontsize=11)
ax.set_title("Estimator variance grows with kernel decay rate (beta)",
             fontsize=12, pad=10)
_rel_sem = 100.0 * sems / TRUE_ALPHA
_lo = float(np.min(rel_biases - _rel_sem))
_hi = float(np.max(rel_biases + _rel_sem))
_pad = 0.12 * (_hi - _lo)
ax.set_ylim(_lo - _pad, _hi + _pad)
ax.grid(True, alpha=0.3, axis="y")

# Value labels only — no box annotations inside the plot area
for i, v in enumerate(rel_biases):
    cap = _rel_sem[i]
    if v >= 0:
        ax.annotate(f"{v:+.1f}%", xy=(i, v + cap), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    else:
        ax.annotate(f"{v:+.1f}%", xy=(i, v - cap), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=10, fontweight="bold")

# Color legend below the chart, outside the plot area
import matplotlib.patches as mpatches
low_patch  = mpatches.Patch(color="#27ae60", label="|bias| < 5%")
med_patch  = mpatches.Patch(color="#e67e22", label="5% < |bias| < 15%")
high_patch = mpatches.Patch(color="#c0392b", label="|bias| > 15%")
ax.legend(handles=[low_patch, med_patch, high_patch],
          loc="upper center", bbox_to_anchor=(0.5, -0.12),
          ncol=3, fontsize=9, framealpha=0.95)

fig.suptitle(
    "Identifiability vs kernel timescale (beta): variance, not bias",
    fontsize=12, fontweight="bold", linespacing=1.4)
plt.savefig("experiments/exp11_bias_ablation.png", dpi=150,
            bbox_inches="tight")
print("Saved: experiments/exp11_bias_ablation.png")
plt.show()