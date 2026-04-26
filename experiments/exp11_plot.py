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

TRUE_ALPHA = 0.4


# =============================================================================
# Two-panel figure
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                         gridspec_kw={"top": 0.85, "bottom": 0.13,
                                      "wspace": 0.32})

# --- Panel 1: recovered alpha_hyper vs beta ---
ax = axes[0]
ax.errorbar(betas, means, yerr=stds, fmt="o-",
            color="steelblue", linewidth=2.5, markersize=10,
            capsize=6, capthick=2, elinewidth=1.5,
            label=r"inferred $\alpha_{(0,1)}$ (mean $\pm$ std)")
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
       edgecolor="black", alpha=0.85, width=0.6)
ax.axhline(0, color="black", linewidth=1)
ax.axhline(5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
ax.axhline(-5, color="gray", linestyle=":", linewidth=1, alpha=0.5)

ax.set_xticks(range(len(betas)))
ax.set_xticklabels([f"$\\beta$={b}" for b in betas], fontsize=10)
ax.set_ylabel("relative bias (%)", fontsize=11)
ax.set_title("Bias is non-monotonic in kernel decay rate",
             fontsize=12, pad=10)
ax.set_ylim(-25, 20)
ax.grid(True, alpha=0.3, axis="y")

# Value labels only — no box annotations inside the plot area
for i, v in enumerate(rel_biases):
    ax.annotate(f"{v:+.1f}%",
                xy=(i, v),
                xytext=(0, 8 if v > 0 else -15),
                textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold")

# Color legend below the chart, outside the plot area
import matplotlib.patches as mpatches
low_patch  = mpatches.Patch(color="#27ae60", label="|bias| < 5%")
med_patch  = mpatches.Patch(color="#e67e22", label="5% < |bias| < 15%")
high_patch = mpatches.Patch(color="#c0392b", label="|bias| > 15%")
ax.legend(handles=[low_patch, med_patch, high_patch],
          loc="upper center", bbox_to_anchor=(0.5, -0.12),
          ncol=3, fontsize=9, framealpha=0.95)

fig.suptitle(
    "Bias ablation: does kernel timescale explain the $-$22% bias?\n"
    "Simple overlap hypothesis not supported; bias minimised at intermediate $\\beta$",
    fontsize=12, fontweight="bold", linespacing=1.4)
plt.savefig("experiments/exp11_bias_ablation.png", dpi=150,
            bbox_inches="tight")
print("Saved: experiments/exp11_bias_ablation.png")
plt.show()