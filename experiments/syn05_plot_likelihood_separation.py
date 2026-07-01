import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("experiments/results/synthetic/syn05_likelihood_separation.pkl", "rb") as f:
    results = pickle.load(f)


# =============================================================================
# Figure: ΔL and BIC difference for both scenarios
# =============================================================================
scenarios = ["A_with_hyperedge", "B_no_hyperedge"]
labels    = ["A: data WITH hyperedge\n(should favour HTH)",
             "B: data WITHOUT hyperedge\n(should favour pairwise)"]

# Larger figure with more top room for the suptitle
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5),
                         gridspec_kw={"top": 0.88, "bottom": 0.13})
x = np.arange(len(scenarios))

# --- Panel 1: log-likelihood gain (Delta L) ---
ax = axes[0]
delta_L_vals = [results[s]["delta_L"] for s in scenarios]
colors_L     = ["#27ae60" if v > 0 else "#7f8c8d" for v in delta_L_vals]

bars = ax.bar(x, delta_L_vals, 0.55,
              color=colors_L, edgecolor="black", alpha=0.85)
ax.axhline(0, color="black", linewidth=1)

# Significance reference: chi^2(df=1) at p=0.05 -> ΔL > 1.92
ax.axhline(1.92, color="darkgreen", linestyle=":", linewidth=1.5,
           label=r"significance threshold ($\Delta L > 1.92$, p<0.05)")
ax.axhline(-1.92, color="darkred", linestyle=":", linewidth=1.5)

# Annotate (all labels positioned outside the threshold lines for clarity)
for i, v in enumerate(delta_L_vals):
    ax.annotate(f"{v:+.2f}",
                xy=(x[i], v),
                xytext=(0, 10 if v > 0 else -18),
                textcoords="offset points",
                ha="center", fontsize=12, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel(r"$\Delta L = L_{\mathrm{HTH}} - L_{\mathrm{pairwise}}$",
              fontsize=11)
ax.set_title("Likelihood gain from including the hyperedge term",
             fontsize=12, pad=10)

ymax = max(abs(min(delta_L_vals)), abs(max(delta_L_vals))) * 1.7
ax.set_ylim(-ymax, ymax)
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.grid(True, alpha=0.3, axis="y")

# --- Panel 2: BIC difference ---
ax = axes[1]
bic_diffs = [results[s]["bic_diff"] for s in scenarios]
colors    = ["#27ae60" if v > 0 else "#7f8c8d" for v in bic_diffs]

bars = ax.bar(x, bic_diffs, 0.55, color=colors,
              alpha=0.85, edgecolor="black")
ax.axhline(0, color="black", linewidth=1)
ax.axhline(6, color="darkgreen", linestyle=":", linewidth=1.5,
           label=r"BIC threshold $\pm 6$ (strong evidence)")
ax.axhline(-6, color="darkred", linestyle=":", linewidth=1.5)

# Place labels well clear of threshold lines
for i, v in enumerate(bic_diffs):
    if v > 0:
        ax.annotate(f"{v:+.2f}",
                    xy=(x[i], v),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", fontsize=12, fontweight="bold")
    else:
        # Place B's label well below -6 line to avoid collision
        ax.annotate(f"{v:+.2f}",
                    xy=(x[i], v),
                    xytext=(0, -22),
                    textcoords="offset points",
                    ha="center", fontsize=12, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("BIC difference (positive favours HTH)", fontsize=11)
ax.set_title("Model selection by BIC", fontsize=12, pad=10)

ymax2 = max(abs(min(bic_diffs)), abs(max(bic_diffs))) * 1.7
ax.set_ylim(-ymax2, ymax2)
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Falsifiability test: does the data demand a hyperedge term?",
             fontsize=14, fontweight="bold")
plt.savefig("figures/synthetic/syn05_likelihood_separation.png", dpi=150,
            bbox_inches="tight")
print("Saved: figures/synthetic/syn05_likelihood_separation.png")
