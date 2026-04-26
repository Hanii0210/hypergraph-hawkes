import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

with open("experiments/exp5_copula.pkl", "rb") as f:
    data = pickle.load(f)

hth_tau     = data["hth_tau"]
null_tau    = data["null_tau"]
hth_probit  = data["hth_probit"]
null_probit = data["null_probit"]


# =============================================================================
# Two-panel figure
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
positions = [1, 2]
xtick_labels = ["HTH\n(with hyperedge)", "Null\n(pairwise only)"]
colors = ["#c0392b", "#2c3e50"]

# --- Panel 1: tau_U boxplot ---
ax = axes[0]
data_box = [hth_tau, null_tau]

bp = ax.boxplot(
    data_box, positions=positions, widths=0.5, patch_artist=True,
    showmeans=True,
    meanprops={
        "marker": "D", "markersize": 9,
        "markerfacecolor": "#f1c40f",
        "markeredgecolor": "black", "markeredgewidth": 1.5,
    },
    medianprops={"color": "black", "linewidth": 1.5},
    flierprops={"marker": "o", "markerfacecolor": "white",
                "markeredgecolor": "black", "markersize": 5},
)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.65)

# Scatter individual replicates (subtle)
np.random.seed(0)
for i, (d, c) in enumerate(zip(data_box, colors)):
    jitter = np.random.uniform(-0.06, 0.06, len(d))
    ax.scatter(np.full(len(d), positions[i]) + jitter, d,
               color=c, alpha=0.6, edgecolor="black", s=40, zorder=3,
               linewidth=0.8)

ax.set_xticks(positions)
ax.set_xticklabels(xtick_labels, fontsize=10)
ax.set_ylabel(r"upper-tail dependence  $\tau_U(X_0, X_1)$", fontsize=11)
ax.set_title("Tail dependence: HTH vs Null", fontsize=12, pad=10)
ax.grid(True, alpha=0.3, axis="y")

_, p_val = ttest_ind(hth_tau, null_tau, equal_var=False)
ax.annotate(f"Welch t-test\np = {p_val:.4f}",
            xy=(0.5, 0.96), xycoords="axes fraction",
            ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5",
                      fc="#fdf6e3", ec="black", linewidth=1))

# --- Panel 2: probit-tail correlation boxplot ---
ax = axes[1]
data_box2 = [hth_probit, null_probit]

bp = ax.boxplot(
    data_box2, positions=positions, widths=0.5, patch_artist=True,
    showmeans=True,
    meanprops={
        "marker": "D", "markersize": 9,
        "markerfacecolor": "#f1c40f",
        "markeredgecolor": "black", "markeredgewidth": 1.5,
    },
    medianprops={"color": "black", "linewidth": 1.5},
    flierprops={"marker": "o", "markerfacecolor": "white",
                "markeredgecolor": "black", "markersize": 5},
)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.65)

np.random.seed(1)
for i, (d, c) in enumerate(zip(data_box2, colors)):
    jitter = np.random.uniform(-0.06, 0.06, len(d))
    ax.scatter(np.full(len(d), positions[i]) + jitter, d,
               color=c, alpha=0.6, edgecolor="black", s=40, zorder=3,
               linewidth=0.8)

ax.set_xticks(positions)
ax.set_xticklabels(xtick_labels, fontsize=10)
ax.set_ylabel(r"probit-transformed tail correlation  $\rho_{\Phi}$",
              fontsize=11)
ax.set_title("Probit copula tail signal", fontsize=12, pad=10)
ax.grid(True, alpha=0.3, axis="y")

_, p_val2 = ttest_ind(hth_probit, null_probit, equal_var=False)
ax.annotate(f"Welch t-test\np = {p_val2:.4f}",
            xy=(0.5, 0.96), xycoords="axes fraction",
            ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5",
                      fc="#fdf6e3", ec="black", linewidth=1))

# Shared legend on the right showing what the diamond is
import matplotlib.lines as mlines
mean_handle = mlines.Line2D([], [], marker="D", color="none",
                             markerfacecolor="#f1c40f",
                             markeredgecolor="black", markersize=9,
                             label="mean")
median_handle = mlines.Line2D([], [], color="black", linewidth=1.5,
                               label="median")
fig.legend(handles=[mean_handle, median_handle],
           loc="lower center", ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, -0.03), frameon=True)

plt.tight_layout()
plt.savefig("experiments/exp5_copula.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/exp5_copula.png")
plt.show()