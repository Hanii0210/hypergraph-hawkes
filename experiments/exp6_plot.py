import sys
sys.path.insert(0, ".")

import pickle
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Load results saved by exp6_3node_hyperedge.py
# =============================================================================
with open("experiments/exp6_3node_hyperedge.pkl", "rb") as f:
    data = pickle.load(f)

edges     = [tuple(e) for e in data["edges"]]
types     = data["types"]
true_vals = np.array(data["true"], dtype=float)
inf_vals  = np.array(data["inferred"], dtype=float)


# =============================================================================
# Two-panel figure
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

x = np.arange(len(edges))
labels = [f"{e}" for e in edges]
colors = ["#c0392b" if t == "TRUE" else "#7f8c8d" for t in types]

# --- Panel 1: side-by-side bars ---
ax = axes[0]
width = 0.38
b1 = ax.bar(x - width/2, true_vals, width,
            color="lightgray", edgecolor="black", label="true")
b2 = ax.bar(x + width/2, inf_vals,  width,
            color=colors, edgecolor="black", alpha=0.85, label="inferred")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, fontsize=10)
ax.set_ylabel(r"hyperedge weight $\alpha_e$")
ax.set_title("3-node hyperedge recovery: 1 truth among 5 candidates")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, axis="y")

# Annotate values
for xi, v in zip(x - width/2, true_vals):
    if v > 0:
        ax.annotate(f"{v:.3f}", xy=(xi, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8)
for xi, v, t in zip(x + width/2, inf_vals, types):
    ax.annotate(f"{v:.3f}", xy=(xi, v),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8,
                fontweight="bold" if t == "TRUE" else "normal")

# --- Panel 2: log-scale to show decoy suppression ---
ax = axes[1]
inferred_safe = np.maximum(inf_vals, 1e-5)  # avoid log(0)
b3 = ax.bar(x, inferred_safe, 0.6,
            color=colors, edgecolor="black", alpha=0.85)
ax.axhline(0.05, color="darkred", linestyle="--", linewidth=1.5,
           label="suppression threshold (0.05)")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, fontsize=10)
ax.set_ylabel(r"inferred $\alpha_e$  (log scale)")
ax.set_title("Decoy suppression: 4 false candidates pushed below threshold")
ax.legend()
ax.grid(True, alpha=0.3, axis="y", which="both")

# Add legend patches for type
import matplotlib.patches as mpatches
true_patch  = mpatches.Patch(color="#c0392b", label="TRUE hyperedge")
decoy_patch = mpatches.Patch(color="#7f8c8d", label="decoy")
ax.legend(handles=[true_patch, decoy_patch,
                   plt.Line2D([], [], color="darkred", linestyle="--",
                              label="threshold = 0.05")],
          loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("experiments/exp6_3node_hyperedge.png", dpi=150,
            bbox_inches="tight")
print("\nSaved: experiments/exp6_3node_hyperedge.png")
plt.show()