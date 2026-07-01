import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("experiments/results/synthetic/syn02_regularization_path.pkl", "rb") as f:
    results = pickle.load(f)

lambdas = np.array([r["lambda"] for r in results])
logL    = np.array([r["logL"]   for r in results])
aic     = np.array([r["aic"]    for r in results])
bic     = np.array([r["bic"]    for r in results])

candidate_edges = list(results[0]["alpha_hyper"].keys())
alpha_paths = {
    e: np.array([r["alpha_hyper"][e] for r in results])
    for e in candidate_edges
}

TRUE_EDGE = (0, 1)


# =============================================================================
# Three-panel figure
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# --- Panel 1: regularisation paths ---
ax = axes[0]

# Plot the TRUE edge first, in red, with thicker line
true_path = alpha_paths[TRUE_EDGE]
ax.plot(lambdas, true_path,
        color="#c0392b", linewidth=2.8, marker="o", markersize=7,
        label=f"{TRUE_EDGE}  TRUE", zorder=4)

# Plot decoys with subdued styling
decoy_color = "#7f8c8d"
for i, (e, path) in enumerate(alpha_paths.items()):
    if e == TRUE_EDGE:
        continue
    ax.plot(lambdas, path,
            color=decoy_color, linewidth=1.2, marker=".", markersize=5,
            alpha=0.65, label=f"{e}" if i < 3 else None)

# Truth reference line
ax.axhline(0.4, color="#c0392b", linestyle="--", linewidth=1, alpha=0.5,
           label=r"true $\alpha_{(0,1)} = 0.4$")

ax.set_xscale("log")
ax.set_xlabel(r"L1 penalty $\lambda$", fontsize=11)
ax.set_ylabel(r"inferred $\alpha_e$", fontsize=11)
ax.set_title("Regularisation paths", fontsize=12, pad=10)

# Two-column legend, smaller font, with a dummy entry to label all decoys
import matplotlib.lines as mlines
true_handle  = mlines.Line2D([], [], color="#c0392b", linewidth=2.8,
                              marker="o", label=f"{TRUE_EDGE}  TRUE")
decoy_handle = mlines.Line2D([], [], color=decoy_color, linewidth=1.2,
                              marker=".", alpha=0.65,
                              label="5 decoy candidates")
truth_handle = mlines.Line2D([], [], color="#c0392b", linestyle="--",
                              linewidth=1,
                              label=r"true $\alpha_{(0,1)} = 0.4$")
ax.legend(handles=[true_handle, decoy_handle, truth_handle],
          loc="center right", fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)

# --- Panel 2: number of survivors ---
ax = axes[1]
n_survivors = np.array([len(r["survivors"]) for r in results])
ax.plot(lambdas, n_survivors, color="darkgreen",
        linewidth=2.5, marker="s", markersize=8)
ax.axhline(1, color="#c0392b", linestyle="--", linewidth=1.5, alpha=0.7,
           label=r"true number = 1")
ax.set_xscale("log")
ax.set_xlabel(r"L1 penalty $\lambda$", fontsize=11)
ax.set_ylabel("number of surviving hyperedges", fontsize=11)
ax.set_title("Sparsity vs penalty", fontsize=12, pad=10)
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_yticks(np.arange(0, max(n_survivors) + 2))

# --- Panel 3: AIC and BIC ---
ax = axes[2]
ax.plot(lambdas, aic, color="darkorange", linewidth=2.5,
        marker="o", markersize=7, label="AIC")
ax.plot(lambdas, bic, color="purple", linewidth=2.5,
        marker="s", markersize=7, label="BIC")

best_aic_idx = np.argmin(aic)
best_bic_idx = np.argmin(bic)

# Combined marker since both criteria pick the same lambda
if best_aic_idx == best_bic_idx:
    ax.axvline(lambdas[best_aic_idx], color="black", linestyle="--",
               linewidth=1.5, alpha=0.7,
               label=f"AIC = BIC optimum at $\\lambda^*$ = {lambdas[best_aic_idx]:.2f}")
else:
    ax.axvline(lambdas[best_aic_idx], color="darkorange", linestyle="--",
               linewidth=1.5, alpha=0.7,
               label=f"AIC* = {lambdas[best_aic_idx]:.2f}")
    ax.axvline(lambdas[best_bic_idx], color="purple", linestyle=":",
               linewidth=1.5, alpha=0.7,
               label=f"BIC* = {lambdas[best_bic_idx]:.2f}")

ax.set_xscale("log")
ax.set_xlabel(r"L1 penalty $\lambda$", fontsize=11)
ax.set_ylabel("information criterion", fontsize=11)
ax.set_title("Model selection", fontsize=12, pad=10)
ax.legend(loc="center right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/synthetic/syn02_regularization_path.png", dpi=150, bbox_inches="tight")
print("Saved: figures/synthetic/syn02_regularization_path.png")
