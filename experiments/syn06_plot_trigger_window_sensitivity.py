import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("experiments/results/synthetic/syn06_trigger_window_sensitivity.pkl", "rb") as f:
    results = pickle.load(f)

deltas    = np.array([r["delta"]              for r in results])
alpha_h   = np.array([r["alpha_hyper"][(0,1)] for r in results])
logLs     = np.array([r["logL"]               for r in results])

TRUE_DELTA = 0.5
TRUE_ALPHA = 0.4


# =============================================================================
# Two-panel figure
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- Panel 1: alpha_hyper vs delta ---
ax = axes[0]
ax.plot(deltas, alpha_h, "o-", color="steelblue", linewidth=2.5,
        markersize=10, zorder=3)

ax.axhline(TRUE_ALPHA, color="#c0392b", linestyle="--", linewidth=2,
           label=f"true $\\alpha = {TRUE_ALPHA}$", zorder=2)
ax.axvline(TRUE_DELTA, color="darkgreen", linestyle=":", linewidth=2,
           label=f"true $\\Delta = {TRUE_DELTA}$", zorder=2)

for x, y in zip(deltas, alpha_h):
    ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(8, 6),
                textcoords="offset points", fontsize=9)

ax.set_xscale("log")
ax.set_xlabel(r"hyperparameter $\Delta$ (window width)", fontsize=11)
ax.set_ylabel(r"inferred $\alpha_{(0,1)}$", fontsize=11)
ax.set_title("Parameter recovery as a function of $\\Delta$",
             fontsize=12, pad=10)
ax.legend(loc="lower left", fontsize=10)
ax.grid(True, alpha=0.3)

# --- Panel 2: log-likelihood vs delta ---
ax = axes[1]
ax.plot(deltas, logLs, "s-", color="purple", linewidth=2.5,
        markersize=10, zorder=3)

best_idx = np.argmax(logLs)

# Single vertical line: the data-driven optimum coincides with the truth
ax.axvline(deltas[best_idx], color="darkgreen", linestyle="--",
           linewidth=2, alpha=0.85,
           label=f"argmax logL = true $\\Delta$ = {deltas[best_idx]}",
           zorder=2)

# Highlight the peak point with a star marker
ax.plot(deltas[best_idx], logLs[best_idx], "*",
        color="#f1c40f", markersize=22,
        markeredgecolor="black", markeredgewidth=1.5, zorder=5)

# Annotate values, with the peak label outside the star
for x, y in zip(deltas, logLs):
    if x == deltas[best_idx]:
        ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(15, -5),
                    textcoords="offset points", fontsize=10,
                    fontweight="bold")
    else:
        ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(8, 6),
                    textcoords="offset points", fontsize=9)

ax.set_xscale("log")
ax.set_xlabel(r"hyperparameter $\Delta$ (window width)", fontsize=11)
ax.set_ylabel("log-likelihood", fontsize=11)
ax.set_title("Likelihood is sharply peaked at true $\\Delta$",
             fontsize=12, pad=10)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/synthetic/syn06_trigger_window_sensitivity.png", dpi=150,
            bbox_inches="tight")
print("Saved: figures/synthetic/syn06_trigger_window_sensitivity.png")
plt.show()