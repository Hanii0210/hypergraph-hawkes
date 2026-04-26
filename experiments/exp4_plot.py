import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("experiments/exp4_phase.pkl", "rb") as f:
    results = pickle.load(f)

alpha_grid    = np.array([r["alpha_hyper_true"] for r in results])
burst         = np.array([r["burst_emp"]        for r in results])
rho_true      = np.array([r["rho_true"]         for r in results])
rho_inf       = np.array([r["rho_inferred"]     for r in results])
n_events      = np.array([r["n_events"]         for r in results])
alpha_inf     = np.array([r["alpha_inferred"]   for r in results])


# =============================================================================
# Two-panel figure
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Panel 1: spectral radius ---
ax = axes[0]
ax.plot(alpha_grid, rho_true,
        color="darkred",  linewidth=2.5, marker="o",
        label=r"$\rho(A)$ from true params")
ax.plot(alpha_grid, rho_inf,
        color="steelblue", linewidth=2.5, marker="s",
        label=r"$\rho(A)$ from inferred params")
ax.axhline(1.0,
           color="black", linestyle="--", linewidth=1.5,
           label=r"stability threshold $\rho = 1$")

# Mark where inferred crosses 1
crossing_idx = np.argmin(np.abs(rho_inf - 1.0))
ax.axvline(alpha_grid[crossing_idx],
           color="steelblue", linestyle=":", alpha=0.6,
           label=f"inferred crossing at α={alpha_grid[crossing_idx]:.2f}")

ax.set_xlabel(r"true hyperedge strength $\alpha_e$")
ax.set_ylabel(r"spectral radius $\rho(A)$")
ax.set_title("Phase transition: spectral radius")
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

# --- Panel 2: burst frequency vs spectral radius ---
ax = axes[1]
ax2 = ax.twinx()

l1 = ax.plot(alpha_grid, burst,
             color="darkgreen", linewidth=2.5, marker="^",
             label="empirical burst frequency")
l2 = ax2.plot(alpha_grid, n_events,
              color="darkorange", linewidth=2, marker="o", alpha=0.7,
              label="total event count")

ax.axvline(1.0,
           color="darkred", linestyle="--", linewidth=1.5, alpha=0.6,
           label=r"true $\rho=1$ at $\alpha=1$")

ax.set_xlabel(r"true hyperedge strength $\alpha_e$")
ax.set_ylabel("burst frequency (events per unit window)", color="darkgreen")
ax2.set_ylabel("total event count over [0, T]", color="darkorange")
ax.tick_params(axis='y', labelcolor="darkgreen")
ax2.tick_params(axis='y', labelcolor="darkorange")
ax.set_title("Cascade emergence: empirical evidence")

# Combine legends
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("experiments/exp4_phase.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/exp4_phase.png")
plt.show()