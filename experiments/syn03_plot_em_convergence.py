import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

trajectories  = np.load("experiments/results/synthetic/syn03_trajectories.npy")
final_logliks = np.load("experiments/results/synthetic/syn03_final_logliks.npy")

n_runs, n_iter = trajectories.shape
print(f"Loaded {n_runs} trajectories, each with {n_iter} points")
print(f"Final logL: mean={final_logliks.mean():.3f}, "
      f"std={final_logliks.std():.4f}")


# =============================================================================
# Figure: log-likelihood trajectories + histogram of final values
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Panel 1: all trajectories ---
ax = axes[0]
for i in range(n_runs):
    ax.plot(trajectories[i], color="steelblue", alpha=0.4, linewidth=1)

mean_traj = trajectories.mean(axis=0)
ax.plot(mean_traj, color="red", linewidth=2.5, label="mean across runs")

ax.set_xlabel("EM iteration", fontsize=11)
ax.set_ylabel("log-likelihood", fontsize=11)
ax.set_title(f"EM convergence over {n_runs} random initialisations",
             fontsize=12, pad=10)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

# --- Panel 2: histogram of final values, plotted on a relative scale ---
ax = axes[1]

mean_val = final_logliks.mean()
# Show as deviation from the mean (in nats), avoids long numeric labels
deviations = final_logliks - mean_val

ax.hist(deviations, bins=8, color="steelblue",
        edgecolor="white", linewidth=1.0, alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=2,
           label="distribution centre")

ax.set_xlabel(f"deviation from mean log-likelihood\n"
              f"(mean = {mean_val:.3f})", fontsize=11)
ax.set_ylabel("count", fontsize=11)
ax.set_title(f"Distribution of final logL  (std = {final_logliks.std():.4f})",
             fontsize=12, pad=10)

# Force y axis to integers only
y_max = ax.get_ylim()[1]
ax.set_yticks(np.arange(0, int(np.ceil(y_max)) + 1))

# Symmetric x range about 0
abs_max = max(abs(deviations.min()), abs(deviations.max())) * 1.15
ax.set_xlim(-abs_max, abs_max)

ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/synthetic/syn03_em_convergence.png", dpi=150, bbox_inches="tight")
print("Saved: figures/synthetic/syn03_em_convergence.png")
plt.show()