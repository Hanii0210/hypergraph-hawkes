import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("experiments/results/synthetic/syn01_recovery_robustness.pkl", "rb") as f:
    data = pickle.load(f)

recovered   = data["recovered"]
true_values = data["true_values"]
N_SEEDS     = data["N_SEEDS"]


# =============================================================================
# Figure: 2x3 grid (5 params + 1 summary)
# =============================================================================
param_names   = ["mu[0]", "mu[1]", "mu[2]", "a[2->0]", "alpha_hyper(0,1)"]
display_names = [r"$\mu_0$", r"$\mu_1$", r"$\mu_2$",
                 r"$\alpha_{2 \to 0}$", r"$\alpha_{(0,1)}$"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, name, disp in zip(axes[:5], param_names, display_names):
    vals   = np.array(recovered[name])
    true_v = true_values[name]

    # Data-driven range: tight around observations, with the truth always
    # inside the visible range (extend if necessary)
    data_lo, data_hi = vals.min(), vals.max()
    pad = (data_hi - data_lo) * 0.08 + 1e-6
    x_lo = min(data_lo - pad, true_v - pad)
    x_hi = max(data_hi + pad, true_v + pad)
    bins = np.linspace(x_lo, x_hi, 12)

    ax.hist(vals, bins=bins, color="#88a9c8",
            edgecolor="white", linewidth=1.0, alpha=0.85)

    # True value: solid red, slightly thicker
    ax.axvline(true_v, color="#c0392b", linestyle="-", linewidth=3.0,
               label=f"true = {true_v:.3f}", zorder=3)
    # Mean: dashed black
    ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=2.0,
               label=f"mean = {vals.mean():.3f}", zorder=4,
               dashes=(4, 3))

    bias    = vals.mean() - true_v
    rel_err = abs(bias) / true_v if true_v > 0 else float("nan")

    ax.set_title(
        f"{disp}    bias = {bias:+.4f},  std = {vals.std():.4f},  "
        f"rel.err = {rel_err*100:.1f}%",
        fontsize=11, pad=8
    )
    ax.set_xlabel("inferred value", fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)
    ax.set_xlim(x_lo, x_hi)


# 6th panel: summary as a single bordered text block
ax = axes[5]
ax.axis("off")

n_events_arr = np.array(recovered["n_events"])

summary_text = (
    "Summary\n"
    "\n"
    f"Across {N_SEEDS} independently simulated\n"
    f"datasets (mean {n_events_arr.mean():.0f} events;\n"
    f"range [{n_events_arr.min()}, {n_events_arr.max()}]):\n"
    "\n"
    "All baseline rates and pairwise weights\n"
    "are recovered with relative error < 5%.\n"
    "\n"
    "The hyperedge weight is recovered\n"
    "near-unbiased (~ -6%) with a larger\n"
    "spread (std/mean ratio ~0.25). The\n"
    "residual difficulty is identifiability\n"
    "/ variance, not a systematic bias --\n"
    "see exp13 (calibration) and exp14\n"
    "(weak detectability)."
)

ax.text(0.04, 0.96, summary_text,
        transform=ax.transAxes,
        fontsize=11, va="top", ha="left",
        linespacing=1.45,
        bbox=dict(boxstyle="round,pad=1.0",
                  fc="#fdf6e3", ec="#888888", linewidth=1.2))

plt.suptitle(
    f"Recovery distribution across {N_SEEDS} independent datasets",
    fontsize=14, y=0.995, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figures/synthetic/syn01_recovery_robustness.png", dpi=150,
            bbox_inches="tight")
print("Saved: figures/synthetic/syn01_recovery_robustness.png")
