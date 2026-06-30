import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("experiments/exp13_calibration.pkl", "rb") as f:
    cal = pickle.load(f)
with open("experiments/exp13_control.pkl", "rb") as f:
    ctrl = pickle.load(f)

fpr_naive = cal["fpr_naive"]      # naive double-dipping, pairwise null
fpr_split = cal["fpr_split"]      # sample-split, pairwise null
fpr_ctrl  = ctrl["fpr_split"]     # sample-split, pure-Poisson control
p_split_null = np.asarray(cal["p_split"], dtype=float)
p_split_ctrl = np.asarray(ctrl["p_split"], dtype=float)
nseed = cal["nseed"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                         gridspec_kw={"top": 0.85, "bottom": 0.13, "wspace": 0.28})

# --- Panel 1: false-positive rates vs the 5% target ---
ax = axes[0]
labels = ["naive\n(double-dipping)", "sample-split\n(pairwise null)", "sample-split\n(Poisson control)"]
vals   = [fpr_naive, fpr_split, fpr_ctrl]
colors = ["#c0392b", "#e67e22", "#27ae60"]
bars = ax.bar(range(3), vals, color=colors, edgecolor="black", alpha=0.85, width=0.6)
ax.axhline(5, color="gray", linestyle="--", linewidth=2, label="5% target")
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Type-I false-positive rate (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.set_title(f"Selective inference calibration ({nseed} seeds)", fontsize=12, pad=10)
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(vals):
    ax.annotate(f"{v:.0f}%", xy=(i, v), xytext=(0, 6),
                textcoords="offset points", ha="center",
                fontsize=12, fontweight="bold")

# --- Panel 2: calibration curve (ECDF of selective p-values vs uniform) ---
ax = axes[1]
def ecdf(p):
    p = np.sort(p)
    return p, np.arange(1, len(p) + 1) / len(p)
xx, yy = ecdf(p_split_null)
ax.step(xx, yy, where="post", color="#e67e22", linewidth=2.5,
        label="pairwise null (confounded)")
xx, yy = ecdf(p_split_ctrl)
ax.step(xx, yy, where="post", color="#27ae60", linewidth=2.5,
        label="Poisson control (calibrated)")
ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5,
        label="uniform (ideal)")
ax.axvline(0.05, color="black", linestyle="--", linewidth=1, alpha=0.6)
ax.set_xlabel("selective p-value", fontsize=11)
ax.set_ylabel("empirical CDF", fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Excess small p-values reveal the pairwise confound", fontsize=12, pad=10)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle(
    "Honest model selection: naive double-dipping is severely anti-conservative\n"
    "sample-split restores calibration on a pure-Poisson null; residual inflation under a "
    "pairwise null is the hyperedge confound",
    fontsize=12, fontweight="bold", linespacing=1.4)
plt.savefig("experiments/exp13_calibration.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/exp13_calibration.png")