import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("experiments/results/synthetic/syn09_identification_diagnostic.pkl", "rb") as f:
    d = pickle.load(f)

rows = d["rows"]
alphas    = np.array([r["alpha"]          for r in rows])
footprint = np.array([r["footprint_mean"] for r in rows])
foot_std  = np.array([r["footprint_std"]  for r in rows])
nom_pct   = np.array([r["nominated_pct"]  for r in rows])
bias_pct  = np.array([r["bias_pct"]       for r in rows])
dL        = np.array([r["dL"]             for r in rows])
bias_sem  = np.array([r.get("bias_sem", 0.0) for r in rows])
dL_sem    = np.array([r.get("dL_sem", 0.0)   for r in rows])
thr       = d["threshold"]
edge      = d["true_edge"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2),
                         gridspec_kw={"top": 0.82, "bottom": 0.14, "wspace": 0.30})

# --- Panel 1: pairwise footprint vs nomination threshold ---
ax = axes[0]
ax.bar(range(len(alphas)), footprint, yerr=foot_std, color="steelblue",
       edgecolor="black", alpha=0.85, width=0.55, capsize=5,
       label="max member-pair footprint")
ax.axhline(thr, color="#c0392b", linestyle="--", linewidth=2,
           label=f"nomination threshold = {thr}")
ax.set_xticks(range(len(alphas)))
ax.set_xticklabels([f"{a}" for a in alphas])
ax.set_xlabel(r"true hyperedge strength $\alpha$", fontsize=11)
ax.set_ylabel("inferred pairwise footprint", fontsize=11)
ax.set_ylim(0, float(np.max(footprint + foot_std)) * 1.55)
ax.set_title("A genuine hyperedge leaves a footprint\nabove threshold (so it is nominated)", fontsize=11, pad=8)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(nom_pct):
    ax.annotate(f"nom {v:.0f}%", xy=(i, footprint[i] + foot_std[i]),
                xytext=(0, 9), textcoords="offset points", ha="center",
                fontsize=9, fontweight="bold", color="#27ae60")

# --- Panel 2: recovery bias vs strength ---
ax = axes[1]
colors = ["#c0392b" if abs(b) > 15 else "#e67e22" if abs(b) > 5 else "#27ae60"
          for b in bias_pct]
ax.bar(range(len(alphas)), bias_pct, color=colors, edgecolor="black",
       alpha=0.85, width=0.55, yerr=bias_sem, capsize=5,
       error_kw={"elinewidth": 1.3, "ecolor": "#333333"})
ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(range(len(alphas)))
ax.set_xticklabels([f"{a}" for a in alphas])
ax.set_xlabel(r"true hyperedge strength $\alpha$", fontsize=11)
ax.set_ylabel("recovery bias (%)", fontsize=11)
ax.set_ylim(float(np.min(bias_pct - bias_sem)) * 1.28, max(3.0, float(np.max(bias_pct)) + 3.0))
ax.set_title("Hyperedge near-unbiased once found;\nweak detectability at low strength", fontsize=11, pad=8)
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(bias_pct):
    ax.annotate(f"{v:+.0f}%", xy=(i, v - bias_sem[i]), xytext=(0, -13),
                textcoords="offset points", ha="center", fontsize=11, fontweight="bold")

# --- Panel 3: detectability vs strength ---
ax = axes[2]
colors = ["#27ae60" if v > 3 else "#e67e22" if v > 0 else "#c0392b" for v in dL]
ax.bar(range(len(alphas)), dL, color=colors, edgecolor="black", alpha=0.85, width=0.55,
       yerr=dL_sem, capsize=5, error_kw={"elinewidth": 1.3, "ecolor": "#333333"})
ax.axhline(0, color="black", linewidth=1)
ax.axhline(3, color="gray", linestyle=":", linewidth=1.5, alpha=0.7,
           label=r"detectability ($\Delta\ell \approx 3$)")
ax.set_xticks(range(len(alphas)))
ax.set_xticklabels([f"{a}" for a in alphas])
ax.set_xlabel(r"true hyperedge strength $\alpha$", fontsize=11)
ax.set_ylabel(r"$\Delta\ell$ (HTH $-$ pairwise)", fontsize=11)
ax.set_ylim(float(np.min(dL - dL_sem)) - 1.1, float(np.max(dL + dL_sem)) + 1.1)
ax.set_title("Weakly detectable until the\nhyperedge is strong", fontsize=11, pad=8)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(dL):
    if v >= 0:
        ax.annotate(f"{v:+.1f}", xy=(i, v + dL_sem[i]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    else:
        ax.annotate(f"{v:+.1f}", xy=(i, v - dL_sem[i]), xytext=(0, -13),
                    textcoords="offset points", ha="center", fontsize=10, fontweight="bold")

fig.suptitle(
    f"Identification diagnostic (true hyperedge {edge}, no pairwise excitation): "
    "nomination is not the bottleneck\u2014identification is",
    fontsize=12, fontweight="bold")
plt.savefig("figures/synthetic/syn09_identification_diagnostic.png", dpi=150, bbox_inches="tight")
print("Saved: figures/synthetic/syn09_identification_diagnostic.png")