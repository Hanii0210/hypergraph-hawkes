import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("experiments/results/synthetic/syn10_interaction_baseline.pkl", "rb") as f:
    d = pickle.load(f)

rows = d["rows"]
alphas   = np.array([r["alpha"]    for r in rows])
dL_inter = np.array([r["dL_inter"] for r in rows])
dL_HTH   = np.array([r["dL_HTH"]   for r in rows])
margin   = np.array([r["margin"]   for r in rows])
dLi_sem  = np.array([r.get("dL_inter_sem", 0.0) for r in rows])
dLh_sem  = np.array([r.get("dL_HTH_sem", 0.0)   for r in rows])
edge     = d["edge"]

fig, ax = plt.subplots(figsize=(9, 6),
                       gridspec_kw={"top": 0.84, "bottom": 0.12})

x = np.arange(len(alphas))
w = 0.38
ax.bar(x - w / 2, dL_inter, width=w, color="#95a5a6", edgecolor="black",
       alpha=0.9, label="3-way interaction baseline",
       yerr=dLi_sem, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#333333"})
ax.bar(x + w / 2, dL_HTH, width=w, color="steelblue", edgecolor="black",
       alpha=0.9, label="HTH (pattern-completion)",
       yerr=dLh_sem, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#333333"})
ax.axhline(0, color="black", linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels([f"{a}" for a in alphas])
ax.set_xlabel(r"true hyperedge strength $\alpha$", fontsize=11)
ax.set_ylabel(r"likelihood gain over pairwise  $\Delta\ell$", fontsize=11)
ax.set_title("HTH vs a parameter-matched non-trivial baseline", fontsize=12, pad=10)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(min(0.0, float(np.min(dL_inter - dLi_sem))) - 0.1,
            float(np.max(dL_HTH + dLh_sem)) * 1.22)

for i in range(len(alphas)):
    ax.annotate(f"{dL_inter[i]:.2f}", xy=(i - w / 2, dL_inter[i] + dLi_sem[i]),
                xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    ax.annotate(f"{dL_HTH[i]:.2f}", xy=(i + w / 2, dL_HTH[i] + dLh_sem[i]),
                xytext=(0, 5), textcoords="offset points", ha="center",
                fontsize=9, fontweight="bold")
    if alphas[i] > 0:
        ytop = dL_HTH[i] + dLh_sem[i]
        ax.annotate(f"margin\n+{margin[i]:.2f}", xy=(i, ytop),
                    xytext=(0, 20), textcoords="offset points", ha="center",
                    fontsize=8, color="#c0392b", fontweight="bold")

fig.suptitle(
    f"Non-trivial baseline (true hyperedge {edge}): the pattern-completion structure\n"
    "wins at every strength and the margin grows; on null data neither invents structure",
    fontsize=12, fontweight="bold", linespacing=1.4)
plt.savefig("figures/synthetic/syn10_interaction_baseline.png", dpi=150, bbox_inches="tight")
print("Saved: figures/synthetic/syn10_interaction_baseline.png")