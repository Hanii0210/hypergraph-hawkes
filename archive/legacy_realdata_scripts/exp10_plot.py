import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

with open("experiments/exp10_realdata.pkl", "rb") as f:
    R = pickle.load(f)

n_cells    = R["n_cells"]
candidates = R["candidates"]


# =============================================================================
# Figure: use explicit GridSpec for full control over spacing
# =============================================================================
fig = plt.figure(figsize=(20, 6))
gs = gridspec.GridSpec(1, 4, figure=fig,
                       width_ratios=[1.1, 0.05, 1.0, 1.0],
                       wspace=0.45,
                       left=0.05, right=0.95, top=0.85, bottom=0.13)

ax_raster = fig.add_subplot(gs[0, 0])
# gs[0,1] is a spacer column
ax_heatmap = fig.add_subplot(gs[0, 2])
ax_hyper   = fig.add_subplot(gs[0, 3])


# --- Panel 1: Raster plot ---
ax = ax_raster
events = []
with open("experiments/exp10_realdata_events.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        events.append((float(row["time"]), int(row["node"])))

t_plot = 10.0
for t, n in events:
    if t <= t_plot:
        ax.plot(t, n, "|", color=f"C{n}", markersize=8, alpha=0.6)

ax.set_xlabel("time (s)", fontsize=11)
ax.set_ylabel("neuron index", fontsize=11)
ax.set_title("Spike raster (first 10 s)", fontsize=12, pad=10)
ax.set_yticks(range(n_cells))
ax.set_xlim(0, t_plot)
ax.grid(True, alpha=0.2, axis="x")

for n in range(n_cells):
    count = sum(1 for _, node in events if node == n)
    rate = count / R["T"]
    ax.text(t_plot + 0.2, n, f"{rate:.0f} Hz",
            fontsize=8, va="center", color=f"C{n}")


# --- Panel 2: Pairwise interaction matrix ---
ax = ax_heatmap
alpha_pw = R["alpha_pw"]
im = ax.imshow(alpha_pw, cmap="Reds", aspect="equal",
               vmin=0, vmax=alpha_pw.max())
ax.set_xlabel("target neuron", fontsize=11)
ax.set_ylabel("source neuron", fontsize=11)
ax.set_title("Inferred pairwise weights", fontsize=12, pad=10)
ax.set_xticks(range(n_cells))
ax.set_yticks(range(n_cells))
plt.colorbar(im, ax=ax, shrink=0.75, pad=0.04, label=r"$\alpha_{ij}$")


# --- Panel 3: Hyperedge weights ---
ax = ax_hyper
alpha_hyper = R["alpha_hyper_hth"]
edges  = list(alpha_hyper.keys())
values = [alpha_hyper[e] for e in edges]
labels = [str(e) for e in edges]

y_cap = 0.6
colors = []
for e, v in zip(edges, values):
    if v > y_cap:
        colors.append("#e67e22")
    elif v > 0.1:
        colors.append("#c0392b")
    else:
        colors.append("#7f8c8d")

display_values = [min(v, y_cap) for v in values]
bars = ax.bar(range(len(edges)), display_values, color=colors,
              edgecolor="black", alpha=0.85)
ax.set_xticks(range(len(edges)))
ax.set_xticklabels(labels, rotation=15, fontsize=9)
ax.set_ylabel(r"hyperedge weight $\alpha_e$", fontsize=11)
ax.set_title("Candidate hyperedge weights", fontsize=12, pad=10)
ax.set_ylim(0, y_cap * 1.25)
ax.grid(True, alpha=0.3, axis="y")

for i, v in enumerate(values):
    if v > y_cap:
        ax.annotate(f"{v:.2f}\n(sparse\nartifact)",
                    xy=(i, y_cap),
                    xytext=(0, 15), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold",
                    color="#e67e22",
                    arrowprops=dict(arrowstyle="->", color="#e67e22"))
    else:
        ax.annotate(f"{v:.3f}", xy=(i, v),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")

summary = (
    f"Model comparison\n"
    f"$\\Delta L$ = {R['delta_L']:+.1f} nats  (p < 0.05)\n"
    f"BIC diff = {R['bic_diff']:+.1f}\n\n"
    f"Likelihood: HTH wins\n"
    f"BIC: pairwise wins\n"
    f"Verdict: suggestive signal"
)
ax.text(0.98, 0.98, summary, transform=ax.transAxes,
        fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.6",
                  fc="#fdf6e3", ec="#888888", linewidth=1))

import matplotlib.patches as mpatches
surv_patch = mpatches.Patch(color="#c0392b", label="surviving (>0.1)")
flag_patch = mpatches.Patch(color="#e67e22", label="flagged outlier")
supp_patch = mpatches.Patch(color="#7f8c8d", label="suppressed (<0.1)")
ax.legend(handles=[surv_patch, flag_patch, supp_patch],
          loc="center right", fontsize=8, framealpha=0.95)

fig.suptitle(
    "Real data: CRCNS ret-1 mouse retinal ganglion cells "
    f"({n_cells} neurons, {R['n_events']} events, {R['T']:.0f} s)",
    fontsize=13, fontweight="bold"
)
plt.savefig("experiments/exp10_realdata.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/exp10_realdata.png")
plt.show()