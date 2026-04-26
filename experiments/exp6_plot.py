import sys
sys.path.insert(0, ".")

import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Run exp6 and parse stdout (since exp6 doesn't save to a pkl)
# =============================================================================
print("Running exp6_3node_hyperedge.py to capture results ...")
result = subprocess.run(
    [sys.executable, "experiments/exp6_3node_hyperedge.py"],
    capture_output=True, text=True
)
output = result.stdout
print(output)


# =============================================================================
# Parse the recovery summary table
# =============================================================================
edges = []
true_vals = []
inf_vals  = []
types     = []

# Look for the table that follows "--- Recovery Summary ---"
pattern = re.compile(
    r"\((\d+(?:,\s*\d+)+)\)\s+(\w+)\s+([\d.]+)\s+([\d.]+)"
)

started = False
for line in output.split("\n"):
    if "Recovery Summary" in line:
        started = True
        continue
    if not started:
        continue
    m = pattern.search(line)
    if m:
        e_str = m.group(1).replace(" ", "")
        e_tup = tuple(int(x) for x in e_str.split(","))
        edges.append(e_tup)
        types.append(m.group(2))
        true_vals.append(float(m.group(3)))
        inf_vals.append(float(m.group(4)))

if len(edges) == 0:
    raise RuntimeError("Could not parse exp6 output")

true_vals = np.array(true_vals)
inf_vals  = np.array(inf_vals)


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