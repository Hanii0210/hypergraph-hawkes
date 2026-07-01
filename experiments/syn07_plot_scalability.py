import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("experiments/results/synthetic/syn07_scalability.pkl", "rb") as f:
    data = pickle.load(f)

scaling_n = data["scaling_n"]
scaling_N = data["scaling_N"]


# =============================================================================
# Figure: scaling with n_events and per-pair cost vs N
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                         gridspec_kw={"top": 0.86, "bottom": 0.13})

# --- Panel 1: scaling with n_events (log-log) ---
ax = axes[0]
ns    = np.array([r["n_events"] for r in scaling_n])
times = np.array([r["time"]     for r in scaling_n])

ax.loglog(ns, times, "o-", color="darkred", linewidth=2.5, markersize=10,
          label="measured")

log_n   = np.log(ns)
log_t   = np.log(times)
p, log_c = np.polyfit(log_n, log_t, 1)
c = np.exp(log_c)

n_smooth = np.linspace(ns.min(), ns.max(), 100)
t_smooth = c * n_smooth ** p
ax.loglog(n_smooth, t_smooth, "--", color="steelblue", linewidth=1.5,
          label=f"fit: $t \\sim n^{{{p:.2f}}}$")

n_ref = ns[0]
t_ref = times[0]
t_theory = t_ref * (ns / n_ref) ** 2.0
ax.loglog(ns, t_theory, ":", color="gray", linewidth=1.5,
          label=r"theoretical $O(n^2)$")

ax.set_xlabel("number of events $n$")
ax.set_ylabel("seconds per EM iteration")
ax.set_title("Scaling with dataset size (fixed N=4 nodes)",
             fontsize=12, pad=10)
ax.legend(loc="upper left")
ax.grid(True, which="both", alpha=0.3)

# --- Panel 2: per-event-pair cost vs N ---
ax = axes[1]
Ns       = np.array([r["N"]        for r in scaling_N])
times_N  = np.array([r["time"]     for r in scaling_N])
n_evts_N = np.array([r["n_events"] for r in scaling_N])

per_pair_us = (times_N / (n_evts_N ** 2)) * 1e6

ax.plot(Ns, per_pair_us, "s-",
        color="darkblue", linewidth=2.5, markersize=10,
        label=r"measured cost per event pair")

# Reference: mean of last three points (the asymptotic regime)
asymptote = per_pair_us[1:].mean()
ax.axhline(asymptote, color="gray", linestyle=":", linewidth=1.5,
           label=f"asymptotic cost $\\approx$ {asymptote:.2f} μs/pair")

ax.set_xlabel("number of nodes $N$")
ax.set_ylabel(r"microseconds per event pair  ($t_{\mathrm{iter}} / n^2$)")
ax.set_title("Per-pair cost stabilises as N grows",
             fontsize=12, pad=10)

# Y-axis: show full range from 0 so the modest variation is honestly framed
ax.set_ylim(0, max(per_pair_us) * 1.25)

for n_val, t_val, n_evt in zip(Ns, per_pair_us, n_evts_N):
    ax.annotate(f"{t_val:.2f}",
                xy=(n_val, t_val),
                xytext=(0, 10),
                textcoords="offset points",
                fontsize=10, ha="center", fontweight="bold")

ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xticks(Ns)

fig.suptitle("Computational scalability of the prototype",
             fontsize=13, fontweight="bold")
plt.savefig("figures/synthetic/syn07_scalability.png", dpi=150,
            bbox_inches="tight")
print("Saved: figures/synthetic/syn07_scalability.png")
