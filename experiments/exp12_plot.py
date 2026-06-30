"""
Experiment 12 — plot. Reads experiments/exp12_rank_sweep.pkl and renders the
rank-selection story for each dataset (R*=2 and R*=3) as a row of panels:

  (1) recovery relative error vs R, with the unconstrained "free" estimator as
      a dashed reference -- CP matches free once R >= R*;
  (2) factor-subspace coverage angle vs R -- collapses at R = R*, showing CP
      recovers the latent structure, not merely the weights;
  (3) AIC and BIC vs R, with the parameter counts -- information criteria and
      their (conservative) selection behaviour.

The true rank R* is marked in every panel.
"""

import sys
sys.path.insert(0, ".")

import pickle
import numpy as np
import matplotlib.pyplot as plt


with open("experiments/exp12_rank_sweep.pkl", "rb") as f:
    results = pickle.load(f)

keys = [k for k in ("R2", "R3") if k in results]
n_rows = len(keys)
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.2 * n_rows))
if n_rows == 1:
    axes = axes[None, :]

C_CP   = "#2c6fbb"
C_FREE = "#888888"
C_TRUE = "#c0392b"
C_AIC  = "#2c6fbb"
C_BIC  = "#e08a1e"


def cp_rows(rows):
    return [r for r in rows if r[0].startswith("CP")]


for row_i, key in enumerate(keys):
    d = results[key]
    R_true = d["R_true"]
    ctrl = d["controlled"]
    full = d["full_em"]
    free = next((r for r in ctrl if r[0] == "free"), None)

    cp = cp_rows(ctrl)
    Rs       = [int(r[0].split("=")[1]) for r in cp]
    relerr   = [r[6] for r in cp]
    cov      = [r[7] for r in cp]
    aic      = [r[4] for r in cp]
    bic      = [r[5] for r in cp]

    cp_f      = cp_rows(full)
    Rs_f      = [int(r[0].split("=")[1]) for r in cp_f]
    relerr_f  = [r[6] for r in cp_f]
    cov_f     = [r[7] for r in cp_f]

    # ---- panel 1: recovery error ----
    ax = axes[row_i, 0]
    ax.plot(Rs, relerr, "o-", color=C_CP, lw=2, label="CP (controlled)")
    ax.plot(Rs_f, relerr_f, "s--", color=C_CP, alpha=0.45, lw=1.5, label="CP (full EM)")
    if free is not None:
        ax.axhline(free[6], color=C_FREE, ls=":", lw=1.8,
                   label=f"free ({free[1]} params)")
    ax.axvline(R_true, color=C_TRUE, ls="-", lw=1.2, alpha=0.6)
    ax.set_xlabel("CP rank  R"); ax.set_ylabel("recovery rel. error (%)")
    ax.set_title(f"R*={R_true}: recovery error")
    ax.set_xticks(Rs); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- panel 2: subspace coverage angle ----
    ax = axes[row_i, 1]
    ax.plot(Rs, cov, "o-", color=C_CP, lw=2, label="controlled")
    ax.plot(Rs_f, cov_f, "s--", color=C_CP, alpha=0.45, lw=1.5, label="full EM")
    ax.axvline(R_true, color=C_TRUE, ls="-", lw=1.2, alpha=0.6,
               label=f"true R*={R_true}")
    ax.set_xlabel("CP rank  R"); ax.set_ylabel("factor coverage angle (deg)")
    ax.set_title(f"R*={R_true}: latent-subspace recovery")
    ax.set_xticks(Rs); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- panel 3: AIC / BIC ----
    ax = axes[row_i, 2]
    ax.plot(Rs, aic, "o-", color=C_AIC, lw=2, label="AIC")
    ax.plot(Rs, bic, "o-", color=C_BIC, lw=2, label="BIC")
    aic_best = Rs[int(np.argmin(aic))]; bic_best = Rs[int(np.argmin(bic))]
    ax.plot(aic_best, min(aic), "*", color=C_AIC, ms=16, zorder=5)
    ax.plot(bic_best, min(bic), "*", color=C_BIC, ms=16, zorder=5)
    ax.axvline(R_true, color=C_TRUE, ls="-", lw=1.2, alpha=0.6,
               label=f"true R*={R_true}")
    ax.set_xlabel("CP rank  R"); ax.set_ylabel("information criterion")
    ax.set_title(f"R*={R_true}: AIC*={aic_best}, BIC*={bic_best}")
    ax.set_xticks(Rs); ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle("Exp 12 — CP rank selection: parameter reduction and "
             "latent-rank identifiability", fontsize=13, y=1.005)
fig.tight_layout()
fig.savefig("experiments/exp12_rank_sweep.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/exp12_rank_sweep.png")
plt.show()