import sys
sys.path.insert(0, ".")

import numpy as np
from tqdm import tqdm
from scipy.stats import norm, rankdata
from models.kernel import ExponentialKernel, HyperedgeAnchor
from simulation.simulator import HawkesSimulator


# =============================================================================
# Experiment 5: Copula-based Cross-Validation
# =============================================================================

TRUE_MU = np.array([0.3, 0.3, 0.3])
TRUE_ALPHA_PAIRWISE = np.zeros((3, 3))
TRUE_ALPHA_PAIRWISE[2, 0] = 0.3
T               = 1500.0
BIN_WIDTH       = 2.0
N_REPLICATES    = 8
BETA            = 1.0
DELTA           = 0.5

HTH_ALPHA_HYPER  = {(0, 1): 0.5}
NULL_ALPHA_HYPER = {}


# =============================================================================
# Setup
# =============================================================================
kernel      = ExponentialKernel(beta=BETA)
anchor_calc = HyperedgeAnchor(delta=DELTA)


def bin_event_counts(events, T, bin_width, n_nodes):
    n_bins = int(T / bin_width)
    counts = np.zeros((n_bins, n_nodes))
    for t, node in events:
        b = int(t / bin_width)
        if b < n_bins:
            counts[b, node] += 1
    return counts


def upper_tail_dependence(x, y, q=0.90):
    n = len(x)
    if n < 10:
        return np.nan
    u = (rankdata(x) - 0.5) / n
    v = (rankdata(y) - 0.5) / n
    in_top_v = (v > q)
    if in_top_v.sum() == 0:
        return 0.0
    return float(((u > q) & in_top_v).sum() / in_top_v.sum())


def probit_copula_density_estimate(x, y, q_high=0.85):
    n = len(x)
    u = (rankdata(x) - 0.5) / n
    v = (rankdata(y) - 0.5) / n
    z_x = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    z_y = norm.ppf(np.clip(v, 1e-6, 1 - 1e-6))
    threshold = norm.ppf(q_high)
    mask = (z_x > threshold) | (z_y > threshold)
    if mask.sum() < 5:
        return np.nan
    return float(np.corrcoef(z_x[mask], z_y[mask])[0, 1])


# =============================================================================
# Run replicates for HTH and Null
# =============================================================================
print(f"Running {N_REPLICATES} replicates each for HTH and Null ...")
print(f"{'Replicate':>10}  {'condition':>10}  {'tauU(0,1)':>10}  "
      f"{'rho_probit(0,1)':>16}  {'n_events':>9}")
print("-" * 65)

hth_tau   = []
null_tau  = []
hth_probit  = []
null_probit = []

for rep in tqdm(range(N_REPLICATES), desc="replicates"):
    sim = HawkesSimulator(
        TRUE_MU, TRUE_ALPHA_PAIRWISE, HTH_ALPHA_HYPER, kernel, anchor_calc
    )
    events = sim.simulate(T=T, seed=300 + rep, max_events=5000)
    counts = bin_event_counts(events, T, BIN_WIDTH, 3)
    t1 = upper_tail_dependence(counts[:, 0], counts[:, 1], q=0.90)
    p1 = probit_copula_density_estimate(counts[:, 0], counts[:, 1], q_high=0.85)
    hth_tau.append(t1)
    hth_probit.append(p1)
    print(f"{rep:>10}  {'HTH':>10}  {t1:>10.4f}  {p1:>16.4f}  {len(events):>9}")

    sim_null = HawkesSimulator(
        TRUE_MU, TRUE_ALPHA_PAIRWISE, NULL_ALPHA_HYPER, kernel, anchor_calc
    )
    events_null = sim_null.simulate(T=T, seed=300 + rep, max_events=5000)
    counts_null = bin_event_counts(events_null, T, BIN_WIDTH, 3)
    t2 = upper_tail_dependence(counts_null[:, 0], counts_null[:, 1], q=0.90)
    p2 = probit_copula_density_estimate(counts_null[:, 0], counts_null[:, 1], q_high=0.85)
    null_tau.append(t2)
    null_probit.append(p2)
    print(f"{rep:>10}  {'Null':>10}  {t2:>10.4f}  {p2:>16.4f}  {len(events_null):>9}")


# =============================================================================
# Summary
# =============================================================================
hth_tau   = np.array(hth_tau)
null_tau  = np.array(null_tau)
hth_probit  = np.array(hth_probit)
null_probit = np.array(null_probit)

print("\n--- Summary ---")
print(f"  Upper-tail dependence  tau_U(X_0, X_1)")
print(f"    HTH  : mean={np.nanmean(hth_tau):.4f}  std={np.nanstd(hth_tau):.4f}")
print(f"    Null : mean={np.nanmean(null_tau):.4f}  std={np.nanstd(null_tau):.4f}")

print(f"\n  Probit-tail correlation in upper tail")
print(f"    HTH  : mean={np.nanmean(hth_probit):.4f}  std={np.nanstd(hth_probit):.4f}")
print(f"    Null : mean={np.nanmean(null_probit):.4f}  std={np.nanstd(null_probit):.4f}")

from scipy.stats import ttest_ind
t_tau, p_tau = ttest_ind(hth_tau, null_tau, equal_var=False, nan_policy="omit")
t_pro, p_pro = ttest_ind(hth_probit, null_probit, equal_var=False, nan_policy="omit")

print(f"\n  Welch t-test  HTH vs Null")
print(f"    tau_U   : t = {t_tau:.3f}   p = {p_tau:.4f}")
print(f"    probit  : t = {t_pro:.3f}   p = {p_pro:.4f}")

import pickle
with open("experiments/exp5_copula.pkl", "wb") as f:
    pickle.dump({
        "hth_tau": hth_tau, "null_tau": null_tau,
        "hth_probit": hth_probit, "null_probit": null_probit,
    }, f)
print("\nSaved: experiments/exp5_copula.pkl")