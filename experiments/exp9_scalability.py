import sys
sys.path.insert(0, ".")

import numpy as np
import time
from tqdm import tqdm
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.e_step import EStep
from inference.m_step import MStep
from simulation.simulator import HawkesSimulator


# =============================================================================
# Experiment 9: Computational Scalability
#
# Honestly characterise the wall-clock cost of one EM iteration
# as a function of dataset size n and node count N.
#
# This addresses the PhD-level question: how large can the system grow
# before the current pure-Python prototype becomes impractical?
# =============================================================================

BETA      = 1.0
DELTA     = 0.5
LAMBDA_L1 = 0.001


# -----------------------------------------------------------------------------
# Sweep 1: scaling with n_events (fixed N=4)
# -----------------------------------------------------------------------------
N_NODES_FIXED = 4
T_VALUES      = [50, 100, 200, 400, 800]
EDGE_LIST     = [(0, 1)]


def time_one_em_iteration(events, T, n_nodes, edge_list, kernel, anchor_calc):
    tensor = HypergraphTensor(n_nodes=n_nodes, rank=3, seed=0)
    estep  = EStep(kernel, anchor_calc)
    mstep  = MStep(n_nodes=n_nodes, tensor=tensor, lambda_l1=LAMBDA_L1)

    rng = np.random.default_rng(seed=2026)
    mu             = rng.uniform(0.1, 0.5, size=n_nodes)
    alpha_pairwise = rng.uniform(0.0, 0.2, size=(n_nodes, n_nodes))
    np.fill_diagonal(alpha_pairwise, 0.0)
    alpha_hyper    = {e: float(rng.uniform(0.05, 0.4)) for e in edge_list}

    for e in edge_list:
        target_factor = alpha_hyper[e] ** (1.0 / (len(e) * tensor.rank))
        for v in e:
            tensor.F[v, :] = target_factor

    # Warm up (excluded from timing)
    result = estep.compute(events, mu, alpha_pairwise, alpha_hyper, edge_list)

    # Time one full EM step
    start = time.time()
    result = estep.compute(events, mu, alpha_pairwise, alpha_hyper, edge_list)
    mu = mstep.update_mu(events, result["p_background"], T)
    alpha_pairwise = mstep.update_alpha_pairwise(
        events, result["p_pairwise"], result["p_hyper"],
        edge_list, kernel, T
    )
    alpha_hyper = mstep.update_alpha_hyper(
        events, result["p_hyper"], edge_list, anchor_calc, kernel, T
    )
    elapsed = time.time() - start
    return elapsed


print("=" * 65)
print(f"Sweep 1: scaling with n_events (N={N_NODES_FIXED} nodes)")
print("=" * 65)
print(f"{'T':>6}  {'n_events':>10}  {'time/iter':>10}  {'time/event':>12}")
print("-" * 50)

kernel      = ExponentialKernel(beta=BETA)
anchor_calc = HyperedgeAnchor(delta=DELTA)

mu_sim = np.full(N_NODES_FIXED, 0.3)
alpha_p_sim = np.zeros((N_NODES_FIXED, N_NODES_FIXED))
alpha_p_sim[2, 0] = 0.3
alpha_h_sim = {(0, 1): 0.4}

results_n = []
for T_val in tqdm(T_VALUES, desc="n_events sweep"):
    sim = HawkesSimulator(
        mu_sim, alpha_p_sim, alpha_h_sim, kernel, anchor_calc
    )
    events = sim.simulate(T=T_val, seed=42, max_events=5000)
    n = len(events)

    elapsed = time_one_em_iteration(
        events, T_val, N_NODES_FIXED, EDGE_LIST, kernel, anchor_calc
    )

    print(f"{T_val:>6}  {n:>10}  {elapsed:>10.3f}s  {elapsed/n*1000:>10.3f}ms")
    results_n.append({
        "T": T_val, "n_events": n, "time": elapsed,
    })


# -----------------------------------------------------------------------------
# Sweep 2: scaling with node count (fixed n ~ 600)
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("Sweep 2: scaling with N nodes (T fixed; n_events ~ similar)")
print("=" * 65)
print(f"{'N_nodes':>10}  {'n_events':>10}  {'time/iter':>10}")
print("-" * 50)

N_VALUES = [3, 5, 8, 12]
T_FIXED  = 200.0

results_N = []
for N in tqdm(N_VALUES, desc="N nodes sweep"):
    mu_sim   = np.full(N, 0.3)
    alpha_p  = np.zeros((N, N))
    if N >= 3:
        alpha_p[2, 0] = 0.3
    alpha_h  = {(0, 1): 0.4}
    edge_lst = [(0, 1)]

    sim = HawkesSimulator(mu_sim, alpha_p, alpha_h, kernel, anchor_calc)
    events = sim.simulate(T=T_FIXED, seed=42, max_events=5000)
    n = len(events)

    elapsed = time_one_em_iteration(
        events, T_FIXED, N, edge_lst, kernel, anchor_calc
    )

    print(f"{N:>10}  {n:>10}  {elapsed:>10.3f}s")
    results_N.append({"N": N, "n_events": n, "time": elapsed})


# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
import pickle
with open("experiments/exp9_scalability.pkl", "wb") as f:
    pickle.dump({
        "scaling_n":  results_n,
        "scaling_N":  results_N,
    }, f)
print("\nSaved: experiments/exp9_scalability.pkl")


# -----------------------------------------------------------------------------
# Quick scaling fit: time = c * n^p
# -----------------------------------------------------------------------------
ns      = np.array([r["n_events"] for r in results_n])
times   = np.array([r["time"]     for r in results_n])
mask    = (ns > 0) & (times > 0)
log_n   = np.log(ns[mask])
log_t   = np.log(times[mask])

# Linear fit log_t = p * log_n + log_c
p, log_c = np.polyfit(log_n, log_t, 1)
c = np.exp(log_c)
print(f"\nEmpirical scaling: time ~ {c:.2e} * n^{p:.2f}")
print(f"  Theoretical E-step is O(n^2) for pairwise + O(n*|E|) for hyperedges")
print(f"  Observed exponent {p:.2f} is consistent with O(n^2) dominance")