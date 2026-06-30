import sys
sys.path.insert(0, ".")

import numpy as np
from tqdm import tqdm
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.e_step import EStep
from inference.m_step import MStep
from inference.em import run_em
from simulation.simulator import HawkesSimulator


# =============================================================================
# Experiment 4: Interaction-Strength Sensitivity (intensity & inferred coupling)
#
# Sweep the hyperedge interaction strength and track how the process intensity
# and the inferred coupling respond.
# At each strength:
#   - simulate a dataset
#   - measure the empirical burst frequency (event density)
#   - run EM, infer parameters
#   - compute spectral radius rho(A) of the effective interaction matrix
# Plot rho vs alpha_hyper, and burst frequency vs alpha_hyper.
# NOTE: rho(A) here is computed from a LINEAR-Hawkes effective matrix and is
# essentially alpha_hyper echoed back; it is a reference quantity, NOT a
# criticality threshold. The HTH hyperedge uses a single most-recent anchor
# (non-accumulating), so the process does not go super-critical: burst
# frequency rises smoothly with alpha_hyper and shows no divergence at rho=1.
# This experiment therefore reports intensity/coupling sensitivity, not a
# phase transition.
# =============================================================================

TRUE_MU = np.array([0.2, 0.2, 0.2, 0.2])
TRUE_ALPHA_PAIRWISE = np.zeros((4, 4))
TRUE_ALPHA_PAIRWISE[3, 0] = 0.2
EDGE_LIST = [(0, 1)]

ALPHA_HYPER_GRID = np.linspace(0.05, 1.5, 12)

T               = 200.0
N_ITER          = 50
N_NODES         = 4
BETA            = 1.0
DELTA           = 0.5
N_SIM_REPEATS   = 3


# =============================================================================
# Setup
# =============================================================================
kernel      = ExponentialKernel(beta=BETA)
anchor_calc = HyperedgeAnchor(delta=DELTA)


def empirical_burst_frequency(events, T, window=2.0):
    if len(events) == 0:
        return 0.0
    times = np.array([t for t, _ in events])
    max_count = 0
    for t_start in times:
        count = int(np.sum((times >= t_start) & (times < t_start + window)))
        if count > max_count:
            max_count = count
    return max_count / window


def effective_interaction_matrix(alpha_pairwise, alpha_hyper, edge_list, n_nodes,
                                  beta):
    A = alpha_pairwise.copy() / beta
    for e in edge_list:
        alpha_e = alpha_hyper.get(e, 0.0)
        for j in e:
            for i in e:
                if i != j:
                    A[j, i] += alpha_e / beta
    return A


# =============================================================================
# Sweep
# =============================================================================
print(f"{'alpha_h':>8}  {'burst_emp':>10}  {'rho_true':>10}  "
      f"{'rho_inf':>10}  {'logL':>10}  {'n_events':>9}")
print("-" * 70)

results = []
for alpha_hyper_true in tqdm(ALPHA_HYPER_GRID, desc="alpha sweep"):
    true_alpha_hyper = {(0, 1): float(alpha_hyper_true)}

    burst_freqs = []
    n_events_list = []
    last_events = None
    for rep in range(N_SIM_REPEATS):
        sim = HawkesSimulator(
            mu=TRUE_MU,
            alpha_pairwise=TRUE_ALPHA_PAIRWISE,
            alpha_hyper=true_alpha_hyper,
            kernel=kernel,
            anchor_calc=anchor_calc,
        )
        events = sim.simulate(T=T, seed=100+rep, max_events=2000)
        burst_freqs.append(empirical_burst_frequency(events, T))
        n_events_list.append(len(events))
        last_events = events

    burst_emp = float(np.mean(burst_freqs))
    n_events_mean = float(np.mean(n_events_list))

    A_true = effective_interaction_matrix(
        TRUE_ALPHA_PAIRWISE, true_alpha_hyper, EDGE_LIST, N_NODES, BETA
    )
    rho_true = float(np.max(np.abs(np.linalg.eigvals(A_true))))

    tensor = HypergraphTensor(n_nodes=N_NODES, rank=3, seed=0)

    rng = np.random.default_rng(seed=2026)
    mu             = rng.uniform(0.1, 0.4, size=N_NODES)
    alpha_pairwise = rng.uniform(0.0, 0.3, size=(N_NODES, N_NODES))
    np.fill_diagonal(alpha_pairwise, 0.0)
    alpha_hyper    = {e: float(rng.uniform(0.05, 0.5)) for e in EDGE_LIST}

    # P8.2: unified EM driver (identical ops to former hand-rolled loop).
    res = run_em(last_events, T, N_NODES, EDGE_LIST, kernel, anchor_calc,
                 mu0=mu, alpha_pairwise0=alpha_pairwise, alpha_hyper0=alpha_hyper,
                 n_iter=N_ITER, lambda_l1=0.001, tensor=tensor, hyper_update="als")
    mu, alpha_pairwise, alpha_hyper = res.mu, res.alpha_pairwise, res.alpha_hyper

    A_inf = effective_interaction_matrix(
        alpha_pairwise, alpha_hyper, EDGE_LIST, N_NODES, BETA
    )
    rho_inf = float(np.max(np.abs(np.linalg.eigvals(A_inf))))

    sim_inf = HawkesSimulator(mu, alpha_pairwise, alpha_hyper, kernel, anchor_calc)
    log_lam_sum = 0.0
    for i, (t_i, n_i) in enumerate(last_events):
        history = last_events[:i]
        lam = sim_inf._intensity(t_i, history)
        if lam[n_i] > 0:
            log_lam_sum += np.log(lam[n_i])

    print(f"{alpha_hyper_true:>8.3f}  {burst_emp:>10.3f}  {rho_true:>10.3f}  "
          f"{rho_inf:>10.3f}  {log_lam_sum:>10.2f}  {n_events_mean:>9.0f}")

    results.append({
        "alpha_hyper_true": float(alpha_hyper_true),
        "burst_emp":        burst_emp,
        "rho_true":         rho_true,
        "rho_inferred":     rho_inf,
        "n_events":         n_events_mean,
        "logL":             log_lam_sum,
        "alpha_inferred":   alpha_hyper[(0, 1)],
    })


import pickle
with open("experiments/exp4_phase.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: experiments/exp4_phase.pkl")