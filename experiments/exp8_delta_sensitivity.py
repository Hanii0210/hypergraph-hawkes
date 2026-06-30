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
from models.likelihood import log_likelihood


# =============================================================================
# Experiment 8: Delta sensitivity analysis
#
# Delta is the time-window width for "pattern completion".
# We use grid search to choose it (proposal section 2.2).
#
# Question: how sensitive are recovered parameters to the Delta hyperparameter?
#
# Generate ONE dataset with a known true Delta = 0.5.
# Refit the model with several Delta values to see:
#   - Delta < truth : pattern completions missed -> alpha_hyper underestimated
#   - Delta = truth : best recovery
#   - Delta > truth : false completions found -> alpha_hyper diluted
# =============================================================================

TRUE_MU = np.array([0.3, 0.3, 0.3])
TRUE_ALPHA_PAIRWISE = np.zeros((3, 3))
TRUE_ALPHA_PAIRWISE[2, 0] = 0.3
TRUE_ALPHA_HYPER = {(0, 1): 0.4}
EDGE_LIST = [(0, 1)]
TRUE_DELTA = 0.5
T          = 500.0
N_ITER     = 60
N_NODES    = 3
BETA       = 1.0

DELTA_GRID = [0.1, 0.25, 0.5, 1.0, 2.0]


# =============================================================================
# Generate data with the TRUE delta
# =============================================================================
kernel       = ExponentialKernel(beta=BETA)
anchor_true  = HyperedgeAnchor(delta=TRUE_DELTA)

sim = HawkesSimulator(
    TRUE_MU, TRUE_ALPHA_PAIRWISE, TRUE_ALPHA_HYPER, kernel, anchor_true
)
events = sim.simulate(T=T, seed=42, max_events=3000)
print(f"Generated {len(events)} events with true Delta = {TRUE_DELTA}\n")


# =============================================================================
# Fit the model under each Delta value
# =============================================================================
print(f"{'Delta':>7}  {'mu[0]':>7}  {'mu[1]':>7}  {'a[2->0]':>9}  "
      f"{'alpha_hyper':>12}  {'logL':>10}")
print("-" * 60)

results = []
for delta_test in tqdm(DELTA_GRID, desc="delta sweep"):
    anchor_test = HyperedgeAnchor(delta=delta_test)

    tensor = HypergraphTensor(n_nodes=N_NODES, rank=3, seed=0)

    rng = np.random.default_rng(seed=2026)
    mu             = rng.uniform(0.1, 0.5, size=N_NODES)
    alpha_pairwise = rng.uniform(0.0, 0.2, size=(N_NODES, N_NODES))
    np.fill_diagonal(alpha_pairwise, 0.0)
    alpha_hyper    = {e: float(rng.uniform(0.05, 0.4)) for e in EDGE_LIST}

    # P8.2: unified EM driver (identical ops to the former hand-rolled loop:
    # E-step -> update_mu -> update_alpha_pairwise -> update_alpha_hyper_als).
    res = run_em(events, T, N_NODES, EDGE_LIST, kernel, anchor_test,
                 mu0=mu, alpha_pairwise0=alpha_pairwise, alpha_hyper0=alpha_hyper,
                 n_iter=N_ITER, lambda_l1=0.001, tensor=tensor, hyper_update="als")
    mu, alpha_pairwise, alpha_hyper = res.mu, res.alpha_pairwise, res.alpha_hyper

    # Log-likelihood under fitted parameters with this Delta.
    # P8: canonical closed-form likelihood (exact piecewise compensator),
    # replacing the legacy 200-point Riemann grid for consistency with
    # exp7 / exp10 / exp12 / exp13. Uses anchor_test so the compensator
    # reflects the Delta being evaluated.
    logL = log_likelihood(events, T, mu, alpha_pairwise, alpha_hyper,
                          EDGE_LIST, kernel, anchor_test, "closed_form")

    print(f"{delta_test:>7.2f}  {mu[0]:>7.4f}  {mu[1]:>7.4f}  "
          f"{alpha_pairwise[2,0]:>9.4f}  {alpha_hyper[(0,1)]:>12.4f}  "
          f"{logL:>10.2f}")

    results.append({
        "delta":         delta_test,
        "mu":            mu.copy(),
        "alpha_pairwise": alpha_pairwise.copy(),
        "alpha_hyper":   dict(alpha_hyper),
        "logL":          logL,
    })


# =============================================================================
# Best Delta by likelihood
# =============================================================================
best_idx = np.argmax([r["logL"] for r in results])
print(f"\nBest Delta (by logL) : {results[best_idx]['delta']}")
print(f"  True Delta          : {TRUE_DELTA}")

import pickle
with open("experiments/exp8_delta_sensitivity.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: experiments/exp8_delta_sensitivity.pkl")