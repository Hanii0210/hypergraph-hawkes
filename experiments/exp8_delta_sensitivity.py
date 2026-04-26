import sys
sys.path.insert(0, ".")

import numpy as np
from tqdm import tqdm
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.e_step import EStep
from inference.m_step import MStep
from simulation.simulator import HawkesSimulator


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
    estep  = EStep(kernel, anchor_test)
    mstep  = MStep(n_nodes=N_NODES, tensor=tensor, lambda_l1=0.001)

    rng = np.random.default_rng(seed=2026)
    mu             = rng.uniform(0.1, 0.5, size=N_NODES)
    alpha_pairwise = rng.uniform(0.0, 0.2, size=(N_NODES, N_NODES))
    np.fill_diagonal(alpha_pairwise, 0.0)
    alpha_hyper    = {e: float(rng.uniform(0.05, 0.4)) for e in EDGE_LIST}

    for e in EDGE_LIST:
        target_factor = alpha_hyper[e] ** (1.0 / (len(e) * tensor.rank))
        for v in e:
            tensor.F[v, :] = target_factor

    for _ in range(N_ITER):
        result = estep.compute(events, mu, alpha_pairwise, alpha_hyper, EDGE_LIST)
        mu = mstep.update_mu(events, result["p_background"], T)
        alpha_pairwise = mstep.update_alpha_pairwise(
            events, result["p_pairwise"], result["p_hyper"],
            EDGE_LIST, kernel, T
        )
        alpha_hyper = mstep.update_alpha_hyper(
            events, result["p_hyper"], EDGE_LIST, anchor_test, kernel, T
        )

    # Log-likelihood under fitted parameters with this Delta
    sim_check = HawkesSimulator(
        mu, alpha_pairwise, alpha_hyper, kernel, anchor_test
    )
    log_lam_sum = 0.0
    for i, (t_i, n_i) in enumerate(events):
        history = events[:i]
        lam = sim_check._intensity(t_i, history)
        if lam[n_i] > 0:
            log_lam_sum += np.log(lam[n_i])

    grid = np.linspace(0, T, 200)
    total_int = 0.0
    for k in range(len(grid) - 1):
        t_mid = 0.5 * (grid[k] + grid[k+1])
        history = [(t, n) for t, n in events if t < t_mid]
        lam = sim_check._intensity(t_mid, history)
        total_int += float(lam.sum()) * (grid[k+1] - grid[k])
    logL = log_lam_sum - total_int

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