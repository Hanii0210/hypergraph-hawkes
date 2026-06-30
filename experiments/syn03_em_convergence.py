import sys
sys.path.insert(0, ".")

import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.em import run_em
from simulation.simulator import HawkesSimulator


# =============================================================================
# Experiment 3: EM Convergence and Initialisation Sensitivity
#
# Run EM (real ALS over the CP factor matrix) from many random
# initialisations on the same dataset. Track the log-likelihood trajectory
# of each run and report the spread of the final log-likelihood.
#
# The trajectory uses the canonical closed-form log-likelihood
# (models.likelihood), which is consistent with the EM objective -- so a
# correct (generalized) EM run is monotone in exactly this quantity.
# =============================================================================

TRUE_MU             = np.array([0.3, 0.3, 0.4])
TRUE_ALPHA_PAIRWISE = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.3, 0.0, 0.0],
])
TRUE_ALPHA_HYPER = {(0, 1): 0.4}
EDGE_LIST        = [(0, 1)]
T                = 500.0
N_ITER           = 80
N_INITS          = 20
BETA             = 1.0
DELTA            = 0.5
N_NODES          = 3


kernel      = ExponentialKernel(beta=BETA)
anchor_calc = HyperedgeAnchor(delta=DELTA)

sim = HawkesSimulator(
    mu=TRUE_MU,
    alpha_pairwise=TRUE_ALPHA_PAIRWISE,
    alpha_hyper=TRUE_ALPHA_HYPER,
    kernel=kernel,
    anchor_calc=anchor_calc,
)
events = sim.simulate(T=T, seed=42, max_events=3000)
print(f"Generated {len(events)} events\n")


# =============================================================================
# Run EM from N_INITS random starting points
# =============================================================================
all_trajectories = []
final_params     = []
final_logliks    = []

print(f"Running EM from {N_INITS} random initialisations ...")
print(f"{'Run':>4}  {'final mu[0]':>12}  {'final a[2->0]':>14}  "
      f"{'final hyper':>12}  {'final logL':>12}")
print("-" * 65)

from tqdm import tqdm
for run_idx in tqdm(range(N_INITS), desc="EM runs"):
    rng = np.random.default_rng(seed=1000 + run_idx)

    init_mu             = rng.uniform(0.1, 0.6, size=N_NODES)
    init_alpha_pairwise = rng.uniform(0.0, 0.4, size=(N_NODES, N_NODES))
    np.fill_diagonal(init_alpha_pairwise, 0.0)
    init_alpha_hyper    = {(0, 1): float(rng.uniform(0.05, 0.7))}

    tensor = HypergraphTensor(n_nodes=N_NODES, rank=3, seed=run_idx)

    res = run_em(
        events, T, N_NODES, EDGE_LIST, kernel, anchor_calc,
        mu0=init_mu, alpha_pairwise0=init_alpha_pairwise,
        alpha_hyper0=init_alpha_hyper,
        n_iter=N_ITER, lambda_l1=0.001, tensor=tensor,
        hyper_update="als", track_loglik=True, assert_ascent=False,
    )

    mu             = res.mu
    alpha_pairwise = res.alpha_pairwise
    alpha_hyper    = res.alpha_hyper
    trajectory     = np.array(res.loglik_history)

    all_trajectories.append(trajectory)
    final_params.append({
        "mu":              mu.copy(),
        "alpha_pairwise":  alpha_pairwise.copy(),
        "alpha_hyper":     dict(alpha_hyper),
    })
    final_logliks.append(trajectory[-1])

    print(f"{run_idx:>4}  {mu[0]:>12.4f}  {alpha_pairwise[2,0]:>14.4f}  "
          f"{alpha_hyper[(0,1)]:>12.4f}  {trajectory[-1]:>12.2f}")


# =============================================================================
# Convergence Analysis
# =============================================================================
final_logliks = np.array(final_logliks)

print("\n--- Convergence Statistics ---")
print(f"  Final log-likelihood:")
print(f"    mean   : {final_logliks.mean():.3f}")
print(f"    std    : {final_logliks.std():.3f}")
print(f"    min    : {final_logliks.min():.3f}")
print(f"    max    : {final_logliks.max():.3f}")
print(f"    range  : {final_logliks.max() - final_logliks.min():.3f}")

best_ll = final_logliks.max()
n_at_best = int(np.sum(final_logliks > best_ll - 1.0))
print(f"\n  Runs within 1 nat of best : {n_at_best} / {N_INITS}")

best_mask = final_logliks > best_ll - 1.0
best_alpha_hyper = np.array([
    final_params[i]["alpha_hyper"][(0, 1)]
    for i in range(N_INITS) if best_mask[i]
])
best_a_2to0 = np.array([
    final_params[i]["alpha_pairwise"][2, 0]
    for i in range(N_INITS) if best_mask[i]
])

print(f"\n  Among best-mode runs:")
print(f"    alpha_hyper(0,1) : mean={best_alpha_hyper.mean():.4f}  "
      f"std={best_alpha_hyper.std():.4f}")
print(f"    alpha[2->0]      : mean={best_a_2to0.mean():.4f}  "
      f"std={best_a_2to0.std():.4f}")

np.save("experiments/results/synthetic/syn03_trajectories.npy", np.array(all_trajectories))
np.save("experiments/results/synthetic/syn03_final_logliks.npy", final_logliks)
print("\n  Trajectories saved to experiments/results/synthetic/syn03_trajectories.npy")
print("  Final logliks  saved to experiments/results/synthetic/syn03_final_logliks.npy")