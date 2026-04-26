import sys
sys.path.insert(0, ".")

import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.e_step import EStep
from inference.m_step import MStep
from simulation.simulator import HawkesSimulator


# =============================================================================
# Experiment 3: EM Convergence and Initialisation Sensitivity
#
# Run EM from many random initialisations on the same dataset.
# Track the log-likelihood trajectory of each run.
# Report the variance of the final log-likelihood and parameter values.
# =============================================================================

# Ground truth (same as Exp 1)
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
N_INITS          = 20            # number of random initialisations
BETA             = 1.0
DELTA            = 0.5
N_NODES          = 3


# =============================================================================
# Setup: simulate ONE dataset, then test different inits on it
# =============================================================================
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
# Helper: compute log-likelihood under given parameters
# =============================================================================
def compute_loglik(events, T, mu, alpha_pairwise, alpha_hyper,
                   kernel, anchor_calc):
    sim_check = HawkesSimulator(
        mu, alpha_pairwise, alpha_hyper, kernel, anchor_calc
    )
    log_lam_sum = 0.0
    for i, (t_i, node_i) in enumerate(events):
        history = events[:i]
        lam = sim_check._intensity(t_i, history)
        if lam[node_i] > 0:
            log_lam_sum += np.log(lam[node_i])

    grid = np.linspace(0, T, 200)
    total_int = 0.0
    for k in range(len(grid) - 1):
        t_mid = 0.5 * (grid[k] + grid[k+1])
        history = [(t, n) for t, n in events if t < t_mid]
        lam = sim_check._intensity(t_mid, history)
        total_int += float(lam.sum()) * (grid[k+1] - grid[k])

    return log_lam_sum - total_int


# =============================================================================
# Run EM from N_INITS random starting points
# =============================================================================
all_trajectories = []     # list of arrays of shape (N_ITER+1,)
final_params     = []     # list of dicts
final_logliks    = []     # list of floats

print(f"Running EM from {N_INITS} random initialisations ...")
print(f"{'Run':>4}  {'final mu[0]':>12}  {'final a[2->0]':>14}  "
      f"{'final hyper':>12}  {'final logL':>12}")
print("-" * 65)

from tqdm import tqdm
for run_idx in tqdm(range(N_INITS), desc="EM runs"):
    rng = np.random.default_rng(seed=1000 + run_idx)

    # Random init: draw uniformly in plausible ranges
    init_mu             = rng.uniform(0.1, 0.6, size=N_NODES)
    init_alpha_pairwise = rng.uniform(0.0, 0.4, size=(N_NODES, N_NODES))
    np.fill_diagonal(init_alpha_pairwise, 0.0)
    init_alpha_hyper    = {(0, 1): float(rng.uniform(0.05, 0.7))}

    # Build fresh inference objects
    tensor = HypergraphTensor(n_nodes=N_NODES, rank=3, seed=run_idx)
    estep  = EStep(kernel, anchor_calc)
    mstep  = MStep(n_nodes=N_NODES, tensor=tensor, lambda_l1=0.001)

    mu             = init_mu.copy()
    alpha_pairwise = init_alpha_pairwise.copy()
    alpha_hyper    = dict(init_alpha_hyper)

    for e in EDGE_LIST:
        target_factor = alpha_hyper[e] ** (1.0 / (len(e) * tensor.rank))
        for v in e:
            tensor.F[v, :] = target_factor

    # Track log-likelihood across iterations
    trajectory = []
    ll = compute_loglik(events, T, mu, alpha_pairwise, alpha_hyper,
                        kernel, anchor_calc)
    trajectory.append(ll)

    # EM loop
    for it in range(N_ITER):
        result = estep.compute(events, mu, alpha_pairwise, alpha_hyper, EDGE_LIST)
        mu = mstep.update_mu(events, result["p_background"], T)
        alpha_pairwise = mstep.update_alpha_pairwise(
            events, result["p_pairwise"], result["p_hyper"], EDGE_LIST, kernel, T
        )
        alpha_hyper = mstep.update_alpha_hyper(
            events, result["p_hyper"], EDGE_LIST, anchor_calc, kernel, T
        )

        ll = compute_loglik(events, T, mu, alpha_pairwise, alpha_hyper,
                            kernel, anchor_calc)
        trajectory.append(ll)

    all_trajectories.append(np.array(trajectory))
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

# Cluster final logliks: count how many are within 1 nat of the best
best_ll = final_logliks.max()
n_at_best = int(np.sum(final_logliks > best_ll - 1.0))
print(f"\n  Runs within 1 nat of best : {n_at_best} / {N_INITS}")

# Parameter spread among "best" runs
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

# Save trajectories for later plotting
np.save("experiments/exp3_trajectories.npy", np.array(all_trajectories))
np.save("experiments/exp3_final_logliks.npy", final_logliks)
print("\n  Trajectories saved to experiments/exp3_trajectories.npy")
print("  Final logliks  saved to experiments/exp3_final_logliks.npy")