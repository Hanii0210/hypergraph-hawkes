import sys
sys.path.insert(0, ".")

import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.e_step import EStep
from inference.m_step import MStep
from simulation.simulator import HawkesSimulator


# =============================================================================
# Experiment 2: Regularisation Path Analysis
#
# Generate ONE dataset with a known true hyperedge.
# Add several "decoy" candidate hyperedges that are NOT in the truth.
# Run EM at many values of lambda_L1.
# Track:
#   - which hyperedges survive (alpha > threshold) at each lambda
#   - log-likelihood at each lambda
#   - AIC and BIC for model selection
# =============================================================================

# Ground truth: only ONE real hyperedge (0, 1)
TRUE_MU             = np.array([0.3, 0.3, 0.3, 0.3])
TRUE_ALPHA_PAIRWISE = np.zeros((4, 4))
TRUE_ALPHA_PAIRWISE[3, 0] = 0.3
TRUE_ALPHA_HYPER = {(0, 1): 0.4}

# Candidates: include the truth + several decoys
CANDIDATE_EDGES = [
    (0, 1),    # TRUE
    (0, 2),    # decoy
    (1, 2),    # decoy
    (2, 3),    # decoy
    (0, 1, 2), # decoy higher-order
    (1, 2, 3), # decoy higher-order
]

T               = 500.0
N_ITER          = 60
N_NODES         = 4
BETA            = 1.0
DELTA           = 0.5
LAMBDA_GRID     = np.logspace(-3, 1, 12)   # 0.001 to 10


# =============================================================================
# Setup
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
# Likelihood helper
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
# Run EM at each lambda_L1 value
# =============================================================================
n_obs = len(events)
results = []

print(f"{'lambda':>10}  {'logL':>10}  {'AIC':>10}  {'BIC':>10}  "
      f"{'survivors':>10}  {'alpha values':>30}")
print("-" * 90)

from tqdm import tqdm
for lam_l1 in tqdm(LAMBDA_GRID, desc="lambda sweep"):
    tensor = HypergraphTensor(n_nodes=N_NODES, rank=3, seed=0)
    estep  = EStep(kernel, anchor_calc)
    mstep  = MStep(n_nodes=N_NODES, tensor=tensor, lambda_l1=lam_l1)

    rng = np.random.default_rng(seed=2026)
    mu             = rng.uniform(0.1, 0.5, size=N_NODES)
    alpha_pairwise = rng.uniform(0.0, 0.3, size=(N_NODES, N_NODES))
    np.fill_diagonal(alpha_pairwise, 0.0)
    alpha_hyper    = {e: float(rng.uniform(0.05, 0.5)) for e in CANDIDATE_EDGES}

    for e in CANDIDATE_EDGES:
        target_factor = alpha_hyper[e] ** (1.0 / (len(e) * tensor.rank))
        for v in e:
            tensor.F[v, :] = target_factor

    for it in range(N_ITER):
        result = estep.compute(events, mu, alpha_pairwise, alpha_hyper, CANDIDATE_EDGES)
        mu = mstep.update_mu(events, result["p_background"], T)
        alpha_pairwise = mstep.update_alpha_pairwise(
            events, result["p_pairwise"], result["p_hyper"], CANDIDATE_EDGES, kernel, T
        )
        alpha_hyper = mstep.update_alpha_hyper(
            events, result["p_hyper"], CANDIDATE_EDGES, anchor_calc, kernel, T
        )

    ll = compute_loglik(events, T, mu, alpha_pairwise, alpha_hyper,
                       kernel, anchor_calc)

    # Count surviving hyperedges (above small threshold)
    survival_threshold = 1e-3
    survivors = [e for e in CANDIDATE_EDGES
                 if alpha_hyper[e] > survival_threshold]

    # Effective parameters: mu (N), alpha_pairwise nonzeros, alpha_hyper nonzeros
    n_params = N_NODES \
             + int(np.sum(alpha_pairwise > 1e-3)) \
             + len(survivors)

    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(n_obs)

    alpha_str = "  ".join(
        f"{e}={alpha_hyper[e]:.3f}" for e in CANDIDATE_EDGES
    )
    print(f"{lam_l1:>10.4f}  {ll:>10.2f}  {aic:>10.2f}  {bic:>10.2f}  "
          f"{len(survivors):>10}  {alpha_str}")

    results.append({
        "lambda":   lam_l1,
        "logL":     ll,
        "aic":      aic,
        "bic":      bic,
        "n_params": n_params,
        "alpha_hyper": dict(alpha_hyper),
        "survivors": survivors,
    })


# =============================================================================
# Save results for plotting
# =============================================================================
import pickle
with open("experiments/exp2_regpath.pkl", "wb") as f:
    pickle.dump(results, f)

print("\nSaved: experiments/exp2_regpath.pkl")
print(f"Optimal AIC at lambda = {results[np.argmin([r['aic'] for r in results])]['lambda']:.4f}")
print(f"Optimal BIC at lambda = {results[np.argmin([r['bic'] for r in results])]['lambda']:.4f}")