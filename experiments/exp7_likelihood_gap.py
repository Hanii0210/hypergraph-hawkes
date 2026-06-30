import sys
sys.path.insert(0, ".")

import numpy as np
from tqdm import tqdm
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.em import run_em
from inference.candidate_filter import fit_pairwise_only
from simulation.simulator import HawkesSimulator
from models.likelihood import log_likelihood


# =============================================================================
# Experiment 7: Likelihood Gap (the project's core falsifiability test)
#
# Compare two fitted models on the same dataset:
#   M1 = pairwise-only Hawkes      (null hypothesis)
#   M2 = HTH (hyperedges + pairwise)
#
# Report:
#   Delta L = L(M2) - L(M1)
#   AIC and BIC differences (penalised by parameter count)
#
# A positive Delta L (above AIC/BIC threshold) supports H1: the data
# requires a hyperedge term.  This is the central decision rule of the
# project proposal section 1 ("falsifiability criterion").
# =============================================================================

# Two scenarios:
#   A: data generated WITH a true hyperedge -> Delta L should be large
#   B: data generated WITHOUT a hyperedge   -> Delta L should be near zero
SCENARIOS = {
    "A_with_hyperedge":    {"alpha_hyper": {(0, 1): 0.4}},
    "B_no_hyperedge":      {"alpha_hyper": {}},
}

TRUE_MU = np.array([0.3, 0.3, 0.3])
TRUE_ALPHA_PAIRWISE = np.zeros((3, 3))
TRUE_ALPHA_PAIRWISE[2, 0] = 0.3
EDGE_LIST = [(0, 1)]
T          = 500.0
N_ITER     = 60
N_NODES    = 3
BETA       = 1.0
DELTA      = 0.5


# =============================================================================
# Setup
# =============================================================================
kernel      = ExponentialKernel(beta=BETA)
anchor_calc = HyperedgeAnchor(delta=DELTA)


def fit_full_HTH(events, T, n_nodes, edge_list, kernel, anchor_calc,
                 n_iter=60, lambda_l1=0.001, seed=0):
    """Fit full HTH model (mu + pairwise + hyperedge)."""
    tensor = HypergraphTensor(n_nodes=n_nodes, rank=3, seed=seed)

    rng = np.random.default_rng(seed)
    mu             = rng.uniform(0.1, 0.5, size=n_nodes)
    alpha_pairwise = rng.uniform(0.0, 0.2, size=(n_nodes, n_nodes))
    np.fill_diagonal(alpha_pairwise, 0.0)
    alpha_hyper    = {e: float(rng.uniform(0.05, 0.4)) for e in edge_list}

    res = run_em(
        events, T, n_nodes, edge_list, kernel, anchor_calc,
        mu0=mu, alpha_pairwise0=alpha_pairwise, alpha_hyper0=alpha_hyper,
        n_iter=n_iter, lambda_l1=lambda_l1, tensor=tensor, hyper_update="als",
    )
    return res.mu, res.alpha_pairwise, res.alpha_hyper


# =============================================================================
# Run both scenarios
# =============================================================================
results = {}

for name, cfg in SCENARIOS.items():
    print(f"\n=== Scenario: {name} ===")
    print(f"  true alpha_hyper = {cfg['alpha_hyper']}")

    sim = HawkesSimulator(
        TRUE_MU, TRUE_ALPHA_PAIRWISE, cfg["alpha_hyper"],
        kernel, anchor_calc,
    )
    events = sim.simulate(T=T, seed=42, max_events=3000)
    n_obs  = len(events)
    print(f"  generated {n_obs} events")

    # Fit pairwise-only (M1)
    mu_p, alpha_p = fit_pairwise_only(
        events, T=T, n_nodes=N_NODES, kernel=kernel, anchor_calc=anchor_calc,
        n_iter=N_ITER, lambda_l1=0.001, seed=0
    )
    L1 = log_likelihood(events, T, mu_p, alpha_p, {}, [],
                        kernel, anchor_calc, "closed_form")
    n_params_M1 = N_NODES + int(np.sum(alpha_p > 1e-3))

    # Fit full HTH (M2)
    mu_h, alpha_ph, alpha_hyp = fit_full_HTH(
        events, T, N_NODES, EDGE_LIST, kernel, anchor_calc,
        n_iter=N_ITER, lambda_l1=0.001, seed=0
    )
    L2 = log_likelihood(events, T, mu_h, alpha_ph, alpha_hyp, EDGE_LIST,
                        kernel, anchor_calc, "closed_form")
    n_params_M2 = N_NODES + int(np.sum(alpha_ph > 1e-3)) + len(EDGE_LIST)

    delta_L = L2 - L1
    aic_diff = (-2*L1 + 2*n_params_M1) - (-2*L2 + 2*n_params_M2)
    bic_diff = (-2*L1 + n_params_M1*np.log(n_obs)) - (-2*L2 + n_params_M2*np.log(n_obs))

    print(f"\n  L(pairwise-only)     = {L1:.3f}   (params={n_params_M1})")
    print(f"  L(full HTH)          = {L2:.3f}   (params={n_params_M2})")
    print(f"  Delta L              = {delta_L:+.3f}")
    print(f"  AIC favours HTH by   = {aic_diff:+.3f}  (positive -> HTH wins)")
    print(f"  BIC favours HTH by   = {bic_diff:+.3f}  (positive -> HTH wins)")

    results[name] = {
        "L1": L1, "L2": L2, "delta_L": delta_L,
        "aic_diff": aic_diff, "bic_diff": bic_diff,
        "n_params_M1": n_params_M1, "n_params_M2": n_params_M2,
        "n_obs": n_obs,
        "alpha_hyper_inferred": dict(alpha_hyp),
    }


# =============================================================================
# Decision
# =============================================================================
print("\n" + "=" * 60)
print("Falsifiability decision (project proposal §1)")
print("=" * 60)

dL_A = results["A_with_hyperedge"]["delta_L"]
dL_B = results["B_no_hyperedge"]["delta_L"]
bic_A = results["A_with_hyperedge"]["bic_diff"]
bic_B = results["B_no_hyperedge"]["bic_diff"]

print(f"\n  Scenario A (true hyperedge):    Delta L = {dL_A:+.3f}, BIC diff = {bic_A:+.3f}")
print(f"  Scenario B (no hyperedge):      Delta L = {dL_B:+.3f}, BIC diff = {bic_B:+.3f}")

print("\n  Expectation:")
print("    A -> large positive Delta L: data demands a hyperedge")
print("    B -> Delta L ~ 0: hyperedge term unjustified")

import pickle
with open("experiments/exp7_likelihood_gap.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: experiments/exp7_likelihood_gap.pkl")