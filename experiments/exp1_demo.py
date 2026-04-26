"""
Experiment 1 (DEMO version)

Quick single-seed demonstration of parameter recovery on the canonical
3-node, 1-hyperedge configuration. Intended as a 1-2 minute "smoke test"
to confirm the inference pipeline runs end-to-end.

For the rigorous statistical claim about recovery accuracy, see
exp1b_recovery_robustness.py, which repeats this experiment across
25 independent datasets and reports the bias and variance of each
parameter.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.e_step import EStep
from inference.m_step import MStep
from simulation.simulator import HawkesSimulator


# =============================================================================
# Ground truth
# =============================================================================
TRUE_MU             = np.array([0.3, 0.3, 0.4])
TRUE_ALPHA_PAIRWISE = np.zeros((3, 3))
TRUE_ALPHA_PAIRWISE[2, 0] = 0.3
TRUE_ALPHA_HYPER = {(0, 1): 0.4}
EDGE_LIST        = [(0, 1)]
T                = 500.0
N_ITER           = 80
BETA             = 1.0
DELTA            = 0.5
N_NODES          = 3


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
print(f"Generated {len(events)} events over T={T}\n")


# =============================================================================
# Initialise inference (random)
# =============================================================================
tensor = HypergraphTensor(n_nodes=N_NODES, rank=3, seed=0)
estep  = EStep(kernel, anchor_calc)
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


# =============================================================================
# EM loop
# =============================================================================
print(f"{'Iter':>4}  {'mu[0]':>7}  {'mu[1]':>7}  {'mu[2]':>7}  "
      f"{'a[2->0]':>9}  {'hyper(0,1)':>12}")
print("-" * 58)

for it in range(1, N_ITER + 1):
    result = estep.compute(events, mu, alpha_pairwise, alpha_hyper, EDGE_LIST)

    mu             = mstep.update_mu(events, result["p_background"], T)
    alpha_pairwise = mstep.update_alpha_pairwise(
        events, result["p_pairwise"], result["p_hyper"],
        EDGE_LIST, kernel, T
    )
    alpha_hyper    = mstep.update_alpha_hyper(
        events, result["p_hyper"], EDGE_LIST, anchor_calc, kernel, T
    )

    if it % 20 == 0 or it == 1:
        print(f"{it:>4}  {mu[0]:>7.4f}  {mu[1]:>7.4f}  {mu[2]:>7.4f}  "
              f"{alpha_pairwise[2,0]:>9.4f}  {alpha_hyper[(0,1)]:>12.6f}")


# =============================================================================
# Final comparison
# =============================================================================
print("\n--- Recovery Summary (single seed) ---")
print(f"{'Parameter':<24} {'True':>7} {'Inferred':>10} {'Error':>8}")
print("-" * 52)

params = [
    ("mu[0]",            TRUE_MU[0],                mu[0]),
    ("mu[1]",            TRUE_MU[1],                mu[1]),
    ("mu[2]",            TRUE_MU[2],                mu[2]),
    ("a[2->0]",          TRUE_ALPHA_PAIRWISE[2, 0], alpha_pairwise[2, 0]),
    ("alpha_hyper(0,1)", TRUE_ALPHA_HYPER[(0, 1)],  alpha_hyper[(0, 1)]),
]

for name, true_val, inf_val in params:
    err = abs(inf_val - true_val)
    flag = "  <-- OK" if err < 0.1 else "  <-- off"
    print(f"  {name:<22} {true_val:>7.3f} {inf_val:>10.4f} {err:>8.4f}{flag}")

print("\nFor distribution-level recovery analysis across 25 datasets,")
print("see: experiments/exp1b_recovery_robustness.py")