import sys
sys.path.insert(0, ".")

import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor
from simulation.simulator import HawkesSimulator
from inference.candidate_filter import (
    fit_pairwise_only,
    generate_candidate_hyperedges,
)


# =============================================================================
# Test 1: generate_candidate_hyperedges with synthetic alpha_pairwise
# =============================================================================
print("--- Test 1: synthetic alpha_pairwise ---")

# Strong connections: (0,1) and (1,2). Weak: (0,3)
A = np.array([
    [0.0, 0.4, 0.0, 0.05],
    [0.4, 0.0, 0.3,  0.0],
    [0.0, 0.3, 0.0,  0.0],
    [0.05, 0.0, 0.0, 0.0],
])

candidates = generate_candidate_hyperedges(
    A, max_edge_size=3, pairwise_threshold=0.2
)
print(f"  candidates : {candidates}")
# Should include (0,1), (1,2), and (0,1,2). Not (0,3) since 0.05 < 0.2.
assert (0, 1) in candidates, "FAILED: missing (0,1)"
assert (1, 2) in candidates, "FAILED: missing (1,2)"
assert (0, 1, 2) in candidates, "FAILED: missing (0,1,2)"
assert (0, 3) not in candidates, "FAILED: (0,3) should be filtered"
print("PASSED\n")


# =============================================================================
# Test 2: top_m_pairs mode
# =============================================================================
print("--- Test 2: top-M filter ---")
candidates = generate_candidate_hyperedges(
    A, max_edge_size=2, top_m_pairs=1
)
print(f"  candidates : {candidates}")
# Top 1 pair is (0,1) with weight 0.4
assert candidates == [(0, 1)], "FAILED"
print("PASSED\n")


# =============================================================================
# Test 3: end-to-end on simulated data with known structure
# =============================================================================
print("--- Test 3: end-to-end pipeline ---")

kernel      = ExponentialKernel(beta=1.0)
anchor_calc = HyperedgeAnchor(delta=0.5)

# Truth: pairwise 2->0 strong, hyperedge (0,1)
true_mu = np.array([0.3, 0.3, 0.3, 0.3])
true_alpha_pairwise = np.zeros((4, 4))
true_alpha_pairwise[2, 0] = 0.3
true_alpha_pairwise[0, 1] = 0.2
true_alpha_pairwise[1, 0] = 0.2
true_alpha_hyper = {(0, 1): 0.4}

sim = HawkesSimulator(
    true_mu, true_alpha_pairwise, true_alpha_hyper, kernel, anchor_calc
)
events = sim.simulate(T=300.0, seed=42, max_events=2000)
print(f"  generated {len(events)} events")

# Stage 1: fit pairwise-only
mu_inf, alpha_pair_inf = fit_pairwise_only(
    events, T=300.0, n_nodes=4, kernel=kernel, anchor_calc=anchor_calc
)
print(f"  inferred alpha_pairwise (after pairwise-only EM):")
print(alpha_pair_inf)

# Stage 2: generate candidates
candidates = generate_candidate_hyperedges(
    alpha_pair_inf, max_edge_size=3, top_m_pairs=3
)
print(f"  candidates : {candidates}")

# Should at least include (0,1) since that pair is reinforced both by
# the pairwise truth and the hyperedge truth
assert (0, 1) in candidates, "FAILED: missed the true hyperedge support"
print("PASSED\n")

print("=== All tests passed ===")