import sys
sys.path.insert(0, ".")

import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.e_step import EStep
from inference.m_step import MStep

kernel      = ExponentialKernel(beta=1.0)
anchor_calc = HyperedgeAnchor(delta=0.5)
tensor      = HypergraphTensor(n_nodes=2, rank=3, seed=42)
estep       = EStep(kernel, anchor_calc)
mstep       = MStep(n_nodes=2, tensor=tensor, lambda_l1=0.01)

events = [
    (0.1, 0),
    (0.3, 1),
    (0.7, 0),
    (1.5, 1),
]
T = 2.0

mu             = np.array([0.5, 0.5])
alpha_pairwise = np.array([[0.3, 0.2],
                            [0.2, 0.3]])
alpha_hyper    = {(0, 1): tensor.get_weight((0, 1))}
edge_list      = [(0, 1)]

# =============================================================================
# Test 1: mu update returns correct shape and positive values
# =============================================================================
print("--- Test 1: mu update ---")
result  = estep.compute(events, mu, alpha_pairwise, alpha_hyper, edge_list)
new_mu  = mstep.update_mu(events, result["p_background"], T)
print(f"  new_mu : {new_mu}")
assert new_mu.shape == (2,), "FAILED: shape"
assert np.all(new_mu > 0),   "FAILED: must be positive"
print("PASSED\n")

# =============================================================================
# Test 2: alpha_pairwise update returns correct shape and non-negative values
# =============================================================================
print("--- Test 2: alpha_pairwise update ---")
new_alpha = mstep.update_alpha_pairwise(
    events, result["p_pairwise"], result["p_hyper"], edge_list, kernel, T
)
print(f"  new_alpha_pairwise :\n{new_alpha}")
assert new_alpha.shape == (2, 2), "FAILED: shape"
assert np.all(new_alpha >= 0),    "FAILED: must be non-negative"
print("PASSED\n")

# =============================================================================
# Test 3: alpha_hyper update returns non-negative weights
# =============================================================================
print("--- Test 3: alpha_hyper update ---")
new_alpha_hyper = mstep.update_alpha_hyper(
    events, result["p_hyper"], edge_list, anchor_calc, kernel, T
)
print(f"  new_alpha_hyper : {new_alpha_hyper}")
assert new_alpha_hyper[(0, 1)] >= 0, "FAILED: must be non-negative"
print("PASSED\n")

# =============================================================================
# Test 4: full EM iteration runs without error
# =============================================================================
print("--- Test 4: full EM iteration ---")
for iteration in range(5):
    alpha_hyper = {e: tensor.get_weight(e) for e in edge_list}
    result      = estep.compute(events, mu, alpha_pairwise, alpha_hyper, edge_list)
    mu             = mstep.update_mu(events, result["p_background"], T)
    alpha_pairwise = mstep.update_alpha_pairwise(
        events, result["p_pairwise"], result["p_hyper"], edge_list, kernel, T
    )
    alpha_hyper    = mstep.update_alpha_hyper(
        events, result["p_hyper"], edge_list, anchor_calc, kernel, T
    )
    print(f"  iter {iteration+1}: mu={mu}, alpha_hyper={alpha_hyper}")

print("PASSED\n")
print("=== All tests passed ===")