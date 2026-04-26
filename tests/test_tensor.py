import sys
sys.path.insert(0, ".")

import numpy as np
from models.tensor_param import HypergraphTensor

# =============================================================================
# Test 1: weight is non-negative
# =============================================================================
print("--- Test 1: weight non-negative ---")

ht = HypergraphTensor(n_nodes=5, rank=4)
w = ht.get_weight((0, 1))
print(f"weight (0,1) : {w:.6f}")
assert w >= 0, "FAILED: weight must be non-negative"
print("PASSED\n")

# =============================================================================
# Test 2: single-node edge equals sum of F row
# =============================================================================
print("--- Test 2: single-node edge ---")

ht = HypergraphTensor(n_nodes=5, rank=4, seed=0)
w = ht.get_weight((2,))
expected = ht.F[2].sum()
print(f"weight (2,)  : {w:.6f}")
print(f"expected     : {expected:.6f}")
assert np.isclose(w, expected), "FAILED"
print("PASSED\n")

# =============================================================================
# Test 3: get_all_weights returns correct shape
# =============================================================================
print("--- Test 3: get_all_weights shape ---")

ht = HypergraphTensor(n_nodes=6, rank=3)
edges = [(0, 1), (0, 2), (1, 2, 3)]
weights = ht.get_all_weights(edges)
print(f"edges   : {edges}")
print(f"weights : {weights}")
assert weights.shape == (3,), "FAILED: shape mismatch"
print("PASSED\n")

# =============================================================================
# Test 4: gradient shape is correct
# =============================================================================
print("--- Test 4: gradient shape ---")

ht = HypergraphTensor(n_nodes=5, rank=4)
grad = ht.gradient_wrt_F(edge=(0, 1, 2), v=0)
print(f"gradient shape : {grad.shape}")
assert grad.shape == (4,), "FAILED: gradient shape mismatch"
print("PASSED\n")

# =============================================================================
# Test 5: gradient is numerically consistent
# =============================================================================
print("--- Test 5: gradient numerical check ---")

ht = HypergraphTensor(n_nodes=5, rank=4, seed=7)
edge = (0, 1)
v = 0
eps = 1e-5

analytic_grad = ht.gradient_wrt_F(edge, v)

numerical_grad = np.zeros(ht.rank)
for r in range(ht.rank):
    F_plus = ht.F.copy()
    F_plus[v, r] += eps
    ht_plus = HypergraphTensor(n_nodes=5, rank=4, seed=7)
    ht_plus.F = F_plus
    w_plus = ht_plus.get_weight(edge)

    F_minus = ht.F.copy()
    F_minus[v, r] -= eps
    ht_minus = HypergraphTensor(n_nodes=5, rank=4, seed=7)
    ht_minus.F = F_minus
    w_minus = ht_minus.get_weight(edge)

    numerical_grad[r] = (w_plus - w_minus) / (2 * eps)

print(f"analytic : {analytic_grad}")
print(f"numerical: {numerical_grad}")
assert np.allclose(analytic_grad, numerical_grad, atol=1e-5), "FAILED: gradient mismatch"
print("PASSED\n")

print("=== All tests passed ===")