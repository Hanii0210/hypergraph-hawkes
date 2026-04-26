import sys
sys.path.insert(0, ".")

import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor
from inference.e_step import EStep

kernel      = ExponentialKernel(beta=1.0)
anchor_calc = HyperedgeAnchor(delta=0.5)
estep       = EStep(kernel, anchor_calc)

# Two nodes, four events
events = [
    (0.1, 0),
    (0.3, 1),
    (0.7, 0),
    (1.5, 1),
]

mu             = np.array([0.5, 0.5])
alpha_pairwise = np.array([[0.3, 0.2],
                            [0.2, 0.3]])
alpha_hyper    = {(0, 1): 0.4}
edge_list      = [(0, 1)]

result = estep.compute(events, mu, alpha_pairwise, alpha_hyper, edge_list)

p_bg   = result["p_background"]
p_pair = result["p_pairwise"]
p_hyp  = result["p_hyper"]

# =============================================================================
# Test 1: probabilities sum to 1 for every event
# =============================================================================
print("--- Test 1: probabilities sum to 1 ---")
for i in range(len(events)):
    total = p_bg[i] + p_pair[i].sum() + sum(p_hyp[e][i] for e in edge_list)
    print(f"  event {i}: sum = {total:.6f}")
    assert np.isclose(total, 1.0), f"FAILED at event {i}"
print("PASSED\n")

# =============================================================================
# Test 2: first event must come entirely from background
# =============================================================================
print("--- Test 2: first event is pure background ---")
print(f"  p_background[0] = {p_bg[0]:.6f}")
assert np.isclose(p_bg[0], 1.0), "FAILED"
print("PASSED\n")

# =============================================================================
# Test 3: all probabilities are in [0, 1]
# =============================================================================
print("--- Test 3: all values in [0, 1] ---")
assert np.all(p_bg >= 0) and np.all(p_bg <= 1)
assert np.all(p_pair >= 0) and np.all(p_pair <= 1)
for e in edge_list:
    assert np.all(p_hyp[e] >= 0) and np.all(p_hyp[e] <= 1)
print("PASSED\n")

# =============================================================================
# Test 4: hyperedge contributes to event 2 (node 0 at t=0.7)
# nodes 0 (t=0.1) and 1 (t=0.3) both fired within delta=0.5 before t=0.7
# =============================================================================
print("--- Test 4: hyperedge contributes to event 2 ---")
print(f"  p_hyper[(0,1)][2] = {p_hyp[(0,1)][2]:.6f}")
assert p_hyp[(0, 1)][2] > 0, "FAILED: expected hyperedge contribution"
print("PASSED\n")

print("=== All tests passed ===")