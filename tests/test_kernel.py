import sys
sys.path.insert(0, ".")

import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor

# =============================================================================
# Test 1: ExponentialKernel basic evaluation
# =============================================================================
print("--- Test 1: ExponentialKernel.__call__ ---")

kernel = ExponentialKernel(beta=1.0)

tau = np.array([0.0, 1.0, 2.0])
result = kernel(tau)
expected = np.array([1.0, np.exp(-1.0), np.exp(-2.0)])

print(f"tau      : {tau}")
print(f"result   : {result}")
print(f"expected : {expected}")
assert np.allclose(result, expected), "FAILED: kernel values mismatch"
print("PASSED\n")

# =============================================================================
# Test 2: ExponentialKernel integral (compensator)
# =============================================================================
print("--- Test 2: ExponentialKernel.integral ---")

kernel = ExponentialKernel(beta=2.0)
t_anchor = 1.0
T = 3.0

analytic = kernel.integral(t_anchor, T)
expected = (1.0 / 2.0) * (1.0 - np.exp(-2.0 * (3.0 - 1.0)))

print(f"analytic : {analytic:.6f}")
print(f"expected : {expected:.6f}")
assert np.isclose(analytic, expected), "FAILED: integral mismatch"
print("PASSED\n")

# =============================================================================
# Test 3: HyperedgeAnchor - simple 2-node case, pattern completes
# =============================================================================
print("--- Test 3: HyperedgeAnchor - pattern completes ---")

anchor_calc = HyperedgeAnchor(delta=0.5)

event_times = {
    0: [0.1, 1.0],
    1: [0.9, 2.0],
}
edge = (0, 1)
t_current = 1.5

anchors = anchor_calc.find_anchors(edge, event_times, t_current)
print(f"event_times : {event_times}")
print(f"edge        : {edge}")
print(f"t_current   : {t_current}")
print(f"anchors     : {anchors}")

# node 0 fires at 1.0, node 1 fires at 0.9 -> gap = 0.1 < 0.5 -> anchor = 1.0
assert 1.0 in anchors, "FAILED: expected anchor at 1.0"
print("PASSED\n")

# =============================================================================
# Test 4: HyperedgeAnchor - window too wide, pattern does NOT complete
# =============================================================================
print("--- Test 4: HyperedgeAnchor - pattern does NOT complete ---")

anchor_calc = HyperedgeAnchor(delta=0.1)

event_times = {
    0: [0.1],
    1: [0.9],
}
edge = (0, 1)
t_current = 2.0

anchors = anchor_calc.find_anchors(edge, event_times, t_current)
print(f"delta       : 0.1  (gap between events is 0.8, too large)")
print(f"anchors     : {anchors}")
assert len(anchors) == 0, "FAILED: should find no anchors"
print("PASSED\n")

# =============================================================================
# Test 5: HyperedgeAnchor - 3-node case
# =============================================================================
print("--- Test 5: HyperedgeAnchor - 3-node pattern completes ---")

anchor_calc = HyperedgeAnchor(delta=0.5)

event_times = {
    0: [1.0],
    1: [1.2],
    2: [1.3],
}
edge = (0, 1, 2)
t_current = 2.0

anchors = anchor_calc.find_anchors(edge, event_times, t_current)
print(f"event_times : {event_times}")
print(f"anchors     : {anchors}")

# all three fire within 0.3 < 0.5 -> anchor = 1.3
assert 1.3 in anchors, "FAILED: expected anchor at 1.3"
print("PASSED\n")

print("=== All tests passed ===")