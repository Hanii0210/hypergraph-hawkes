import sys
sys.path.insert(0, ".")

import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor
from simulation.simulator import HawkesSimulator

kernel      = ExponentialKernel(beta=1.0)
anchor_calc = HyperedgeAnchor(delta=0.5)

mu             = np.array([0.3, 0.3, 0.3])
alpha_pairwise = np.array([
    [0.0, 0.2, 0.1],
    [0.2, 0.0, 0.2],
    [0.1, 0.2, 0.0],
])
alpha_hyper = {(0, 1): 0.4, (1, 2): 0.3}

sim = HawkesSimulator(mu, alpha_pairwise, alpha_hyper, kernel, anchor_calc)

# =============================================================================
# Test 1: simulation runs and returns a non-empty list
# =============================================================================
print("--- Test 1: simulation produces events ---")
events = sim.simulate(T=20.0, seed=0)
print(f"  total events : {len(events)}")
assert len(events) > 0, "FAILED: no events generated"
print("PASSED\n")

# =============================================================================
# Test 2: all event times are in (0, T]
# =============================================================================
print("--- Test 2: event times within window ---")
T = 20.0
for t, node in events:
    assert 0 < t <= T, f"FAILED: time {t} out of range"
print("PASSED\n")

# =============================================================================
# Test 3: all node indices are valid
# =============================================================================
print("--- Test 3: node indices valid ---")
for t, node in events:
    assert 0 <= node < 3, f"FAILED: invalid node {node}"
print("PASSED\n")

# =============================================================================
# Test 4: events are sorted by time
# =============================================================================
print("--- Test 4: events sorted by time ---")
times = [t for t, _ in events]
assert times == sorted(times), "FAILED: events not sorted"
print("PASSED\n")

# =============================================================================
# Test 5: longer window produces more events (sub-critical parameters)
# =============================================================================
print("--- Test 5: longer window produces more events ---")

mu_stable             = np.array([0.2, 0.2, 0.2])
alpha_pairwise_stable = np.array([
    [0.0, 0.1, 0.05],
    [0.1, 0.0, 0.1],
    [0.05, 0.1, 0.0],
])
alpha_hyper_stable = {(0, 1): 0.1, (1, 2): 0.1}

sim_stable = HawkesSimulator(
    mu_stable, alpha_pairwise_stable, alpha_hyper_stable, kernel, anchor_calc
)

events_long  = sim_stable.simulate(T=100.0, seed=1)
events_short = sim_stable.simulate(T=10.0,  seed=1)
print(f"  T=100 events: {len(events_long)}")
print(f"  T=10  events: {len(events_short)}")
assert len(events_long) > len(events_short), "FAILED"
print("PASSED\n")

# =============================================================================
# Test 6: print a sample of events to visually inspect
# =============================================================================
print("--- Test 6: sample output ---")
for t, node in events[:8]:
    print(f"  t={t:.4f}  node={node}")
print()

print("=== All tests passed ===")