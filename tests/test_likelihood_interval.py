import sys
sys.path.insert(0, ".")

import numpy as np

from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.likelihood import (
    log_likelihood,
    log_likelihood_interval,
    pairwise_event_exposures,
    piecewise_compensator_interval,
)


def assert_close(actual, expected, tol=1e-10, label="value"):
    if not np.isclose(actual, expected, atol=tol, rtol=tol):
        raise AssertionError(f"{label}: {actual:.15g} != {expected:.15g}")


kernel = ExponentialKernel(beta=1.3)
anchor = HyperedgeAnchor(delta=0.35)
events = [
    (0.10, 0),
    (0.28, 1),       # completes (0,1) before the split
    (0.74, 0),
    (1.00, 1),       # event exactly at split; belongs to left interval/history
    (1.18, 0),
    (1.36, 1),       # new completion inside the right interval
    (1.92, 0),
    (2.35, 1),
]
T = 2.5
split = 1.0
mu = np.array([0.24, 0.31])
alpha_pairwise = np.array([[0.08, 0.17], [0.12, 0.05]])
edge_list = [(0, 1)]
alpha_hyper = {(0, 1): 0.27}

print("--- Test 1: interval_start=0 matches canonical likelihood ---")
full = log_likelihood(
    events, T, mu, alpha_pairwise, alpha_hyper,
    edge_list, kernel, anchor,
)
initial_interval = log_likelihood_interval(
    events, 0.0, T, mu, alpha_pairwise, alpha_hyper,
    edge_list, kernel, anchor,
)
print(f"  canonical : {full:.12f}")
print(f"  interval  : {initial_interval:.12f}")
assert_close(initial_interval, full, label="full-window compatibility")
print("PASSED\n")

print("--- Test 2: conditional likelihood is additive across a split ---")
left = log_likelihood_interval(
    events, 0.0, split, mu, alpha_pairwise, alpha_hyper,
    edge_list, kernel, anchor,
)
right = log_likelihood_interval(
    events, split, T, mu, alpha_pairwise, alpha_hyper,
    edge_list, kernel, anchor,
)
print(f"  full       : {full:.12f}")
print(f"  left+right : {left + right:.12f}")
assert_close(left + right, full, tol=2e-10, label="interval additivity")
print("PASSED\n")

print("--- Test 3: pairwise excitation crosses the interval boundary analytically ---")
beta = 2.0
source_times = np.array([0.2, 1.4, 2.0])
exposure = pairwise_event_exposures(source_times, 1.0, 2.0, beta)
expected_first = (
    np.exp(-beta * (1.0 - 0.2)) - np.exp(-beta * (2.0 - 0.2))
) / beta
expected_second = (1.0 - np.exp(-beta * (2.0 - 1.4))) / beta
print(f"  exposure : {exposure}")
assert_close(exposure[0], expected_first, label="pre-interval source exposure")
assert_close(exposure[1], expected_second, label="within-interval source exposure")
assert_close(exposure[2], 0.0, label="end-boundary source exposure")

pair_events = [(0.2, 0)]
pair_ll = log_likelihood_interval(
    pair_events,
    1.0,
    2.0,
    mu=np.array([0.0]),
    alpha_pairwise=np.array([[0.7]]),
    alpha_hyper={},
    edge_list=[],
    kernel=ExponentialKernel(beta=beta),
    anchor_calc=HyperedgeAnchor(delta=0.2),
)
assert_close(pair_ll, -0.7 * expected_first, label="pairwise interval likelihood")
print("PASSED\n")

print("--- Test 4: a pre-interval hyperedge anchor retains residual exposure ---")
hyper_beta = 1.7
completion = 0.9
hyper_exposure = piecewise_compensator_interval(
    [completion], 1.0, 2.0, hyper_beta
)
expected_hyper = (
    np.exp(-hyper_beta * (1.0 - completion))
    - np.exp(-hyper_beta * (2.0 - completion))
) / hyper_beta
print(f"  exposure : {hyper_exposure:.12f}")
assert_close(hyper_exposure, expected_hyper, label="hyperedge residual exposure")

hyper_events = [(0.80, 0), (0.90, 1)]
hyper_ll = log_likelihood_interval(
    hyper_events,
    1.0,
    2.0,
    mu=np.zeros(2),
    alpha_pairwise=np.zeros((2, 2)),
    alpha_hyper={(0, 1): 0.6},
    edge_list=[(0, 1)],
    kernel=ExponentialKernel(beta=hyper_beta),
    anchor_calc=HyperedgeAnchor(delta=0.2),
)
assert_close(
    hyper_ll,
    -2.0 * 0.6 * expected_hyper,
    label="hyperedge interval likelihood",
)
print("PASSED\n")

print("=== All interval likelihood tests passed ===")
