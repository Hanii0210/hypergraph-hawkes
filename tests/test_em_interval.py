import sys
sys.path.insert(0, ".")

import numpy as np

from inference.e_step import EStep
from inference.em import run_em
from inference.m_step import MStep
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.likelihood import (
    all_completion_times,
    event_times_by_node,
    piecewise_compensator_interval,
)
from models.tensor_param import HypergraphTensor


kernel = ExponentialKernel(beta=1.0)
anchor = HyperedgeAnchor(delta=0.45)
edge = (0, 1)
edge_list = [edge]
events = [
    (0.10, 0),
    (0.30, 1),
    (0.72, 0),
    (0.95, 1),
    (1.10, 0),
    (1.28, 1),
    (1.55, 0),
    (1.82, 1),
    (2.10, 0),
    (2.42, 1),
    (2.70, 0),
    (2.88, 1),
]
T = 3.0
start = 1.0
end = 3.0
mu0 = np.array([0.35, 0.30])
ap0 = np.array([[0.12, 0.18], [0.16, 0.10]])
ah0 = {edge: 0.20}

print("--- Test 1: interval E-step uses history but zeros non-target rows ---")
estep = EStep(kernel, anchor)
result = estep.compute(
    events, mu0, ap0, ah0, edge_list,
    target_start=start, target_end=end,
)
times = np.array([t for t, _ in events])
target = (times > start) & (times <= end)
assert np.array_equal(result["target_mask"], target)
assert np.all(result["p_background"][~target] == 0.0)
assert np.all(result["p_pairwise"][~target, :] == 0.0)
assert np.all(result["p_hyper"][edge][~target] == 0.0)
for i in np.flatnonzero(target):
    total = (
        result["p_background"][i]
        + result["p_pairwise"][i].sum()
        + result["p_hyper"][edge][i]
    )
    if not np.isclose(total, 1.0, atol=1e-12):
        raise AssertionError(f"target responsibility row {i} sums to {total}")
# First target event can use parents and an anchor from before target_start.
first_target = int(np.flatnonzero(target)[0])
assert result["p_pairwise"][first_target, :first_target].sum() > 0.0
assert result["p_hyper"][edge][first_target] > 0.0
print("PASSED\n")

print("--- Test 2: interval free-hyperedge M-step equals A/L exactly ---")
tensor = HypergraphTensor(n_nodes=2, rank=2, seed=3)
mstep = MStep(n_nodes=2, tensor=tensor, lambda_l1=0.0)
manual_p_hyper = {edge: np.zeros(len(events))}
manual_p_hyper[edge][target] = np.linspace(0.05, 0.25, target.sum())
updated = mstep.update_alpha_hyper(
    events,
    manual_p_hyper,
    edge_list,
    anchor,
    kernel,
    T,
    target_start=start,
    target_end=end,
)
completions = all_completion_times(
    edge, event_times_by_node(events), anchor.delta
)
C = piecewise_compensator_interval(completions, start, end, kernel.beta)
expected = manual_p_hyper[edge][target].sum() / (len(edge) * C)
print(f"  update   : {updated[edge]:.12f}")
print(f"  expected : {expected:.12f}")
assert np.isclose(updated[edge], expected, atol=1e-12, rtol=1e-12)
print("PASSED\n")

print("--- Test 3: interval pairwise M-step uses residual source exposure ---")
pair_events = [(0.20, 0), (1.40, 1)]
p_pair = np.zeros((2, 2))
p_pair[1, 0] = 0.7
pair_update = mstep.update_alpha_pairwise(
    pair_events,
    p_pair,
    {},
    [],
    ExponentialKernel(beta=2.0),
    T=2.0,
    target_start=1.0,
    target_end=2.0,
)
source_exposure = (
    np.exp(-2.0 * (1.0 - 0.20)) - np.exp(-2.0 * (2.0 - 0.20))
) / 2.0
expected_pair = 0.7 / source_exposure
print(f"  update   : {pair_update[0, 1]:.12f}")
print(f"  expected : {expected_pair:.12f}")
assert np.isclose(pair_update[0, 1], expected_pair, atol=1e-12, rtol=1e-12)
print("PASSED\n")

print("--- Test 4: conditional-interval EM is monotone and numerically converges ---")
fit = run_em(
    events=events,
    T=T,
    n_nodes=2,
    edge_list=edge_list,
    kernel=kernel,
    anchor_calc=anchor,
    mu0=mu0,
    alpha_pairwise0=ap0,
    alpha_hyper0=ah0,
    lambda_l1=0.0,
    hyper_update="free",
    target_start=start,
    target_end=end,
    max_iter=700,
    min_iter=8,
    convergence_tol=1e-8,
    consecutive_convergence=4,
    track_loglik=True,
    assert_ascent=True,
    ascent_tol=2e-8,
)
objective = np.asarray(fit.objective_history)
increments = np.diff(objective)
print(f"  converged : {fit.converged}")
print(f"  n_iter    : {fit.n_iter}")
print(f"  final rel : {fit.final_relative_increment:.3e}")
print(f"  min step  : {increments.min():.3e}")
assert fit.converged
assert fit.n_iter <= 700
assert fit.final_relative_increment < 1e-8
assert np.all(increments >= -2e-8)
assert fit.ascent_violations == []
print("PASSED\n")

print("--- Test 5: legacy fixed-iteration mode remains available ---")
legacy = run_em(
    events=events,
    T=T,
    n_nodes=2,
    edge_list=edge_list,
    kernel=kernel,
    anchor_calc=anchor,
    mu0=mu0,
    alpha_pairwise0=ap0,
    alpha_hyper0=ah0,
    n_iter=3,
    lambda_l1=0.0,
    hyper_update="free",
    track_loglik=True,
)
assert legacy.n_iter == 3
assert not legacy.converged
assert len(legacy.loglik_history) == 3
print("PASSED\n")

print("=== All interval EM tests passed ===")
