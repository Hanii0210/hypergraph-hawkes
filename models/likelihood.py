"""
Canonical likelihood evaluators for the hyperedge-triggered Hawkes process.

The full-window log-likelihood is

    log L_[0,T] = sum_i log lambda_{n_i}(t_i)
                  - integral_0^T sum_n lambda_n(t) dt.

For sample-split and blocked predictive evaluation we also need the conditional
likelihood on an interval (a, b] given the complete event history up to a:

    log L_(a,b] | H_a = sum_{i: a < t_i <= b} log lambda_{n_i}(t_i)
                        - integral_a^b sum_n lambda_n(t) dt.

The event stream passed to :func:`log_likelihood_interval` must contain the
history required to evaluate intensities in the target interval. Events after
``interval_end`` are harmless and ignored by the target score and compensator.

The closed-form compensator decomposes into:

* baseline exposure over the target duration;
* pairwise kernel exposure from every source event before the interval end,
  including residual excitation from events before the interval start;
* most-recent-anchor hyperedge exposure, integrating each completion only until
  the next completion and intersecting that activity interval with the target
  interval.

The same interval exposure helpers are reused by the M-step, keeping likelihood
and EM updates mathematically aligned.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _validate_interval(interval_start: float, interval_end: float) -> tuple[float, float]:
    """Validate and return a finite, non-empty observation interval."""
    a = float(interval_start)
    b = float(interval_end)
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("interval bounds must be finite")
    if b <= a:
        raise ValueError(
            f"interval_end must be greater than interval_start; got {a} >= {b}"
        )
    return a, b


def event_target_mask(
    times: np.ndarray,
    interval_start: float,
    interval_end: float,
) -> np.ndarray:
    """Return the event mask for the target interval.

    The conditional interval convention is ``(start, end]``.  At the origin we
    include time zero so that ``log_likelihood_interval(..., 0, T)`` is exactly
    backward compatible with the canonical full-window likelihood, even for a
    synthetic stream containing an event at zero.
    """
    a, b = _validate_interval(interval_start, interval_end)
    times = np.asarray(times, dtype=float)
    left = times >= a if a == 0.0 else times > a
    return left & (times <= b)


def event_times_by_node(events: Sequence[tuple[float, int]]) -> dict[int, np.ndarray]:
    """Build sorted per-node event-time arrays from a time-sorted event stream."""
    out: dict[int, list[float]] = {}
    for t, node in events:
        out.setdefault(int(node), []).append(float(t))
    return {node: np.asarray(ts, dtype=float) for node, ts in out.items()}


def all_completion_times(
    edge: tuple[int, ...],
    per_node_times: dict[int, np.ndarray],
    delta: float,
) -> np.ndarray:
    """Return all most-recent-anchor pattern completion times for ``edge``.

    A time ``t_last`` is a completion when the event at that time is from one
    member of the edge and every other member has at least one event in
    ``[t_last - delta, t_last]``.  Duplicate completion times are collapsed.
    """
    completions: set[float] = set()
    for anchor_node in edge:
        anchor_times = per_node_times.get(anchor_node)
        if anchor_times is None:
            continue
        for t_last in anchor_times:
            window_start = float(t_last) - float(delta)
            complete = True
            for member in edge:
                if member == anchor_node:
                    continue
                member_times = per_node_times.get(member)
                if member_times is None:
                    complete = False
                    break
                lo = np.searchsorted(member_times, window_start, side="left")
                hi = np.searchsorted(member_times, t_last, side="right")
                if hi <= lo:
                    complete = False
                    break
            if complete:
                completions.add(float(t_last))
    return np.asarray(sorted(completions), dtype=float)



# Backward-compatible private aliases used by the existing simulator-validity
# diagnostic and archived scripts.  New code should use the public helpers.
def _all_completion_times(edge, event_times_by_node, delta):
    return list(all_completion_times(edge, event_times_by_node, delta))


def _piecewise_compensator(completion_times, T, beta):
    return piecewise_compensator_interval(completion_times, 0.0, T, beta)

def exponential_interval_integral(
    anchor_time: float,
    interval_start: float,
    interval_end: float,
    beta: float,
) -> float:
    """Integrate ``exp(-beta * (t-anchor_time))`` on an interval intersection.

    The source is active only after ``anchor_time``.  Therefore the effective
    lower bound is ``max(interval_start, anchor_time)``.
    """
    a, b = _validate_interval(interval_start, interval_end)
    c = float(anchor_time)
    if c >= b:
        return 0.0
    lo = max(a, c)
    if b <= lo:
        return 0.0
    return float(
        (np.exp(-beta * (lo - c)) - np.exp(-beta * (b - c))) / beta
    )


def pairwise_event_exposures(
    source_times: np.ndarray,
    interval_start: float,
    interval_end: float,
    beta: float,
) -> np.ndarray:
    """Kernel exposure of each source event over the target interval.

    Events before the interval start retain their exponentially decayed residual
    contribution.  Events at or after the interval end have zero exposure.
    """
    a, b = _validate_interval(interval_start, interval_end)
    times = np.asarray(source_times, dtype=float)
    exposure = np.zeros(times.shape, dtype=float)
    active = times < b
    if not np.any(active):
        return exposure
    t = times[active]
    lo = np.maximum(a, t)
    exposure[active] = (
        np.exp(-beta * (lo - t)) - np.exp(-beta * (b - t))
    ) / beta
    return exposure


def piecewise_compensator_interval(
    completion_times: Iterable[float],
    interval_start: float,
    interval_end: float,
    beta: float,
) -> float:
    """Most-recent-anchor compensator restricted to a target interval.

    Completion ``c_k`` is active only until ``c_{k+1}``.  This function
    intersects each such activity interval with ``[interval_start,
    interval_end]``.  Consequently a completion before the target interval can
    contribute residual excitation inside it, while a completion inside the
    interval supersedes the previous anchor.
    """
    a, b = _validate_interval(interval_start, interval_end)
    completions = np.asarray(sorted(set(float(x) for x in completion_times)), dtype=float)
    if completions.size == 0:
        return 0.0

    total = 0.0
    for k, c in enumerate(completions):
        if c >= b:
            break
        next_c = completions[k + 1] if k + 1 < completions.size else b
        lo = max(a, float(c))
        hi = min(b, float(next_c))
        if hi <= lo:
            continue
        total += (
            np.exp(-beta * (lo - c)) - np.exp(-beta * (hi - c))
        ) / beta
    return float(total)


def log_likelihood_components(
    events, interval_start, interval_end, mu, alpha_pairwise, alpha_hyper,
    edge_list, kernel, anchor_calc, baseline_basis=None, include_start=None,
):
    """Exact conditional score, retaining its event and compensator parts."""
    from models.baseline import as_baseline
    from models.features import EventFeatures
    baseline = as_baseline(mu, baseline_basis)
    features = EventFeatures(events, baseline.n_nodes, edge_list, kernel.beta,
                             anchor_calc.delta, interval_start, interval_end,
                             baseline.basis, include_start)
    return features.score(baseline.coefficients, alpha_pairwise, alpha_hyper)


def log_likelihood_interval(
    events, interval_start, interval_end, mu, alpha_pairwise, alpha_hyper,
    edge_list, kernel, anchor_calc, integral_method="closed_form", grid_points=200,
    baseline_basis=None, include_start=None,
):
    """Conditional score on (start,end]; legacy origin includes zero by default.

    Pass include_start=False for the strictly right-closed revision protocol.
    Time-varying baselines are TemporalBaseline instances or coefficient arrays
    accompanied by baseline_basis. Parameters are never re-estimated here.
    """
    from models.baseline import as_baseline
    from models.features import EventFeatures
    a,b = _validate_interval(interval_start, interval_end)
    baseline = as_baseline(mu, baseline_basis)
    features = EventFeatures(events, baseline.n_nodes, edge_list, kernel.beta,
                             anchor_calc.delta, a, b, baseline.basis, include_start)
    parts = features.score(baseline.coefficients, alpha_pairwise, alpha_hyper)
    if integral_method == "closed_form":
        return parts["loglik"]
    if integral_method != "grid":
        raise ValueError(f"unknown integral_method: {integral_method!r}")
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    # Independent midpoint reference retained for small validation problems.
    per_node = event_times_by_node(events)
    completions = {e: all_completion_times(e, per_node, anchor_calc.delta) for e in features.edges}
    total = 0.0
    grid = np.linspace(a, b, int(grid_points))
    for left,right in zip(grid[:-1],grid[1:]):
        t = (left+right)/2
        value = float(baseline.values([t]).sum())
        for source_t, source_node in events:
            if source_t >= t:
                break
            value += float(np.asarray(alpha_pairwise)[source_node].sum())*np.exp(-kernel.beta*(t-source_t))
        for e,c in completions.items():
            idx = np.searchsorted(c,t,side="left")-1
            if idx >= 0:
                value += len(e)*alpha_hyper.get(e,0.0)*np.exp(-kernel.beta*(t-c[idx]))
        total += value*(right-left)
    return float(parts["event_log_intensity"]-total)


def log_likelihood(
    events, T, mu, alpha_pairwise, alpha_hyper, edge_list, kernel, anchor_calc,
    integral_method="closed_form", grid_points=200, baseline_basis=None,
    include_start=None,
):
    """Backward-compatible full-window likelihood."""
    return log_likelihood_interval(
        events, 0.0, T, mu, alpha_pairwise, alpha_hyper, edge_list, kernel,
        anchor_calc, integral_method, grid_points, baseline_basis, include_start,
    )
