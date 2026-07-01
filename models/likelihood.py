"""
Canonical log-likelihood evaluator for the HTH process.

This is the single source of truth for the model log-likelihood. Before
this module existed, the likelihood was hand-written inline inside a few
experiment files (exp7, exp10, exp2), each with its own integral
approximation. That made the numbers (a) inconsistent across experiments
and (b) inconsistent with the M-step, whose closed-form updates assume a
*closed-form* piecewise compensator while exp7 used a coarse 200-point
Riemann grid.

The log-likelihood of a multivariate point process on [0, T] is

    log L = sum_i log lambda_{n_i}(t_i)  -  int_0^T sum_n lambda_n(t) dt.

The integral (compensator) term decomposes exactly in closed form:

  * baseline:   sum_n mu_n * T
  * pairwise:   sum_j (sum_n alpha[node_j, n]) * (1/beta)(1 - e^{-beta (T - t_j)})
  * hyperedge:  sum_e |e| * alpha_e * C_e

where C_e is the *piecewise* compensator (each anchor active only until the
next pattern completion), identical to the one used in the M-step. The
hyperedge term carries the factor |e| because the most-recent anchor adds
alpha_e * phi(.) to the intensity of every member node of e.

Two integral methods are provided:
  * "closed_form" (default): the exact decomposition above. Consistent with
    the M-step, so EM is a genuine ascent on *this* objective.
  * "grid": replicates the legacy 200-point Riemann approximation used in the
    old inline exp7 code, for parity-checking the Phase-0 refactor only.
"""

import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor


def _intensity_at(t, node, events_before, mu, alpha_pairwise, alpha_hyper,
                  edge_list, kernel, anchor_calc, event_times_by_node):
    """Intensity of a single `node` at time `t`, given history `events_before`.

    Matches simulation.simulator._intensity and inference.e_step exactly:
    background + pairwise (all prior events) + hyperedge (most-recent anchor).
    """
    lam = mu[node]
    for t_j, node_j in events_before:
        if t_j >= t:
            break
        lam += alpha_pairwise[node_j, node] * float(kernel(np.array([t - t_j]))[0])
    for e in edge_list:
        if node not in e:
            continue
        a_e = alpha_hyper.get(e, 0.0)
        if a_e == 0.0:
            continue
        anchors = anchor_calc.find_anchors(e, event_times_by_node, t)
        if len(anchors) == 0:
            continue
        taus = np.array([t - t_a for t_a in anchors])
        lam += a_e * float(kernel(taus).sum())
    return lam


def _all_completion_times(edge, event_times_by_node, delta):
    """All times at which `edge` completes its pattern in [0, T].

    Identical logic to m_step.update_alpha_hyper.all_completion_times, so the
    compensator used by the likelihood and by the M-step are the same object.
    """
    completions = set()
    for anchor_node in edge:
        if anchor_node not in event_times_by_node:
            continue
        for t_last in event_times_by_node[anchor_node]:
            window_start = t_last - delta
            complete = True
            for v in edge:
                if v == anchor_node:
                    continue
                if v not in event_times_by_node:
                    complete = False
                    break
                in_window = [t for t in event_times_by_node[v]
                             if window_start <= t <= t_last]
                if len(in_window) == 0:
                    complete = False
                    break
            if complete:
                completions.add(t_last)
    return sorted(completions)


def _piecewise_compensator(completion_times, T, beta):
    """C_e: each anchor active only until the next completion (or T)."""
    if len(completion_times) == 0:
        return 0.0
    st = sorted(completion_times)
    total = 0.0
    for k in range(len(st)):
        t_start = st[k]
        t_end = st[k + 1] if k + 1 < len(st) else T
        if t_end > t_start:
            total += (1.0 / beta) * (1.0 - np.exp(-beta * (t_end - t_start)))
    return total


def log_likelihood(events, T, mu, alpha_pairwise, alpha_hyper, edge_list,
                   kernel, anchor_calc, integral_method="closed_form",
                   grid_points=200):
    """Total log-likelihood of `events` under the given parameters.

    Parameters
    ----------
    events          : list of (time, node), sorted by time
    T               : float, observation window
    mu              : (N,) baseline rates
    alpha_pairwise  : (N, N) array, alpha_pairwise[j, i] = j -> i
    alpha_hyper     : dict edge -> weight
    edge_list       : list of candidate edges
    kernel          : ExponentialKernel
    anchor_calc     : HyperedgeAnchor
    integral_method : "closed_form" (exact, M-step consistent) or
                      "grid" (legacy 200-pt Riemann, parity check only)

    Returns
    -------
    float : log L
    """
    mu = np.asarray(mu, dtype=float)
    alpha_pairwise = np.asarray(alpha_pairwise, dtype=float)
    n = len(events)
    beta = kernel.beta

    event_times_by_node = {}
    for t_j, node_j in events:
        event_times_by_node.setdefault(node_j, []).append(t_j)
    for k in event_times_by_node:
        event_times_by_node[k] = np.asarray(event_times_by_node[k], dtype=float)

    # --- sum of log-intensities at events (vectorised) ---
    # Intensity at event i equals the same quantity the E-step computes
    # (background + pairwise over strictly-earlier events + most-recent
    # hyperedge anchor); broadcasting it here is numerically identical to the
    # per-event loop but O(n^2) in NumPy rather than in Python.
    if n == 0:
        log_lam_sum = 0.0
    else:
        times = np.array([t for t, _ in events], dtype=float)
        nodes = np.array([node for _, node in events], dtype=int)

        dt = times[:, None] - times[None, :]          # dt[i, j] = t_i - t_j
        mask = dt > 0.0                                # strictly t_j < t_i
        P = alpha_pairwise[np.ix_(nodes, nodes)]       # P[i, j] = alpha[n_i, n_j]
        alpha_mat = P.T                                # [i, j] = alpha[n_j, n_i]
        safe_dt = np.where(mask, dt, 0.0)
        kern = np.where(mask, np.exp(-beta * safe_dt), 0.0)
        pair_sum = (alpha_mat * kern).sum(axis=1)      # (n,)

        lam = mu[nodes] + pair_sum                     # background + pairwise
        for e in edge_list:
            a_e = float(alpha_hyper.get(e, 0.0))
            if a_e == 0.0:
                continue
            comps = _all_completion_times(e, event_times_by_node,
                                          anchor_calc.delta)
            if len(comps) == 0:
                continue
            comps = np.asarray(comps, dtype=float)
            member = np.array([nodes[i] in e for i in range(n)], dtype=bool)
            idx = np.searchsorted(comps, times, side="left") - 1
            use = member & (idx >= 0)
            if np.any(use):
                tau = times[use] - comps[idx[use]]
                lam[use] += a_e * np.exp(-beta * tau)

        lam_arr = np.asarray(lam)
        bad_lam = lam_arr <= 0
        if np.any(bad_lam):
            raise FloatingPointError(
                f"Non-positive intensity encountered in E-step: "
                f"min(lambda)={float(np.min(lam_arr)):.3e}, "
                f"count={int(np.sum(bad_lam))}. "
                "Check baseline rates, excitation weights, and kernel values."
            )
        pos = lam > 0
        log_lam_sum = float(np.log(lam[pos]).sum())

    # --- compensator (integral) term ---
    if integral_method == "closed_form":
        beta = kernel.beta
        total_int = float(mu.sum()) * T
        for t_j, node_j in events:
            row_sum = float(alpha_pairwise[node_j, :].sum())
            total_int += row_sum * (1.0 / beta) * (1.0 - np.exp(-beta * (T - t_j)))
        for e in edge_list:
            a_e = alpha_hyper.get(e, 0.0)
            if a_e == 0.0:
                continue
            comps = _all_completion_times(e, event_times_by_node, anchor_calc.delta)
            C_e = _piecewise_compensator(comps, T, beta)
            total_int += len(e) * a_e * C_e

    elif integral_method == "grid":
        grid = np.linspace(0, T, grid_points)
        total_int = 0.0
        N = len(mu)
        for k in range(len(grid) - 1):
            t_mid = 0.5 * (grid[k] + grid[k + 1])
            hist = [(t, n) for t, n in events if t < t_mid]
            lam_sum = 0.0
            for node in range(N):
                lam_sum += _intensity_at(t_mid, node, hist, mu, alpha_pairwise,
                                         alpha_hyper, edge_list, kernel,
                                         anchor_calc, event_times_by_node)
            total_int += lam_sum * (grid[k + 1] - grid[k])
    else:
        raise ValueError(f"unknown integral_method: {integral_method!r}")

    return log_lam_sum - total_int