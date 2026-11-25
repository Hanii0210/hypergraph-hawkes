"""Coordinate maximisation of the same nonnegative fixed-feature likelihood.

For coordinate v with offset c_k, the derivative is
    sum_k x_k / (c_k + v*x_k) - d.
It is decreasing. A nonpositive derivative at the lower bound makes that
bound a coordinate maximiser; otherwise a safeguarded Newton solve finds
the positive root. All coordinates remain eligible on subsequent sweeps.
This is an optimisation refinement, not a variable-selection rule or EM step.
"""
from time import perf_counter
import numpy as np


def _coordinate_maximum(offset, feature, exposure, lower, current):
    if exposure <= 0:
        if np.any(feature):
            raise ValueError("positive event feature with zero exposure")
        return lower
    if not len(feature):
        return lower
    low_intensity = offset + lower*feature
    if np.any(low_intensity <= 0):
        raise FloatingPointError("baseline lower bounds must keep event intensities positive")
    derivative = float(np.sum(feature/low_intensity)-exposure)
    if derivative <= 0:
        return lower
    # Offset is nonnegative, so sum x/(offset+v*x) <= n/v.
    left, right = float(lower), float(lower+len(feature)/exposure)
    value = float(np.clip(current, left, right))
    if value <= left or value >= right:
        value = (left+right)/2
    for _ in range(100):
        ratio = feature/(offset+value*feature)
        gradient = float(ratio.sum()-exposure)
        if abs(gradient)/(1+exposure) <= 1e-10:
            return value
        if gradient > 0:
            left = value
        else:
            right = value
        curvature = float(ratio @ ratio)
        proposal = value+gradient/curvature if curvature > 0 else np.nan
        if not np.isfinite(proposal) or not left < proposal < right:
            proposal = (left+right)/2
        if proposal == value or right-left <= 1e-14*(1+abs(value)):
            return proposal
        value = proposal
    raise RuntimeError("coordinate root did not converge within its bracket")


def refine_likelihood(features, coefficients, pairwise, hyper, pairwise_mask, *,
                      lambda_l1=0.0, max_sweeps=200, kkt_tol=1e-4,
                      convergence_tol=1e-7, consecutive_convergence=5, ascent_tol=1e-6):
    """Refine an EM endpoint without changing features, objective or KKT rule."""
    started = perf_counter()
    f = features
    b, a, h = f.weights(coefficients, pairwise, hyper)
    mask = np.asarray(pairwise_mask, dtype=bool)
    if mask.shape != a.shape or np.any(a[~mask] != 0):
        raise ValueError("pairwise structural zeros must match the supplied mask")
    if max_sweeps < 1 or kkt_tol <= 0 or lambda_l1 < 0 or consecutive_convergence < 1:
        raise ValueError("invalid refinement settings")
    columns, theta, lower, exposure, labels = [], [], [], [], []
    for node in range(f.n_nodes):
        for column in range(f.basis.size):
            columns.append(f.background[:, column]*(f.nodes == node))
            theta.append(b[node, column])
            lower.append(f.basis.lower_bounds(f.n_nodes)[node, column])
            exposure.append(f.background_exposure[column])
            labels.append(("baseline", node, column))
    for source, target in zip(*np.nonzero(mask)):
        columns.append(f.pair[:, source]*(f.nodes == target))
        theta.append(a[source, target])
        lower.append(0.0)
        exposure.append(f.pair_exposure[source])
        labels.append(("pairwise", source, target))
    for index in range(len(f.edges)):
        columns.append(f.hyper[:, index])
        theta.append(h[index])
        lower.append(0.0)
        exposure.append(f.hyper_exposure[index]+lambda_l1)
        labels.append(("hyperedge", index, 0))
    design = np.column_stack(columns)
    theta, lower, exposure = map(lambda x: np.asarray(x, dtype=float), (theta, lower, exposure))
    if np.any(theta < lower):
        raise ValueError("starting point is below a coefficient bound")
    support = [(np.flatnonzero(x), x[x != 0]) for x in columns]
    lam = design @ theta

    def objective(intensity, values):
        if not np.isfinite(intensity).all() or np.any(intensity <= 0):
            raise FloatingPointError("invalid intensity during refinement")
        return float(np.log(intensity).sum()-exposure @ values)

    def residual(intensity, values):
        gradient = design.T @ (1.0/intensity)-exposure
        scaled = np.where(values-lower > 1e-8, np.abs(gradient), np.maximum(gradient, 0))/(1+exposure)
        return float(scaled.max(initial=0.0))

    initial_objective = previous = objective(lam, theta)
    objective_history, kkt_history, relative_history = [], [], []
    stable, converged = 0, False
    bound_hits = reentries = 0
    for sweep in range(1, int(max_sweeps)+1):
        for index, (active, values) in enumerate(support):
            old = theta[index]
            # Recompute the offset from the other contributions if subtraction
            # loses nonnegativity through roundoff. The baseline floor remains.
            offset = lam[active]-old*values
            if np.any(offset < 0) or np.any(offset+lower[index]*values <= 0):
                other = theta.copy()
                other[index] = 0
                offset = design[active] @ other
            new = _coordinate_maximum(offset, values, exposure[index], lower[index], old)
            bound_hits += int(new == lower[index] and old > lower[index])
            reentries += int(new > lower[index] and old == lower[index])
            theta[index] = new
            lam[active] = offset+new*values
        # Independent full reconstruction avoids accumulating update error.
        lam = design @ theta
        current = objective(lam, theta)
        increment = current-previous
        if increment < -ascent_tol:
            raise AssertionError(f"coordinate objective decreased by {-increment:g}")
        relative = abs(increment)/(1+abs(previous))
        kkt = residual(lam, theta)
        objective_history.append(current)
        kkt_history.append(kkt)
        relative_history.append(relative)
        stable = stable+1 if relative < convergence_tol else 0
        previous = current
        if stable >= consecutive_convergence and kkt <= kkt_tol:
            converged = True
            break
    background, pair, weights = b.copy(), a.copy(), h.copy()
    for value, (kind, first, second) in zip(theta, labels):
        if kind == "baseline":
            background[first, second] = value
        elif kind == "pairwise":
            pair[first, second] = value
        else:
            weights[first] = value
    refined_hyper = dict(zip(f.edges, map(float, weights)))
    score = f.score(background, pair, refined_hyper)
    if not np.isclose(score["loglik"]-lambda_l1*weights.sum(), previous, atol=1e-7, rtol=1e-10):
        raise AssertionError("refinement objective and model score disagree")
    return {"baseline_coefficients": background, "alpha_pairwise": pair,
            "alpha_hyper": refined_hyper, "converged": converged,
            "n_sweeps": sweep, "kkt_residual": kkt,
            "initial_objective": initial_objective, "final_objective": previous,
            "objective_history": objective_history, "kkt_history": kkt_history,
            "relative_increment_history": relative_history,
            "bound_hits": bound_hits, "bound_reentries": reentries,
            "training_score": score, "design_bytes": design.nbytes,
            "elapsed_seconds": perf_counter()-started}
