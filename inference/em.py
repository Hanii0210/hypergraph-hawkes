"""Unified EM driver with optional conditional-interval convergence control."""

from __future__ import annotations

import numpy as np

from inference.e_step import EStep
from inference.m_step import MStep
from models.likelihood import log_likelihood_interval


class EMResult:
    """Container for fitted parameters and numerical diagnostics."""

    def __init__(
        self,
        mu,
        alpha_pairwise,
        alpha_hyper,
        loglik_history,
        objective_history,
        n_iter,
        converged=False,
        final_absolute_increment=None,
        final_relative_increment=None,
        ascent_violations=None,
        target_start=0.0,
        target_end=None,
        max_iter=None,
    ):
        self.mu = mu
        self.alpha_pairwise = alpha_pairwise
        self.alpha_hyper = alpha_hyper
        self.loglik_history = loglik_history
        self.objective_history = objective_history
        self.n_iter = int(n_iter)
        self.converged = bool(converged)
        self.final_absolute_increment = final_absolute_increment
        self.final_relative_increment = final_relative_increment
        self.ascent_violations = list(ascent_violations or [])
        self.target_start = float(target_start)
        self.target_end = None if target_end is None else float(target_end)
        self.max_iter = max_iter

    @property
    def final_loglik(self):
        return self.loglik_history[-1] if self.loglik_history else None

    @property
    def final_objective(self):
        return self.objective_history[-1] if self.objective_history else None


def run_em(
    events,
    T,
    n_nodes,
    edge_list,
    kernel,
    anchor_calc,
    mu0,
    alpha_pairwise0,
    alpha_hyper0,
    n_iter=80,
    lambda_l1=0.001,
    tensor=None,
    track_loglik=False,
    assert_ascent=False,
    ascent_tol=1e-6,
    integral_method="closed_form",
    verbose=False,
    hyper_update="als",
    target_start=None,
    target_end=None,
    max_iter=None,
    min_iter=10,
    convergence_tol=1e-7,
    consecutive_convergence=5,
    baseline_basis=None,
    pairwise_mask=None,
    positive_start=False,
    kkt_tol=None,
    max_total_iter=None,
    include_start=None,
):
    """Run full-window or conditional-interval EM.

    Backward compatibility
    ----------------------
    Existing calls that only pass ``n_iter`` retain fixed-iteration behaviour.
    Supplying ``max_iter`` activates numerical stopping: the penalised objective
    must have relative increment below ``convergence_tol`` for
    ``consecutive_convergence`` iterations after ``min_iter``.

    Interval mode
    -------------
    When ``target_start`` and ``target_end`` are supplied, the entire ``events``
    list is used as history, while only events in the target interval enter the
    conditional likelihood and sufficient-statistic numerators.
    """
    if hyper_update == "free" and integral_method == "closed_form":
        return _run_free_em(
            events, T, n_nodes, edge_list, kernel, anchor_calc, mu0,
            alpha_pairwise0, alpha_hyper0, n_iter, lambda_l1,
            track_loglik, assert_ascent, ascent_tol, verbose,
            target_start, target_end, max_iter, min_iter, convergence_tol,
            consecutive_convergence, baseline_basis, pairwise_mask,
            positive_start, kkt_tol, max_total_iter, include_start,
        )
    if baseline_basis is not None or pairwise_mask is not None or kkt_tol is not None or max_total_iter is not None or positive_start or include_start is not None:
        raise ValueError("new protocol options require free weights and closed-form integrals")

    from models.tensor_param import HypergraphTensor

    if tensor is None:
        tensor = HypergraphTensor(n_nodes=n_nodes, rank=3, seed=0)

    if (target_start is None) != (target_end is None):
        raise ValueError("target_start and target_end must be supplied together")
    interval_start = 0.0 if target_start is None else float(target_start)
    interval_end = float(T) if target_end is None else float(target_end)
    if interval_end <= interval_start:
        raise ValueError("target interval must have positive length")

    numerical_stopping = max_iter is not None
    iteration_limit = int(max_iter if numerical_stopping else n_iter)
    if iteration_limit <= 0:
        raise ValueError("iteration limit must be positive")
    if numerical_stopping:
        if min_iter < 1 or min_iter > iteration_limit:
            raise ValueError("min_iter must lie in [1, max_iter]")
        if convergence_tol <= 0.0:
            raise ValueError("convergence_tol must be positive")
        if consecutive_convergence < 1:
            raise ValueError("consecutive_convergence must be positive")

    estep = EStep(kernel, anchor_calc)
    mstep = MStep(n_nodes=n_nodes, tensor=tensor, lambda_l1=lambda_l1)

    mu = np.asarray(mu0, dtype=float).copy()
    alpha_pairwise = np.asarray(alpha_pairwise0, dtype=float).copy()
    alpha_hyper = {
        edge: float(alpha_hyper0.get(edge, 0.0)) for edge in edge_list
    }

    # Numerical stopping and ascent assertions require the objective even when
    # the caller did not request legacy trajectory tracking.
    need_objective = track_loglik or numerical_stopping or assert_ascent
    loglik_history = []
    objective_history = []
    ascent_violations = []
    previous_objective = None
    final_absolute_increment = None
    final_relative_increment = None
    stable_count = 0
    converged = False
    completed = 0

    for iteration in range(1, iteration_limit + 1):
        responsibilities = estep.compute(
            events,
            mu,
            alpha_pairwise,
            alpha_hyper,
            edge_list,
            target_start=interval_start,
            target_end=interval_end,
        )

        mu = mstep.update_mu(
            events,
            responsibilities["p_background"],
            T,
            target_start=interval_start,
            target_end=interval_end,
        )
        alpha_pairwise = mstep.update_alpha_pairwise(
            events,
            responsibilities["p_pairwise"],
            responsibilities["p_hyper"],
            edge_list,
            kernel,
            T,
            target_start=interval_start,
            target_end=interval_end,
        )

        if hyper_update == "free":
            alpha_hyper = mstep.update_alpha_hyper(
                events,
                responsibilities["p_hyper"],
                edge_list,
                anchor_calc,
                kernel,
                T,
                target_start=interval_start,
                target_end=interval_end,
            )
        elif hyper_update == "als":
            alpha_hyper = mstep.update_alpha_hyper_als(
                events,
                responsibilities["p_hyper"],
                edge_list,
                anchor_calc,
                kernel,
                T,
                target_start=interval_start,
                target_end=interval_end,
            )
        else:
            raise ValueError(f"unknown hyper_update: {hyper_update!r}")

        completed = iteration
        if need_objective:
            ll = log_likelihood_interval(
                events=events,
                interval_start=interval_start,
                interval_end=interval_end,
                mu=mu,
                alpha_pairwise=alpha_pairwise,
                alpha_hyper=alpha_hyper,
                edge_list=edge_list,
                kernel=kernel,
                anchor_calc=anchor_calc,
                integral_method=integral_method,
            )
            penalty = float(lambda_l1) * float(sum(alpha_hyper.values()))
            objective = ll - penalty
            loglik_history.append(ll)
            objective_history.append(objective)

            if previous_objective is not None:
                signed_increment = objective - previous_objective
                final_absolute_increment = abs(signed_increment)
                final_relative_increment = final_absolute_increment / (
                    1.0 + abs(previous_objective)
                )
                if signed_increment < -float(ascent_tol):
                    violation = {
                        "iteration": iteration,
                        "previous": previous_objective,
                        "current": objective,
                        "drop": -signed_increment,
                    }
                    ascent_violations.append(violation)
                    if assert_ascent:
                        raise AssertionError(
                            "EM penalised objective decreased at iteration "
                            f"{iteration}: {previous_objective:.9f} -> "
                            f"{objective:.9f} (drop {-signed_increment:.3e})"
                        )

                if numerical_stopping and iteration >= min_iter:
                    if final_relative_increment < convergence_tol:
                        stable_count += 1
                    else:
                        stable_count = 0
                    if stable_count >= consecutive_convergence:
                        converged = True
                        if verbose:
                            print(
                                f"  converged at iter {iteration}: "
                                f"relative increment={final_relative_increment:.3e}"
                            )
                        break

            previous_objective = objective
            if verbose:
                print(
                    f"  iter {iteration:>3}  logL={ll:.9f}  "
                    f"objective={objective:.9f}"
                )

    return EMResult(
        mu=mu,
        alpha_pairwise=alpha_pairwise,
        alpha_hyper=alpha_hyper,
        loglik_history=loglik_history,
        objective_history=objective_history,
        n_iter=completed,
        converged=converged,
        final_absolute_increment=final_absolute_increment,
        final_relative_increment=final_relative_increment,
        ascent_violations=ascent_violations,
        target_start=interval_start,
        target_end=interval_end,
        max_iter=iteration_limit,
    )


def _run_free_em(
    events, T, n_nodes, edge_list, kernel, anchor_calc, mu0,
    alpha_pairwise0, alpha_hyper0, n_iter, lambda_l1,
    track_loglik, assert_ascent, ascent_tol, verbose,
    target_start, target_end, max_iter, min_iter, convergence_tol,
    consecutive_convergence, baseline_basis, pairwise_mask,
    positive_start, kkt_tol, max_total_iter, include_start,
):
    """Exact free-weight EM using cached source and completion features."""
    from models.baseline import BaselineBasis, TemporalBaseline
    from models.features import EventFeatures
    from models.diagnostics import stability_diagnostics
    from time import perf_counter
    started = perf_counter()
    if (target_start is None) != (target_end is None):
        raise ValueError("both interval bounds are required")
    a = 0.0 if target_start is None else float(target_start)
    b = float(T) if target_end is None else float(target_end)
    numerical = max_iter is not None
    first_limit = int(max_iter if numerical else n_iter)
    total_limit = int(max_total_iter if max_total_iter is not None else first_limit)
    if first_limit < 1 or total_limit < first_limit or (max_total_iter is not None and not numerical):
        raise ValueError("invalid iteration budget")
    if numerical and (min_iter < 1 or min_iter > first_limit or convergence_tol <= 0 or consecutive_convergence < 1):
        raise ValueError("invalid stopping criteria")
    if not np.isfinite(lambda_l1) or lambda_l1 < 0 or (kkt_tol is not None and kkt_tol <= 0):
        raise ValueError("invalid penalty or KKT tolerance")
    if isinstance(mu0, TemporalBaseline):
        basis = mu0.basis
        if baseline_basis is not None and basis != baseline_basis:
            raise ValueError("conflicting initial baseline")
        background = mu0.coefficients.copy()
    else:
        basis = BaselineBasis() if baseline_basis is None else baseline_basis
        initial = np.asarray(mu0, dtype=float)
        background = basis.constant_coefficients(initial) if initial.ndim == 1 else initial.copy()
    features = EventFeatures(events, n_nodes, edge_list, kernel.beta, anchor_calc.delta, a, b, basis, include_start)
    pair = np.asarray(alpha_pairwise0, dtype=float).copy()
    hyper = {e: float(alpha_hyper0.get(e, 0.0)) for e in features.edges}
    features.weights(background, pair, hyper)  # Validate before clipping.
    mask = np.ones_like(pair,dtype=bool) if pairwise_mask is None else np.asarray(pairwise_mask,dtype=bool)
    if mask.shape != pair.shape:
        raise ValueError("pairwise mask shape mismatch")
    lower = basis.lower_bounds(n_nodes)
    background = np.maximum(background, lower)
    if positive_start:
        pair[mask] = np.maximum(pair[mask],1e-8)
        hyper = {e:max(weight,1e-8) for e,weight in hyper.items()}
    pair[~mask] = 0.0
    preparation_seconds = perf_counter()-started
    db = features.background_exposure
    da = features.pair_exposure[:,None]
    dh = features.hyper_exposure + lambda_l1
    identifiable_hyper = features.hyper_exposure > 1e-12
    need_history = track_loglik or numerical or assert_ascent
    ll_history, obj_history, violations, checkpoints = [],[],[],[]
    previous = features.score(background,pair,hyper)["loglik"] - lambda_l1*sum(hyper.values())
    stable, converged = 0,False
    abs_increment = rel_increment = None

    def residual(stats):
        def value(theta, lo, gradient, denominator, allowed=None):
            scaled = np.where(theta-lo > 1e-8, np.abs(gradient), np.maximum(gradient,0))/(1+denominator)
            if allowed is not None:
                scaled = scaled[allowed]
            return float(np.max(scaled,initial=0.0))
        hb = np.array([hyper[e] for e in features.edges])
        return max(value(background,lower,stats["background_score"],db),
                   value(pair,0,stats["pairwise_score"],da,mask),
                   value(hb,0,stats["hyperedge_score"]-lambda_l1,dh))

    for iteration in range(1,total_limit+1):
        stats = EStep.compute_sufficient(features,background,pair,hyper)
        background = np.maximum(np.divide(stats["background_counts"],db,
            out=np.zeros_like(background),where=db>0),lower)
        pair = np.divide(stats["pairwise_counts"],da,out=np.zeros_like(pair),where=da>0)
        pair[~mask] = 0.0
        h = np.divide(stats["hyperedge_counts"],dh,out=np.zeros_like(dh),where=dh>1e-12)
        h[~identifiable_hyper] = 0.0
        hyper = dict(zip(features.edges,map(float,h)))
        ll = features.score(background,pair,hyper)["loglik"]
        objective = ll-lambda_l1*float(h.sum())
        signed = objective-previous
        abs_increment = abs(signed)
        rel_increment = abs_increment/(1+abs(previous))
        if need_history:
            ll_history.append(ll)
            obj_history.append(objective)
        if signed < -ascent_tol:
            violations.append({"iteration":iteration,"previous":previous,"current":objective,"drop":-signed})
            if assert_ascent:
                raise AssertionError(f"EM objective decreased at iteration {iteration}: {signed}")
        previous = objective
        if numerical:
            stable = stable+1 if iteration >= min_iter and rel_increment < convergence_tol else 0
            if stable >= consecutive_convergence:
                kkt = residual(features.statistics(background,pair,hyper))
                if (kkt_tol is None or kkt <= kkt_tol) and not violations:
                    converged = True
        if iteration == first_limit:
            checkpoints.append({"iteration":iteration,"objective":objective,"converged":converged})
        if verbose:
            print(f"iter {iteration:4d} logL={ll:.9f} relative={rel_increment:.3e}")
        if converged:
            break
    kkt = residual(features.statistics(background,pair,hyper))
    result = EMResult(
        mu=background[:,0].copy() if basis.kind == "constant" else background.copy(),
        alpha_pairwise=pair,alpha_hyper=hyper,loglik_history=ll_history,
        objective_history=obj_history,n_iter=iteration,converged=converged,
        final_absolute_increment=abs_increment,final_relative_increment=rel_increment,
        ascent_violations=violations,target_start=a,target_end=b,max_iter=total_limit,
    )
    result.baseline = TemporalBaseline(background,basis)
    result.baseline_basis = basis
    result.baseline_coefficients = background.copy()
    result.kkt_residual = kkt
    result.kkt_tolerance = kkt_tol
    result.identifiable_hyperedges = dict(zip(features.edges,map(bool,identifiable_hyper)))
    result.zero_pairwise_exposure = np.flatnonzero(features.pair_exposure <= 0).tolist()
    result.stability = stability_diagnostics(result.baseline,pair,hyper,kernel.beta)
    result.checkpoints = checkpoints
    result.feature_bytes = features.nbytes
    result.preparation_seconds = preparation_seconds
    result.elapsed_seconds = perf_counter()-started
    result.score_components = features.score(background,pair,hyper)
    result.pairwise_mask = mask.copy()
    return result
