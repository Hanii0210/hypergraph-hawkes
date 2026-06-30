"""
Unified EM driver for the HTH process.

Before this module, the EM loop was copy-pasted into all 11 experiment
files, each followed by a redundant (and, for node-sharing hyperedges,
buggy) block that back-filled the CP factor matrix F from the freely
solved alpha_e. Centralising the loop here means every later fix
(real ALS M-step, ascent monitoring, held-out evaluation) lands in one
place instead of eleven.

The driver also tracks the canonical log-likelihood (models.likelihood)
every iteration and can assert monotone ascent. This is the empirical
backbone for the convergence claim: with the closed-form integral
consistent with the M-step, a correct (generalized) EM step must not
decrease the objective.

Phase 0 contract: with `track_loglik=False` and the default M-step calls,
this driver reproduces the legacy inline loops bit-for-bit. The legacy
`target_factor` F back-fill is intentionally NOT carried over, because the
E-step never reads F (it reads the alpha_hyper dict), so dropping it
changes no inference output. It will be replaced by a real ALS update in
Phase 1.
"""

import numpy as np
from inference.e_step import EStep
from inference.m_step import MStep
from models.likelihood import log_likelihood


class EMResult:
    def __init__(self, mu, alpha_pairwise, alpha_hyper, loglik_history,
                 n_iter):
        self.mu = mu
        self.alpha_pairwise = alpha_pairwise
        self.alpha_hyper = alpha_hyper
        self.loglik_history = loglik_history   # list (possibly empty)
        self.n_iter = n_iter

    @property
    def final_loglik(self):
        return self.loglik_history[-1] if self.loglik_history else None

def run_em(events, T, n_nodes, edge_list, kernel, anchor_calc,
           mu0, alpha_pairwise0, alpha_hyper0,
           n_iter=80, lambda_l1=0.001, tensor=None,
           track_loglik=False, assert_ascent=False, ascent_tol=1e-6,
           integral_method="closed_form", verbose=False,
           hyper_update="als"):
    """Run EM to convergence (fixed iteration count).

    Parameters
    ----------
    events, T, n_nodes, edge_list, kernel, anchor_calc : model + data
    mu0, alpha_pairwise0, alpha_hyper0 : initial parameters
    n_iter      : number of EM iterations
    lambda_l1   : L1 penalty on hyperedge weights
    tensor      : HypergraphTensor (created internally if None)
    track_loglik: compute canonical log-likelihood each iteration
    assert_ascent: raise if log-likelihood ever decreases by > ascent_tol
                   (requires track_loglik=True)
    integral_method: passed to log_likelihood ("closed_form" or "grid")
    hyper_update: "free"  -> legacy independent closed-form alpha_e (Phase 0)
                  "als"   -> real ALS over CP factor F (Phase 1, once added)

    Returns
    -------
    EMResult
    """
    from models.tensor_param import HypergraphTensor
    if tensor is None:
        tensor = HypergraphTensor(n_nodes=n_nodes, rank=3, seed=0)

    estep = EStep(kernel, anchor_calc)
    mstep = MStep(n_nodes=n_nodes, tensor=tensor, lambda_l1=lambda_l1)

    mu = np.array(mu0, dtype=float)
    alpha_pairwise = np.array(alpha_pairwise0, dtype=float)
    alpha_hyper = {e: float(alpha_hyper0.get(e, 0.0)) for e in edge_list}

    loglik_history = []
    prev_ll = None

    for it in range(1, n_iter + 1):
        result = estep.compute(events, mu, alpha_pairwise, alpha_hyper, edge_list)

        mu = mstep.update_mu(events, result["p_background"], T)
        alpha_pairwise = mstep.update_alpha_pairwise(
            events, result["p_pairwise"], result["p_hyper"],
            edge_list, kernel, T)

        if hyper_update == "free":
            alpha_hyper = mstep.update_alpha_hyper(
                events, result["p_hyper"], edge_list, anchor_calc, kernel, T)
        elif hyper_update == "als":
            # Phase 1: real ALS over CP factor matrix. Implemented in m_step
            # as update_alpha_hyper_als; falls back with a clear error until then.
            if not hasattr(mstep, "update_alpha_hyper_als"):
                raise NotImplementedError(
                    "hyper_update='als' requires MStep.update_alpha_hyper_als "
                    "(added in Phase 1).")
            alpha_hyper = mstep.update_alpha_hyper_als(
                events, result["p_hyper"], edge_list, anchor_calc, kernel, T)
        else:
            raise ValueError(f"unknown hyper_update: {hyper_update!r}")

        if track_loglik:
            ll = log_likelihood(events, T, mu, alpha_pairwise, alpha_hyper,
                                edge_list, kernel, anchor_calc,
                                integral_method=integral_method)
            loglik_history.append(ll)
            if assert_ascent and prev_ll is not None:
                if ll < prev_ll - ascent_tol:
                    raise AssertionError(
                        f"EM log-likelihood decreased at iter {it}: "
                        f"{prev_ll:.6f} -> {ll:.6f} (drop {prev_ll - ll:.2e})")
            prev_ll = ll
            if verbose:
                print(f"  iter {it:>3}  logL = {ll:.6f}")

    return EMResult(mu, alpha_pairwise, alpha_hyper, loglik_history, n_iter)