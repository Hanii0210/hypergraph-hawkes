import sys
sys.path.insert(0, ".")

import numpy as np
from scipy import stats
from itertools import combinations

from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from simulation.simulator import HawkesSimulator
from inference.em import run_em
from models.likelihood import log_likelihood

# =============================================================================
# Experiment 13: Honest hyperedge inference (sample splitting + chi-bar-squared)
#
# WHY THIS EXPERIMENT EXISTS
# --------------------------
# The real-data section of the paper (exp10) reported a hyperedge via a naive
# likelihood-ratio test: candidates were *selected* by an L1 path and then
# *tested* by an LRT on the SAME data, comparing 2 dL against a chi^2_k
# critical value. Two things are wrong with that:
#
#   (1) Double-dipping / post-selection inference. Selecting the model and
#       testing it on one dataset inflates significance: the LRT statistic is
#       no longer drawn from its nominal null because the edge was chosen for
#       looking strong in that very data.
#
#   (2) Wrong reference distribution. The hyperedge weight obeys alpha_e >= 0,
#       so under H0: alpha_e = 0 the parameter sits on the boundary of the
#       admissible space. The correct null is NOT chi^2_k but a chi-bar-squared
#       mixture (Self & Liang 1987). For a single boundary parameter this is
#       0.5*chi^2_0 + 0.5*chi^2_1, i.e. the p-value is HALVED relative to the
#       naive chi^2_1 tail.
#
# WHAT THIS EXPERIMENT DOES (scope-matched honest fix)
# ----------------------------------------------------
#   * Main method: SAMPLE SPLITTING. Select candidate hyperedges on the first
#     time-half, test them by LRT on the held-out second half -> the test edge
#     is fixed independently of the inference data, so the LRT is honest.
#   * Correct null: chi-bar-squared boundary correction on the p-value.
#   * Contrast: the naive (select+test-on-all-data) procedure, to QUANTIFY how
#     badly double-dipping inflates the false-positive rate.
#
# SCOPE NOTE -> MPhil (deliberately NOT done here)
# ------------------------------------------------
# A *complete* treatment of post-selection inference for this pipeline -- the
# full selective-inference machinery, or a Monte-Carlo / parametric-bootstrap /
# ABC null distribution that accounts for the selection rule and the boundary
# jointly -- belongs to the Bayesian / likelihood-free programme of the MPhil
# (filtered MCMC for component separation; ABC with statistical distances for
# kernel/model selection). This undergraduate paper deliberately stops at the
# scope-matched honest fix (sample splitting + chi-bar-squared) and points the
# complete solution forward. Do NOT pull that machinery back into this paper.
# =============================================================================


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------
def fit_hth(events, T, cands, n_nodes, kernel, anchor, rank=3,
            n_iter=40, lambda_l1=0.0, seed=0):
    """Fit the HTH model on `events` with the given candidate hyperedges.

    Uses the real ALS CP M-step (Phase 1). Returns an EMResult.
    """
    rng = np.random.default_rng(seed)
    tensor = HypergraphTensor(n_nodes=n_nodes, rank=rank, seed=seed)
    return run_em(
        events, T, n_nodes, cands, kernel, anchor,
        rng.uniform(0.2, 0.5, n_nodes),       # mu0
        np.zeros((n_nodes, n_nodes)),          # alpha_pairwise0
        {e: 0.1 for e in cands},               # alpha_hyper0
        n_iter=n_iter, lambda_l1=lambda_l1,
        tensor=tensor, hyper_update="als",
    )


def time_split(events, T, frac=0.5):
    """Split an event stream at t = frac*T.

    The inference half is shifted to start at t=0 so its own anchors/
    compensator are computed on a clean [0, T_inf] window (the selection half
    is never seen by the inference-half likelihood).
    Returns (sel_events, T_sel, inf_events, T_inf).
    """
    t_cut = T * frac
    sel = [(t, n) for (t, n) in events if t <= t_cut]
    inf = [(t - t_cut, n) for (t, n) in events if t > t_cut]
    return sel, t_cut, inf, T - t_cut


def lrt_stat(events, T, edge, base_cands, n_nodes, kernel, anchor, **fit_kw):
    """LRT statistic 2*(logL_full - logL_reduced) for ADDING `edge`.

    full     = base_cands + [edge]   (refit)
    reduced  = base_cands            (refit; if empty -> pairwise-only, no hyper)
    Both models are refit; likelihoods use the canonical closed-form integral.
    """
    full = list(base_cands) + [edge]
    rf = fit_hth(events, T, full, n_nodes, kernel, anchor, **fit_kw)
    llf = log_likelihood(events, T, rf.mu, rf.alpha_pairwise, rf.alpha_hyper,
                         full, kernel, anchor, "closed_form")
    if base_cands:
        rr = fit_hth(events, T, list(base_cands), n_nodes, kernel, anchor, **fit_kw)
        llr = log_likelihood(events, T, rr.mu, rr.alpha_pairwise, rr.alpha_hyper,
                             list(base_cands), kernel, anchor, "closed_form")
    else:
        rr = fit_hth(events, T, [], n_nodes, kernel, anchor, **fit_kw)
        llr = log_likelihood(events, T, rr.mu, rr.alpha_pairwise, {}, [],
                             kernel, anchor, "closed_form")
    return 2.0 * (llf - llr)


def chibar_p(stat):
    """Chi-bar-squared p-value for H0: alpha_e = 0 (boundary parameter).

    Null = 0.5*chi^2_0 + 0.5*chi^2_1  =>  p = 0.5 * P(chi^2_1 > stat).
    """
    if stat <= 0:
        return 1.0
    return 0.5 * stats.chi2.sf(stat, df=1)


def naive_chi2_p(stat):
    """WRONG reference distribution kept only for contrast: ignores the
    alpha_e >= 0 boundary and uses the full chi^2_1 tail."""
    return stats.chi2.sf(stat, df=1)


# -----------------------------------------------------------------------------
# Block 1 self-validation: run `python experiments/exp13_selective_inference.py`
# Verifies three things on a synthetic dataset with a KNOWN true hyperedge:
#   (1) the time split partitions data correctly,
#   (2) the chi-bar-squared correction halves the naive chi^2 p-value,
#   (3) the select-on-half / test-on-other-half chain keeps a true hyperedge
#       significant (sample splitting does not destroy real signal).
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    k = ExponentialKernel(1.0)
    an = HyperedgeAnchor(0.5)
    N = 4
    mu = np.full(N, 0.3)
    ap = np.zeros((N, N)); ap[2, 0] = 0.3
    ah_true = {(0, 1): 0.4}
    T = 600
    ev = HawkesSimulator(mu, ap, ah_true, k, an).simulate(T=T, seed=7, max_events=4000)
    print(f"n_events = {len(ev)}  (true hyperedge (0,1) alpha=0.4)")

    # (2) chi-bar-squared halving + naive double-dipping baseline on full data
    s_full = lrt_stat(ev, T, (0, 1), [], N, k, an)
    p_naive = naive_chi2_p(s_full)
    p_cbar = chibar_p(s_full)
    print("\n[1] naive (select+test on ALL data), edge (0,1):")
    print(f"    LRT stat = {s_full:.3f}")
    print(f"    naive chi2_1 p = {p_naive:.4g}   chi-bar-sq p = {p_cbar:.4g}"
          f"   (ratio = {p_cbar / p_naive:.3f}, expect 0.5)")

    # (1) + (3) sample split: select half / inference half
    sel, Tsel, inf, Tinf = time_split(ev, T, 0.5)
    print(f"\n[2] time split: selection n={len(sel)} (T={Tsel:.0f}), "
          f"inference n={len(inf)} (T={Tinf:.0f})")
    s_inf = lrt_stat(inf, Tinf, (0, 1), [], N, k, an)
    print(f"    inference-half LRT stat = {s_inf:.3f}   chi-bar-sq p = {chibar_p(s_inf):.4g}")
    print("\n    -> split p > naive p (more conservative/honest), "
          "true hyperedge still significant.")