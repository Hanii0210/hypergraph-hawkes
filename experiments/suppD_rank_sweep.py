"""
Experiment 12 — Part B: CP rank selection and parameter reduction.

Reads the cached datasets from exp12_generate_data.py (whose hyperedge
weights are exactly rank-R* CP tensors) and asks, for R = 1..R_MAX:

  * how well does a rank-R CP parameterisation recover the true hyperedge
    weights, compared with the unconstrained "free" estimator;
  * how many parameters each uses (N*R for CP vs |candidates| for free);
  * whether AIC / BIC / the recovery elbow identify the true rank R*;
  * whether the recovered factor subspace aligns with the true one
    (principal/coverage angle) -- i.e. whether CP recovers the latent
    structure, not merely the weights.

Two evaluation modes (see design note):
  * CONTROLLED: a single full EM provides one frozen set of E-step
    responsibilities; free and every rank-R CP are then fit to the SAME
    responsibilities. This isolates the effect of the *parameterisation*.
  * FULL_EM:    each rank-R CP runs its own complete EM from scratch (the
    realistic deployment), as a sanity check that the controlled conclusion
    survives end-to-end.

CP is non-convex, so each CP fit uses several random initialisations and
keeps the one with the highest hyperedge objective Q_hyper.
"""

import sys
sys.path.insert(0, ".")

import pickle
import time
from itertools import combinations

import numpy as np
from numpy.linalg import qr, svd

from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.e_step import EStep
from inference.m_step import MStep
from inference.em import run_em
from models.likelihood import log_likelihood


R_MAX        = 5
LAMBDA_L1    = 0.01
EM_ITERS     = 40
N_INIT_CTRL  = 5     # CP restarts in controlled mode (cheap)
N_INIT_FULL  = 2     # CP restarts in full-EM mode (each is a whole EM)


def build_candidates(N, true_triples):
    """All pairwise edges + true 3-node edges + a handful of decoy 3-node edges."""
    pairs = list(combinations(range(N), 2))
    all_triples = list(combinations(range(N), 3))
    decoys = [e for e in all_triples if e not in true_triples][:6]
    return pairs + list(true_triples) + decoys, pairs, list(true_triples), decoys


def coverage_angle_deg(F_true, F_rec):
    """Largest angle (degrees) by which a true latent direction fails to lie in
    the recovered factor subspace. ~0 means the recovered subspace covers the
    true one; ~90 means at least one true direction is missed (e.g. R < R*)."""
    Ft = F_true[:, np.linalg.norm(F_true, axis=0) > 1e-8]
    Fr = F_rec[:,  np.linalg.norm(F_rec,  axis=0) > 1e-8]
    if Ft.shape[1] == 0 or Fr.shape[1] == 0:
        return 90.0
    Qt, _ = qr(Ft)
    Qr, _ = qr(Fr)
    # column norms of Qr^T Qt = cos(angle) of each true direction's projection
    proj = np.linalg.norm(Qr.T @ Qt, axis=0)
    proj = np.clip(proj, 0.0, 1.0)
    return float(np.degrees(np.arccos(proj.min())))


def Qhyper(alpha_vec, Avec, Lvec):
    a = np.maximum(alpha_vec, 1e-12)
    return float(np.sum(Avec * np.log(a) - a * Lvec))


def fit_cp_controlled(ev, p_hyper, CAND, an, k, T, N, R, n_init):
    """Fit rank-R CP to FROZEN responsibilities; keep best of n_init restarts."""
    best_a, best_F, best_q = None, None, -np.inf
    Avec = None
    for s in range(n_init):
        ten = HypergraphTensor(N, R, seed=100 + s)
        m = MStep(N, ten, lambda_l1=LAMBDA_L1)
        if Avec is None:
            A, L = m._hyper_sufficient_stats(ev, p_hyper, CAND, an, k, T)
            Avec = np.array([A[e] for e in CAND]); Lvec = np.array([L[e] for e in CAND])
        ah = m.update_alpha_hyper_als(ev, p_hyper, CAND, an, k, T)
        a = np.array([ah[e] for e in CAND])
        q = Qhyper(a, Avec, Lvec)
        if q > best_q:
            best_q, best_a, best_F = q, a, ten.F.copy()
    return best_a, best_F, best_q, Avec, Lvec


def fit_cp_full_em(ev, CAND, an, k, T, N, R, mu0_fn, n_init):
    """Run a complete EM (rank-R CP) from scratch; keep best of n_init restarts."""
    best = None; best_ll = -np.inf
    for s in range(n_init):
        ten = HypergraphTensor(N, R, seed=200 + s)
        rng = np.random.default_rng(300 + s)
        mu0 = rng.uniform(0.2, 0.6, N)
        ap0 = np.zeros((N, N))
        ah0 = {e: 0.1 for e in CAND}
        res = run_em(ev, T, N, CAND, k, an, mu0, ap0, ah0,
                     n_iter=EM_ITERS, lambda_l1=LAMBDA_L1, tensor=ten,
                     hyper_update="als", track_loglik=True)
        ll = res.final_loglik
        if ll > best_ll:
            best_ll = ll
            best = (res.mu, res.alpha_pairwise,
                    np.array([res.alpha_hyper[e] for e in CAND]), ten.F.copy())
    return best, best_ll


def run_dataset(path):
    d = pickle.load(open(path, "rb"))
    N, T, ev = d["N"], d["T"], d["events"]
    F_true, R_TRUE = d["F_true"], d["r_true"]
    k = ExponentialKernel(d["beta"]); an = HyperedgeAnchor(d["delta"])

    CAND, pairs, true_triples, decoys = build_candidates(N, d["strong_edges_tuples"]
                                                         and [tuple(e) for e in d["strong_edges_tuples"] if len(e) == 3])
    true_alpha = np.array([float(np.prod(np.stack([F_true[v] for v in e]), axis=0).sum())
                           for e in CAND])
    strong_mask = true_alpha > 0.05
    n = len(ev)

    print("\n" + "=" * 70)
    print(f"  Dataset R*={R_TRUE}   N={N}  n_events={n}  candidates={len(CAND)}")
    print(f"  free params={len(CAND)};  CP params N*R: " +
          ", ".join(f"R{r}={N*r}" for r in range(1, R_MAX + 1)))
    print("=" * 70)

    def relerr(a):
        return float(np.mean(np.abs(a[strong_mask] - true_alpha[strong_mask]) /
                             true_alpha[strong_mask]))
    def aic_bic(ll, p):
        return -2 * ll + 2 * p, -2 * ll + p * np.log(n)

    # ---- one full EM to obtain the frozen controlled state ----
    print("\n[controlled] one full EM to freeze responsibilities ...", flush=True)
    t0 = time.time()
    ten0 = HypergraphTensor(N, 3, 0); rng = np.random.default_rng(0)
    res0 = run_em(ev, T, N, CAND, k, an, rng.uniform(0.2, 0.6, N), np.zeros((N, N)),
                  {e: 0.1 for e in CAND}, n_iter=EM_ITERS, lambda_l1=LAMBDA_L1,
                  tensor=ten0, hyper_update="als")
    mu_f, ap_f = res0.mu, res0.alpha_pairwise
    p_hyper = EStep(k, an).compute(ev, mu_f, ap_f, res0.alpha_hyper, CAND)["p_hyper"]
    print(f"  done ({time.time()-t0:.0f}s)")

    def logL_ctrl(a):
        ah = {e: float(a[i]) for i, e in enumerate(CAND)}
        return log_likelihood(ev, T, mu_f, ap_f, ah, CAND, k, an, "closed_form")

    rows_ctrl = []
    # free baseline (controlled): per-edge closed-form optimum A_e / L_e
    ms = MStep(N, HypergraphTensor(N, 3, 0), lambda_l1=LAMBDA_L1)
    A, L = ms._hyper_sufficient_stats(ev, p_hyper, CAND, an, k, T)
    Avec = np.array([A[e] for e in CAND]); Lvec = np.array([L[e] for e in CAND])
    a_free = Avec / Lvec
    ll = logL_ctrl(a_free); aic, bic = aic_bic(ll, len(CAND))
    rows_ctrl.append(("free", len(CAND), Qhyper(a_free, Avec, Lvec), ll, aic, bic,
                      relerr(a_free) * 100, np.nan))
    for R in range(1, R_MAX + 1):
        a, F, q, _, _ = fit_cp_controlled(ev, p_hyper, CAND, an, k, T, N, R, N_INIT_CTRL)
        ll = logL_ctrl(a); aic, bic = aic_bic(ll, N * R)
        rows_ctrl.append((f"CP R={R}", N * R, q, ll, aic, bic,
                          relerr(a) * 100, coverage_angle_deg(F_true, F)))

    def show(title, rows):
        print(f"\n  --- {title} ---")
        print(f"  {'mode':<9}{'params':>7}{'Qhyper':>11}{'logL':>11}"
              f"{'AIC':>10}{'BIC':>10}{'relerr%':>9}{'cov.angle':>10}")
        for name, p, q, ll, aic, bic, re, ang in rows:
            qs = f"{q:.1f}" if np.isfinite(q) else "   -"
            angs = f"{ang:.1f}" if np.isfinite(ang) else "   -"
            print(f"  {name:<9}{p:>7}{qs:>11}{ll:>11.1f}{aic:>10.1f}{bic:>10.1f}"
                  f"{re:>9.1f}{angs:>10}")
        cp = [r for r in rows if r[0].startswith("CP")]
        aic_best = min(cp, key=lambda r: r[4])[0]
        bic_best = min(cp, key=lambda r: r[5])[0]
        print(f"  -> AIC selects {aic_best};  BIC selects {bic_best}  (true R*={R_TRUE})")

    show("CONTROLLED (same frozen responsibilities)", rows_ctrl)

    # ---- full-EM mode ----
    print("\n[full-EM] each rank runs its own complete EM ...", flush=True)
    rows_full = []
    t0 = time.time()
    for R in range(1, R_MAX + 1):
        tr = time.time()
        (mu_r, ap_r, a, F), ll = fit_cp_full_em(ev, CAND, an, k, T, N, R,
                                                None, N_INIT_FULL)
        aic, bic = aic_bic(ll, N * R)
        rows_full.append((f"CP R={R}", N * R, np.nan, ll, aic, bic,
                          relerr(a) * 100, coverage_angle_deg(F_true, F)))
        print(f"    R={R} done  logL={ll:.1f}  relerr={relerr(a)*100:.1f}%  "
              f"cov={coverage_angle_deg(F_true, F):.1f}deg  ({time.time()-tr:.0f}s)", flush=True)
    print(f"  done ({time.time()-t0:.0f}s)")


    show("FULL EM (independent EM per rank)", rows_full)

    return {
        "R_true": R_TRUE, "N": N, "n_events": n, "n_candidates": len(CAND),
        "controlled": rows_ctrl, "full_em": rows_full,
    }


def main():
    import os
    only = sys.argv[1] if len(sys.argv) > 1 else None
    rtrues = (int(only),) if only else (2, 3)
    cache = "experiments/results/synthetic/suppD_rank_sweep.pkl"
    results = {}
    if os.path.exists(cache):
        results = pickle.load(open(cache, "rb"))
    for r_true in rtrues:
        results[f"R{r_true}"] = run_dataset(f"experiments/results/synthetic/suppD_data_R{r_true}.pkl")
    with open(cache, "wb") as f:
        pickle.dump(results, f)
    print("\nSaved: experiments/results/synthetic/suppD_rank_sweep.pkl")


if __name__ == "__main__":
    main()