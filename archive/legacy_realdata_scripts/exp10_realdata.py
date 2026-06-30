"""
Experiment 10: Real Neural Data Application (CRCNS ret-1)

Apply the HTH inference framework to multi-electrode array recordings of
mouse retinal ganglion cells (CRCNS ret-1 dataset, Harvard; Zhang & Meister
2008). 7 simultaneously recorded neurons under binary white-noise stimulus.

P5 HONEST-INFERENCE REVISION
----------------------------
The original analysis selected candidate hyperedges and then tested them by a
likelihood-ratio test on the SAME 40 s of data, comparing 2 dL against chi^2_k.
That naive procedure is invalid twice over: (i) double-dipping (select + test
on one dataset inflates significance), and (ii) wrong reference distribution
(alpha_e >= 0 puts H0 on the boundary -> chi-bar-squared, not chi^2_k). This
revision therefore:

  1. Computes ALL log-likelihoods with the canonical closed-form integral
     (models.likelihood, the exact piecewise compensator) rather than a
     200-point Riemann grid, for consistency with exp7 / exp13.
  2. Adds an honest SAMPLE-SPLIT analysis: candidates are selected on the first
     half of the recording and the hyperedge term is tested on the held-out
     second half (chi-bar-squared p-value).
  3. Reports the naive full-data LRT only as a (flagged) reference, and bases
     the verdict on BIC, which is conservative and does not double-dip.

IMPORTANT CAVEAT (from exp13 calibration). Sample splitting removes the
SELECTION-induced inflation but NOT the pairwise->hyperedge identifiability
confound: exp13 showed the split false-positive rate is ~5% when no pairwise
structure is present but ~15% when it is (control: 2% pure-Poisson). Real 
retinal data has strong pairwise coupling (a hub neuron driving several others), 
i.e. exactly the regime where the split p-value is anti-conservative. We therefore 
treat the split LRT as a DIAGNOSTIC, not as calibrated evidence, and defer to BIC. 
Fully honest hyperedge inference here requires inference that separates the pairwise 
and hyperedge components -- a Bayesian latent-branching / filtered-MCMC treatment
with likelihood-free (ABC) null calibration -- which we leave to future work.
"""

import sys
sys.path.insert(0, ".")

import pickle
import numpy as np
import scipy.io as sio
from scipy import stats
from tqdm import tqdm
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.e_step import EStep
from inference.m_step import MStep
from models.likelihood import log_likelihood
from simulation.data_loader import save_events_to_csv


# =============================================================================
# Config
# =============================================================================
MAT_PATH   = "data/crcns_ret-1/crcns_ret-1/Data/20080516_R1.mat"
STIM_IDX   = 0
T_START    = 10.0
T_END      = 50.0
BETA       = 2.0
DELTA      = 0.05
N_ITER     = 40
LAMBDA_L1  = 0.01
THRESHOLD  = 0.02     # pairwise strength to call a pair "strong"
MAX_CANDS  = 6
SEED       = 2026

kernel      = ExponentialKernel(beta=BETA)
anchor_calc = HyperedgeAnchor(delta=DELTA)


# =============================================================================
# Fitting helpers (shared by the full-data and the split analyses)
# =============================================================================
def fit_pairwise(events, T, n_cells, n_iter=N_ITER, lambda_l1=LAMBDA_L1, seed=SEED):
    """Hand-rolled EM for the pairwise-only Hawkes model (no hyperedges)."""
    tensor = HypergraphTensor(n_nodes=n_cells, rank=3, seed=0)
    estep = EStep(kernel, anchor_calc)
    mstep = MStep(n_nodes=n_cells, tensor=tensor, lambda_l1=lambda_l1)
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.5, 5.0, size=n_cells)
    ap = rng.uniform(0.0, 0.1, size=(n_cells, n_cells))
    np.fill_diagonal(ap, 0.0)
    for _ in range(n_iter):
        r = estep.compute(events, mu, ap, {}, [])
        mu = mstep.update_mu(events, r["p_background"], T)
        ap = mstep.update_alpha_pairwise(events, r["p_pairwise"], r["p_hyper"], [], kernel, T)
    return mu, ap


def gen_candidates(ap, n_cells, threshold=THRESHOLD, cap=MAX_CANDS):
    """Two-stage candidate generation: 2-node hyperedges from strong pairs.
    (Unchanged pairwise-clique mechanism; its circularity is addressed in P4.)"""
    cands = []
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            if ap[i, j] > threshold or ap[j, i] > threshold:
                cands.append((i, j))
    if not cands:   # fallback: top pairs by mutual strength
        mutual = sorted(
            ((i, j, ap[i, j] + ap[j, i]) for i in range(n_cells) for j in range(i + 1, n_cells)),
            key=lambda x: -x[2])
        cands = [(i, j) for i, j, _ in mutual[:3]]
    return cands[:cap]


def fit_hth(events, T, candidates, n_cells, mu_init, ap_init,
            n_iter=N_ITER, lambda_l1=LAMBDA_L1):
    """Hand-rolled EM for the full HTH model (real ALS hyperedge update, P1)."""
    tensor = HypergraphTensor(n_nodes=n_cells, rank=3, seed=0)
    estep = EStep(kernel, anchor_calc)
    mstep = MStep(n_nodes=n_cells, tensor=tensor, lambda_l1=lambda_l1)
    mu = mu_init.copy()
    ap = ap_init.copy()
    ah = {e: 0.1 for e in candidates}
    for _ in range(n_iter):
        r = estep.compute(events, mu, ap, ah, candidates)
        mu = mstep.update_mu(events, r["p_background"], T)
        ap = mstep.update_alpha_pairwise(events, r["p_pairwise"], r["p_hyper"], candidates, kernel, T)
        ah = mstep.update_alpha_hyper_als(events, r["p_hyper"], candidates, anchor_calc, kernel, T)
    return mu, ap, ah


def n_pairwise_params(mu, ap, tol=0.01):
    return len(mu) + int(np.sum((ap > tol) & ~np.eye(ap.shape[0], dtype=bool)))


def chibar_p(stat):
    """chi-bar-squared p-value for H0: alpha_e=0 (boundary): 0.5*P(chi^2_1>stat)."""
    return 1.0 if stat <= 0 else 0.5 * stats.chi2.sf(stat, df=1)


# =============================================================================
# 1. Load CRCNS ret-1
# =============================================================================
print("=" * 65)
print("  Experiment 10: Real Neural Data (CRCNS ret-1) -- P5 honest inference")
print("=" * 65)
print(f"\nLoading {MAT_PATH} ...")
mat = sio.loadmat(MAT_PATH)
spikes_raw = mat["spikes"]
n_cells = spikes_raw.shape[0]

events_all = []
for node in range(n_cells):
    for t in spikes_raw[node, STIM_IDX].flatten():
        if T_START <= t <= T_END:
            events_all.append((t - T_START, node))
events_all.sort(key=lambda x: x[0])
T = T_END - T_START
n_events = len(events_all)

print(f"  {n_cells} neurons, window [{T_START},{T_END}]s (T={T}s), {n_events} events")
for node in range(n_cells):
    c = sum(1 for _, n in events_all if n == node)
    print(f"    neuron {node}: {c:>5} spikes ({c / T:.1f} Hz)")
save_events_to_csv(events_all, "experiments/exp10_realdata_events.csv")


# =============================================================================
# 2. FULL-DATA fit (reference only; the LRT here is the NAIVE, invalid one)
# =============================================================================
print("\n--- Full-data fit (reference; naive LRT is flagged invalid) ---")
mu_pw, ap_pw = fit_pairwise(events_all, T, n_cells)
candidates = gen_candidates(ap_pw, n_cells)
print(f"  candidate hyperedges (selected on full data): {candidates}")
mu_hth, ap_hth, ah_hth = fit_hth(events_all, T, candidates, n_cells, mu_pw, ap_pw)

logL_pw = log_likelihood(events_all, T, mu_pw, ap_pw, {}, [], kernel, anchor_calc, "closed_form")
logL_hth = log_likelihood(events_all, T, mu_hth, ap_hth, ah_hth, candidates, kernel, anchor_calc, "closed_form")
n_params_pw = n_pairwise_params(mu_pw, ap_pw)
surv_full = [e for e in candidates if ah_hth[e] > 0.01]
n_params_hth = n_params_pw + len(surv_full)
delta_L = logL_hth - logL_pw
naive_lrt = 2.0 * delta_L
bic_diff = (-2 * logL_pw + n_params_pw * np.log(n_events)) - \
           (-2 * logL_hth + n_params_hth * np.log(n_events))

print(f"  logL_pw={logL_pw:.3f} (p={n_params_pw})  logL_hth={logL_hth:.3f} (p={n_params_hth})")
print(f"  dL={delta_L:+.3f}  BICdiff={bic_diff:+.3f} (positive favours HTH)")
print(f"  surviving hyperedges: {[(e, round(ah_hth[e],3)) for e in surv_full]}")
print(f"  [NAIVE LRT 2dL={naive_lrt:.2f}, chi-bar-sq p={chibar_p(naive_lrt):.3g}] "
      f"-- INVALID (double-dipping: candidates selected on the same data). "
      f"Reported only for transparency; not used for the verdict.")


# =============================================================================
# 3. SAMPLE-SPLIT honest analysis: select on [0,T/2], test on [T/2,T]
# =============================================================================
print("\n--- Sample-split honest inference (select first half / test second half) ---")
t_cut = T / 2.0
sel = [(t, n) for (t, n) in events_all if t <= t_cut]
inf = [(t - t_cut, n) for (t, n) in events_all if t > t_cut]
T_half = T - t_cut
print(f"  selection half: {len(sel)} events (T={t_cut:.0f}s)   "
      f"inference half: {len(inf)} events (T={T_half:.0f}s)")

mu_sel, ap_sel = fit_pairwise(sel, t_cut, n_cells)
cand_split = gen_candidates(ap_sel, n_cells)
print(f"  candidates selected on first half: {cand_split}")

mu_pw_i, ap_pw_i = fit_pairwise(inf, T_half, n_cells)
mu_hth_i, ap_hth_i, ah_hth_i = fit_hth(inf, T_half, cand_split, n_cells, mu_pw_i, ap_pw_i)
logL_pw_i = log_likelihood(inf, T_half, mu_pw_i, ap_pw_i, {}, [], kernel, anchor_calc, "closed_form")
logL_hth_i = log_likelihood(inf, T_half, mu_hth_i, ap_hth_i, ah_hth_i, cand_split, kernel, anchor_calc, "closed_form")

n_pw_i = n_pairwise_params(mu_pw_i, ap_pw_i)
surv_split = [e for e in cand_split if ah_hth_i[e] > 0.01]
n_hth_i = n_pw_i + len(surv_split)
split_dL = logL_hth_i - logL_pw_i
split_lrt = 2.0 * split_dL
split_p = chibar_p(split_lrt)
n_inf = len(inf)
split_bic_diff = (-2 * logL_pw_i + n_pw_i * np.log(n_inf)) - \
                 (-2 * logL_hth_i + n_hth_i * np.log(n_inf))

print(f"  held-out logL_pw={logL_pw_i:.3f}  logL_hth={logL_hth_i:.3f}")
print(f"  split dL={split_dL:+.3f}   LRT 2dL={split_lrt:.2f}   chi-bar-sq p={split_p:.3g}")
print(f"  split BICdiff={split_bic_diff:+.3f} (positive favours HTH)")
print(f"  surviving on held-out half: {[(e, round(ah_hth_i[e],3)) for e in surv_split]}")


# =============================================================================
# 4. Honest verdict
# =============================================================================
print("\n--- Verdict ---")
print("  The split p-value above is a DIAGNOSTIC, not calibrated evidence: real")
print("  retinal data has strong pairwise coupling, the regime where exp13 showed")
print("  the split false-positive rate is inflated (~15% vs ~5% nominal) by the")
print("  pairwise->hyperedge confound. We therefore defer to BIC.")
bic_verdict = "favours HTH" if split_bic_diff > 0 else "favours pairwise-only"
print(f"  BIC on held-out half {bic_verdict} (BICdiff={split_bic_diff:+.3f}).")
print("  Overall: suggestive but not decisive. Fully honest inference requires")
print("  component separation (Bayesian latent-branching / filtered MCMC) with")
print("  ABC null calibration -- left to future work.")


# =============================================================================
# 5. Save (keeps the keys snapshot_golden.py reads; adds split fields)
# =============================================================================
results = {
    "mat_path": MAT_PATH, "T_start": T_START, "T_end": T_END, "T": T,
    "n_events": n_events, "n_cells": n_cells, "beta": BETA, "delta": DELTA,
    "mu_pw": mu_pw, "alpha_pw": ap_pw,
    "mu_hth": mu_hth, "alpha_pw_hth": ap_hth, "alpha_hyper_hth": ah_hth,
    "candidates": candidates,
    "logL_pw": logL_pw, "logL_hth": logL_hth,
    "delta_L": delta_L, "bic_diff": bic_diff,
    "n_params_pw": n_params_pw, "n_params_hth": n_params_hth,
    "naive_lrt_2dL": naive_lrt, "naive_chibar_p": chibar_p(naive_lrt),
    "split_t_cut": t_cut, "split_n_sel": len(sel), "split_n_inf": n_inf,
    "split_candidates": cand_split,
    "split_logL_pw": logL_pw_i, "split_logL_hth": logL_hth_i,
    "split_delta_L": split_dL, "split_lrt_2dL": split_lrt, "split_chibar_p": split_p,
    "split_bic_diff": split_bic_diff,
    "split_alpha_hyper": ah_hth_i,
}
with open("experiments/exp10_realdata.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: experiments/exp10_realdata.pkl")