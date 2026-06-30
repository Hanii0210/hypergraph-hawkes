"""
Experiment 14: Candidate-generation identification diagnostic

PURPOSE
-------
The two-stage candidate pipeline (inference/candidate_filter) fits a pairwise-
only model, flags strong pairs, and forms hyperedge candidates as cliques of
those pairs. A natural worry (the "circularity" critique) is that this can only
nominate hyperedges whose members already show strong PAIRWISE excitation, and
would therefore MISS genuinely higher-order structure.

This experiment tests that worry directly. We generate data with a single TRUE
3-node hyperedge (0,1,2) and NO pairwise excitation anywhere, sweep the
hyperedge weight, and measure:
  * footprint   : the largest inferred pairwise weight among the member pairs
                  {(0,1),(0,2),(1,2)} from a pairwise-only fit;
  * nominated   : whether generate_candidate_hyperedges (at the standard
                  threshold) actually proposes (0,1,2);
  * fitted alpha: the HTH-recovered hyperedge weight (and its bias vs truth);
  * dL          : closed-form log-likelihood gain of adding the TRUE hyperedge
                  over the pairwise-only model.

FINDING (see table printed below)
---------------------------------
The candidate pipeline does NOT miss the hyperedge: a genuine hyperedge induces
member co-firing, which the pairwise-only fit absorbs as a pairwise footprint
(~0.05-0.07) above the nomination threshold (0.02), so (0,1,2) is nominated
79-92% of the time. The pathology is NOT recall but IDENTIFICATION: the only
signal the pipeline can use to find the hyperedge is the same co-firing that

the pairwise model explains away, so the hyperedge is only weakly detectable 
at moderate strength (dL ~ 0.5-2.5; clearly detectable only once dL > ~3, i.e. 
alpha >= 0.8). With the corrected simulator the recovered weight is near-unbiased 
(-1% to -5%): the identification difficulty now shows up as weak detectability / 
large variance, not as a systematic point-estimate bias.

This is the same pairwise<->hyperedge confound quantified in exp13. With the
corrected simulator the confound manifests primarily as WEAK DETECTABILITY,
not as a point-estimate bias: the candidate-generation "circularity" (a
hyperedge is only findable through the member co-firing the pairwise model
also explains) and the calibration inflation of exp13 (split FPR ~15% vs ~5%
nominal, collapsing to ~2% once the pairwise structure is removed) are two
faces of one identifiability limitation. The recovered weight itself is now
near-unbiased (-1% to -5% here; cf. exp1b), so the earlier "systematic
recovery bias" is no longer a third face -- it was largely an artifact of the
previous simulator.

SCOPE NOTE
----------
A genuinely higher-order structure with NO pairwise footprint (an XOR-type
co-activation with marginally independent members) is invisible not only to
this nomination step but to the HTH detection mechanism itself, which is driven
by member co-firing. Resolving the identification problem -- separating the
pairwise and hyperedge components -- requires inference beyond this pipeline
(a Bayesian latent-branching / filtered-MCMC treatment with ABC null
calibration) and is left to future work. This experiment only DIAGNOSES the
limitation; it does not attempt a new nomination mechanism or a fix.
"""

import sys
sys.path.insert(0, ".")

import pickle
import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from simulation.simulator import HawkesSimulator
from inference.candidate_filter import fit_pairwise_only, generate_candidate_hyperedges
from inference.em import run_em
from models.likelihood import log_likelihood

# ---- config ----
N_NODES    = 4
TRUE_EDGE  = (0, 1, 2)
MEMBER_PAIRS = [(0, 1), (0, 2), (1, 2)]
THRESHOLD  = 0.02          # standard pairwise-nomination threshold
T          = 600.0
ALPHA_GRID = [0.4, 0.6, 0.8]
SEEDS      = list(range(1, 25))
BETA       = 1.0
DELTA      = 0.5
N_ITER     = 40
DETECT_DL  = 3.0           # dL above this -> "detectable" (rough 2dL>~6)

kernel = ExponentialKernel(beta=BETA)
anchor = HyperedgeAnchor(delta=DELTA)
MU = np.full(N_NODES, 0.3)
AP_ZERO = np.zeros((N_NODES, N_NODES))   # NO pairwise excitation anywhere


def fit_hth_true(events, T):
    """Fit HTH with only the TRUE hyperedge as candidate (real ALS M-step)."""
    rng = np.random.default_rng(0)
    tensor = HypergraphTensor(n_nodes=N_NODES, rank=3, seed=0)
    return run_em(events, T, N_NODES, [TRUE_EDGE], kernel, anchor,
                  rng.uniform(0.2, 0.5, N_NODES), np.zeros((N_NODES, N_NODES)),
                  {TRUE_EDGE: 0.1}, n_iter=N_ITER, lambda_l1=0.0,
                  tensor=tensor, hyper_update="als")


def main():
    print("=" * 70)
    print("  Experiment 14: candidate-generation identification diagnostic")
    print("  (true hyperedge (0,1,2), NO pairwise; threshold = %.2f)" % THRESHOLD)
    print("=" * 70)
    print(f"\n{'a_true':>7} {'footprint':>16} {'nom%':>5} "
          f"{'fit_ahyp':>9} {'bias':>7} {'dL':>7} {'detect':>7}")

    rows = []
    for a in ALPHA_GRID:
        foots, noms, fits, dls = [], [], [], []
        for s in SEEDS:
            ev = HawkesSimulator(MU, AP_ZERO, {TRUE_EDGE: a}, kernel, anchor).simulate(
                T=T, seed=s, max_events=5000)
            mu_p, ap_p = fit_pairwise_only(ev, T, N_NODES, kernel, anchor,
                                           n_iter=N_ITER, lambda_l1=0.001, seed=0)
            A = (ap_p + ap_p.T) / 2.0
            foots.append(max(A[i, j] for i, j in MEMBER_PAIRS))
            cands = generate_candidate_hyperedges(ap_p, max_edge_size=3,
                                                  pairwise_threshold=THRESHOLD)
            noms.append(TRUE_EDGE in cands)
            res = fit_hth_true(ev, T)
            fits.append(res.alpha_hyper[TRUE_EDGE])
            llf = log_likelihood(ev, T, res.mu, res.alpha_pairwise, res.alpha_hyper,
                                 [TRUE_EDGE], kernel, anchor, "closed_form")
            llr = log_likelihood(ev, T, mu_p, ap_p, {}, [], kernel, anchor, "closed_form")
            dls.append(llf - llr)

        foot_m, foot_s = float(np.mean(foots)), float(np.std(foots))
        nom_pct = 100.0 * float(np.mean(noms))
        fit_m = float(np.mean(fits))
        bias = 100.0 * (fit_m - a) / a
        n = len(fits)
        bias_sem = float(np.std(100.0 * (np.array(fits) - a) / a) / np.sqrt(n))
        dl_m = float(np.mean(dls))
        dl_sem = float(np.std(dls) / np.sqrt(len(dls)))
        detect = dl_m > DETECT_DL
        rows.append(dict(alpha=a, footprint_mean=foot_m, footprint_std=foot_s,
                         threshold=THRESHOLD, nominated_pct=nom_pct,
                         fit_alpha=fit_m, bias_pct=bias, bias_sem=bias_sem,
                         dL=dl_m, dL_sem=dl_sem, n_seeds=n, detectable=detect))
        print(f"{a:>7.1f} {foot_m:>8.4f}±{foot_s:.3f} {nom_pct:>4.0f}% "
              f"{fit_m:>9.3f} {bias:>+6.0f}% {dl_m:>7.2f} {str(detect):>7}")

    print("\n  Conclusion: nomination recall is high (a genuine hyperedge usually")
    print("  leaves a pairwise footprint above threshold), but the hyperedge is only weakly")
    print("  detectable at moderate strength (near-unbiased once found) -- an IDENTIFICATION")
    print("  limitation (pairwise<->hyperedge confound), not a recall failure.")

    with open("experiments/exp14_identification_diagnostic.pkl", "wb") as f:
        pickle.dump({"rows": rows, "true_edge": TRUE_EDGE, "threshold": THRESHOLD,
                     "T": T, "seeds": SEEDS, "alpha_grid": ALPHA_GRID}, f)
    print("\nSaved: experiments/exp14_identification_diagnostic.pkl")


if __name__ == "__main__":
    main()