"""
Experiment 12 — Part A: data generation (run once, caches to disk).

Generates two synthetic datasets whose hyperedge weights are EXACTLY a
low-rank CP tensor, so that "recovering the true rank" is a well-posed
question:

  * dataset R*=2 : two latent factors, supports {0,1,2} and {2,3,4} (overlap
                   at node 2) on N=6 nodes.
  * dataset R*=3 : three latent factors (a control, to show that AIC/BIC
                   select the true rank in general, not only at R*=2).

The hyperedge weight of edge e is  alpha_e = sum_r prod_{v in e} F_true[v, r].
Only edges with non-negligible weight drive the dynamics, so the simulator is
given just those (the near-zero "decoy" edges contribute nothing and are
withheld from the generator for speed; they re-enter as decoys at inference
time in Part B).

The Ogata simulator is O(n^2), so each dataset costs ~30-60 s. We pay that
ONCE here and cache (events + ground truth) to experiments/results/synthetic/suppD_data_R*.pkl;
the rank-sweep inference in Part B reads the cache and never re-simulates.
"""

import sys
sys.path.insert(0, ".")

import os
import pickle
import time
from itertools import combinations

import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor
from simulation.simulator import HawkesSimulator


N         = 6
T         = 300.0
DELTA     = 0.75
BETA      = 1.0
MU0       = 0.4
SEED_SIM  = 1
MAX_EVENTS = 8000
STRONG_THRESH = 0.05   # edges above this weight drive the dynamics


def alpha_of(edge, F):
    return float(np.prod(np.stack([F[v] for v in edge], axis=0), axis=0).sum())


def make_factor_matrix(r_true):
    """Construct a non-negative N x r_true factor matrix with overlapping
    supports, so the resulting hyperedge-weight tensor has exactly rank r_true.
    """
    F = np.zeros((N, r_true))
    if r_true == 2:
        F[[0, 1, 2], 0] = [0.80, 0.75, 0.70]   # factor 1: nodes 0,1,2
        F[[2, 3, 4], 1] = [0.70, 0.80, 0.75]   # factor 2: nodes 2,3,4
    elif r_true == 3:
        F[[0, 1, 2], 0] = [0.80, 0.75, 0.70]   # factor 1: nodes 0,1,2
        F[[2, 3, 4], 1] = [0.70, 0.80, 0.75]   # factor 2: nodes 2,3,4
        F[[1, 4, 5], 2] = [0.65, 0.70, 0.80]   # factor 3: nodes 1,4,5
    else:
        raise ValueError("only r_true in {2, 3} are constructed here")
    return F


def strong_edges(F):
    """All 2- and 3-node edges whose true CP weight exceeds STRONG_THRESH."""
    cand = list(combinations(range(N), 2)) + list(combinations(range(N), 3))
    return {e: alpha_of(e, F) for e in cand if alpha_of(e, F) > STRONG_THRESH}


def generate(r_true):
    F_true = make_factor_matrix(r_true)
    strong = strong_edges(F_true)

    kernel = ExponentialKernel(beta=BETA)
    anchor = HyperedgeAnchor(delta=DELTA)
    mu_true = np.full(N, MU0)
    ap_true = np.zeros((N, N))            # no pairwise: isolate hyperedge signal

    print(f"\n=== Generating dataset  R*={r_true} ===")
    print(f"  {len(strong)} strong (data-generating) edges:")
    for e, w in sorted(strong.items(), key=lambda x: -x[1]):
        print(f"    {str(e):<10} alpha={w:.3f}")

    print("  simulating (Ogata, one-off cost) ...", flush=True)
    t0 = time.time()
    sim = HawkesSimulator(mu_true, ap_true, strong, kernel, anchor)
    events = sim.simulate(T=T, seed=SEED_SIM, max_events=MAX_EVENTS)
    print(f"  n_events = {len(events)}   ({time.time()-t0:.0f} s)")

    payload = {
        "r_true":   r_true,
        "N":        N,
        "T":        T,
        "delta":    DELTA,
        "beta":     BETA,
        "mu_true":  mu_true,
        "ap_true":  ap_true,
        "F_true":   F_true,
        "strong_edges":  {str(k): v for k, v in strong.items()},
        "strong_edges_tuples": list(strong.keys()),
        "events":   events,
    }
    path = f"experiments/results/synthetic/suppD_data_R{r_true}.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  cached -> {path}")
    return path


def main():
    os.makedirs("experiments", exist_ok=True)
    for r_true in (2, 3):
        path = f"experiments/results/synthetic/suppD_data_R{r_true}.pkl"
        if os.path.exists(path):
            print(f"[skip] {path} already exists (delete it to regenerate).")
            continue
        generate(r_true)
    print("\nData generation complete. Part B (rank sweep) reads these caches.")


if __name__ == "__main__":
    main()