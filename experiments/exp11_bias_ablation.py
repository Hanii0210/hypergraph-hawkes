"""
Experiment 11: Bias Ablation

Investigate the mechanism behind the systematic -22% bias on alpha_hyper.

Hypothesis: the bias arises because pairwise and hyperedge contributions
overlap temporally, causing the EM to under-attribute events to the
hyperedge branch.

Test: if we increase beta (faster decay), the temporal overlap between
pairwise and hyperedge contributions shrinks, and the bias should decrease.
Conversely, if we decrease beta (slower decay), overlap increases and
bias should worsen.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from tqdm import tqdm
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.e_step import EStep
from inference.m_step import MStep
from simulation.simulator import HawkesSimulator


# =============================================================================
# Ground truth (same as Exp 1/1b)
# =============================================================================
TRUE_MU             = np.array([0.3, 0.3, 0.4])
TRUE_ALPHA_PAIRWISE = np.zeros((3, 3))
TRUE_ALPHA_PAIRWISE[2, 0] = 0.3
TRUE_ALPHA_HYPER    = {(0, 1): 0.4}
EDGE_LIST           = [(0, 1)]
T                   = 200.0
N_ITER              = 60
N_NODES             = 3
DELTA               = 0.5
N_SEEDS             = 5

# Sweep beta: small beta = slow decay = more overlap
#              large beta = fast decay = less overlap
BETA_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0]


# =============================================================================
# Run recovery at each beta
# =============================================================================
print("=" * 65)
print("  Experiment 11: Bias Ablation (beta sweep)")
print("=" * 65)
print(f"\n  Hypothesis: larger beta -> less temporal overlap -> smaller bias")
print(f"  Sweeping beta over {BETA_VALUES}")
print(f"  {N_SEEDS} seeds per beta value\n")

results = []

for beta in tqdm(BETA_VALUES, desc="beta sweep"):
    kernel      = ExponentialKernel(beta=beta)
    anchor_calc = HyperedgeAnchor(delta=DELTA)

    alpha_hyper_recovered = []

    for seed in range(N_SEEDS):
        sim = HawkesSimulator(
            TRUE_MU, TRUE_ALPHA_PAIRWISE, TRUE_ALPHA_HYPER, kernel, anchor_calc
        )
        events = sim.simulate(T=T, seed=seed, max_events=3000)

        tensor = HypergraphTensor(n_nodes=N_NODES, rank=3, seed=0)
        estep  = EStep(kernel, anchor_calc)
        mstep  = MStep(n_nodes=N_NODES, tensor=tensor, lambda_l1=0.001)

        rng = np.random.default_rng(seed=2026 + seed)
        mu             = rng.uniform(0.1, 0.5, size=N_NODES)
        alpha_pairwise = rng.uniform(0.0, 0.2, size=(N_NODES, N_NODES))
        np.fill_diagonal(alpha_pairwise, 0.0)
        alpha_hyper    = {e: float(rng.uniform(0.05, 0.4)) for e in EDGE_LIST}

        for e in EDGE_LIST:
            target_factor = alpha_hyper[e] ** (1.0 / (len(e) * tensor.rank))
            for v in e:
                tensor.F[v, :] = target_factor

        for _ in range(N_ITER):
            result = estep.compute(events, mu, alpha_pairwise, alpha_hyper, EDGE_LIST)
            mu = mstep.update_mu(events, result["p_background"], T)
            alpha_pairwise = mstep.update_alpha_pairwise(
                events, result["p_pairwise"], result["p_hyper"],
                EDGE_LIST, kernel, T
            )
            alpha_hyper = mstep.update_alpha_hyper(
                events, result["p_hyper"], EDGE_LIST, anchor_calc, kernel, T
            )

        alpha_hyper_recovered.append(alpha_hyper[(0, 1)])

    arr = np.array(alpha_hyper_recovered)
    bias = arr.mean() - TRUE_ALPHA_HYPER[(0, 1)]
    rel_bias = bias / TRUE_ALPHA_HYPER[(0, 1)] * 100

    print(f"  beta={beta:<5.2f}  mean(alpha_hyper)={arr.mean():.4f}  "
          f"std={arr.std():.4f}  bias={bias:+.4f}  rel_bias={rel_bias:+.1f}%")

    results.append({
        "beta":      beta,
        "mean":      arr.mean(),
        "std":       arr.std(),
        "bias":      bias,
        "rel_bias":  rel_bias,
        "values":    arr.copy(),
    })


# =============================================================================
# Save
# =============================================================================
import pickle
with open("experiments/exp11_bias_ablation.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: experiments/exp11_bias_ablation.pkl")

# Summary
print("\n" + "=" * 65)
print("  Summary: bias vs temporal overlap")
print("=" * 65)
print(f"  {'beta':>6}  {'1/beta (timescale)':>18}  {'rel_bias':>10}")
print("-" * 40)
for r in results:
    print(f"  {r['beta']:>6.2f}  {1/r['beta']:>18.2f} s  {r['rel_bias']:>+10.1f}%")

if results[-1]["rel_bias"] > results[0]["rel_bias"]:
    print("\n  Result: bias INCREASES with beta -> hypothesis NOT supported")
else:
    print("\n  Result: bias DECREASES with beta -> hypothesis SUPPORTED")
    print("  Interpretation: faster decay reduces temporal overlap between")
    print("  pairwise and hyperedge branches, allowing EM to correctly")
    print("  attribute more responsibility to the hyperedge term.")