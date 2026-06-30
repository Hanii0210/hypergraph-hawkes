import sys
sys.path.insert(0, ".")

import numpy as np
from tqdm import tqdm
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.em import run_em
from simulation.simulator import HawkesSimulator


# =============================================================================
# Experiment 1b: Recovery Robustness Across Random Datasets
#
# Re-run the recovery test from Exp 1 across N_SEEDS different
# simulation seeds, each generating a fresh dataset from the same true
# parameters. Report the distribution of recovery errors.
#
# This addresses a question: is the 7% error in Exp 1 a
# stable property of the inference, or a one-seed artefact?
# =============================================================================

# Same ground truth as Exp 1
TRUE_MU             = np.array([0.3, 0.3, 0.4])
TRUE_ALPHA_PAIRWISE = np.zeros((3, 3))
TRUE_ALPHA_PAIRWISE[2, 0] = 0.3
TRUE_ALPHA_HYPER    = {(0, 1): 0.4}
EDGE_LIST           = [(0, 1)]
T                   = 500.0
N_ITER              = 60
N_NODES             = 3
BETA                = 1.0
DELTA               = 0.5
N_SEEDS             = 25         # number of different datasets


# =============================================================================
# Setup
# =============================================================================
kernel      = ExponentialKernel(beta=BETA)
anchor_calc = HyperedgeAnchor(delta=DELTA)


# =============================================================================
# Run recovery on each seed
# =============================================================================
print(f"Running recovery test across {N_SEEDS} different datasets ...")

recovered = {
    "mu[0]":            [],
    "mu[1]":            [],
    "mu[2]":            [],
    "a[2->0]":          [],
    "alpha_hyper(0,1)": [],
    "n_events":         [],
}

for seed in tqdm(range(N_SEEDS), desc="seeds"):
    sim = HawkesSimulator(
        TRUE_MU, TRUE_ALPHA_PAIRWISE, TRUE_ALPHA_HYPER, kernel, anchor_calc
    )
    events = sim.simulate(T=T, seed=seed, max_events=3000)
    n_events = len(events)

    tensor = HypergraphTensor(n_nodes=N_NODES, rank=3, seed=0)

    rng = np.random.default_rng(seed=2026 + seed)
    mu             = rng.uniform(0.1, 0.5, size=N_NODES)
    alpha_pairwise = rng.uniform(0.0, 0.2, size=(N_NODES, N_NODES))
    np.fill_diagonal(alpha_pairwise, 0.0)
    alpha_hyper    = {e: float(rng.uniform(0.05, 0.4)) for e in EDGE_LIST}

    res = run_em(
        events, T, N_NODES, EDGE_LIST, kernel, anchor_calc,
        mu0=mu, alpha_pairwise0=alpha_pairwise, alpha_hyper0=alpha_hyper,
        n_iter=N_ITER, lambda_l1=0.001, tensor=tensor,
        hyper_update="als",
    )
    mu             = res.mu
    alpha_pairwise = res.alpha_pairwise
    alpha_hyper    = res.alpha_hyper

    recovered["mu[0]"].append(mu[0])
    recovered["mu[1]"].append(mu[1])
    recovered["mu[2]"].append(mu[2])
    recovered["a[2->0]"].append(alpha_pairwise[2, 0])
    recovered["alpha_hyper(0,1)"].append(alpha_hyper[(0, 1)])
    recovered["n_events"].append(n_events)


# =============================================================================
# Report distribution
# =============================================================================
true_values = {
    "mu[0]":            TRUE_MU[0],
    "mu[1]":            TRUE_MU[1],
    "mu[2]":            TRUE_MU[2],
    "a[2->0]":          TRUE_ALPHA_PAIRWISE[2, 0],
    "alpha_hyper(0,1)": TRUE_ALPHA_HYPER[(0, 1)],
}

print("\n" + "=" * 70)
print(f"Recovery distribution across {N_SEEDS} datasets")
print("=" * 70)
print(f"{'Parameter':<22} {'True':>7} {'Mean':>9} {'Std':>9} "
      f"{'Bias':>9} {'RelErr':>9}")
print("-" * 70)

for name in ["mu[0]", "mu[1]", "mu[2]", "a[2->0]", "alpha_hyper(0,1)"]:
    vals = np.array(recovered[name])
    true_v = true_values[name]
    mean_v = vals.mean()
    std_v  = vals.std()
    bias   = mean_v - true_v
    rel_err = abs(bias) / true_v if true_v > 0 else float("nan")
    print(f"  {name:<20} {true_v:>7.3f} {mean_v:>9.4f} {std_v:>9.4f} "
          f"{bias:>+9.4f} {rel_err*100:>7.2f}%")

print("-" * 70)
n_events_arr = np.array(recovered["n_events"])
print(f"  events per dataset: mean={n_events_arr.mean():.0f}  "
      f"std={n_events_arr.std():.0f}  "
      f"range=[{n_events_arr.min()}, {n_events_arr.max()}]")


# =============================================================================
# Save for plotting
# =============================================================================
import pickle
with open("experiments/exp1b_recovery_robustness.pkl", "wb") as f:
    pickle.dump({
        "recovered":   recovered,
        "true_values": true_values,
        "N_SEEDS":     N_SEEDS,
    }, f)
print("\nSaved: experiments/exp1b_recovery_robustness.pkl")