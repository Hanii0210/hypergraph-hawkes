"""
Experiment 10: Real Neural Data Application

Apply the HTH inference framework to multi-electrode array recordings
of mouse retinal ganglion cells (CRCNS ret-1 dataset, Harvard).

Source: Zhang & Meister, 2008. 61-electrode array, isolated mouse retina.
Data:   7 simultaneously recorded neurons under binary white noise stimulus.

We fit both a pairwise-only Hawkes model and the full HTH model to
a segment of the recording, then compare via likelihood and BIC.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.e_step import EStep
from inference.m_step import MStep
from simulation.data_loader import save_events_to_csv


# =============================================================================
# 1. Load and convert the CRCNS ret-1 data
# =============================================================================
MAT_PATH   = "data/crcns_ret-1/crcns_ret-1/Data/20080516_R1.mat"
STIM_IDX   = 0          # first stimulus condition
T_START    = 10.0        # skip first 10s (transient response)
T_END      = 50.0        # use 40 seconds of data
BETA       = 2.0         # faster decay for neural data (spikes are brief)
DELTA      = 0.05        # 50 ms co-activation window (biologically motivated)
N_ITER     = 40          # EM iterations
LAMBDA_L1  = 0.01

print("=" * 65)
print("  Experiment 10: Real Neural Data (CRCNS ret-1)")
print("=" * 65)

print(f"\nLoading {MAT_PATH} ...")
mat = sio.loadmat(MAT_PATH)
spikes_raw = mat["spikes"]
n_cells = spikes_raw.shape[0]
print(f"  Found {n_cells} neurons, using stimulus condition {STIM_IDX}")

# Convert to our (time, node) format
events_all = []
for node in range(n_cells):
    times = spikes_raw[node, STIM_IDX].flatten()
    for t in times:
        if T_START <= t <= T_END:
            events_all.append((t - T_START, node))

events_all.sort(key=lambda x: x[0])
T = T_END - T_START
n_events = len(events_all)

print(f"  Time window: [{T_START}, {T_END}] s  (duration = {T} s)")
print(f"  Total events in window: {n_events}")
print(f"  Events per neuron:")
for node in range(n_cells):
    count = sum(1 for _, n in events_all if n == node)
    rate = count / T
    print(f"    neuron {node}: {count:>5} spikes  ({rate:.1f} Hz)")

# Save as CSV for reproducibility
csv_path = "experiments/exp10_realdata_events.csv"
save_events_to_csv(events_all, csv_path)
print(f"\n  Saved events to {csv_path}")


# =============================================================================
# 2. Fit pairwise-only Hawkes model
# =============================================================================
print("\n--- Phase 1: Pairwise-only Hawkes model ---")
kernel      = ExponentialKernel(beta=BETA)
anchor_calc = HyperedgeAnchor(delta=DELTA)

# No hyperedges
tensor_pw = HypergraphTensor(n_nodes=n_cells, rank=3, seed=0)
estep_pw  = EStep(kernel, anchor_calc)
mstep_pw  = MStep(n_nodes=n_cells, tensor=tensor_pw, lambda_l1=LAMBDA_L1)

rng = np.random.default_rng(seed=2026)
mu_pw             = rng.uniform(0.5, 5.0, size=n_cells)
alpha_pairwise_pw = rng.uniform(0.0, 0.1, size=(n_cells, n_cells))
np.fill_diagonal(alpha_pairwise_pw, 0.0)
alpha_hyper_pw    = {}
edge_list_pw      = []

for it in tqdm(range(N_ITER), desc="pairwise EM"):
    result = estep_pw.compute(
        events_all, mu_pw, alpha_pairwise_pw, alpha_hyper_pw, edge_list_pw
    )
    mu_pw = mstep_pw.update_mu(events_all, result["p_background"], T)
    alpha_pairwise_pw = mstep_pw.update_alpha_pairwise(
        events_all, result["p_pairwise"], result["p_hyper"],
        edge_list_pw, kernel, T
    )

# Count pairwise parameters (mu + nonzero alpha)
n_params_pw = n_cells  # mu
sig_pairs = []
for i in range(n_cells):
    for j in range(n_cells):
        if i != j and alpha_pairwise_pw[i, j] > 0.01:
            n_params_pw += 1
            sig_pairs.append((i, j, alpha_pairwise_pw[i, j]))

print(f"\n  Fitted baseline rates (Hz):")
for i in range(n_cells):
    print(f"    mu[{i}] = {mu_pw[i]:.3f}")
print(f"  Significant pairwise connections (alpha > 0.01):")
for i, j, a in sorted(sig_pairs, key=lambda x: -x[2])[:10]:
    print(f"    {i} -> {j}: alpha = {a:.4f}")


# =============================================================================
# 3. Identify candidate hyperedges from pairwise results
# =============================================================================
print("\n--- Phase 2: Candidate hyperedge generation ---")
threshold = 0.02
strong_pairs = [(i, j) for i, j, a in sig_pairs if a > threshold]
print(f"  Strong pairs (alpha > {threshold}): {len(strong_pairs)}")

# Build candidate 2-node hyperedges from mutual excitation
candidates = []
for i in range(n_cells):
    for j in range(i+1, n_cells):
        a_ij = alpha_pairwise_pw[i, j]
        a_ji = alpha_pairwise_pw[j, i]
        if a_ij > threshold or a_ji > threshold:
            candidates.append((i, j))

# Limit candidates
candidates = candidates[:6]
print(f"  Candidate hyperedges: {candidates}")

if len(candidates) == 0:
    print("  No candidate hyperedges found. Using top 3 pairs by mutual strength.")
    mutual = []
    for i in range(n_cells):
        for j in range(i+1, n_cells):
            mutual.append((i, j, alpha_pairwise_pw[i,j] + alpha_pairwise_pw[j,i]))
    mutual.sort(key=lambda x: -x[2])
    candidates = [(i, j) for i, j, _ in mutual[:3]]
    print(f"  Fallback candidates: {candidates}")


# =============================================================================
# 4. Fit full HTH model with candidate hyperedges
# =============================================================================
print("\n--- Phase 3: Full HTH model ---")
tensor_hth = HypergraphTensor(n_nodes=n_cells, rank=3, seed=0)
estep_hth  = EStep(kernel, anchor_calc)
mstep_hth  = MStep(n_nodes=n_cells, tensor=tensor_hth, lambda_l1=LAMBDA_L1)

mu_hth             = mu_pw.copy()
alpha_pairwise_hth = alpha_pairwise_pw.copy()
alpha_hyper_hth    = {e: 0.1 for e in candidates}

for e in candidates:
    target_factor = alpha_hyper_hth[e] ** (1.0 / (len(e) * tensor_hth.rank))
    for v in e:
        tensor_hth.F[v, :] = target_factor

for it in tqdm(range(N_ITER), desc="HTH EM"):
    result = estep_hth.compute(
        events_all, mu_hth, alpha_pairwise_hth, alpha_hyper_hth, candidates
    )
    mu_hth = mstep_hth.update_mu(events_all, result["p_background"], T)
    alpha_pairwise_hth = mstep_hth.update_alpha_pairwise(
        events_all, result["p_pairwise"], result["p_hyper"],
        candidates, kernel, T
    )
    alpha_hyper_hth = mstep_hth.update_alpha_hyper(
        events_all, result["p_hyper"], candidates, anchor_calc, kernel, T
    )

print(f"\n  HTH baseline rates (Hz):")
for i in range(n_cells):
    print(f"    mu[{i}] = {mu_hth[i]:.3f}")
print(f"  Hyperedge weights:")
surviving = []
for e in candidates:
    a = alpha_hyper_hth[e]
    status = "SURVIVING" if a > 0.01 else "suppressed"
    print(f"    {e}: alpha = {a:.4f}  [{status}]")
    surviving.append((e, a))


# =============================================================================
# 5. Model comparison: log-likelihood + BIC
# =============================================================================
print("\n--- Phase 4: Model comparison ---")

def compute_loglik(events, mu, alpha_pw, alpha_hyp, edge_list, kernel, anchor, T):
    from simulation.simulator import HawkesSimulator
    sim = HawkesSimulator(mu, alpha_pw, alpha_hyp, kernel, anchor)
    log_lam_sum = 0.0
    for i, (t_i, n_i) in enumerate(events):
        history = events[:i]
        lam = sim._intensity(t_i, history)
        if lam[n_i] > 1e-10:
            log_lam_sum += np.log(lam[n_i])
    grid = np.linspace(0, T, 200)
    total_int = 0.0
    for k in range(len(grid) - 1):
        t_mid = 0.5 * (grid[k] + grid[k+1])
        history = [(t, n) for t, n in events if t < t_mid]
        lam = sim._intensity(t_mid, history)
        total_int += float(lam.sum()) * (grid[k+1] - grid[k])
    return log_lam_sum - total_int

print("  Computing pairwise-only log-likelihood ...")
logL_pw = compute_loglik(
    events_all, mu_pw, alpha_pairwise_pw, {}, [], kernel, anchor_calc, T
)

print("  Computing HTH log-likelihood ...")
logL_hth = compute_loglik(
    events_all, mu_hth, alpha_pairwise_hth, alpha_hyper_hth, candidates,
    kernel, anchor_calc, T
)

n_params_hth = n_params_pw + len([e for e in candidates if alpha_hyper_hth[e] > 0.01])
delta_L  = logL_hth - logL_pw
bic_pw   = -2 * logL_pw  + n_params_pw  * np.log(n_events)
bic_hth  = -2 * logL_hth + n_params_hth * np.log(n_events)
bic_diff = bic_pw - bic_hth  # positive favours HTH

print(f"\n  Pairwise-only:  logL = {logL_pw:.3f}   (params = {n_params_pw})")
print(f"  Full HTH:       logL = {logL_hth:.3f}   (params = {n_params_hth})")
print(f"  Delta L         = {delta_L:+.3f}")
print(f"  BIC difference  = {bic_diff:+.3f}  (positive favours HTH)")


# =============================================================================
# 6. Save results
# =============================================================================
import pickle
results = {
    "mat_path":        MAT_PATH,
    "T_start":         T_START,
    "T_end":           T_END,
    "T":               T,
    "n_events":        n_events,
    "n_cells":         n_cells,
    "beta":            BETA,
    "delta":           DELTA,
    "mu_pw":           mu_pw,
    "alpha_pw":        alpha_pairwise_pw,
    "mu_hth":          mu_hth,
    "alpha_pw_hth":    alpha_pairwise_hth,
    "alpha_hyper_hth": alpha_hyper_hth,
    "candidates":      candidates,
    "logL_pw":         logL_pw,
    "logL_hth":        logL_hth,
    "delta_L":         delta_L,
    "bic_diff":        bic_diff,
    "n_params_pw":     n_params_pw,
    "n_params_hth":    n_params_hth,
}

with open("experiments/exp10_realdata.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: experiments/exp10_realdata.pkl")