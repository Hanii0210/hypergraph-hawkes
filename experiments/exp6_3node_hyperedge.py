import sys
sys.path.insert(0, ".")

import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from inference.e_step import EStep
from inference.m_step import MStep
from simulation.simulator import HawkesSimulator


# =============================================================================
# Experiment 6: 3-node hyperedge recovery
#
# This is the first true test of higher-than-2 hyperedges in this project.
# Truth: a single 3-node hyperedge (0, 1, 2). No pairwise interactions.
# We test whether EM can recover the 3-node interaction strength.
# =============================================================================

TRUE_MU = np.array([0.3, 0.3, 0.3, 0.3])
TRUE_ALPHA_PAIRWISE = np.zeros((4, 4))
TRUE_ALPHA_HYPER = {(0, 1, 2): 0.6}     # genuine 3-node hyperedge

# Candidates include the truth + decoys (some 2-node, some 3-node)
CANDIDATE_EDGES = [
    (0, 1),         # 2-node decoy
    (1, 2),         # 2-node decoy
    (0, 1, 2),      # TRUE 3-node
    (0, 1, 3),      # 3-node decoy
    (1, 2, 3),      # 3-node decoy
]

T          = 2000.0
N_ITER     = 60
N_NODES    = 4
BETA       = 1.0
DELTA      = 0.5
LAMBDA_L1  = 0.001
SEED       = 42


# =============================================================================
# Setup
# =============================================================================
kernel      = ExponentialKernel(beta=BETA)
anchor_calc = HyperedgeAnchor(delta=DELTA)

sim = HawkesSimulator(
    TRUE_MU, TRUE_ALPHA_PAIRWISE, TRUE_ALPHA_HYPER, kernel, anchor_calc
)
events = sim.simulate(T=T, seed=SEED, max_events=3000)
print(f"Generated {len(events)} events over T={T}")

# Count actual completions of the true 3-node pattern
event_times_by_node = {}
for t, node in events:
    if node not in event_times_by_node:
        event_times_by_node[node] = []
    event_times_by_node[node].append(t)

n_completions_true = 0
for anchor_node in (0, 1, 2):
    for t_last in event_times_by_node.get(anchor_node, []):
        window_start = t_last - DELTA
        complete = True
        for v in (0, 1, 2):
            if v == anchor_node:
                continue
            in_window = [
                t for t in event_times_by_node.get(v, [])
                if window_start <= t <= t_last
            ]
            if len(in_window) == 0:
                complete = False
                break
        if complete:
            n_completions_true += 1
print(f"True 3-node pattern completions in data: {n_completions_true}\n")


# =============================================================================
# Initialise inference (random)
# =============================================================================
tensor = HypergraphTensor(n_nodes=N_NODES, rank=3, seed=0)
estep  = EStep(kernel, anchor_calc)
mstep  = MStep(n_nodes=N_NODES, tensor=tensor, lambda_l1=LAMBDA_L1)

rng = np.random.default_rng(seed=2026)
mu             = rng.uniform(0.1, 0.5, size=N_NODES)
alpha_pairwise = rng.uniform(0.0, 0.2, size=(N_NODES, N_NODES))
np.fill_diagonal(alpha_pairwise, 0.0)
alpha_hyper    = {e: float(rng.uniform(0.05, 0.4)) for e in CANDIDATE_EDGES}

# =============================================================================
# EM loop
# =============================================================================
print(f"{'Iter':>4}  " + "  ".join(f"{str(e):>14}" for e in CANDIDATE_EDGES))
print("-" * (6 + 16 * len(CANDIDATE_EDGES)))

for it in range(1, N_ITER + 1):
    result = estep.compute(events, mu, alpha_pairwise, alpha_hyper, CANDIDATE_EDGES)

    mu = mstep.update_mu(events, result["p_background"], T)
    alpha_pairwise = mstep.update_alpha_pairwise(
        events, result["p_pairwise"], result["p_hyper"],
        CANDIDATE_EDGES, kernel, T
    )
    alpha_hyper = mstep.update_alpha_hyper_als(
        events, result["p_hyper"], CANDIDATE_EDGES, anchor_calc, kernel, T
    )

    if it % 15 == 0 or it == 1:
        line = f"{it:>4}  " + "  ".join(
            f"{alpha_hyper[e]:>14.4f}" for e in CANDIDATE_EDGES
        )
        print(line)


# =============================================================================
# Summary
# =============================================================================
print("\n--- Recovery Summary ---")
print(f"{'Edge':<14} {'Type':<10} {'True':>7} {'Inferred':>10}")
print("-" * 44)

for e in CANDIDATE_EDGES:
    true_val = TRUE_ALPHA_HYPER.get(e, 0.0)
    inf_val  = alpha_hyper[e]
    edge_type = "TRUE" if e in TRUE_ALPHA_HYPER else "decoy"
    flag = ""
    if edge_type == "TRUE":
        err = abs(inf_val - true_val)
        flag = "  <-- recovered" if err < 0.15 else "  <-- off"
    elif inf_val > 0.05:
        flag = "  <-- false positive!"
    print(f"  {str(e):<12} {edge_type:<10} {true_val:>7.3f} {inf_val:>10.4f}{flag}")


# =============================================================================
# Diagnosis
# =============================================================================
print("\n--- E-step share ---")
result = estep.compute(events, mu, alpha_pairwise, alpha_hyper, CANDIDATE_EDGES)
total_bg    = result["p_background"].sum()
total_pair  = result["p_pairwise"].sum()
total_hyper_per_e = {e: float(result["p_hyper"][e].sum()) for e in CANDIDATE_EDGES}
total = total_bg + total_pair + sum(total_hyper_per_e.values())

print(f"  background  : {total_bg/total*100:.1f}%")
print(f"  pairwise    : {total_pair/total*100:.1f}%")
for e in CANDIDATE_EDGES:
    print(f"  hyper {str(e):<12}: {total_hyper_per_e[e]/total*100:.1f}%")


# =============================================================================
# Save results (read by exp6_plot.py)
# =============================================================================
import pickle
results = {
    "edges":     [tuple(e) for e in CANDIDATE_EDGES],
    "types":     ["TRUE" if e in TRUE_ALPHA_HYPER else "decoy"
                  for e in CANDIDATE_EDGES],
    "true":      [TRUE_ALPHA_HYPER.get(e, 0.0) for e in CANDIDATE_EDGES],
    "inferred":  [float(alpha_hyper[e]) for e in CANDIDATE_EDGES],
    "estep_share": {
        "background": float(total_bg / total),
        "pairwise":   float(total_pair / total),
        "hyper":      {str(e): float(total_hyper_per_e[e] / total)
                       for e in CANDIDATE_EDGES},
    },
}
with open("experiments/exp6_3node_hyperedge.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved: experiments/exp6_3node_hyperedge.pkl")