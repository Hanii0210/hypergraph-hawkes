import numpy as np
from itertools import combinations
from models.kernel import ExponentialKernel
from models.tensor_param import HypergraphTensor
from inference.e_step import EStep
from inference.m_step import MStep


def fit_pairwise_only(events, T, n_nodes, kernel, anchor_calc,
                      n_iter=30, lambda_l1=0.001, seed=0):
    """
    Run EM with NO hyperedges (pairwise model only).
    Returns the inferred mu and alpha_pairwise.

    This is the baseline against which hyperedge candidates are tested.
    """
    tensor = HypergraphTensor(n_nodes=n_nodes, rank=3, seed=seed)
    estep  = EStep(kernel, anchor_calc)
    mstep  = MStep(n_nodes=n_nodes, tensor=tensor, lambda_l1=lambda_l1)

    rng = np.random.default_rng(seed)
    mu             = rng.uniform(0.1, 0.5, size=n_nodes)
    alpha_pairwise = rng.uniform(0.0, 0.2, size=(n_nodes, n_nodes))
    np.fill_diagonal(alpha_pairwise, 0.0)
    alpha_hyper    = {}
    edge_list      = []

    for _ in range(n_iter):
        result = estep.compute(events, mu, alpha_pairwise, alpha_hyper, edge_list)
        mu = mstep.update_mu(events, result["p_background"], T)
        alpha_pairwise = mstep.update_alpha_pairwise(
            events, result["p_pairwise"], result["p_hyper"],
            edge_list, kernel, T
        )

    return mu, alpha_pairwise


def generate_candidate_hyperedges(
    alpha_pairwise: np.ndarray,
    max_edge_size: int = 3,
    top_m_pairs: int = None,
    pairwise_threshold: float = None,
) -> list:
    """
    Construct candidate hyperedges from strongly connected node pairs.

    Strategy:
      1. Identify "strong" pairs: alpha_pairwise[j, i] above threshold
         (or top-M by magnitude)
      2. For each subset of size 2 to max_edge_size from the union of
         strong-pair endpoints, generate a candidate hyperedge

    Parameters
    ----------
    alpha_pairwise     : (N, N) inferred pairwise weights
    max_edge_size      : K, maximum hyperedge cardinality (default 3)
    top_m_pairs        : keep only the top-M pairs (mutually exclusive
                         with pairwise_threshold)
    pairwise_threshold : minimum alpha to call a pair "strong"

    Returns
    -------
    list of tuples : sorted, deduplicated candidate hyperedges
    """
    N = alpha_pairwise.shape[0]

    # Symmetrise (since hyperedges are unordered)
    A = (alpha_pairwise + alpha_pairwise.T) / 2.0

    # Get all (i, j, weight) with i < j
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j, A[i, j]))

    # Filter to "strong" pairs
    if top_m_pairs is not None:
        pairs.sort(key=lambda x: -x[2])
        strong_pairs = pairs[:top_m_pairs]
    elif pairwise_threshold is not None:
        strong_pairs = [p for p in pairs if p[2] > pairwise_threshold]
    else:
        # Default: above median nonzero
        nonzero = [p[2] for p in pairs if p[2] > 1e-6]
        if len(nonzero) == 0:
            return []
        thresh = float(np.median(nonzero))
        strong_pairs = [p for p in pairs if p[2] > thresh]

    if len(strong_pairs) == 0:
        return []

    # Pool of nodes that participate in any strong pair
    strong_nodes = set()
    for i, j, _ in strong_pairs:
        strong_nodes.add(i)
        strong_nodes.add(j)
    strong_nodes = sorted(strong_nodes)

    # Generate all candidate hyperedges of size 2..K from strong_nodes
    candidates = set()
    for size in range(2, max_edge_size + 1):
        for combo in combinations(strong_nodes, size):
            candidates.add(tuple(sorted(combo)))

    # For size 2, only keep those that were actually in strong_pairs
    strong_pair_set = set(tuple(sorted([i, j])) for i, j, _ in strong_pairs)
    final = []
    for e in sorted(candidates, key=lambda x: (len(x), x)):
        if len(e) == 2 and e not in strong_pair_set:
            continue
        final.append(e)

    return final