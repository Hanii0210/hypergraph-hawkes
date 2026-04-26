import numpy as np


class HypergraphTensor:
    """
    Low-rank CP decomposition for hyperedge interaction weights.

    Instead of storing a full O(N^K) tensor, we parameterise each
    hyperedge weight as:
        alpha_e = sum_r prod_{v in e} F[v, r]

    where F is an (N x R) factor matrix. This reduces the parameter
    count from O(N^K) to O(N * R).

    Note: in this project the M-step writes the factor matrix F directly
    (see m_step.update_alpha_hyper). The class therefore exposes F as a
    plain attribute and provides analytic weight / gradient evaluation,
    but no internal optimiser.

    Parameters
    ----------
    n_nodes : int   number of nodes in the system
    rank    : int   number of latent factors R
    seed    : int   random seed for reproducibility
    """

    def __init__(self, n_nodes: int, rank: int = 5, seed: int = 42):
        assert n_nodes > 0, "n_nodes must be positive"
        assert rank > 0, "rank must be positive"
        self.n_nodes = n_nodes
        self.rank = rank
        rng = np.random.default_rng(seed)
        self.F = rng.uniform(0.0, 0.1, size=(n_nodes, rank))

    def get_weight(self, edge: tuple) -> float:
        """
        Compute the interaction weight for a given hyperedge.

            alpha_e = sum_r prod_{v in e} F[v, r]

        Parameters
        ----------
        edge : tuple of int, e.g. (0, 1) or (0, 1, 2)

        Returns
        -------
        float : interaction weight, >= 0 by non-negative initialisation
        """
        factors = np.stack([self.F[v] for v in edge], axis=0)
        return float(np.prod(factors, axis=0).sum())

    def get_all_weights(self, edges: list) -> np.ndarray:
        """Vectorised version of get_weight over a list of edges."""
        return np.array([self.get_weight(e) for e in edges])

    def gradient_wrt_F(self, edge: tuple, v: int) -> np.ndarray:
        """
        Gradient of alpha_e with respect to factor row F[v, :].

            d(alpha_e) / d(F[v, r]) = prod_{u in e, u != v} F[u, r]

        Parameters
        ----------
        edge : tuple of int
        v    : int, the node whose factor row we differentiate

        Returns
        -------
        np.ndarray of shape (R,)
        """
        assert v in edge, "node v must be a member of the edge"
        other_nodes = [u for u in edge if u != v]
        if len(other_nodes) == 0:
            return np.ones(self.rank)
        factors = np.stack([self.F[u] for u in other_nodes], axis=0)
        return np.prod(factors, axis=0)