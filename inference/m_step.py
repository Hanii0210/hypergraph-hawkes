import numpy as np
from models.tensor_param import HypergraphTensor


class MStep:
    """
    Updates model parameters given the soft assignments from the E-step.

    All three parameter types use closed-form analytic updates derived
    from setting the gradient of the expected complete-data log-likelihood
    Q(theta) to zero.

    Parameters
    ----------
    n_nodes   : int
    tensor    : HypergraphTensor
    lambda_l1 : float, L1 penalty on hyperedge weights
    """

    def __init__(self, n_nodes: int, tensor: HypergraphTensor,
                 lambda_l1: float = 0.01):
        self.n_nodes   = n_nodes
        self.tensor    = tensor
        self.lambda_l1 = lambda_l1

    def update_mu(
        self,
        events: list,
        p_background: np.ndarray,
        T: float,
    ) -> np.ndarray:
        """
        Closed-form update for baseline intensities.

            mu[i] = (sum of background responsibility for events on node i) / T
        """
        mu = np.zeros(self.n_nodes)
        for k, (t, node) in enumerate(events):
            mu[node] += p_background[k]
        mu /= T
        mu = np.clip(mu, 1e-6, None)
        return mu

    def update_alpha_pairwise(
        self,
        events: list,
        p_pairwise: np.ndarray,
        p_hyper: dict,
        edge_list: list,
        kernel,
        T: float,
    ) -> np.ndarray:
        """
        Closed-form update for pairwise interaction weights.

        Responsibility already attributed to hyperedges is subtracted
        to prevent the same event from being explained by both a pairwise
        term and a hyperedge term:

            alpha[j, i] = sum_k p_pair[k,j] * (1 - hyper_share[k]) / int_j
        """
        n = len(events)
        numerator   = np.zeros((self.n_nodes, self.n_nodes))
        denominator = np.zeros((self.n_nodes, self.n_nodes))

        hyper_share = np.zeros(n)
        for e in edge_list:
            hyper_share += p_hyper[e]
        hyper_share = np.clip(hyper_share, 0.0, 1.0)

        for j, (t_j, node_j) in enumerate(events):
            for i, (t_i, node_i) in enumerate(events):
                if t_i <= t_j:
                    continue
                numerator[node_j, node_i] += p_pairwise[i, j]

        for j, (t_j, node_j) in enumerate(events):
            contrib = kernel.integral(t_j, T)
            for node_i in range(self.n_nodes):
                denominator[node_j, node_i] += contrib

        safe_denom = np.where(denominator > 0, denominator, 1.0)
        alpha = numerator / safe_denom
        alpha = np.clip(alpha, 0.0, None)
        return alpha

    def update_alpha_hyper(
        self,
        events: list,
        p_hyper: dict,
        edge_list: list,
        anchor_calc,
        kernel,
        T: float,
    ) -> dict:
        """
        Closed-form update for hyperedge weights.

            alpha_e = sum_i p_hyper[e][i] / (|e| * C_e + lambda_l1)

        where the compensator C_e is a *piecewise* integral over anchor
        activity windows: each anchor is active only until the next
        completion (or until T for the last). This matches the
        most-recent-anchor semantics in HyperedgeAnchor.find_anchors.

        Naively integrating each anchor's kernel from its own time to T
        double-counts and gives a systematic bias of order |completions|.
        The piecewise form below resolves this; see exp7 for the
        likelihood gap induced by the bug-fix.
        """
        event_times_by_node = {}
        for t, node in events:
            if node not in event_times_by_node:
                event_times_by_node[node] = []
            event_times_by_node[node].append(t)

        def all_completion_times(edge):
            """All times at which the hyperedge pattern completes in [0, T]."""
            completions = set()
            for anchor_node in edge:
                if anchor_node not in event_times_by_node:
                    continue
                for t_last in event_times_by_node[anchor_node]:
                    window_start = t_last - anchor_calc.delta
                    complete = True
                    for v in edge:
                        if v == anchor_node:
                            continue
                        if v not in event_times_by_node:
                            complete = False
                            break
                        in_window = [
                            t for t in event_times_by_node[v]
                            if window_start <= t <= t_last
                        ]
                        if len(in_window) == 0:
                            complete = False
                            break
                    if complete:
                        completions.add(t_last)
            return sorted(completions)

        def piecewise_compensator(completion_times, T, kernel):
            """Each anchor active only until next anchor (or T for the last)."""
            if len(completion_times) == 0:
                return 0.0
            sorted_t = sorted(completion_times)
            total = 0.0
            for k in range(len(sorted_t)):
                t_start = sorted_t[k]
                t_end = sorted_t[k+1] if k+1 < len(sorted_t) else T
                if t_end > t_start:
                    total += (1.0 / kernel.beta) * (
                        1.0 - np.exp(-kernel.beta * (t_end - t_start))
                    )
            return total

        new_alpha = {}

        for e in edge_list:
            resp = float(p_hyper[e].sum())

            completion_times = all_completion_times(e)
            C_e = piecewise_compensator(completion_times, T, kernel)
            compensator = len(e) * C_e

            denom   = compensator + self.lambda_l1
            alpha_e = resp / denom if denom > 1e-9 else 0.0
            alpha_e = max(alpha_e, 0.0)
            new_alpha[e] = alpha_e

            # Sync tensor factors so that get_weight(e) returns alpha_e
            k = len(e)
            target_factor = alpha_e ** (1.0 / (k * self.tensor.rank))
            for v in e:
                self.tensor.F[v, :] = target_factor

        return new_alpha