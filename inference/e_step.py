import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor


class EStep:
    """
    Computes soft branch probabilities for each observed event.

    For event i at time t_i on node n_i, the probability that it was
    triggered by source s is:

        p[i, s] = lambda[i, s] / sum_s' lambda[i, s']

    Sources are:
        - background (baseline mu)
        - pairwise parent j (node j fired before t_i)
        - hyperedge e (pattern completed before t_i)

    Parameters
    ----------
    kernel       : ExponentialKernel
    anchor_calc  : HyperedgeAnchor
    """

    def __init__(self, kernel: ExponentialKernel, anchor_calc: HyperedgeAnchor):
        self.kernel = kernel
        self.anchor_calc = anchor_calc

    def compute(
        self,
        events: list,
        mu: np.ndarray,
        alpha_pairwise: np.ndarray,
        alpha_hyper: dict,
        edge_list: list,
    ) -> dict:
        """
        Run the E-step over all observed events.

        Parameters
        ----------
        events : list of (time, node) tuples, sorted by time
            e.g. [(0.1, 0), (0.3, 1), (0.9, 0), ...]

        mu : np.ndarray of shape (N,)
            Baseline intensity for each node.

        alpha_pairwise : np.ndarray of shape (N, N)
            alpha_pairwise[j, i] = pairwise influence of node j on node i.

        alpha_hyper : dict mapping edge tuple -> float
            Hyperedge interaction weights, e.g. {(0,1): 0.3, (0,1,2): 0.1}

        edge_list : list of tuples
            All candidate hyperedges to consider.

        Returns
        -------
        dict with keys:
            'p_background' : np.ndarray (n_events,)
                Probability each event came from background.
            'p_pairwise'   : np.ndarray (n_events, n_events)
                p_pairwise[i, j] = probability event i was triggered by event j.
            'p_hyper'      : dict mapping edge -> np.ndarray (n_events,)
                p_hyper[e][i] = probability event i was triggered by hyperedge e.
        """
        n = len(events)

        # Build lookup: node -> list of (time, event_index)
        event_times_by_node = {}
        for idx, (t, node) in enumerate(events):
            if node not in event_times_by_node:
                event_times_by_node[node] = []
            event_times_by_node[node].append(t)

        p_background = np.zeros(n)
        p_pairwise   = np.zeros((n, n))
        p_hyper      = {e: np.zeros(n) for e in edge_list}

        for i, (t_i, node_i) in enumerate(events):

            # --- background contribution ---
            contrib_bg = mu[node_i]

            # --- pairwise contributions ---
            contrib_pair = np.zeros(n)
            for j, (t_j, node_j) in enumerate(events):
                if t_j >= t_i:
                    break
                tau = t_i - t_j
                contrib_pair[j] = alpha_pairwise[node_j, node_i] * self.kernel(tau)

            # --- hyperedge contributions ---
            contrib_hyper = {}
            for e in edge_list:
                if node_i not in e:
                    contrib_hyper[e] = 0.0
                    continue

                anchors = self.anchor_calc.find_anchors(
                    edge=e,
                    event_times=event_times_by_node,
                    t_current=t_i,
                )
                if len(anchors) == 0:
                    contrib_hyper[e] = 0.0
                    continue

                taus = np.array([t_i - t_a for t_a in anchors])
                contrib_hyper[e] = alpha_hyper.get(e, 0.0) * float(
                    self.kernel(taus).sum()
                )

            # --- normalise ---
            total = contrib_bg + contrib_pair.sum() + sum(contrib_hyper.values())

            if total <= 0:
                p_background[i] = 1.0
                continue

            p_background[i] = contrib_bg / total

            for j in range(i):
                p_pairwise[i, j] = contrib_pair[j] / total

            for e in edge_list:
                p_hyper[e][i] = contrib_hyper[e] / total

        return {
            "p_background": p_background,
            "p_pairwise":   p_pairwise,
            "p_hyper":      p_hyper,
        }