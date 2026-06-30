import numpy as np
from models.kernel import ExponentialKernel, HyperedgeAnchor


class EStep:
    """
    Computes soft branch probabilities for each observed event.

    Vectorised implementation. Numerically identical to the original
    triple-loop version, but the O(n^2) pairwise computation is done with
    NumPy broadcasting and the per-event anchor search is replaced by a
    one-off precomputation of each hyperedge's completion times plus a
    binary search (searchsorted).

    Correctness rests on one fact: whether a time t_last is a valid pattern
    completion for edge e depends only on whether every member fired in
    [t_last - delta, t_last]; those member events are at times <= t_last, so
    the validity of t_last does not depend on the query time t_current (as
    long as t_last < t_current). Hence the set of completions can be computed
    once over the whole stream, and the anchor for event i is simply the
    largest completion strictly before t_i -- exactly the max(...) the
    original find_anchors returns.

    For event i at time t_i on node n_i, the probability that it was
    triggered by source s is:

        p[i, s] = lambda[i, s] / sum_s' lambda[i, s']

    Sources: background (mu), pairwise parent j, hyperedge e.

    Parameters
    ----------
    kernel       : ExponentialKernel
    anchor_calc  : HyperedgeAnchor
    """

    def __init__(self, kernel: ExponentialKernel, anchor_calc: HyperedgeAnchor):
        self.kernel = kernel
        self.anchor_calc = anchor_calc

    def _completion_times(self, edge, event_times_by_node):
        """All times at which `edge` completes its pattern, sorted ascending.

        Identical logic to HyperedgeAnchor.find_anchors' inner test and to
        m_step / likelihood, so the anchor set is the same object everywhere.
        """
        delta = self.anchor_calc.delta
        completions = set()
        for anchor_node in edge:
            if anchor_node not in event_times_by_node:
                continue
            for t_last in event_times_by_node[anchor_node]:
                window_start = t_last - delta
                complete = True
                for v in edge:
                    if v == anchor_node:
                        continue
                    if v not in event_times_by_node:
                        complete = False
                        break
                    member_t = event_times_by_node[v]
                    # any member event in [window_start, t_last] ?
                    lo = np.searchsorted(member_t, window_start, side="left")
                    hi = np.searchsorted(member_t, t_last, side="right")
                    if hi - lo == 0:
                        complete = False
                        break
                if complete:
                    completions.add(t_last)
        return np.array(sorted(completions), dtype=float)

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

        Parameters / return values are identical to the original loop
        implementation (see git history): returns a dict with keys
        'p_background' (n,), 'p_pairwise' (n, n), 'p_hyper' (edge -> (n,)).
        """
        n = len(events)
        beta = self.kernel.beta

        p_background = np.zeros(n)
        p_pairwise = np.zeros((n, n))
        p_hyper = {e: np.zeros(n) for e in edge_list}

        if n == 0:
            return {"p_background": p_background,
                    "p_pairwise": p_pairwise,
                    "p_hyper": p_hyper}

        times = np.array([t for t, _ in events], dtype=float)
        nodes = np.array([node for _, node in events], dtype=int)

        # node -> sorted array of its event times (events are time-sorted, so
        # each per-node list is already ascending)
        event_times_by_node = {}
        for t, node in events:
            event_times_by_node.setdefault(node, []).append(t)
        for k in event_times_by_node:
            event_times_by_node[k] = np.asarray(event_times_by_node[k], dtype=float)

        # ---------------- pairwise contribution matrix ----------------
        # contrib_pair[i, j] = alpha_pairwise[node_j, node_i] * exp(-beta dt)
        #                      for t_j < t_i, else 0.
        dt = times[:, None] - times[None, :]          # dt[i, j] = t_i - t_j
        mask = dt > 0.0                                # strictly t_j < t_i
        # alpha_pairwise[node_j, node_i] as [i, j]: build P[a,b]=alpha[nodes[a],nodes[b]]
        P = alpha_pairwise[np.ix_(nodes, nodes)]       # P[i, j] = alpha[node_i, node_j]
        alpha_mat = P.T                                # [i, j] = alpha[node_j, node_i]
        # guard exp against negative tau (upper triangle) before masking
        safe_dt = np.where(mask, dt, 0.0)
        kern = np.where(mask, np.exp(-beta * safe_dt), 0.0)
        contrib_pair = alpha_mat * kern                # (n, n)

        # ---------------- hyperedge contribution ----------------
        # contrib_hyper_col[e] : (n,) contribution of edge e to each event
        contrib_hyper = {}
        for e in edge_list:
            col = np.zeros(n)
            a_e = float(alpha_hyper.get(e, 0.0))
            comps = self._completion_times(e, event_times_by_node)
            if a_e != 0.0 and comps.size > 0:
                member = np.array([1 if nodes[i] in e else 0 for i in range(n)],
                                  dtype=bool)
                # most-recent completion strictly before t_i
                idx = np.searchsorted(comps, times, side="left") - 1
                has_anchor = idx >= 0
                use = member & has_anchor
                if np.any(use):
                    anchor_t = np.empty(n)
                    anchor_t[use] = comps[idx[use]]
                    tau = times[use] - anchor_t[use]
                    col[use] = a_e * np.exp(-beta * tau)
            contrib_hyper[e] = col

        # ---------------- totals and normalisation ----------------
        bg = mu[nodes]                                  # (n,)
        pair_sum = contrib_pair.sum(axis=1)             # (n,)
        hyper_sum = np.zeros(n)
        for e in edge_list:
            hyper_sum += contrib_hyper[e]
        total = bg + pair_sum + hyper_sum               # (n,)

        pos = total > 0.0
        # events with total <= 0: p_background = 1, all else 0 (original behaviour)
        p_background[~pos] = 1.0

        inv = np.zeros(n)
        inv[pos] = 1.0 / total[pos]

        p_background[pos] = bg[pos] * inv[pos]
        p_pairwise = contrib_pair * inv[:, None]        # rows with total<=0 are 0
        # zero out any pairwise row where total<=0 (inv=0 already does this)
        for e in edge_list:
            p_hyper[e] = contrib_hyper[e] * inv

        return {
            "p_background": p_background,
            "p_pairwise":   p_pairwise,
            "p_hyper":      p_hyper,
        }