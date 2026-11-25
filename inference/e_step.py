"""Expectation step for full-window and conditional-interval HTH inference."""

from __future__ import annotations

import numpy as np

from models.kernel import ExponentialKernel, HyperedgeAnchor


class EStep:
    """Compute soft branching responsibilities for observed events.

    ``compute`` is backward compatible with the original full-window API.  When
    ``target_start`` and ``target_end`` are supplied, the complete event stream
    is still used as history, but responsibilities are returned only for target
    events in ``(target_start, target_end]`` (with time zero included when the
    start is zero).  Rows outside the target interval are exactly zero.
    """

    def __init__(self, kernel: ExponentialKernel, anchor_calc: HyperedgeAnchor):
        self.kernel = kernel
        self.anchor_calc = anchor_calc

    @staticmethod
    def compute_sufficient(features, coefficients, alpha_pairwise, alpha_hyper):
        """Aggregate latent parent responsibilities by source node.

        The dense compute API remains available for existing CP code and
        independent small-data comparisons. Free-weight EM uses this API.
        """
        return features.statistics(coefficients, alpha_pairwise, alpha_hyper)

    def _completion_times(self, edge, event_times_by_node):
        """All completion times for an edge, sorted and deduplicated."""
        delta = self.anchor_calc.delta
        completions = set()
        for anchor_node in edge:
            member_anchor_times = event_times_by_node.get(anchor_node)
            if member_anchor_times is None:
                continue
            for t_last in member_anchor_times:
                window_start = t_last - delta
                complete = True
                for member in edge:
                    if member == anchor_node:
                        continue
                    member_times = event_times_by_node.get(member)
                    if member_times is None:
                        complete = False
                        break
                    lo = np.searchsorted(member_times, window_start, side="left")
                    hi = np.searchsorted(member_times, t_last, side="right")
                    if hi <= lo:
                        complete = False
                        break
                if complete:
                    completions.add(float(t_last))
        return np.asarray(sorted(completions), dtype=float)

    @staticmethod
    def _target_mask(times, target_start, target_end):
        if target_start is None and target_end is None:
            return np.ones(len(times), dtype=bool)
        if target_start is None or target_end is None:
            raise ValueError("target_start and target_end must be supplied together")
        a = float(target_start)
        b = float(target_end)
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            raise ValueError("target interval must be finite and have positive length")
        left = times >= a if a == 0.0 else times > a
        return left & (times <= b)

    def compute(
        self,
        events: list,
        mu: np.ndarray,
        alpha_pairwise: np.ndarray,
        alpha_hyper: dict,
        edge_list: list,
        target_start: float | None = None,
        target_end: float | None = None,
    ) -> dict:
        """Run the E-step using the full stream as history.

        Returns
        -------
        dict
            ``p_background`` with shape ``(n,)``, ``p_pairwise`` with shape
            ``(n,n)``, ``p_hyper`` mapping each edge to shape ``(n,)``, and
            ``target_mask``.  Responsibility rows outside the target interval
            are zero.
        """
        n = len(events)
        beta = self.kernel.beta

        p_background = np.zeros(n)
        p_pairwise = np.zeros((n, n))
        p_hyper = {edge: np.zeros(n) for edge in edge_list}

        if n == 0:
            return {
                "p_background": p_background,
                "p_pairwise": p_pairwise,
                "p_hyper": p_hyper,
                "target_mask": np.zeros(0, dtype=bool),
            }

        times = np.asarray([t for t, _ in events], dtype=float)
        nodes = np.asarray([node for _, node in events], dtype=int)
        target_mask = self._target_mask(times, target_start, target_end)

        event_times_by_node = {}
        for t, node in events:
            event_times_by_node.setdefault(node, []).append(t)
        for node in event_times_by_node:
            event_times_by_node[node] = np.asarray(
                event_times_by_node[node], dtype=float
            )

        # Pairwise contribution matrix: target event i, potential parent j.
        dt = times[:, None] - times[None, :]
        prior = dt > 0.0
        pair_weights = alpha_pairwise[np.ix_(nodes, nodes)].T
        safe_dt = np.where(prior, dt, 0.0)
        pair_kernel = np.where(prior, np.exp(-beta * safe_dt), 0.0)
        pair_contribution = pair_weights * pair_kernel

        # Hyperedge contribution for the most recent completion strictly before
        # each event.
        hyper_contribution = {}
        for edge in edge_list:
            col = np.zeros(n)
            weight = float(alpha_hyper.get(edge, 0.0))
            completions = self._completion_times(edge, event_times_by_node)
            if weight != 0.0 and completions.size:
                member = np.fromiter(
                    (node in edge for node in nodes), dtype=bool, count=n
                )
                idx = np.searchsorted(completions, times, side="left") - 1
                use = member & (idx >= 0)
                if np.any(use):
                    tau = times[use] - completions[idx[use]]
                    col[use] = weight * np.exp(-beta * tau)
            hyper_contribution[edge] = col

        background = mu[nodes]
        total = background + pair_contribution.sum(axis=1)
        for edge in edge_list:
            total += hyper_contribution[edge]

        positive_target = target_mask & (total > 0.0)
        nonpositive_target = target_mask & ~positive_target
        p_background[nonpositive_target] = 1.0

        inv = np.zeros(n)
        inv[positive_target] = 1.0 / total[positive_target]
        p_background[positive_target] = background[positive_target] * inv[positive_target]
        p_pairwise = pair_contribution * inv[:, None]
        for edge in edge_list:
            p_hyper[edge] = hyper_contribution[edge] * inv

        return {
            "p_background": p_background,
            "p_pairwise": p_pairwise,
            "p_hyper": p_hyper,
            "target_mask": target_mask,
        }
