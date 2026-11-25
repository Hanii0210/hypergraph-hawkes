"""Maximisation steps for full-window and conditional-interval HTH EM."""

from __future__ import annotations

import numpy as np

from models.likelihood import (
    all_completion_times,
    event_target_mask,
    event_times_by_node,
    pairwise_event_exposures,
    piecewise_compensator_interval,
)
from models.tensor_param import HypergraphTensor


class MStep:
    """Update model parameters from soft branching responsibilities.

    The optional target interval lets the complete stream act as conditioning
    history while only sufficient statistics from ``(target_start,target_end]``
    enter the numerators.  Every denominator is the exact kernel exposure on
    that same interval.
    """

    def __init__(
        self,
        n_nodes: int,
        tensor: HypergraphTensor,
        lambda_l1: float = 0.001,
    ):
        self.n_nodes = int(n_nodes)
        self.tensor = tensor
        self.lambda_l1 = float(lambda_l1)

    @staticmethod
    def _interval(T, target_start, target_end):
        if target_start is None and target_end is None:
            return 0.0, float(T)
        if target_start is None or target_end is None:
            raise ValueError("target_start and target_end must be supplied together")
        a = float(target_start)
        b = float(target_end)
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            raise ValueError("target interval must be finite and have positive length")
        return a, b

    @staticmethod
    def _mask(events, a, b):
        times = np.asarray([t for t, _ in events], dtype=float)
        return times, event_target_mask(times, a, b)

    def update_mu(
        self,
        events: list,
        p_background: np.ndarray,
        T: float,
        target_start: float | None = None,
        target_end: float | None = None,
    ) -> np.ndarray:
        """Closed-form baseline update on the target interval."""
        a, b = self._interval(T, target_start, target_end)
        _, target_mask = self._mask(events, a, b)
        mu = np.zeros(self.n_nodes, dtype=float)
        for k, (_, node) in enumerate(events):
            if target_mask[k]:
                mu[node] += float(p_background[k])
        mu /= b - a
        return np.clip(mu, 1e-6, None)

    def update_alpha_pairwise(
        self,
        events: list,
        p_pairwise: np.ndarray,
        p_hyper: dict,
        edge_list: list,
        kernel,
        T: float,
        target_start: float | None = None,
        target_end: float | None = None,
    ) -> np.ndarray:
        """Closed-form pairwise update with exact interval exposure.

        Source events before the target interval are retained as possible
        parents and contribute their decayed exposure to the denominator.
        Events after the target interval have zero exposure.
        """
        del p_hyper, edge_list  # responsibilities already partition the E-step
        n = len(events)
        if n == 0:
            return np.zeros((self.n_nodes, self.n_nodes), dtype=float)

        a, b = self._interval(T, target_start, target_end)
        times, target_mask = self._mask(events, a, b)
        nodes = np.asarray([node for _, node in events], dtype=int)

        membership = np.zeros((n, self.n_nodes), dtype=float)
        membership[np.arange(n), nodes] = 1.0

        prior = times[:, None] > times[None, :]
        row_mask = target_mask[:, None]
        p_masked = p_pairwise * prior * row_mask
        numerator = (membership.T @ p_masked @ membership).T

        source_exposure = pairwise_event_exposures(
            times, a, b, float(kernel.beta)
        )
        exposure_by_source_node = membership.T @ source_exposure
        denominator = np.tile(
            exposure_by_source_node[:, None], (1, self.n_nodes)
        )

        alpha = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )
        return np.clip(alpha, 0.0, None)

    def _hyper_sufficient_stats(
        self,
        events,
        p_hyper,
        edge_list,
        anchor_calc,
        kernel,
        T,
        target_start=None,
        target_end=None,
    ):
        """Return ``A_e`` and interval exposure ``L_e`` for every edge."""
        a, b = self._interval(T, target_start, target_end)
        _, target_mask = self._mask(events, a, b)
        per_node = event_times_by_node(events)

        A = {}
        L = {}
        for edge in edge_list:
            responsibilities = np.asarray(p_hyper[edge], dtype=float)
            A[edge] = float(responsibilities[target_mask].sum())
            completions = all_completion_times(
                edge, per_node, float(anchor_calc.delta)
            )
            exposure = piecewise_compensator_interval(
                completions, a, b, float(kernel.beta)
            )
            L[edge] = len(edge) * exposure + self.lambda_l1
        return A, L

    def update_alpha_hyper(
        self,
        events: list,
        p_hyper: dict,
        edge_list: list,
        anchor_calc,
        kernel,
        T: float,
        target_start: float | None = None,
        target_end: float | None = None,
    ) -> dict:
        """Closed-form free-hyperedge update on the target interval."""
        A, L = self._hyper_sufficient_stats(
            events,
            p_hyper,
            edge_list,
            anchor_calc,
            kernel,
            T,
            target_start=target_start,
            target_end=target_end,
        )
        out = {}
        for edge in edge_list:
            out[edge] = max(A[edge] / L[edge], 0.0) if L[edge] > 1e-12 else 0.0
        return out

    # Non-negative CP block-coordinate ascent.  For one scalar x=F[v,r], each
    # edge weight is affine, alpha_e=g_e*x+h_e, making the block objective
    # concave.  The exact constrained maximum is found by bisection on f'.
    F_MAX = 1e4

    @classmethod
    def _solve_block_1d(cls, g, h, A, L, x_cur, tol=1e-12, max_bisect=200):
        g = np.asarray(g, dtype=float)
        h = np.asarray(h, dtype=float)
        A = np.asarray(A, dtype=float)
        L = np.asarray(L, dtype=float)

        active = g > 1e-15
        if not np.any(active):
            return min(max(float(x_cur), 0.0), cls.F_MAX)
        g, h, A, L = g[active], h[active], A[active], L[active]
        linear_cost = float(np.sum(g * L))

        def derivative(x):
            denom = np.maximum(g * x + h, 1e-300)
            with np.errstate(over="ignore", invalid="ignore"):
                terms = A * g / denom
            terms = np.where(np.isfinite(terms), terms, 0.0)
            return float(np.sum(terms) - linear_cost)

        if derivative(0.0) <= 0.0:
            return 0.0
        if derivative(cls.F_MAX) >= 0.0:
            return cls.F_MAX

        lo, hi = 0.0, cls.F_MAX
        for _ in range(max_bisect):
            mid = 0.5 * (lo + hi)
            if derivative(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol * max(1.0, hi):
                break
        return 0.5 * (lo + hi)

    def update_alpha_hyper_als(
        self,
        events,
        p_hyper,
        edge_list,
        anchor_calc,
        kernel,
        T,
        als_max_iter=100,
        als_tol=1e-6,
        target_start=None,
        target_end=None,
    ):
        """Generalised-EM CP update using interval sufficient statistics."""
        A, L = self._hyper_sufficient_stats(
            events,
            p_hyper,
            edge_list,
            anchor_calc,
            kernel,
            T,
            target_start=target_start,
            target_end=target_end,
        )
        F = self.tensor.F
        rank = self.tensor.rank

        edges_of = {}
        for edge_index, edge in enumerate(edge_list):
            for vertex in edge:
                edges_of.setdefault(vertex, []).append(edge_index)

        def alpha_of(edge):
            product = np.ones(rank)
            for vertex in edge:
                product *= F[vertex]
            return float(product.sum())

        alpha_previous = {edge: alpha_of(edge) for edge in edge_list}
        for _ in range(int(als_max_iter)):
            for vertex, edge_indices in edges_of.items():
                for component in range(rank):
                    gs, hs, As, Ls = [], [], [], []
                    for edge_index in edge_indices:
                        edge = edge_list[edge_index]
                        g = 1.0
                        for other in edge:
                            if other != vertex:
                                g *= F[other, component]
                        current_alpha = alpha_of(edge)
                        h = max(current_alpha - F[vertex, component] * g, 0.0)
                        gs.append(g)
                        hs.append(h)
                        As.append(A[edge])
                        Ls.append(L[edge])
                    F[vertex, component] = np.clip(
                        self._solve_block_1d(
                            gs, hs, As, Ls, F[vertex, component]
                        ),
                        0.0,
                        self.F_MAX,
                    )

            alpha_current = {edge: alpha_of(edge) for edge in edge_list}
            max_relative = 0.0
            for edge in edge_list:
                denominator = max(abs(alpha_previous[edge]), 1e-12)
                max_relative = max(
                    max_relative,
                    abs(alpha_current[edge] - alpha_previous[edge]) / denominator,
                )
            alpha_previous = alpha_current
            if max_relative < als_tol:
                break

        return {edge: alpha_of(edge) for edge in edge_list}
