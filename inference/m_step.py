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
                 lambda_l1: float = 0.001):
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

        The split between background / pairwise / hyperedge is ALREADY enforced
        by the E-step, which normalises p_background + sum_j p_pairwise +
        sum_e p_hyper = 1 per event. So p_pairwise already excludes the
        hyperedge share and NO explicit subtraction is applied here:
 
            alpha[j, i] = (sum_k p_pairwise[k, j], t_k > t_j) / int_j
 
        (p_hyper and edge_list are accepted for a uniform M-step signature but
        are not used by this update.)
        """
        n = len(events)
        if n == 0:
            return np.zeros((self.n_nodes, self.n_nodes))

        times = np.array([t for t, _ in events], dtype=float)
        nodes = np.array([node for _, node in events], dtype=int)

        # One-hot node-membership matrix G (n x N): G[k, c] = 1 iff node_k == c.
        G = np.zeros((n, self.n_nodes))
        G[np.arange(n), nodes] = 1.0

        # numerator[node_j, node_i] = sum over (i, j) with t_i > t_j of
        # p_pairwise[i, j]. The original triple loop is a scatter-add; with the
        # one-hot G it is a pair of matrix products. The strict t_i > t_j mask
        # is applied explicitly so the result is identical to the loop even if
        # p_pairwise carries (spurious) mass where t_i <= t_j.
        mask = (times[:, None] > times[None, :]).astype(float)   # [i, j]: t_i>t_j
        p_masked = p_pairwise * mask
        numerator = (G.T @ p_masked @ G).T                       # (N, N)

        # denominator[node_j, node_i] = sum over events j on node_j of
        # kernel.integral(t_j, T), identical across the node_i axis.
        beta = kernel.beta
        integ = (1.0 / beta) * (1.0 - np.exp(-beta * (T - times)))
        per_node = G.T @ integ                                   # (N,)
        denominator = np.tile(per_node[:, None], (1, self.n_nodes))

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

        return new_alpha
    # =====================================================================
    # Phase 1: real ALS over the CP factor matrix F.
    #
    # The legacy update_alpha_hyper (above) solves each alpha_e independently
    # in closed form and then *back-fills* F via F[v,:] = alpha_e**(1/(k R)).
    # That is not a CP decomposition: it fits the weights first and fabricates
    # factors afterwards, and for hyperedges that share a node the back-fill
    # overwrites the shared row, so get_weight() disagrees with alpha for all
    # but the last such edge. The E-step never reads F (it reads the
    # alpha_hyper dict), which is the only reason that inconsistency stayed
    # hidden.
    #
    # update_alpha_hyper_als below makes the low-rank structure
    #       alpha_e = sum_r prod_{v in e} F[v, r],   F >= 0
    # genuinely binding: the M-step maximises the expected complete-data
    # hyperedge log-likelihood
    #       Q_hyper(F) = sum_e [ A_e log alpha_e - alpha_e L_e ]
    #   with A_e = sum_{i: e ni n_i} p_hyper[e][i]   (aggregate responsibility)
    #        L_e = |e| C_e + lambda_l1               (piecewise compensator + L1)
    # directly over F, by exact block-coordinate ascent (ALS).
    #
    # Block update. Fix every entry of F except the scalar x = F[v, r]. Then
    # for each edge e containing v, alpha_e is affine in x:
    #       alpha_e = g_e x + h_e,
    #       g_e = prod_{u in e, u != v} F[u, r]      (coeff of x in rank r)
    #       h_e = sum_{r' != r} prod_{u in e} F[u, r']   (other ranks; const)
    # so the block objective
    #       f(x) = sum_{e ni v} [ A_e log(g_e x + h_e) - (g_e x + h_e) L_e ]
    # has f''(x) = - sum A_e g_e^2 / (g_e x + h_e)^2 <= 0 (concave) and a
    # strictly decreasing f'(x); the constrained maximiser on x >= 0 is unique
    # and is found exactly by bisection on f'. For a single isolated edge this
    # reduces to alpha_e = A_e / L_e -- identical to the legacy free update --
    # so single-hyperedge recovery is provably unchanged.
    #
    # Convergence / rigor. Block-coordinate ascent on a smooth concave-in-each-
    # block objective is monotone; iterating to a tolerance gives a stationary
    # point of Q_hyper in the given E-step. Warm-starting F from the previous
    # EM iteration (F persists on self.tensor) keeps successive M-steps near
    # one another, so ALS converges in a few sweeps and does not jump between
    # factor local optima -- this is the standard non-negative-CP practice and
    # makes the whole procedure a genuine (generalized) EM whose ascent the
    # driver verifies. CP is non-convex, so convergence is to a local optimum;
    # this is intrinsic to CP, handled empirically by the multi-init agreement
    # check (exp3), not a shortcut in this implementation.
    #
    # Note (deferred): column scaling of F is non-identifiable (F[:,r]*c and
    # the rest /c leave every alpha_e unchanged). alpha_e -- the only quantity
    # used downstream -- is invariant to this, so no column normalisation is
    # applied here. If the *factors themselves* are later interpreted (P2),
    # normalise each column to unit norm and carry a separate weight vector.
    # =====================================================================
    def _hyper_sufficient_stats(self, events, p_hyper, edge_list, anchor_calc,
                                kernel, T):
        """Compute (A_e, L_e) for every edge. A_e = aggregate responsibility,
        L_e = |e| C_e + lambda_l1, with C_e the *piecewise* compensator --
        identical to the legacy update_alpha_hyper, so both paths share stats.
        """
        event_times_by_node = {}
        for t, node in events:
            event_times_by_node.setdefault(node, []).append(t)

        def all_completion_times(edge):
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
                        in_window = [t for t in event_times_by_node[v]
                                     if window_start <= t <= t_last]
                        if len(in_window) == 0:
                            complete = False
                            break
                    if complete:
                        completions.add(t_last)
            return sorted(completions)

        def piecewise_compensator(completion_times, T, kernel):
            if len(completion_times) == 0:
                return 0.0
            sorted_t = sorted(completion_times)
            total = 0.0
            for k in range(len(sorted_t)):
                t_start = sorted_t[k]
                t_end = sorted_t[k + 1] if k + 1 < len(sorted_t) else T
                if t_end > t_start:
                    total += (1.0 / kernel.beta) * (
                        1.0 - np.exp(-kernel.beta * (t_end - t_start)))
            return total

        A = {}
        L = {}
        for e in edge_list:
            A[e] = float(p_hyper[e].sum())
            C_e = piecewise_compensator(all_completion_times(e), T, kernel)
            L[e] = len(e) * C_e + self.lambda_l1
        return A, L

    # F_MAX bounds each CP factor. alpha_e is invariant to rescaling a column,
    # so an over-parameterised rank can satisfy a finite alpha_e with one factor
    # blown up and its partner shrunk to near zero; that degenerate split has no
    # effect on alpha_e but overflows g = prod of partner factors. Constraining
    # F to [0, F_MAX] removes the degeneracy. F_MAX is set far above any factor
    # a realistic weight requires (a balanced factor for alpha ~ O(1) is O(1)),
    # so the box never binds on the identifiable alpha_e -- it only rules out
    # the unphysical scale explosion. Box-constrained block-coordinate ascent on
    # a concave objective is still monotone, so EM ascent is preserved.
    F_MAX = 1e4

    @classmethod
    def _solve_block_1d(cls, g, h, A, L, x_cur, tol=1e-12, max_bisect=200):
        """Exact maximiser on x in [0, F_MAX] of
              sum_e [ A_e log(g_e x + h_e) - (g_e x + h_e) L_e ].
        f'(x) = sum_e A_e g_e/(g_e x + h_e) - sum_e g_e L_e is strictly
        decreasing, so the constrained maximiser is x=0 (if f'(0)<=0), F_MAX
        (if f'(F_MAX)>=0), or the interior root found by bisection.
        """
        g = np.asarray(g, dtype=float)
        h = np.asarray(h, dtype=float)
        A = np.asarray(A, dtype=float)
        L = np.asarray(L, dtype=float)

        active = g > 1e-15
        if not np.any(active):
            return min(max(x_cur, 0.0), cls.F_MAX)  # block cannot affect alpha
        g, h, A, L = g[active], h[active], A[active], L[active]
        C = float(np.sum(g * L))  # constant RHS, >= 0

        def fprime(x):
            denom = np.maximum(g * x + h, 1e-300)
            with np.errstate(over="ignore", invalid="ignore"):
                terms = A * g / denom
            terms = np.where(np.isfinite(terms), terms, 0.0)
            return float(np.sum(terms) - C)

        if fprime(0.0) <= 0.0:
            return 0.0                         # boundary optimum at 0
        if fprime(cls.F_MAX) >= 0.0:
            return cls.F_MAX                   # boundary optimum at F_MAX

        lo, hi = 0.0, cls.F_MAX                # root is bracketed in [0, F_MAX]
        for _ in range(max_bisect):
            mid = 0.5 * (lo + hi)
            if fprime(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol * max(1.0, hi):
                break
        return 0.5 * (lo + hi)

    def update_alpha_hyper_als(self, events, p_hyper, edge_list, anchor_calc,
                               kernel, T, als_max_iter=100, als_tol=1e-6):
        """Real ALS update of the CP factor matrix F (warm-started, to tol).

        Returns the reconstructed {edge: alpha_e} dict and leaves self.tensor.F
        holding the converged factors (so the next EM step warm-starts).
        """
        A, L = self._hyper_sufficient_stats(events, p_hyper, edge_list,
                                            anchor_calc, kernel, T)
        F = self.tensor.F
        R = self.tensor.rank

        # Scale stabilisation is provided by the box constraint F in [0, F_MAX]
        # in _solve_block_1d: it caps the non-identifiable scale drift that the
        # over-parameterised CP would otherwise accumulate when F is warm-started
        # across EM steps. With the box in place we do NOT renormalise columns
        # here, so the warm start (both direction and magnitude) is preserved and
        # ALS reconverges in a few sweeps per M-step.

        edges_of = {}
        for ei, e in enumerate(edge_list):
            for v in e:
                edges_of.setdefault(v, []).append(ei)

        def alpha_of(e):
            prod = np.ones(R)
            for u in e:
                prod = prod * F[u]
            return float(prod.sum())

        # Convergence is judged on alpha_e (identifiable), NOT on F. The CP
        # factorisation is scale non-identifiable, so |Delta F| can stay above
        # tol forever while every alpha_e is already fixed; tracking alpha
        # avoids that and stops as soon as the quantity we actually want has
        # converged.
        alpha_prev = {e: alpha_of(e) for e in edge_list}
        for _ in range(als_max_iter):
            for v, eis in edges_of.items():
                for r in range(R):
                    gs, hs, As, Ls = [], [], [], []
                    for ei in eis:
                        e = edge_list[ei]
                        g = 1.0
                        for u in e:
                            if u != v:
                                g *= F[u, r]
                        a_cur = alpha_of(e)
                        h = a_cur - F[v, r] * g
                        if h < 0.0:
                            h = 0.0  # guard tiny negative round-off
                        gs.append(g); hs.append(h)
                        As.append(A[e]); Ls.append(L[e])
                    x_new = self._solve_block_1d(gs, hs, As, Ls, F[v, r])
                    if x_new < 0.0:
                        x_new = 0.0
                    elif x_new > self.F_MAX:
                        x_new = self.F_MAX
                    F[v, r] = x_new
            alpha_now = {e: alpha_of(e) for e in edge_list}
            max_rel = 0.0
            for e in edge_list:
                denom = max(abs(alpha_prev[e]), 1e-12)
                max_rel = max(max_rel, abs(alpha_now[e] - alpha_prev[e]) / denom)
            alpha_prev = alpha_now
            if max_rel < als_tol:
                break

        return {e: alpha_of(e) for e in edge_list}