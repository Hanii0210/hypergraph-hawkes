import numpy as np


class HawkesSimulator:
    """
    Simulates a Hyperedge-triggered Hawkes (HTH) process.

    Conditional intensity for node n at time t:

        lambda_n(t) = mu_n
                    + sum_{j: t_j < t} alpha_pairwise[n_j, n] * phi(t - t_j)
                    + sum_{e in edges, n in e} alpha_e
                          * sum_{k anchor} phi(t - t_anchor_k)

    Each hyperedge contribution is added to every member node once, which
    matches the intensity decomposition used by the E-step.

    Simulation uses Ogata's thinning algorithm with a refreshing upper
    bound and a hard event cap to handle super-critical regimes safely.
    """

    def __init__(self, mu, alpha_pairwise, alpha_hyper, kernel, anchor_calc):
        self.mu             = np.asarray(mu, dtype=float)
        self.alpha_pairwise = np.asarray(alpha_pairwise, dtype=float)
        self.alpha_hyper    = alpha_hyper
        self.kernel         = kernel
        self.anchor_calc    = anchor_calc
        self.n_nodes        = len(mu)

    def _intensity(self, t: float, events: list) -> np.ndarray:
        """
        Compute lambda(t) given the event history.

        Parameters
        ----------
        t      : float, current time
        events : list of (time, node) observed strictly before t

        Returns
        -------
        np.ndarray of shape (N,) with per-node intensity at t
        """
        lam = self.mu.copy()

        # Pairwise contributions
        for t_j, node_j in events:
            if t_j >= t:
                break
            tau = t - t_j
            lam += self.alpha_pairwise[node_j] * self.kernel(np.array([tau]))[0]

        # Hyperedge contributions
        event_times_by_node = {}
        for t_j, node_j in events:
            if node_j not in event_times_by_node:
                event_times_by_node[node_j] = []
            event_times_by_node[node_j].append(t_j)

        for e, alpha_e in self.alpha_hyper.items():
            anchors = self.anchor_calc.find_anchors(e, event_times_by_node, t)
            if len(anchors) == 0:
                continue
            taus = np.array([t - t_a for t_a in anchors])
            contribution = alpha_e * float(self.kernel(taus).sum())
            for node_i in e:
                lam[node_i] += contribution

        return lam

    def simulate(self, T: float, seed: int = 0,
                 max_events: int = 2000) -> list:
        """
        Generate an event sequence on [0, T] via Ogata thinning.

        Parameters
        ----------
        T          : float, observation window length
        seed       : int,   random seed
        max_events : int,   hard cap to prevent super-critical explosion

        Returns
        -------
        list of (time, node) tuples sorted by time
        """
        rng    = np.random.default_rng(seed)
        events = []
        t      = 0.0

        lambda_bar = float(self.mu.sum()) + 1.0

        while t < T:
            if len(events) >= max_events:
                break

            dt = rng.exponential(1.0 / lambda_bar)
            t  = t + dt
            if t > T:
                break

            lam       = self._intensity(t, events)
            lam_total = float(lam.sum())
            lambda_bar = lam_total + float(self.mu.sum()) + 1.0

            u = rng.uniform()
            if u * lambda_bar <= lam_total:
                prob = lam / lam_total
                node = int(rng.choice(self.n_nodes, p=prob))
                events.append((t, node))

        return events