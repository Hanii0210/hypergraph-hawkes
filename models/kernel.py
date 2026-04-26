import numpy as np


class ExponentialKernel:
    """
    Exponential decay kernel: phi(tau) = exp(-beta * tau)

    Parameters
    ----------
    beta : float
        Decay rate, must be > 0. Larger beta means faster decay of past influence.
    """

    def __init__(self, beta: float):
        assert beta > 0, "beta must be positive"
        self.beta = beta

    def __call__(self, tau: np.ndarray) -> np.ndarray:
        """
        Evaluate the kernel at time differences tau.

        Parameters
        ----------
        tau : array-like, all elements >= 0

        Returns
        -------
        np.ndarray : exp(-beta * tau), same shape as tau
        """
        tau = np.asarray(tau, dtype=float)
        assert np.all(tau >= 0), "tau must be non-negative"
        return np.exp(-self.beta * tau)

    def integral(self, t_anchor: float, T: float) -> float:
        """
        Closed-form compensator integral from t_anchor to T:
            integral_{t_anchor}^{T} exp(-beta * (t - t_anchor)) dt
            = (1 / beta) * (1 - exp(-beta * (T - t_anchor)))

        Parameters
        ----------
        t_anchor : float  anchor event time
        T        : float  end of observation window

        Returns
        -------
        float : compensator value for this anchor
        """
        assert T >= t_anchor, "T must be >= t_anchor"
        return (1.0 / self.beta) * (1.0 - np.exp(-self.beta * (T - t_anchor)))


class HyperedgeAnchor:
    """
    Computes pattern-completion anchor times for a hyperedge.

    A hyperedge e = {v1, v2, ..., vk} fires at time t if all members
    have an event within a time window of width delta ending at t.
    The anchor time is the time of the last member to arrive (the max).

    Only the single most recent anchor is returned per query. This prevents
    intensity inflation from accumulating many historical pattern completions,
    which would cause the hyperedge term to dominate the likelihood.

    Parameters
    ----------
    delta : float
        Window width. All member events must fall within delta to count.
    reuse : bool
        If True (default), the same anchor can contribute to multiple
        future events. This preserves the additive Hawkes structure and
        keeps the EM derivation exact.
        If False (single-use), the likelihood form changes non-trivially.
        Not implemented yet; reserved for future extension.
    """

    def __init__(self, delta: float, reuse: bool = True):
        assert delta > 0, "delta must be positive"
        self.delta = delta
        self.reuse = reuse

        if not reuse:
            raise NotImplementedError(
                "reuse=False requires a modified likelihood structure. "
                "Only reuse=True is supported in this version."
            )

    def find_anchors(
        self,
        edge: tuple,
        event_times: dict,
        t_current: float
    ) -> list:
        """
        Find the most recent anchor time for a hyperedge before t_current.

        Only the single most recent valid anchor is returned to prevent
        cumulative intensity inflation from stale pattern completions.

        Parameters
        ----------
        edge        : tuple of int, e.g. (0, 1) or (0, 1, 2)
        event_times : dict mapping node index -> list of event times
        t_current   : float, only events strictly before this time are used

        Returns
        -------
        list of float : at most one element (the most recent anchor time),
                        or empty list if no pattern completion found
        """
        member_events = {}
        for v in edge:
            if v not in event_times:
                return []
            past = [t for t in event_times[v] if t < t_current]
            if len(past) == 0:
                return []
            member_events[v] = past

        anchors = []

        for anchor_node in edge:
            for t_last in member_events[anchor_node]:
                window_start = t_last - self.delta
                pattern_complete = True

                for v in edge:
                    if v == anchor_node:
                        continue
                    in_window = [
                        t for t in member_events[v]
                        if window_start <= t <= t_last
                    ]
                    if len(in_window) == 0:
                        pattern_complete = False
                        break

                if pattern_complete:
                    anchors.append(t_last)

        if len(anchors) == 0:
            return []

        # Return only the most recent anchor to prevent intensity inflation
        return [max(anchors)]