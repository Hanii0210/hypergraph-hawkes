"""Fixed history features; memory is O(events * (nodes + candidates))."""
import numpy as np
from models.baseline import BaselineBasis


def validate_events(events, n_nodes):
    rows = np.asarray(list(events), dtype=float)
    if rows.size == 0:
        return np.empty(0), np.empty(0, dtype=int)
    if rows.ndim != 2 or rows.shape[1] != 2 or not np.isfinite(rows).all():
        raise ValueError("events must be finite (time, node) pairs")
    t, v = rows[:, 0], rows[:, 1]
    if np.any(np.diff(t) < 0) or np.any(v != np.floor(v)):
        raise ValueError("events must be time sorted with integer nodes")
    if np.any(v < 0) or np.any(v >= n_nodes):
        raise ValueError("event node outside model")
    return t, v.astype(int)


class EventFeatures:
    def __init__(self, events, n_nodes, edge_list, beta, delta,
                 interval_start, interval_end, basis=None, include_start=None):
        # Local import keeps public likelihood helper APIs backwards compatible.
        from models.likelihood import pairwise_event_exposures, piecewise_compensator_interval, all_completion_times
        self.n_nodes = int(n_nodes)
        self.beta, self.delta = float(beta), float(delta)
        self.start, self.end = float(interval_start), float(interval_end)
        if self.n_nodes < 1 or not np.isfinite([self.beta, self.delta, self.start, self.end]).all():
            raise ValueError("invalid model dimensions or interval")
        if self.beta <= 0 or self.delta <= 0 or self.end <= self.start:
            raise ValueError("positive beta, delta and interval duration required")
        self.basis = BaselineBasis() if basis is None else basis
        self.edges = [tuple(sorted(e)) for e in edge_list]
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("duplicate candidates")
        for e in self.edges:
            if len(e) < 2 or len(set(e)) != len(e) or any(int(v) != v or not 0 <= v < self.n_nodes for v in e):
                raise ValueError("invalid candidate edge")
        times, nodes = validate_events(events, self.n_nodes)
        keep = times <= self.end
        times, nodes = times[keep], nodes[keep]
        self.include_start = self.start == 0 if include_start is None else bool(include_start)
        target = (times >= self.start if self.include_start else times > self.start)
        self.times, self.nodes = times[target], nodes[target]
        self.pair = np.empty((int(target.sum()), self.n_nodes))
        z = np.zeros(self.n_nodes)
        previous = times[0] if len(times) else self.start
        i = cursor = 0
        while i < len(times):
            t = times[i]
            j = int(np.searchsorted(times, t, side="right"))
            z *= np.exp(-self.beta*(t-previous))
            # Every tied record sees the same strict past.
            if target[i]:
                self.pair[cursor:cursor+j-i] = z
                cursor += j-i
            z += np.bincount(nodes[i:j], minlength=self.n_nodes)
            previous, i = t, j
        self.pair_exposure = np.bincount(
            nodes, weights=pairwise_event_exposures(times, self.start, self.end, self.beta),
            minlength=self.n_nodes,
        )
        self.hyper = np.zeros((len(self.times), len(self.edges)))
        self.hyper_exposure = np.zeros(len(self.edges))
        self.completion_counts = np.zeros(len(self.edges), dtype=int)
        per_node = {n: times[nodes == n] for n in range(self.n_nodes)}
        for k, e in enumerate(self.edges):
            c = all_completion_times(e, per_node, self.delta)
            self.completion_counts[k] = np.sum((c > self.start) & (c <= self.end))
            self.hyper_exposure[k] = len(e)*piecewise_compensator_interval(c, self.start, self.end, self.beta)
            index = np.searchsorted(c, self.times, side="left") - 1
            use = (index >= 0) & np.isin(self.nodes, e)
            if np.any(use):
                self.hyper[use, k] = np.exp(-self.beta*(self.times[use]-c[index[use]]))
        self.background = self.basis.values(self.times)
        self.background_exposure = self.basis.integral(self.start, self.end)

    def weights(self, coefficients, pairwise, hyper):
        b = np.asarray(coefficients, dtype=float)
        if b.ndim == 1:
            b = b[:, None]
        a = np.asarray(pairwise, dtype=float)
        h = np.array([hyper.get(e, 0.0) for e in self.edges], dtype=float)
        if b.shape != (self.n_nodes, self.basis.size) or a.shape != (self.n_nodes, self.n_nodes):
            raise ValueError("coefficient shape does not match cached features")
        if any(not np.isfinite(x).all() or np.any(x < 0) for x in (b,a,h)):
            raise ValueError("model weights must be finite and nonnegative")
        return b,a,h

    def intensity(self, coefficients, pairwise, hyper):
        b,a,h = self.weights(coefficients, pairwise, hyper)
        lam = (self.background*b[self.nodes]).sum(axis=1)
        lam += (self.pair*a[:, self.nodes].T).sum(axis=1)
        lam += self.hyper @ h
        if not np.isfinite(lam).all() or np.any(lam <= 0):
            raise FloatingPointError("non-positive or non-finite event intensity")
        return lam

    def score(self, coefficients, pairwise, hyper):
        b,a,h = self.weights(coefficients, pairwise, hyper)
        event = float(np.log(self.intensity(b,a,hyper)).sum())
        background = float(b.sum(axis=0) @ self.background_exposure)
        pair = float(a.sum(axis=1) @ self.pair_exposure)
        higher = float(h @ self.hyper_exposure)
        integral = background + pair + higher
        if not np.isfinite(integral):
            raise FloatingPointError("non-finite compensator")
        return {"event_log_intensity": event, "baseline_integral": background,
                "pairwise_integral": pair, "hyperedge_integral": higher,
                "integral": integral, "loglik": event-integral,
                "n_events": len(self.times), "duration": self.end-self.start}

    def statistics(self, coefficients, pairwise, hyper):
        b,a,h = self.weights(coefficients,pairwise,hyper)
        inv = 1.0/self.intensity(b,a,hyper)
        sb = np.zeros_like(b)
        sa = np.zeros_like(a)
        np.add.at(sb, self.nodes, self.background*inv[:,None])
        np.add.at(sa.T, self.nodes, self.pair*inv[:,None])
        sh = self.hyper.T @ inv
        return {"background_counts": b*sb, "pairwise_counts": a*sa,
                "hyperedge_counts": h*sh,
                "background_score": sb-self.background_exposure,
                "pairwise_score": sa-self.pair_exposure[:,None],
                "hyperedge_score": sh-self.hyper_exposure}

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self.__dict__.values() if isinstance(x, np.ndarray))
