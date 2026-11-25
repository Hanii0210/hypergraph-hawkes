"""Ogata thinning for the most-recent-completion HTH intensity.

The proposal rate is a predictable bound valid until the next proposal; the
same bound is used in the acceptance ratio. See Ogata (1981), pp. 24-27.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from models.baseline import as_baseline


@dataclass
class SimulationResult:
    events: list
    completed: bool
    status: str
    requested_end: float
    actual_end: float
    history_start: float
    max_events: int
    seed: int
    n_proposals: int
    n_rejected: int
    maximum_acceptance_ratio: float
    elapsed_seconds: float
    proposal_trace: list

    def as_dict(self):
        return asdict(self)


class SimulationTruncatedError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(f"event cap reached at {result.actual_end:g}, before {result.requested_end:g}; partial path is not complete")


class HawkesSimulator:
    """Simulate non-accumulating completion excitation on all edge members."""
    def __init__(self, mu, alpha_pairwise, alpha_hyper, kernel, anchor_calc, baseline_basis=None):
        self.baseline = as_baseline(mu, baseline_basis)
        self.mu = self.baseline.coefficients[:,0].copy() if self.baseline.basis.kind == "constant" else self.baseline.coefficients.copy()
        self.alpha_pairwise = np.asarray(alpha_pairwise,dtype=float).copy()
        self.alpha_hyper = {tuple(sorted(e)):float(v) for e,v in alpha_hyper.items()}
        if len(self.alpha_hyper) != len(alpha_hyper):
            raise ValueError("duplicate candidate edges")
        self.kernel, self.anchor_calc = kernel, anchor_calc
        self.n_nodes = self.baseline.n_nodes
        self.beta = float(kernel.beta)
        if self.alpha_pairwise.shape != (self.n_nodes,self.n_nodes) or self.beta <= 0 or not np.isfinite(self.beta):
            raise ValueError("invalid matrix or decay")
        if not np.isfinite(self.alpha_pairwise).all() or np.any(self.alpha_pairwise < 0):
            raise ValueError("pairwise weights must be finite and nonnegative")
        self.edges = list(self.alpha_hyper)
        self.weights = np.array([self.alpha_hyper[e] for e in self.edges])
        if not np.isfinite(self.weights).all() or np.any(self.weights < 0):
            raise ValueError("hyperedge weights must be finite and nonnegative")
        self.membership = np.zeros((len(self.edges),self.n_nodes))
        for k,e in enumerate(self.edges):
            if len(e)<2 or len(set(e))!=len(e) or any(int(v)!=v or not 0<=v<self.n_nodes for v in e):
                raise ValueError("invalid hyperedge")
            self.membership[k,list(e)] = 1
        self.incident = [np.flatnonzero(self.membership[:,n]).tolist() for n in range(self.n_nodes)]

    def _intensity(self,t,events):
        """Small-history reference, preserving the existing diagnostic API."""
        value = self.baseline.values([t])[0].copy()
        per_node = {}
        for s,n in events:
            if s >= t:
                break
            value += self.alpha_pairwise[n]*np.exp(-self.beta*(t-s))
            per_node.setdefault(n,[]).append(s)
        for e,w in self.alpha_hyper.items():
            anchors = self.anchor_calc.find_anchors(e,per_node,t)
            if anchors:
                value[list(e)] += w*np.exp(-self.beta*(t-anchors[0]))
        return value

    def _intensity_upper_bound(self,t,events):
        value = self.baseline.upper_bound().copy()
        per_node = {}
        for s,n in events:
            if s > t:
                break
            value += self.alpha_pairwise[n]*np.exp(-self.beta*(t-s))
            per_node.setdefault(n,[]).append(s)
        for e,w in self.alpha_hyper.items():
            anchors = self.anchor_calc.find_anchors(e,per_node,np.nextafter(float(t),np.inf))
            if anchors:
                value[list(e)] += w*np.exp(-self.beta*(t-anchors[0]))
        return float(value.sum())

    def simulate(self,T,seed=0,max_events=2000,*,burn_in=0.0,return_info=False,record_trace=False):
        """Generate on (-burn_in,T]; retain negative-time conditioning history.

        Legacy calls return an event list if complete. A truncated legacy call
        raises SimulationTruncatedError instead of silently returning a partial
        record. With return_info=True, partial paths have explicit status.
        """
        T,burn_in = float(T),float(burn_in)
        if not np.isfinite([T,burn_in]).all() or T <= 0 or burn_in < 0:
            raise ValueError("invalid simulation horizon")
        if int(max_events) != max_events or max_events < 1:
            raise ValueError("positive integer event cap required")
        started = perf_counter()
        rng = np.random.default_rng(seed)
        t = -burn_in
        events,trace = [],[]
        pair = np.zeros(self.n_nodes)
        q = np.zeros(len(self.edges))
        last = np.full(self.n_nodes,-np.inf)
        baseline_bound = float(self.baseline.upper_bound().sum())
        proposals = rejected = 0
        maximum_ratio = 0.0
        completed = False
        while t < T and len(events) < max_events:
            higher = (q*self.weights) @ self.membership
            bound = baseline_bound + float(pair.sum()+higher.sum())
            if not np.isfinite(bound) or bound < 0:
                raise FloatingPointError("invalid proposal bound")
            if bound == 0:
                t,completed = T,True
                break
            proposal = t+float(rng.exponential(1.0/bound))
            if proposal <= t:
                raise FloatingPointError("simulation clock did not advance")
            if proposal > T:
                t,completed = T,True
                break
            decay = np.exp(-self.beta*(proposal-t))
            pair *= decay
            q *= decay
            t = proposal
            lam = self.baseline.values([t])[0] + pair + (q*self.weights) @ self.membership
            total = float(lam.sum())
            ratio = total/bound
            if not np.isfinite(ratio) or ratio < 0 or ratio > 1+1e-12:
                raise FloatingPointError(f"invalid thinning acceptance ratio: {ratio}")
            maximum_ratio = max(maximum_ratio,ratio)
            proposals += 1
            accepted = bool(rng.uniform()*bound < total)
            if record_trace:
                trace.append({"time":t,"bound":bound,"intensity":total,"ratio":ratio,"accepted":accepted})
            if accepted:
                node = int(rng.choice(self.n_nodes,p=lam/total))
                events.append((t,node))
                pair += self.alpha_pairwise[node]
                last[node] = t
                for k in self.incident[node]:
                    if np.all(last[list(self.edges[k])] >= t-self.anchor_calc.delta):
                        q[k] = 1.0  # Replace, never accumulate, old completion.
            else:
                rejected += 1
        if t >= T:
            completed = True
        result = SimulationResult(events,completed,"completed" if completed else "event_cap",
            T,t,-burn_in,int(max_events),int(seed),proposals,rejected,maximum_ratio,
            perf_counter()-started,trace)
        self.last_result = result
        if return_info:
            return result
        if not completed:
            raise SimulationTruncatedError(result)
        return events
