"""
Time-rescaling validity gate for HawkesSimulator.

Rationale
---------
The structural checks in test_simulator.py (events non-empty, in-window,
valid nodes, sorted, monotone in T) are ALL passed by a statistically WRONG
sampler -- a biased thinning still produces sorted in-window events. They
therefore cannot detect an incorrect Ogata thinning, which is exactly how a
KS p=4e-8 defect previously survived into the frozen golden master.

This module closes that hole using the time-rescaling theorem: if events are
drawn from a point process with compensator Lambda, then the rescaled
inter-event increments Lambda(t_{k+1}) - Lambda(t_k) are i.i.d. Exp(1). We
test this with a one-sample Kolmogorov-Smirnov test against Exp(1), pooled
over independent realisations, for two regimes:

  (A) 1-D self-exciting (branching 0.8) -- the high-intensity LINEAR regime
      where the old thinning failed (this is the discriminating test).
  (B) multivariate pairwise + hyperedge, per-node compensator -- the actual
      experiment regime.

A correct simulator yields p well above the threshold; the previous buggy
simulator yields p ~ 1e-8 in regime (A).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import stats

from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.likelihood import _all_completion_times
from simulation.simulator import HawkesSimulator

KS_P_MIN = 1e-3  # reject the simulator only on overwhelming evidence


def _increments_1d(times, mu, a, beta):
    """Compensator increments for a 1-D exponential-kernel Hawkes process."""
    times = np.asarray(times)
    z = []
    for k in range(len(times) - 1):
        ta, tb = times[k], times[k + 1]
        past = times[: k + 1]
        def Lam(x):
            p = past[past < x]
            return mu * x + (a / beta) * np.sum(1.0 - np.exp(-beta * (x - p)))
        z.append(Lam(tb) - Lam(ta))
    return np.array(z)


def _increments_per_node(events, mu, ap, ah, beta, delta):
    """Per-node compensator increments for the full HTH model."""
    etbn = {}
    for t, n in events:
        etbn.setdefault(n, []).append(t)
    for k in etbn:
        etbn[k] = np.asarray(etbn[k])
    comps = {e: np.asarray(_all_completion_times(e, etbn, delta)) for e in ah}

    def Lam_n(n, t):
        val = mu[n] * t
        for t_j, n_j in events:
            if t_j >= t:
                break
            val += ap[n_j, n] * (1.0 / beta) * (1.0 - np.exp(-beta * (t - t_j)))
        for e, a_e in ah.items():
            if n not in e:
                continue
            c = comps[e]
            c = c[c < t]
            for k in range(len(c)):
                ts = c[k]
                te = c[k + 1] if k + 1 < len(c) else t
                te = min(te, t)
                if te > ts:
                    val += a_e * (1.0 / beta) * (1.0 - np.exp(-beta * (te - ts)))
        return val

    z = []
    for n in range(len(mu)):
        tn = sorted(etbn.get(n, []))
        for k in range(len(tn) - 1):
            z.append(Lam_n(n, tn[k + 1]) - Lam_n(n, tn[k]))
    return np.array(z)


def test_time_rescaling_1d_self_exciting():
    """1-D, branching 0.8: discriminating regime for the thinning bug."""
    mu, a, beta, T, n_seeds = 0.5, 0.8, 1.0, 300.0, 18
    theo_rate = mu / (1.0 - a / beta)
    kernel = ExponentialKernel(beta=beta)
    anchor = HyperedgeAnchor(delta=0.5)
    sim = HawkesSimulator(np.array([mu]), np.array([[a]]), {}, kernel, anchor)

    rates, all_z = [], []
    for s in range(n_seeds):
        ev = sim.simulate(T=T, seed=s, max_events=10 ** 7)
        tm = [t for t, _ in ev]
        rates.append(len(tm) / T)
        if len(tm) > 5:
            all_z.append(_increments_1d(tm, mu, a, beta))
    all_z = np.concatenate(all_z)
    ks = stats.kstest(all_z, "expon")

    assert ks.pvalue > KS_P_MIN, (
        f"Simulator fails time-rescaling KS in 1-D branching-0.8 regime: "
        f"p={ks.pvalue:.2e} (mean increment {all_z.mean():.3f}, should be ~1). "
        f"The thinning is not a valid Ogata sampler."
    )
    assert abs(np.mean(rates) - theo_rate) / theo_rate < 0.10, (
        f"Stationary rate {np.mean(rates):.3f} deviates >10% from theory "
        f"{theo_rate:.3f}."
    )


def test_time_rescaling_multivariate_with_hyperedge():
    """3-node pairwise + hyperedge, per-node compensator (experiment regime)."""
    mu = np.array([0.3, 0.3, 0.3])
    ap = np.zeros((3, 3))
    ap[2, 0] = 0.3
    ah = {(0, 1): 0.4}
    beta, delta, T, n_seeds = 1.0, 0.5, 300.0, 20
    kernel = ExponentialKernel(beta=beta)
    anchor = HyperedgeAnchor(delta=delta)
    sim = HawkesSimulator(mu, ap, ah, kernel, anchor)

    all_z = []
    for s in range(n_seeds):
        ev = sim.simulate(T=T, seed=s, max_events=20000)
        if len(ev) > 30:
            all_z.append(_increments_per_node(ev, mu, ap, ah, beta, delta))
    all_z = np.concatenate(all_z)
    ks = stats.kstest(all_z, "expon")

    assert ks.pvalue > KS_P_MIN, (
        f"Simulator fails per-node time-rescaling KS in the multivariate "
        f"pairwise+hyperedge regime: p={ks.pvalue:.2e} "
        f"(mean increment {all_z.mean():.3f}, should be ~1)."
    )


if __name__ == "__main__":
    test_time_rescaling_1d_self_exciting()
    test_time_rescaling_multivariate_with_hyperedge()
    print("Both time-rescaling gates PASS.")