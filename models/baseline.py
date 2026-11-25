"""Non-negative temporal bases with analytic integrals and simulation bounds."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class BaselineBasis:
    kind: str = "constant"
    origin: float = 0.0
    end: float | None = None
    period: float | None = None

    def __post_init__(self):
        if self.kind not in {"constant", "sinusoidal", "linear", "quadratic"}:
            raise ValueError("unknown baseline basis")
        if not np.isfinite(self.origin):
            raise ValueError("baseline origin must be finite")
        if self.kind == "sinusoidal" and (
            self.period is None or not np.isfinite(self.period) or self.period <= 0
        ):
            raise ValueError("sinusoidal basis requires a positive finite period")
        if self.kind in {"linear", "quadratic"} and (
            self.end is None or not np.isfinite(self.end) or self.end <= self.origin
        ):
            raise ValueError("Bernstein basis requires end > origin")

    @property
    def size(self):
        return {"constant": 1, "sinusoidal": 2, "linear": 2, "quadratic": 3}[self.kind]

    def values(self, times):
        t = np.asarray(times, dtype=float).reshape(-1)
        if not np.all(np.isfinite(t)):
            raise ValueError("baseline evaluation times must be finite")
        if self.kind == "constant":
            return np.ones((len(t), 1))
        if self.kind == "sinusoidal":
            s = np.sin(2 * np.pi * (t - self.origin) / self.period)
            return np.column_stack((1 + s, 1 - s))
        # Outside the predeclared window, hold the endpoint value constant.
        # In particular this defines the linear experiment's prehistory.
        u = np.clip((t - self.origin) / (self.end - self.origin), 0, 1)
        if self.kind == "linear":
            return np.column_stack((1 - u, u))
        return np.column_stack(((1 - u)**2, 2*u*(1-u), u**2))

    def integral(self, start, end):
        a, b = float(start), float(end)
        if not np.isfinite([a, b]).all() or b <= a:
            raise ValueError("integration interval must be finite with end > start")
        if self.kind == "constant":
            return np.array([b - a])
        if self.kind == "sinusoidal":
            w = 2*np.pi/self.period
            s = (np.cos(w*(a-self.origin))-np.cos(w*(b-self.origin))) / w
            return np.array([b-a+s, b-a-s])
        d = self.end - self.origin
        out = np.zeros(self.size)
        out[0] += max(0.0, min(b, self.origin)-a)
        out[-1] += max(0.0, b-max(a, self.end))
        lo, hi = max(a, self.origin), min(b, self.end)
        if hi > lo:
            def primitive(u):
                if self.kind == "linear":
                    return np.array([u-u*u/2, u*u/2])
                return np.array([u-u*u+u**3/3, u*u-2*u**3/3, u**3/3])
            out += d*(primitive((hi-self.origin)/d)-primitive((lo-self.origin)/d))
        return np.maximum(out, 0.0)

    def constant_coefficients(self, rates):
        rates = np.asarray(rates, dtype=float).reshape(-1)
        scale = 0.5 if self.kind == "sinusoidal" else 1.0
        return np.repeat((rates*scale)[:, None], self.size, axis=1)

    def lower_bounds(self, n_nodes, rate_floor=1e-6):
        return self.constant_coefficients(np.full(n_nodes, rate_floor))

    def as_dict(self):
        return asdict(self)


class TemporalBaseline:
    def __init__(self, coefficients, basis=None):
        self.basis = BaselineBasis() if basis is None else basis
        c = np.asarray(coefficients, dtype=float)
        if c.ndim == 1:
            if self.basis.size != 1:
                raise ValueError("time-varying coefficients must have shape (nodes, basis size)")
            c = c[:, None]
        if c.ndim != 2 or c.shape[0] < 1 or c.shape[1] != self.basis.size:
            raise ValueError("invalid baseline coefficient shape")
        if not np.isfinite(c).all() or np.any(c < 0):
            raise ValueError("baseline coefficients must be finite and nonnegative")
        self.coefficients = c.copy()

    @property
    def n_nodes(self):
        return self.coefficients.shape[0]

    def values(self, times):
        return self.basis.values(times) @ self.coefficients.T

    def integral(self, start, end):
        return self.coefficients @ self.basis.integral(start, end)

    def upper_bound(self):
        """Per-node bound valid over the entire declared time axis."""
        if self.basis.kind == "sinusoidal":
            return 2*np.max(self.coefficients, axis=1)
        return np.max(self.coefficients, axis=1)


def as_baseline(value, basis=None):
    if isinstance(value, TemporalBaseline):
        if basis is not None and basis != value.basis:
            raise ValueError("conflicting baseline bases")
        return value
    return TemporalBaseline(value, basis)
