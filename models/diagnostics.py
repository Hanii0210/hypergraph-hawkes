"""Sufficient-condition checks, not tests for explosion or stationarity."""
import numpy as np
from models.baseline import as_baseline


def stability_diagnostics(mu, alpha_pairwise, alpha_hyper, beta, baseline_basis=None):
    background = as_baseline(mu, baseline_basis)
    a = np.asarray(alpha_pairwise, dtype=float)
    if a.shape != (background.n_nodes, background.n_nodes) or beta <= 0:
        raise ValueError("invalid pairwise matrix or decay")
    if not np.isfinite(a).all() or np.any(a < 0) or not np.isfinite(beta):
        raise ValueError("invalid stability parameters")
    h = np.zeros(background.n_nodes)
    for edge, weight in alpha_hyper.items():
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("invalid hyperedge weight")
        for node in edge:
            h[node] += weight
    b = a.T / beta
    rho = float(np.max(np.abs(np.linalg.eigvals(b))))
    stationary = background.basis.kind == "constant" and rho < 1.0
    bound = np.linalg.solve(np.eye(len(h))-b, background.coefficients[:,0]+h) if stationary else None
    return {"rho_pairwise": rho, "pairwise_subcritical": rho < 1,
            "maximum_source_row_sum": float((a/beta).sum(axis=1).max()),
            "maximum_target_column_sum": float((a/beta).sum(axis=0).max()),
            "hyperedge_upper_per_node": h.tolist(),
            "stationary_existence_sufficient": stationary,
            "stationary_mean_upper": None if bound is None else bound.tolist(),
            "baseline_kind": background.basis.kind,
            "interpretation": "sufficient condition only; no claim of stationary uniqueness"}
