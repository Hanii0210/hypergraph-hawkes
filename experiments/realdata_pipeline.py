"""Shared event-reader, independent node-screening and initialisation helpers."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from inference.em import EMResult
Event = Tuple[float, int]
Edge = Tuple[int, ...]


def as_1d_float(x: Any) -> np.ndarray:
    """Convert a MATLAB/HDF/list object to a sorted finite 1D array."""
    if x is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(x, dtype=float).ravel()
    except Exception:
        return np.array([], dtype=float)
    arr = arr[np.isfinite(arr)]
    return np.sort(arr)


def select_fixed_subset(
    trains: Sequence[np.ndarray],
    top_n: int,
    span_start: float,
    span_end: float,
    original_labels: Optional[Sequence[Any]] = None,
    label_key: str = "original_index",
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Select top-N trains using only an independent screening span."""
    if span_end <= span_start:
        raise ValueError("screening span must have positive length")
    if top_n <= 0 or top_n > len(trains):
        raise ValueError("top_n must lie between 1 and the number of trains")

    counts = []
    for idx, train in enumerate(trains):
        arr = as_1d_float(train)
        count = int(np.sum((arr >= span_start) & (arr < span_end)))
        counts.append((idx, count))
    selected = sorted(counts, key=lambda x: (-x[1], x[0]))[: int(top_n)]
    selected_indices = [idx for idx, _ in selected]

    labels = list(range(len(trains))) if original_labels is None else list(original_labels)
    mapping: List[Dict[str, Any]] = []
    for new_node, (source_index, count) in enumerate(selected):
        mapping.append(
            {
                "new_node": int(new_node),
                label_key: labels[source_index],
                "source_index": int(source_index),
                "count_in_screening_span": int(count),
                "screening_span_start": float(span_start),
                "screening_span_end": float(span_end),
            }
        )
    return selected_indices, mapping


def _base_initial_state(
    events: Sequence[Event],
    n_nodes: int,
    interval_start: float,
    interval_end: float,
    edge_list: Sequence[Edge],
    seed: int,
    warm: Optional[EMResult] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[Edge, float]]:
    rng = np.random.default_rng(int(seed))
    duration = float(interval_end - interval_start)
    counts = np.zeros(int(n_nodes), dtype=float)
    for t, node in events:
        inside = (t >= interval_start if interval_start == 0.0 else t > interval_start) and t <= interval_end
        if inside:
            counts[int(node)] += 1.0
    empirical_rate = np.maximum(counts / max(duration, 1e-12), 1e-3)

    if warm is None:
        mu = empirical_rate * rng.uniform(0.35, 0.75, size=n_nodes)
        alpha_pairwise = rng.uniform(0.005, 0.08, size=(n_nodes, n_nodes))
        alpha_hyper = {tuple(e): float(rng.uniform(0.01, 0.08)) for e in edge_list}
    else:
        scale_mu = rng.lognormal(mean=0.0, sigma=0.08, size=n_nodes)
        scale_ap = rng.lognormal(mean=0.0, sigma=0.08, size=(n_nodes, n_nodes))
        warm_mu = np.asarray(warm.mu, dtype=float)
        mu = np.maximum(warm_mu * (scale_mu[:, None] if warm_mu.ndim == 2 else scale_mu), 1e-6)
        alpha_pairwise = np.maximum(
            np.asarray(warm.alpha_pairwise, dtype=float) * scale_ap, 1e-8
        )
        alpha_hyper = {}
        for edge in edge_list:
            inherited = float(warm.alpha_hyper.get(tuple(edge), 0.0))
            if inherited > 0.0:
                alpha_hyper[tuple(edge)] = float(
                    max(inherited * rng.lognormal(0.0, 0.12), 1e-8)
                )
            else:
                alpha_hyper[tuple(edge)] = float(rng.uniform(0.01, 0.08))

    return mu, alpha_pairwise, alpha_hyper
