"""
Shared real-data pipeline for HTH sample-split/window-stability experiments.

This module centralises the common logic that was duplicated across:
    - exp16_pvc3_window_stability.py
    - exp17_pvc11_window_stability.py
    - exp18_ret1_window_stability.py
    - smoke_hth_bic_checked.py

It intentionally does NOT contain dataset-specific loaders. Dataset scripts should:
    1. load spike trains as a list of 1D numpy arrays in a common time unit;
    2. build a mapping list for selected neurons/cells;
    3. call run_window_grid(...).

Primary model-comparison statistic:
    candidate-count BIC difference

        2 * (logL_HTH - logL_pairwise) - n_candidates * log(n_events)

Positive values favour HTH. Active-edge-count BIC is diagnostic only.

The implementation uses the existing exact E-step and therefore remains O(n_events^2).
Use moderate windows/subsets for real data.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from models.likelihood import log_likelihood
from inference.e_step import EStep
from inference.m_step import MStep
from inference.candidate_filter import fit_pairwise_only, generate_candidate_hyperedges


Event = Tuple[float, int]
Edge = Tuple[int, ...]
Row = Dict[str, Any]


# ---------------------------------------------------------------------
# Small parsing / I/O helpers
# ---------------------------------------------------------------------

def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def as_1d_float(x: Any) -> np.ndarray:
    """Convert a MATLAB/HDF/list object to a sorted finite 1D float array."""
    if x is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(x, dtype=float).ravel()
    except Exception:
        return np.array([], dtype=float)
    arr = arr[np.isfinite(arr)]
    return np.sort(arr)


def load_events_csv(path: str | Path) -> List[Event]:
    """Load a time,node CSV event stream."""
    events: List[Event] = []
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}. Expected columns: time,node")
        names = {c.lower(): c for c in reader.fieldnames}
        if "time" not in names or "node" not in names:
            raise ValueError(f"CSV must contain time,node columns. Found: {reader.fieldnames}")
        for row in reader:
            t = float(row[names["time"]])
            node = int(row[names["node"]])
            if np.isfinite(t):
                events.append((t, node))
    events.sort(key=lambda z: (z[0], z[1]))
    return events


def write_rows_csv(rows: Sequence[Row], path: str | Path) -> None:
    """Write a list of dict rows to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as f:
            f.write("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Event construction and splitting
# ---------------------------------------------------------------------

def infer_n_nodes(events: Sequence[Event]) -> int:
    return 0 if not events else max(n for _, n in events) + 1


def count_events_by_node(events: Sequence[Event], n_nodes: Optional[int] = None) -> List[int]:
    if n_nodes is None:
        n_nodes = infer_n_nodes(events)
    counts = [0] * n_nodes
    for _, n in events:
        counts[int(n)] += 1
    return counts


def summarise_events(events: Sequence[Event], T: float) -> Dict[str, Any]:
    n_nodes = infer_n_nodes(events)
    counts = count_events_by_node(events, n_nodes)
    if events:
        t_min = min(t for t, _ in events)
        t_max = max(t for t, _ in events)
    else:
        t_min = float("nan")
        t_max = float("nan")
    return {
        "n_events": int(len(events)),
        "n_nodes": int(n_nodes),
        "counts": counts,
        "time_min": float(t_min),
        "time_max": float(t_max),
        "T": float(T),
        "rate_per_T": float(len(events) / T) if T > 0 else float("nan"),
    }


def select_fixed_subset(
    trains: Sequence[np.ndarray],
    top_n: int,
    span_start: float,
    span_end: float,
    original_labels: Optional[Sequence[Any]] = None,
    label_key: str = "original_index",
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Select top_n spike trains by count over [span_start, span_end).

    Returns:
        selected_indices: indices into trains
        mapping: list of dicts with new_node, original label, and count
    """
    counts = []
    for idx, t in enumerate(trains):
        arr = as_1d_float(t)
        c = int(np.sum((arr >= span_start) & (arr < span_end)))
        counts.append((idx, c))

    selected = sorted(counts, key=lambda x: x[1], reverse=True)[: int(top_n)]
    selected_indices = [idx for idx, _ in selected]

    if original_labels is None:
        original_labels = list(range(len(trains)))

    mapping = []
    for new_node, (old_idx, count) in enumerate(selected):
        mapping.append({
            "new_node": int(new_node),
            label_key: original_labels[old_idx],
            "source_index": int(old_idx),
            "count_in_analysis_span": int(count),
        })

    return selected_indices, mapping


def make_events_from_trains(
    trains: Sequence[np.ndarray],
    selected_indices: Sequence[int],
    start: float,
    duration: float,
) -> Tuple[List[Event], List[int]]:
    """Build a local event list for one window, with time shifted to window start."""
    end = float(start + duration)
    events: List[Event] = []
    counts: List[int] = []

    for new_node, old_idx in enumerate(selected_indices):
        t = as_1d_float(trains[int(old_idx)])
        w = t[(t >= start) & (t < end)] - float(start)
        counts.append(int(len(w)))
        for x in w:
            events.append((float(x), int(new_node)))

    events.sort(key=lambda z: (z[0], z[1]))
    return events, counts


def split_events(events: Sequence[Event], T: float) -> Tuple[List[Event], List[Event], float, float]:
    """Split a local event list into first-half selection and second-half inference."""
    cut = float(T) / 2.0
    selection = [(float(t), int(n)) for t, n in events if float(t) <= cut]
    inference = [(float(t - cut), int(n)) for t, n in events if float(t) > cut]
    return selection, inference, cut, float(T) - cut


# ---------------------------------------------------------------------
# Fitting and model comparison
# ---------------------------------------------------------------------

def fit_pairwise_model(
    events: Sequence[Event],
    T: float,
    n_nodes: int,
    kernel: ExponentialKernel,
    anchor_calc: HyperedgeAnchor,
    n_iter: int,
    lambda_l1: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    mu, alpha_pairwise = fit_pairwise_only(
        events=list(events),
        T=float(T),
        n_nodes=int(n_nodes),
        kernel=kernel,
        anchor_calc=anchor_calc,
        n_iter=int(n_iter),
        lambda_l1=float(lambda_l1),
        seed=int(seed),
    )
    ll = log_likelihood(list(events), float(T), mu, alpha_pairwise, {}, [], kernel, anchor_calc)
    return mu, alpha_pairwise, float(ll)


def fit_hth_model(
    events: Sequence[Event],
    T: float,
    n_nodes: int,
    candidates: Sequence[Edge],
    kernel: ExponentialKernel,
    anchor_calc: HyperedgeAnchor,
    mu_init: np.ndarray,
    alpha_pairwise_init: np.ndarray,
    n_iter: int,
    lambda_l1: float,
    seed: int,
    verbose: bool = False,
    log_every: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Dict[Edge, float], List[float]]:
    tensor = HypergraphTensor(n_nodes=int(n_nodes), rank=3, seed=int(seed))
    estep = EStep(kernel, anchor_calc)
    mstep = MStep(n_nodes=int(n_nodes), tensor=tensor, lambda_l1=float(lambda_l1))

    mu = mu_init.copy()
    alpha_pairwise = alpha_pairwise_init.copy()
    alpha_hyper: Dict[Edge, float] = {tuple(e): 0.1 for e in candidates}
    history: List[float] = []

    for it in range(1, int(n_iter) + 1):
        r = estep.compute(list(events), mu, alpha_pairwise, alpha_hyper, list(candidates))
        mu = mstep.update_mu(list(events), r["p_background"], float(T))
        alpha_pairwise = mstep.update_alpha_pairwise(
            list(events), r["p_pairwise"], r["p_hyper"], list(candidates), kernel, float(T)
        )
        alpha_hyper = mstep.update_alpha_hyper_als(
            list(events), r["p_hyper"], list(candidates), anchor_calc, kernel, float(T)
        )
        ll = log_likelihood(list(events), float(T), mu, alpha_pairwise, alpha_hyper, list(candidates), kernel, anchor_calc)
        history.append(float(ll))

        if verbose and (it == 1 or it == int(n_iter) or it % int(log_every) == 0):
            print(f"      HTH iter {it:>3}: logL={ll:.3f}")

    return mu, alpha_pairwise, alpha_hyper, history


def candidate_count_bic(delta_L: float, n_events: int, n_candidates: int) -> float:
    """Primary BIC difference: positive favours HTH."""
    if n_events <= 1:
        return float("nan")
    return float(2.0 * float(delta_L) - int(n_candidates) * np.log(int(n_events)))


def active_edges(alpha_hyper: Dict[Edge, float], threshold: float = 1e-2) -> List[Tuple[Edge, float]]:
    return [(tuple(e), float(a)) for e, a in alpha_hyper.items() if float(a) > float(threshold)]


def active_count_bic(delta_L: float, n_events: int, n_active: int) -> float:
    """Diagnostic only. Do not use as primary evidence after candidate selection."""
    if n_events <= 1:
        return float("nan")
    return float(2.0 * float(delta_L) - int(n_active) * np.log(int(n_events)))


def compare_models(
    ll_pairwise: float,
    ll_hth: float,
    candidates: Sequence[Edge],
    alpha_hyper: Dict[Edge, float],
    n_events: int,
    active_threshold: float = 1e-2,
) -> Dict[str, Any]:
    dL = float(ll_hth - ll_pairwise)
    act = active_edges(alpha_hyper, active_threshold)
    return {
        "logL_pairwise": float(ll_pairwise),
        "logL_HTH": float(ll_hth),
        "delta_L": dL,
        "n_events": int(n_events),
        "n_candidates": int(len(candidates)),
        "n_active_edges": int(len(act)),
        "bicdiff_candidate_count": candidate_count_bic(dL, int(n_events), len(candidates)),
        "bicdiff_active_count": active_count_bic(dL, int(n_events), len(act)),
        "active_edges": [(tuple(e), float(a)) for e, a in act],
    }


# ---------------------------------------------------------------------
# Exposure diagnostics
# ---------------------------------------------------------------------

def completion_times(edge: Edge, events: Sequence[Event], delta: float) -> List[float]:
    by_node: Dict[int, List[float]] = {}
    for t, node in events:
        by_node.setdefault(int(node), []).append(float(t))

    arrays = {k: np.asarray(v, dtype=float) for k, v in by_node.items()}

    completions = set()
    for anchor_node in edge:
        if anchor_node not in arrays:
            continue
        for t_last in arrays[anchor_node]:
            lo_t = t_last - float(delta)
            ok = True
            for v in edge:
                if v == anchor_node:
                    continue
                if v not in arrays:
                    ok = False
                    break
                arr = arrays[v]
                lo = np.searchsorted(arr, lo_t, side="left")
                hi = np.searchsorted(arr, t_last, side="right")
                if hi - lo == 0:
                    ok = False
                    break
            if ok:
                completions.add(float(t_last))

    return sorted(completions)


def piecewise_C(completions: Sequence[float], T: float, beta: float) -> float:
    if not completions:
        return 0.0

    comps = sorted(float(x) for x in completions)
    total = 0.0
    for k, t0 in enumerate(comps):
        t1 = comps[k + 1] if k + 1 < len(comps) else float(T)
        if t1 > t0:
            total += (1.0 / float(beta)) * (1.0 - np.exp(-float(beta) * (t1 - t0)))
    return float(total)


def exposure_diagnostics(
    events: Sequence[Event],
    T: float,
    candidates: Sequence[Edge],
    beta: float,
    delta: float,
    alpha_hyper: Optional[Dict[Edge, float]] = None,
    min_completions_threshold: int = 20,
    min_spikes_threshold: int = 20,
    min_compensator_threshold: float = 1e-3,
) -> List[Dict[str, Any]]:
    counts_by_node: Dict[int, int] = {}
    for _, node in events:
        counts_by_node[int(node)] = counts_by_node.get(int(node), 0) + 1

    rows = []
    for e in candidates:
        edge = tuple(e)
        comps = completion_times(edge, events, delta)
        C_e = piecewise_C(comps, T, beta)
        min_spk = min(counts_by_node.get(int(v), 0) for v in edge)
        alpha = None if alpha_hyper is None else float(alpha_hyper.get(edge, 0.0))
        sparse = (
            len(comps) < int(min_completions_threshold)
            or C_e < float(min_compensator_threshold)
            or min_spk < int(min_spikes_threshold)
        )
        rows.append({
            "edge": edge,
            "n_completions": int(len(comps)),
            "C_e": float(C_e),
            "min_member_spikes": int(min_spk),
            "alpha": alpha,
            "sparse_flag": bool(sparse),
        })
    return rows


def main_exposure_row(exposure_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not exposure_rows:
        return {}
    return max(exposure_rows, key=lambda r: 0.0 if r.get("alpha") is None else float(r["alpha"]))


# ---------------------------------------------------------------------
# Window-stability protocol
# ---------------------------------------------------------------------

def edge_to_mapping_labels(edge: Edge, mapping: Sequence[Dict[str, Any]], label_key: str) -> Tuple[Any, ...]:
    return tuple(mapping[int(i)].get(label_key, mapping[int(i)].get("source_index", int(i))) for i in edge)


def run_one_window(
    events: Sequence[Event],
    T: float,
    n_nodes: int,
    top_m: int,
    beta: float,
    delta: float,
    n_iter: int,
    lambda_l1: float,
    seed: int,
    active_threshold: float = 1e-2,
    verbose_hth: bool = False,
) -> Dict[str, Any]:
    """
    Run the sample-split protocol for a single event window.

    Candidates are selected on the first half and evaluated on the second half.
    """
    kernel = ExponentialKernel(beta=float(beta))
    anchor_calc = HyperedgeAnchor(delta=float(delta))

    selection, inference, T_selection, T_inference = split_events(events, T)
    if not selection or not inference:
        raise ValueError("Selection or inference half is empty.")

    mu_sel, ap_sel, ll_sel = fit_pairwise_model(
        selection, T_selection, n_nodes, kernel, anchor_calc,
        n_iter, lambda_l1, seed
    )
    candidates = generate_candidate_hyperedges(
        ap_sel,
        max_edge_size=2,
        top_m_pairs=int(top_m),
    )
    candidates = [tuple(e) for e in candidates]

    if not candidates:
        raise ValueError("No candidate hyperedges selected.")

    mu_pw, ap_pw, ll_pw = fit_pairwise_model(
        inference, T_inference, n_nodes, kernel, anchor_calc,
        n_iter, lambda_l1, seed + 1
    )
    mu_h, ap_h, ah, hist = fit_hth_model(
        inference, T_inference, n_nodes, candidates, kernel, anchor_calc,
        mu_pw, ap_pw, n_iter, lambda_l1, seed + 2, verbose=verbose_hth
    )
    ll_h = log_likelihood(inference, T_inference, mu_h, ap_h, ah, candidates, kernel, anchor_calc)

    comparison = compare_models(ll_pw, float(ll_h), candidates, ah, len(inference), active_threshold)
    exposure = exposure_diagnostics(inference, T_inference, candidates, beta, delta, ah)
    main = main_exposure_row(exposure)

    return {
        "selection_events": selection,
        "inference_events": inference,
        "T_selection": float(T_selection),
        "T_inference": float(T_inference),
        "n_events_selection": int(len(selection)),
        "n_events_inference": int(len(inference)),
        "candidates": candidates,
        "alpha_hyper": ah,
        "comparison": comparison,
        "exposure": exposure,
        "main_exposure": main,
        "history": hist,
    }


def make_window_row(
    dataset: str,
    window_start: float,
    duration: float,
    top_n: int,
    top_m: int,
    events: Sequence[Event],
    result: Dict[str, Any],
    mapping: Optional[Sequence[Dict[str, Any]]] = None,
    label_key: str = "source_index",
    extra: Optional[Dict[str, Any]] = None,
) -> Row:
    comp = result["comparison"]
    main = result["main_exposure"]
    candidates = result["candidates"]

    def original(edge: Edge) -> str:
        if mapping is None:
            return str(tuple(edge))
        return str(edge_to_mapping_labels(tuple(edge), mapping, label_key))

    row: Row = {
        "dataset": dataset,
        "window_start": float(window_start),
        "window_end": float(window_start + duration),
        "duration": float(duration),
        "top_n": int(top_n),
        "top_m": int(top_m),
        "n_events_total": int(len(events)),
        "n_events_selection": int(result["n_events_selection"]),
        "n_events_inference": int(result["n_events_inference"]),
        "candidate_edges": ";".join(str(tuple(e)) for e in candidates),
        "candidate_edges_original": ";".join(original(tuple(e)) for e in candidates),
        "logL_pairwise_heldout": float(comp["logL_pairwise"]),
        "logL_HTH_heldout": float(comp["logL_HTH"]),
        "delta_L_heldout": float(comp["delta_L"]),
        "bicdiff_candidate_count": float(comp["bicdiff_candidate_count"]),
        "bicdiff_active_count": float(comp["bicdiff_active_count"]),
        "n_active_edges": int(comp["n_active_edges"]),
        "main_edge": str(main.get("edge", "")),
        "main_edge_original": original(tuple(main["edge"])) if main else "",
        "main_alpha": float(main.get("alpha", float("nan"))) if main else float("nan"),
        "main_n_completions": int(main.get("n_completions", 0)) if main else 0,
        "main_C_e": float(main.get("C_e", float("nan"))) if main else float("nan"),
        "main_min_member_spikes": int(main.get("min_member_spikes", 0)) if main else 0,
        "main_sparse_flag": bool(main.get("sparse_flag", False)) if main else False,
    }
    if extra:
        row.update(extra)
    return row


def run_window_grid(
    trains: Sequence[np.ndarray],
    selected_indices: Sequence[int],
    mapping: Sequence[Dict[str, Any]],
    starts: Sequence[float],
    duration: float,
    top_n: int,
    top_m_list: Sequence[int],
    dataset: str,
    beta: float,
    delta: float,
    n_iter: int,
    lambda_l1: float,
    seed: int,
    label_key: str = "source_index",
    extra: Optional[Dict[str, Any]] = None,
    verbose_hth: bool = False,
    print_progress: bool = True,
) -> Tuple[List[Row], List[Dict[str, Any]]]:
    """
    Run window-stability analysis over multiple starts and top_m values.
    """
    rows: List[Row] = []
    results: List[Dict[str, Any]] = []
    n_nodes = int(top_n)

    for w_idx, start in enumerate(starts):
        events, window_counts = make_events_from_trains(trains, selected_indices, float(start), float(duration))

        if not events:
            if print_progress:
                print(f"[WARN] window {start}-{start + duration}: no events, skipped")
            continue

        for top_m in top_m_list:
            run_seed = int(seed) + 1000 * int(w_idx) + int(top_m)
            try:
                result = run_one_window(
                    events=events,
                    T=float(duration),
                    n_nodes=n_nodes,
                    top_m=int(top_m),
                    beta=float(beta),
                    delta=float(delta),
                    n_iter=int(n_iter),
                    lambda_l1=float(lambda_l1),
                    seed=run_seed,
                    verbose_hth=verbose_hth,
                )
            except Exception as exc:
                if print_progress:
                    print(f"[WARN] window {start}-{start + duration}, top_m={top_m}: {exc}")
                continue

            row = make_window_row(
                dataset=dataset,
                window_start=float(start),
                duration=float(duration),
                top_n=int(top_n),
                top_m=int(top_m),
                events=events,
                result=result,
                mapping=mapping,
                label_key=label_key,
                extra=extra,
            )
            rows.append(row)
            results.append({
                "row": row,
                "window_counts": window_counts,
                "mapping": list(mapping),
                "result": result,
            })

            if print_progress:
                print(
                    f"{float(start):>6.1f}-{float(start + duration):<6.1f} "
                    f"top_m={int(top_m):>2} "
                    f"n_sel={row['n_events_selection']:>5} "
                    f"n_inf={row['n_events_inference']:>5} "
                    f"dL={row['delta_L_heldout']:>8.3f} "
                    f"BICcand={row['bicdiff_candidate_count']:>8.3f} "
                    f"main={row['main_edge']:<10} "
                    f"alpha={row['main_alpha']:>8.5f} "
                    f"flag={'SPARSE' if row['main_sparse_flag'] else 'ok'}"
                )

    return rows, results


def summarise_bic_by_top_m(rows: Sequence[Row]) -> List[Dict[str, Any]]:
    out = []
    top_ms = sorted({int(r["top_m"]) for r in rows})
    for top_m in top_ms:
        vals = np.asarray([float(r["bicdiff_candidate_count"]) for r in rows if int(r["top_m"]) == top_m], dtype=float)
        if vals.size == 0:
            continue
        out.append({
            "top_m": int(top_m),
            "n": int(vals.size),
            "mean_bicdiff": float(np.mean(vals)),
            "median_bicdiff": float(np.median(vals)),
            "positive_count": int(np.sum(vals > 0)),
        })
    return out


def print_bic_summary(rows: Sequence[Row]) -> None:
    total = len(rows)
    pos = sum(float(r["bicdiff_candidate_count"]) > 0 for r in rows)
    print(f"\nPositive candidate-count BIC rows: {pos}/{total}")
    for s in summarise_bic_by_top_m(rows):
        print(
            f"  top_m={s['top_m']}: mean BICdiff={s['mean_bicdiff']:+.3f}, "
            f"median={s['median_bicdiff']:+.3f}, positive={s['positive_count']}/{s['n']}"
        )
