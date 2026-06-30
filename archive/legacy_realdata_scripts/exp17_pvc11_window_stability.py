"""
Experiment 17: PVC-11 spontaneous window-stability analysis.

This is an additive experiment script. It does not modify exp10 or any existing
experiment. It mirrors exp16 but loads CRCNS pvc-11 spontaneous spike-time data.

Default target:
    monkey 2 spontaneous activity, 40s non-overlapping windows.

Protocol:
    1. Load PVC-11 spontaneous spike trains.
    2. Choose a fixed top-N neuron subset over the whole analysis span.
    3. For each window:
         - split the window into selection half and inference half
         - select top-m pairwise-supported candidate hyperedges on selection half
         - fit pairwise-only and HTH on inference half
         - report held-out delta log-likelihood and candidate-count BIC difference
         - report exposure diagnostics for selected candidates

Important:
    BICdiff uses conservative candidate-count penalty:

        BICdiff = 2 * (logL_HTH - logL_pairwise) - (# selected candidates) * log(n_events_inference)

    Positive BICdiff favours HTH. This avoids the active-edge-count ambiguity
    seen in smoke tests when L1 shrinks alpha_e close to zero.

Quick run:
    python experiments/exp17_pvc11_window_stability.py --raw-root data/raw --monkey 2 --top-m-list 1 --n-iter 20

Fuller run:
    python experiments/exp17_pvc11_window_stability.py --raw-root data/raw --monkey 2 --top-m-list 1,2,3 --n-iter 20
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, ".")

from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from models.likelihood import log_likelihood
from inference.e_step import EStep
from inference.m_step import MStep
from inference.candidate_filter import fit_pairwise_only, generate_candidate_hyperedges


def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def iter_event_list(events):
    if isinstance(events, np.ndarray):
        if events.dtype == object:
            return [events.flat[i] for i in range(events.size)]
        if events.ndim == 1:
            return [events]
    if isinstance(events, (list, tuple)):
        return list(events)
    return []


def as_1d_float(x):
    if x is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(x, dtype=float).ravel()
    except Exception:
        return np.array([], dtype=float)
    arr = arr[np.isfinite(arr)]
    return np.sort(arr)


def find_pvc11_file(raw_root: Path, monkey: int) -> Path:
    pattern = f"**/data_and_scripts/spikes_spontaneous/spiketimesmonkey{monkey}spont.mat"
    matches = sorted(raw_root.glob(pattern))
    matches = [m for m in matches if "__MACOSX" not in m.parts and not m.name.startswith("._")]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {pattern} under {raw_root}. "
            "Check that pvc-11 data_and_scripts.tar.gz is extracted."
        )
    return matches[0]


def load_pvc11_spont(raw_root: Path, monkey: int):
    mat_path = find_pvc11_file(raw_root, monkey)
    mat = sio.loadmat(mat_path, simplify_cells=True)
    data = mat.get("data", {})
    if not isinstance(data, dict) or "EVENTS" not in data:
        raise KeyError(
            f"Could not find data['EVENTS'] in {mat_path}. "
            f"Available top-level keys: {list(mat.keys())}"
        )

    trains = [as_1d_float(x) for x in iter_event_list(data["EVENTS"])]
    keep = [(idx, t) for idx, t in enumerate(trains) if len(t) > 0]
    original_indices = [idx for idx, _ in keep]
    trains = [t for _, t in keep]

    if not trains:
        raise ValueError(f"No nonempty spike trains found in {mat_path}")

    return mat_path, trains, original_indices


def select_fixed_subset(trains, original_indices, top_n, span_start, span_end):
    counts = []
    for local_idx, t in enumerate(trains):
        c = int(np.sum((t >= span_start) & (t < span_end)))
        counts.append((local_idx, c))

    selected = sorted(counts, key=lambda x: x[1], reverse=True)[:top_n]
    selected_local_indices = [idx for idx, _ in selected]

    mapping = []
    for new_node, (local_idx, count) in enumerate(selected):
        mapping.append({
            "new_node": int(new_node),
            "original_index": int(original_indices[local_idx]),
            "local_nonempty_index": int(local_idx),
            "count_in_analysis_span": int(count),
        })

    return selected_local_indices, mapping


def make_events(trains, selected_local_indices, start, duration):
    end = start + duration
    events = []
    counts = []

    for new_node, local_idx in enumerate(selected_local_indices):
        t = trains[local_idx]
        w = t[(t >= start) & (t < end)] - start
        counts.append(int(len(w)))
        for x in w:
            events.append((float(x), int(new_node)))

    events.sort(key=lambda z: (z[0], z[1]))
    return events, counts


def split_events(events, T):
    cut = T / 2.0
    sel = [(t, n) for t, n in events if t <= cut]
    inf = [(t - cut, n) for t, n in events if t > cut]
    return sel, inf, cut, T - cut


def fit_pairwise(events, T, n_nodes, kernel, anchor_calc, n_iter, lambda_l1, seed):
    mu, ap = fit_pairwise_only(
        events=events,
        T=T,
        n_nodes=n_nodes,
        kernel=kernel,
        anchor_calc=anchor_calc,
        n_iter=n_iter,
        lambda_l1=lambda_l1,
        seed=seed,
    )
    ll = log_likelihood(events, T, mu, ap, {}, [], kernel, anchor_calc)
    return mu, ap, float(ll)


def fit_hth(events, T, n_nodes, candidates, kernel, anchor_calc,
            mu_init, ap_init, n_iter, lambda_l1, seed, quiet=True):
    tensor = HypergraphTensor(n_nodes=n_nodes, rank=3, seed=seed)
    estep = EStep(kernel, anchor_calc)
    mstep = MStep(n_nodes=n_nodes, tensor=tensor, lambda_l1=lambda_l1)

    mu = mu_init.copy()
    ap = ap_init.copy()
    ah = {e: 0.1 for e in candidates}
    hist = []

    for it in range(1, n_iter + 1):
        r = estep.compute(events, mu, ap, ah, candidates)
        mu = mstep.update_mu(events, r["p_background"], T)
        ap = mstep.update_alpha_pairwise(
            events, r["p_pairwise"], r["p_hyper"], candidates, kernel, T
        )
        ah = mstep.update_alpha_hyper_als(
            events, r["p_hyper"], candidates, anchor_calc, kernel, T
        )
        ll = log_likelihood(events, T, mu, ap, ah, candidates, kernel, anchor_calc)
        hist.append(float(ll))

        if not quiet and (it == 1 or it == n_iter or it % 5 == 0):
            print(f"      HTH iter {it:>3}: logL={ll:.3f}")

    return mu, ap, ah, hist


def completion_times(edge, events, delta):
    by_node = {}
    for t, node in events:
        by_node.setdefault(node, []).append(t)

    for k in by_node:
        by_node[k] = np.asarray(by_node[k], dtype=float)

    completions = set()
    for anchor_node in edge:
        if anchor_node not in by_node:
            continue
        for t_last in by_node[anchor_node]:
            lo_t = t_last - delta
            ok = True
            for v in edge:
                if v == anchor_node:
                    continue
                if v not in by_node:
                    ok = False
                    break
                arr = by_node[v]
                lo = np.searchsorted(arr, lo_t, side="left")
                hi = np.searchsorted(arr, t_last, side="right")
                if hi - lo == 0:
                    ok = False
                    break
            if ok:
                completions.add(float(t_last))

    return sorted(completions)


def piecewise_C(comps, T, beta):
    if not comps:
        return 0.0

    total = 0.0
    for k, t0 in enumerate(comps):
        t1 = comps[k + 1] if k + 1 < len(comps) else T
        if t1 > t0:
            total += (1.0 / beta) * (1.0 - np.exp(-beta * (t1 - t0)))
    return float(total)


def exposure_rows(events, T, candidates, beta, delta, alpha_hyper=None):
    counts_by_node = {}
    for _, node in events:
        counts_by_node[node] = counts_by_node.get(node, 0) + 1

    rows = []
    for e in candidates:
        comps = completion_times(e, events, delta)
        C_e = piecewise_C(comps, T, beta)
        min_spk = min(counts_by_node.get(v, 0) for v in e)
        alpha = None if alpha_hyper is None else float(alpha_hyper.get(e, 0.0))
        sparse = (len(comps) < 20) or (C_e < 1e-3) or (min_spk < 20)
        rows.append({
            "edge": e,
            "n_completions": int(len(comps)),
            "C_e": C_e,
            "min_member_spikes": int(min_spk),
            "alpha": alpha,
            "sparse_flag": bool(sparse),
        })
    return rows


def edge_to_original(edge, mapping):
    return tuple(mapping[i]["original_index"] for i in edge)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--monkey", type=int, default=2)
    parser.add_argument("--starts", default="0,40,80,120,160")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-m-list", default="1,2,3")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--verbose-hth", action="store_true")
    args = parser.parse_args()

    starts = parse_float_list(args.starts)
    top_m_list = parse_int_list(args.top_m_list)

    span_start = min(starts)
    span_end = max(s + args.duration for s in starts)

    raw_root = Path(args.raw_root)
    mat_path, trains, original_indices = load_pvc11_spont(raw_root, args.monkey)

    selected_local_indices, mapping = select_fixed_subset(
        trains, original_indices, args.top_n, span_start, span_end
    )

    kernel = ExponentialKernel(beta=args.beta)
    anchor_calc = HyperedgeAnchor(delta=args.delta)

    all_results = []
    rows = []

    print("=" * 114)
    print("Experiment 17: PVC-11 spontaneous window-stability analysis")
    print("=" * 114)
    print(f"source_file     : {mat_path}")
    print(f"monkey          : {args.monkey}")
    print(f"analysis span   : [{span_start}, {span_end}) sec")
    print(f"starts          : {starts}")
    print(f"duration        : {args.duration}")
    print(f"top_n fixed     : {args.top_n}")
    print(f"top_m_list      : {top_m_list}")
    print(f"beta/delta      : {args.beta} / {args.delta}")
    print(f"n_iter/lambda   : {args.n_iter} / {args.lambda_l1}")
    print("\nFixed neuron mapping:")
    for m in mapping:
        print(f"  node {m['new_node']:2d} <- original {m['original_index']:3d}, "
              f"count_span={m['count_in_analysis_span']}")

    print("\nRunning windows...")
    print("-" * 114)
    header = (
        f"{'window':<12} {'top_m':>5} {'n_sel':>7} {'n_inf':>7} "
        f"{'candidates':<30} {'dL':>9} {'BICd_cand':>10} {'main_edge':<12} "
        f"{'main_alpha':>10} {'main_Ce':>9} {'flag':>8}"
    )
    print(header)
    print("-" * 114)

    for w_idx, start in enumerate(starts):
        events, window_counts = make_events(trains, selected_local_indices, start, args.duration)
        sel, inf, T_sel, T_inf = split_events(events, args.duration)

        if len(sel) == 0 or len(inf) == 0:
            print(f"[WARN] window {start}-{start+args.duration}: empty split, skipped")
            continue

        for top_m in top_m_list:
            seed = args.seed + 1000 * w_idx + top_m

            mu_sel, ap_sel, ll_sel = fit_pairwise(
                sel, T_sel, args.top_n, kernel, anchor_calc,
                args.n_iter, args.lambda_l1, seed
            )

            candidates = generate_candidate_hyperedges(
                ap_sel,
                max_edge_size=2,
                top_m_pairs=top_m,
            )

            if not candidates:
                print(f"[WARN] window {start}-{start+args.duration}, top_m={top_m}: no candidates")
                continue

            mu_pw_i, ap_pw_i, ll_pw_i = fit_pairwise(
                inf, T_inf, args.top_n, kernel, anchor_calc,
                args.n_iter, args.lambda_l1, seed + 1
            )

            mu_hth_i, ap_hth_i, ah_i, hist_i = fit_hth(
                inf, T_inf, args.top_n, candidates, kernel, anchor_calc,
                mu_pw_i, ap_pw_i,
                n_iter=args.n_iter,
                lambda_l1=args.lambda_l1,
                seed=seed + 2,
                quiet=not args.verbose_hth,
            )
            ll_hth_i = log_likelihood(inf, T_inf, mu_hth_i, ap_hth_i, ah_i, candidates, kernel, anchor_calc)

            dL = float(ll_hth_i - ll_pw_i)
            bic_candidate = float(2.0 * dL - len(candidates) * np.log(len(inf)))

            exp_rows = exposure_rows(inf, T_inf, candidates, args.beta, args.delta, ah_i)
            main = max(exp_rows, key=lambda r: (0.0 if r["alpha"] is None else r["alpha"]))

            candidate_str = ";".join(str(e) for e in candidates)
            main_edge = str(main["edge"])
            flag = "SPARSE" if main["sparse_flag"] else "ok"
            print(
                f"{start:>4.0f}-{start+args.duration:<7.0f} {top_m:>5} {len(sel):>7} {len(inf):>7} "
                f"{candidate_str:<30.30} {dL:>9.3f} {bic_candidate:>10.3f} "
                f"{main_edge:<12} {main['alpha']:>10.5f} {main['C_e']:>9.4f} {flag:>8}"
            )

            row = {
                "dataset": "pvc-11",
                "condition": "spontaneous",
                "monkey": int(args.monkey),
                "window_start": float(start),
                "window_end": float(start + args.duration),
                "duration": float(args.duration),
                "top_n": int(args.top_n),
                "top_m": int(top_m),
                "n_events_total": int(len(events)),
                "n_events_selection": int(len(sel)),
                "n_events_inference": int(len(inf)),
                "candidate_edges": candidate_str,
                "candidate_edges_original_indices": ";".join(str(edge_to_original(e, mapping)) for e in candidates),
                "logL_pairwise_heldout": float(ll_pw_i),
                "logL_HTH_heldout": float(ll_hth_i),
                "delta_L_heldout": dL,
                "bicdiff_candidate_count": bic_candidate,
                "main_edge": main_edge,
                "main_edge_original_indices": str(edge_to_original(main["edge"], mapping)),
                "main_alpha": float(main["alpha"]),
                "main_n_completions": int(main["n_completions"]),
                "main_C_e": float(main["C_e"]),
                "main_min_member_spikes": int(main["min_member_spikes"]),
                "main_sparse_flag": bool(main["sparse_flag"]),
            }
            rows.append(row)

            all_results.append({
                "row": row,
                "window_counts": window_counts,
                "mapping": mapping,
                "candidates": candidates,
                "alpha_hyper": ah_i,
                "exposure": exp_rows,
                "logL_pairwise": float(ll_pw_i),
                "logL_HTH": float(ll_hth_i),
            })

    out_prefix = Path(args.out_prefix) if args.out_prefix else Path(
        f"experiments/exp17_pvc11_monkey{args.monkey}_window_stability"
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_prefix.with_suffix(".csv")
    pkl_path = out_prefix.with_suffix(".pkl")
    json_path = out_prefix.with_name(out_prefix.name + "_mapping.json")

    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with pkl_path.open("wb") as f:
        pickle.dump({
            "args": vars(args),
            "source_file": str(mat_path),
            "mapping": mapping,
            "rows": rows,
            "results": all_results,
        }, f)

    with json_path.open("w") as f:
        json.dump({
            "source_file": str(mat_path),
            "analysis_span": [span_start, span_end],
            "mapping": mapping,
        }, f, indent=2)

    print("-" * 114)
    print(f"Saved CSV    : {csv_path}")
    print(f"Saved PKL    : {pkl_path}")
    print(f"Saved mapping: {json_path}")

    if rows:
        n_pos = sum(1 for r in rows if r["bicdiff_candidate_count"] > 0)
        print(f"\nPositive candidate-count BIC rows: {n_pos}/{len(rows)}")
        by_top = {}
        for r in rows:
            by_top.setdefault(r["top_m"], []).append(r["bicdiff_candidate_count"])
        for top_m, vals in sorted(by_top.items()):
            vals = np.asarray(vals, dtype=float)
            print(f"  top_m={top_m}: mean BICdiff={vals.mean():+.3f}, "
                  f"median={np.median(vals):+.3f}, positive={np.sum(vals > 0)}/{len(vals)}")


if __name__ == "__main__":
    main()
