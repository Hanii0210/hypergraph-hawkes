"""
General HTH smoke test with corrected BIC reporting.

This script is additive and does not modify the older smoke_pvc3_hth.py.
Use it to avoid the active-edge-count ambiguity we observed when L1 shrinks
hyperedge coefficients close to zero.

Main decision statistic:
    candidate-count BIC difference

        BICdiff_candidate = 2 * (logL_HTH - logL_pairwise)
                            - (# selected candidate hyperedges) * log(n_events)

    Positive values favour HTH.

The script still reports active hyperedges for interpretation, but active-edge
counts are diagnostic only and should not be used as the primary BIC decision.

Examples:
    python experiments/smoke_hth_bic_checked.py --csv data/processed/pvc3_area17_top10_0to20s.csv --T 20 --n-iter 20 --lambda-l1 1 --top-m-pairs 1

    python experiments/smoke_hth_bic_checked.py --csv data/processed/gnode_natimg_top10_img0to5_rep13_gap10.csv --T 20150 --beta 0.2 --delta 2 --n-iter 20 --lambda-l1 1 --top-m-pairs 3
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.tensor_param import HypergraphTensor
from models.likelihood import log_likelihood
from inference.e_step import EStep
from inference.m_step import MStep
from inference.candidate_filter import fit_pairwise_only, generate_candidate_hyperedges


def load_events_csv(path):
    events = []
    with open(path, "r", newline="") as f:
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
    events.sort(key=lambda x: (x[0], x[1]))
    return events


def infer_n_nodes(events):
    if not events:
        return 0
    return max(n for _, n in events) + 1


def print_summary(events, T):
    n_nodes = infer_n_nodes(events)
    counts = [0] * n_nodes
    for _, n in events:
        counts[n] += 1
    t0 = min(t for t, _ in events) if events else float("nan")
    t1 = max(t for t, _ in events) if events else float("nan")
    print(f"  total events  : {len(events)}")
    print(f"  time span     : [{t0:.3f}, {t1:.3f}]   T={T}")
    print(f"  n_nodes       : {n_nodes}")
    print(f"  events/node   : {counts}")
    print(f"  rate (per T)  : {len(events) / T:.3f}")
    return n_nodes, counts


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
            mu_init, ap_init, n_iter, lambda_l1, seed, verbose=True):
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

        if verbose and (it == 1 or it == n_iter or it % 5 == 0):
            print(f"    HTH iter {it:>3}: logL={ll:.3f}")

    return mu, ap, ah, hist


def split_events(events, T):
    cut = T / 2.0
    sel = [(t, n) for t, n in events if t <= cut]
    inf = [(t - cut, n) for t, n in events if t > cut]
    return sel, inf, cut, T - cut


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


def print_exposure(rows):
    print("\nHyperedge exposure diagnostics")
    print("-" * 88)
    print(f"{'edge':<14} {'n_comp':>10} {'C_e':>12} {'min_spk':>8} {'alpha':>12} {'flag':>8}")
    for r in rows:
        alpha = "NA" if r["alpha"] is None else f"{r['alpha']:.5f}"
        flag = "SPARSE" if r["sparse_flag"] else "ok"
        print(
            f"{str(r['edge']):<14} {r['n_completions']:>10} "
            f"{r['C_e']:>12.5f} {r['min_member_spikes']:>8} "
            f"{alpha:>12} {flag:>8}"
        )


def bicdiff_candidate_count(delta_L, n_events, n_candidates):
    if n_events <= 1:
        return float("nan")
    return float(2.0 * delta_L - n_candidates * np.log(n_events))


def bicdiff_active_count(delta_L, n_events, n_active):
    if n_events <= 1:
        return float("nan")
    return float(2.0 * delta_L - n_active * np.log(n_events))


def active_edges(alpha_hyper, threshold):
    return [(e, float(a)) for e, a in alpha_hyper.items() if float(a) > threshold]


def comparison_block(title, ll_pw, ll_hth, candidates, alpha_hyper, n_events, active_threshold):
    dL = float(ll_hth - ll_pw)
    act = active_edges(alpha_hyper, active_threshold)
    bic_cand = bicdiff_candidate_count(dL, n_events, len(candidates))
    bic_act = bicdiff_active_count(dL, n_events, len(act))

    print(f"\n{title}")
    print("-" * 88)
    print(f"logL_pairwise           = {ll_pw:.3f}")
    print(f"logL_HTH                = {ll_hth:.3f}")
    print(f"delta_L                 = {dL:+.3f}")
    print(f"n_events                = {n_events}")
    print(f"n_candidates            = {len(candidates)}")
    print(f"n_active_edges          = {len(act)}  threshold={active_threshold}")
    print(f"BICdiff_candidate_count = {bic_cand:+.3f}  (PRIMARY; positive favours HTH)")
    print(f"BICdiff_active_count    = {bic_act:+.3f}  (diagnostic only)")
    print(f"active hyperedges       = {[(e, round(a, 5)) for e, a in act]}")

    if bic_act > 0 and bic_cand <= 0:
        print("WARNING: active-count BIC is positive but candidate-count BIC is not.")
        print("         Do not interpret this as decisive HTH evidence.")

    if len(act) == 0 and dL > 0:
        print("WARNING: no active hyperedges but HTH logL is higher.")
        print("         Treat gain as optimisation/refitting effect, not hyperedge evidence.")

    return {
        "logL_pairwise": ll_pw,
        "logL_HTH": ll_hth,
        "delta_L": dL,
        "n_events": int(n_events),
        "n_candidates": int(len(candidates)),
        "n_active_edges": int(len(act)),
        "bicdiff_candidate_count": bic_cand,
        "bicdiff_active_count": bic_act,
        "active_edges": act,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--T", type=float, required=True)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--top-m-pairs", type=int, default=8)
    parser.add_argument("--active-threshold", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-events-warning", type=int, default=6000)
    parser.add_argument("--out", default="experiments/hth_bic_checked_smoke.pkl")
    args = parser.parse_args()

    events = load_events_csv(args.csv)
    n_nodes = infer_n_nodes(events)
    kernel = ExponentialKernel(beta=args.beta)
    anchor_calc = HyperedgeAnchor(delta=args.delta)

    print("=" * 88)
    print("HTH smoke test with corrected BIC reporting")
    print("=" * 88)
    print(f"csv              : {args.csv}")
    print(f"T                : {args.T}")
    print(f"beta/delta       : {args.beta} / {args.delta}")
    print(f"n_iter/lambda    : {args.n_iter} / {args.lambda_l1}")
    print(f"top_m_pairs      : {args.top_m_pairs}")
    print(f"active_threshold : {args.active_threshold}")
    print()
    n_nodes, counts = print_summary(events, args.T)

    if len(events) > args.max_events_warning:
        print()
        print(f"WARNING: {len(events)} events may be large for exact O(n^2) E-step.")
        print("         Consider a smaller window/block if memory becomes an issue.")

    results = {"args": vars(args), "summary": {"n_nodes": n_nodes, "counts": counts}}

    print("\n=== Full-data diagnostic fit ===")
    print("Fitting pairwise-only model...")
    mu_pw, ap_pw, ll_pw = fit_pairwise(
        events, args.T, n_nodes, kernel, anchor_calc,
        args.n_iter, args.lambda_l1, args.seed
    )

    candidates_full = generate_candidate_hyperedges(
        ap_pw, max_edge_size=2, top_m_pairs=args.top_m_pairs
    )
    print(f"Candidates from full data: {candidates_full}")
    print_exposure(exposure_rows(events, args.T, candidates_full, args.beta, args.delta))

    print("\nFitting HTH model...")
    mu_h, ap_h, ah_h, hist_h = fit_hth(
        events, args.T, n_nodes, candidates_full, kernel, anchor_calc,
        mu_pw, ap_pw, args.n_iter, args.lambda_l1, args.seed + 1
    )
    ll_h = log_likelihood(events, args.T, mu_h, ap_h, ah_h, candidates_full, kernel, anchor_calc)

    results["full_data"] = comparison_block(
        "Full-data comparison -- diagnostic only, selected/tested on same data",
        ll_pw, float(ll_h), candidates_full, ah_h, len(events), args.active_threshold
    )
    print_exposure(exposure_rows(events, args.T, candidates_full, args.beta, args.delta, ah_h))

    print("\n=== Sample-split smoke test ===")
    sel, inf, T_sel, T_inf = split_events(events, args.T)
    print(f"selection half: {len(sel)} events, T={T_sel}")
    print(f"inference half: {len(inf)} events, T={T_inf}")

    print("Fitting pairwise on selection half...")
    mu_sel, ap_sel, ll_sel = fit_pairwise(
        sel, T_sel, n_nodes, kernel, anchor_calc,
        args.n_iter, args.lambda_l1, args.seed + 10
    )
    candidates_split = generate_candidate_hyperedges(
        ap_sel, max_edge_size=2, top_m_pairs=args.top_m_pairs
    )
    print(f"Candidates selected on first half: {candidates_split}")
    print_exposure(exposure_rows(sel, T_sel, candidates_split, args.beta, args.delta))

    print("\nFitting pairwise on inference half...")
    mu_i, ap_i, ll_i = fit_pairwise(
        inf, T_inf, n_nodes, kernel, anchor_calc,
        args.n_iter, args.lambda_l1, args.seed + 11
    )
    print("Fitting HTH on inference half...")
    mu_hi, ap_hi, ah_i, hist_i = fit_hth(
        inf, T_inf, n_nodes, candidates_split, kernel, anchor_calc,
        mu_i, ap_i, args.n_iter, args.lambda_l1, args.seed + 12
    )
    ll_hi = log_likelihood(inf, T_inf, mu_hi, ap_hi, ah_i, candidates_split, kernel, anchor_calc)

    results["sample_split"] = comparison_block(
        "Held-out comparison",
        ll_i, float(ll_hi), candidates_split, ah_i, len(inf), args.active_threshold
    )
    print_exposure(exposure_rows(inf, T_inf, candidates_split, args.beta, args.delta, ah_i))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
