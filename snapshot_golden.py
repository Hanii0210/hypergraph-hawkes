"""
Golden-master snapshot.

Reads the result artifacts that each experiment writes (experiments/*.pkl,
*.npy) and distils the headline numbers into a single timestamped JSON plus a
human-readable summary. Run this AFTER a full clean run of the suite to freeze
a reference point (e.g. "post-P1"). Later, after P2 / P5 / etc., run it again
and diff the summaries to see exactly which numbers moved and by how much.

Usage:
    python run_all.py          # (re)generate all artifacts first
    python snapshot_golden.py  # then freeze this snapshot

Output:
    golden/golden_<timestamp>.json   machine-readable
    golden/golden_<timestamp>.txt    human-readable summary
"""

import os
import json
import pickle
import datetime
import numpy as np

EXP_DIR = "experiments"
OUT_DIR = "golden"


def _load_pkl(name):
    path = os.path.join(EXP_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _f(x):
    """JSON-safe float."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def collect():
    snap = {}

    # exp1b: recovery bias across seeds
    d = _load_pkl("exp1b_recovery_robustness.pkl")
    if d:
        rec = d["recovered"]; tv = d["true_values"]
        ah = np.array(rec["alpha_hyper(0,1)"])
        snap["exp1b"] = {
            "alpha_hyper_mean": _f(ah.mean()),
            "alpha_hyper_std": _f(ah.std()),
            "rel_bias_pct": _f((ah.mean() - tv["alpha_hyper(0,1)"]) /
                               tv["alpha_hyper(0,1)"] * 100),
            "a_2to0_mean": _f(np.mean(rec["a[2->0]"])),
        }

    # exp2: regularisation path -> optimal lambda, survivor counts
    d = _load_pkl("exp2_regpath.pkl")
    if d:
        aic = [r["aic"] for r in d]; bic = [r["bic"] for r in d]
        snap["exp2"] = {
            "opt_lambda_aic": _f(d[int(np.argmin(aic))]["lambda"]),
            "opt_lambda_bic": _f(d[int(np.argmin(bic))]["lambda"]),
            "survivors_at_min_lambda": int(len(d[0]["survivors"])),
            "survivors_at_opt": int(len(d[int(np.argmin(bic))]["survivors"])),
        }

    # exp3: convergence spread
    p = os.path.join(EXP_DIR, "exp3_final_logliks.npy")
    if os.path.exists(p):
        ll = np.load(p)
        best = ll.max()
        snap["exp3"] = {
            "final_logL_mean": _f(ll.mean()),
            "final_logL_std": _f(ll.std()),
            "n_within_1nat": int(np.sum(ll > best - 1.0)),
            "n_total": int(ll.size),
        }

    # exp4: phase transition -> inferred rho crossing
    d = _load_pkl("exp4_phase.pkl")
    if d:
        cross_true = next((r["alpha_hyper_true"] for r in d if r["rho_true"] >= 1.0), None)
        cross_inf = next((r["alpha_hyper_true"] for r in d if r["rho_inferred"] >= 1.0), None)
        snap["exp4"] = {
            "alpha_at_rho_true_cross": _f(cross_true),
            "alpha_at_rho_inf_cross": _f(cross_inf),
        }

    # exp7: falsifiability deltas
    d = _load_pkl("exp7_likelihood_gap.pkl")
    if d:
        snap["exp7"] = {
            "deltaL_A": _f(d["A_with_hyperedge"]["delta_L"]),
            "bic_diff_A": _f(d["A_with_hyperedge"]["bic_diff"]),
            "deltaL_B": _f(d["B_no_hyperedge"]["delta_L"]),
            "bic_diff_B": _f(d["B_no_hyperedge"]["bic_diff"]),
        }

    # exp8: delta sensitivity -> best delta and alpha there
    d = _load_pkl("exp8_delta_sensitivity.pkl")
    if d:
        best = max(d, key=lambda r: r["logL"])
        snap["exp8"] = {
            "best_delta": _f(best["delta"]),
            "alpha_at_best_delta": _f(best["alpha_hyper"].get((0, 1))),
        }

    # exp9: scaling exponents
    d = _load_pkl("exp9_scalability.pkl")
    if d:
        sn = d["scaling_n"]
        ns = np.array([r["n_events"] for r in sn], dtype=float)
        ts = np.array([r["time"] for r in sn], dtype=float)
        m = (ns > 0) & (ts > 0)
        full = np.polyfit(np.log(ns[m]), np.log(ts[m]), 1)[0]
        asym = np.polyfit(np.log(ns[m][-3:]), np.log(ts[m][-3:]), 1)[0] if m.sum() >= 3 else full
        snap["exp9"] = {
            "exponent_full": _f(full),
            "exponent_asymptotic": _f(asym),
            "max_n": _f(ns.max()),
        }

    # exp10: real-data model comparison
    d = _load_pkl("exp10_realdata.pkl")
    if d:
        snap["exp10"] = {
            "logL_pw": _f(d["logL_pw"]),
            "logL_hth": _f(d["logL_hth"]),
            "delta_L": _f(d["delta_L"]),
            "bic_diff": _f(d["bic_diff"]),
            "alpha_hyper": {str(k): _f(v) for k, v in d["alpha_hyper_hth"].items()},
        }
        if d.get("split_delta_L") is not None:
            snap["exp10"]["split_delta_L"] = _f(d["split_delta_L"])
            snap["exp10"]["split_chibar_p"] = _f(d["split_chibar_p"])
            snap["exp10"]["split_bic_diff"] = _f(d["split_bic_diff"])
            snap["exp10"]["naive_chibar_p"] = _f(d.get("naive_chibar_p"))

    # exp11: bias ablation across beta
    d = _load_pkl("exp11_bias_ablation.pkl")
    if d:
        snap["exp11"] = {
            f"rel_bias_beta_{r['beta']}": _f(r["rel_bias"]) for r in d
        }

    # exp12: CP rank selection (per dataset R*=2, R*=3)
    d = _load_pkl("exp12_rank_sweep.pkl")
    if d:
        def _rank_of(name):           # "CP R=3" -> 3
            return int(name.split("=")[1])
        def _summarise(rows, r_true):
            cp = [r for r in rows if r[0].startswith("CP")]
            free = next((r for r in rows if r[0] == "free"), None)
            aic_best = _rank_of(min(cp, key=lambda r: r[4])[0])
            bic_best = _rank_of(min(cp, key=lambda r: r[5])[0])
            at_true = next((r for r in cp if _rank_of(r[0]) == r_true), None)
            # coverage-angle elbow: largest drop between consecutive ranks
            covs = [(_rank_of(r[0]), r[7]) for r in cp
                    if r[7] is not None and np.isfinite(r[7])]
            elbow = None
            if len(covs) >= 2:
                drops = [(covs[i - 1][1] - covs[i][1], covs[i][0])
                         for i in range(1, len(covs))]
                elbow = max(drops)[1]
            return {
                "aic_selected_rank": aic_best,
                "bic_selected_rank": bic_best,
                "cov_angle_elbow_rank": elbow,
                "relerr_at_true": _f(at_true[6]) if at_true else None,
                "cov_angle_at_true": _f(at_true[7]) if at_true else None,
                "relerr_free": _f(free[6]) if free else None,
                "params_free": int(free[1]) if free else None,
                "params_cp_at_true": _f(at_true[1]) if at_true else None,
            }
        snap["exp12"] = {}
        for key in ("R2", "R3"):
            if key in d:
                e = d[key]
                snap["exp12"][key] = {
                    "r_true": e["R_true"],
                    "controlled": _summarise(e["controlled"], e["R_true"]),
                    "full_em": _summarise(e["full_em"], e["R_true"]),
                }

    # exp13: Type-I error calibration of hyperedge discovery (P5)
    #   exp13_calibration.pkl -> naive vs split FPR on the 2-edge pairwise null
    #   exp13_control.pkl     -> split FPR on the pure-Poisson control
    cal = _load_pkl("exp13_calibration.pkl")
    ctl = _load_pkl("exp13_control.pkl")
    if cal or ctl:
        e13 = {}
        if cal:
            e13["naive_fpr_pct"] = _f(cal.get("fpr_naive"))
            e13["split_fpr_pct"] = _f(cal.get("fpr_split"))
            e13["nseed"] = int(cal.get("nseed", 0))
            e13["T"] = _f(cal.get("T"))
        if ctl:
            e13["control_split_fpr_pct"] = _f(ctl.get("fpr_split"))
            e13["control_nseed"] = int(ctl.get("nseed", 0))
            e13["control_T"] = _f(ctl.get("T"))
        snap["exp13"] = e13

    # exp14: candidate-generation identification diagnostic
    d = _load_pkl("exp14_identification_diagnostic.pkl")
    if d:
        snap["exp14"] = {
            "true_edge": str(d["true_edge"]),
            "threshold": _f(d["threshold"]),
            "rows": [
                {"alpha": _f(r["alpha"]), "footprint": _f(r["footprint_mean"]),
                 "nominated_pct": _f(r["nominated_pct"]),
                 "bias_pct": _f(r["bias_pct"]), "dL": _f(r["dL"])}
                for r in d["rows"]
            ],
        }

    # exp15: non-trivial baseline (HTH vs third-order interaction Hawkes)
    d = _load_pkl("exp15_interaction_baseline.pkl")
    if d:
        snap["exp15"] = {
            "edge": str(d["edge"]),
            "rows": [
                {"alpha": _f(r["alpha"]), "dL_inter": _f(r["dL_inter"]),
                 "dL_HTH": _f(r["dL_HTH"]), "winner": r["winner"],
                 "margin": _f(r["margin"])}
                for r in d["rows"]
            ],
        }

    return snap


def render(snap):
    lines = []
    def p(s=""): lines.append(s)
    p("GOLDEN MASTER SNAPSHOT")
    p("=" * 60)
    if "exp1b" in snap:
        e = snap["exp1b"]
        p(f"exp1b  hyperedge recovery: mean={e['alpha_hyper_mean']:.4f} "
          f"std={e['alpha_hyper_std']:.4f}  rel_bias={e['rel_bias_pct']:+.2f}%")
    if "exp2" in snap:
        e = snap["exp2"]
        p(f"exp2   opt_lambda(BIC)={e['opt_lambda_bic']:.4f}  "
          f"survivors@min_lambda={e['survivors_at_min_lambda']}")
    if "exp3" in snap:
        e = snap["exp3"]
        p(f"exp3   final logL mean={e['final_logL_mean']:.3f} std={e['final_logL_std']:.3f}  "
          f"{e['n_within_1nat']}/{e['n_total']} within 1 nat")
    if "exp4" in snap:
        e = snap["exp4"]
        p(f"exp4   rho_true crosses 1 at alpha={e['alpha_at_rho_true_cross']}, "
          f"rho_inferred at alpha={e['alpha_at_rho_inf_cross']}")
    if "exp7" in snap:
        e = snap["exp7"]
        p(f"exp7   A: dL={e['deltaL_A']:+.3f} BICd={e['bic_diff_A']:+.3f}   "
          f"B: dL={e['deltaL_B']:+.3f} BICd={e['bic_diff_B']:+.3f}")
    if "exp8" in snap:
        e = snap["exp8"]
        p(f"exp8   best_delta={e['best_delta']}  alpha_there={e['alpha_at_best_delta']:.4f}")
    if "exp9" in snap:
        e = snap["exp9"]
        p(f"exp9   exponent full={e['exponent_full']:.2f}  "
          f"asymptotic={e['exponent_asymptotic']:.2f}  (max n={e['max_n']:.0f})")
    if "exp10" in snap:
        e = snap["exp10"]
        p(f"exp10  logL_pw={e['logL_pw']:.3f}  logL_hth={e['logL_hth']:.3f}  "
          f"dL={e['delta_L']:+.3f}  BICd={e['bic_diff']:+.3f}")
        for k, v in e["alpha_hyper"].items():
            p(f"         hyper {k}: {v:.4f}")
        if "split_chibar_p" in e:
            p(f"         split: dL={e['split_delta_L']:+.3f} chibar_p={e['split_chibar_p']:.3g} "
              f"BICd={e['split_bic_diff']:+.3f}  (naive chibar_p={e['naive_chibar_p']:.3g}, invalid)")
    if "exp11" in snap:
        p("exp11  rel_bias by beta: " +
          "  ".join(f"{k.split('_')[-1]}={v:+.1f}%" for k, v in snap["exp11"].items()))
    if "exp12" in snap:
        for key, e in snap["exp12"].items():
            c = e["controlled"]; fe = e["full_em"]
            p(f"exp12 {key} (R*={e['r_true']})  controlled: AIC*={c['aic_selected_rank']} "
              f"BIC*={c['bic_selected_rank']} cov-elbow=R{c['cov_angle_elbow_rank']}  "
              f"relerr@R*={c['relerr_at_true']:.1f}% cov@R*={c['cov_angle_at_true']:.1f}deg "
              f"(free relerr={c['relerr_free']:.1f}%, params {c['params_cp_at_true']:.0f} vs {c['params_free']})")
            p(f"        full-EM: AIC*={fe['aic_selected_rank']} "
              f"BIC*={fe['bic_selected_rank']} cov-elbow=R{fe['cov_angle_elbow_rank']} "
              f"relerr@R*={fe['relerr_at_true']:.1f}% cov@R*={fe['cov_angle_at_true']:.1f}deg")
    if "exp13" in snap:
        e = snap["exp13"]
        parts = []
        if "naive_fpr_pct" in e:
            parts.append(f"naive={e['naive_fpr_pct']:.0f}% split={e['split_fpr_pct']:.0f}% "
                         f"(pairwise null, {e['nseed']} seeds T={e['T']:.0f})")
        if "control_split_fpr_pct" in e:
            parts.append(f"control split={e['control_split_fpr_pct']:.0f}% "
                         f"(pure-Poisson, {e['control_nseed']} seeds T={e['control_T']:.0f})")
        p("exp13  Type-I FPR (target 5%): " + "  |  ".join(parts))
    if "exp14" in snap:
        e = snap["exp14"]
        segs = [f"a{r['alpha']:.1f} foot={r['footprint']:.3f} nom={r['nominated_pct']:.0f}% "
                f"bias={r['bias_pct']:+.0f}% dL={r['dL']:+.1f}" for r in e["rows"]]
        p(f"exp14  ident.diag (true {e['true_edge']}, no pairwise, thr={e['threshold']:.2f}): "
          + "  |  ".join(segs))
    if "exp15" in snap:
        e = snap["exp15"]
        segs = [f"a{r['alpha']:.1f} inter={r['dL_inter']:.2f} HTH={r['dL_HTH']:.2f}"
                for r in e["rows"]]
        allwin = all(r["winner"] == "HTH" for r in e["rows"] if r["alpha"] > 0)
        p(f"exp15  baseline HTH vs 3-way interaction (true {e['edge']}): "
          + "  |  ".join(segs) + f"   [HTH wins all a>0: {allwin}]")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    snap = collect()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snap["_timestamp"] = stamp

    json_path = os.path.join(OUT_DIR, f"golden_{stamp}.json")
    txt_path = os.path.join(OUT_DIR, f"golden_{stamp}.txt")
    with open(json_path, "w") as f:
        json.dump(snap, f, indent=2)
    summary = render(snap)
    with open(txt_path, "w") as f:
        f.write(summary + "\n")

    print(summary)
    print("\n" + "=" * 60)
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"\nCaptured {len([k for k in snap if k.startswith('exp')])} experiments.")
    missing = [e for e in ["exp1b","exp2","exp3","exp4","exp7","exp8","exp9","exp10","exp11","exp12","exp13","exp14","exp15"]
               if e not in snap]
    if missing:
        print(f"NOTE: no artifact found for {missing} "
              f"(run those experiments first to include them).")


if __name__ == "__main__":
    main()