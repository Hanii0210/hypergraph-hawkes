"""Checkpointed fixed-candidate recovery and baseline-confounding studies.

Generation, fitting and prediction have separate records. Candidate sets and
basis definitions are fixed before any event is generated. No test score is
used to choose an initialisation. Each worker runs one generation or one start.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import zipfile

import numpy as np

from experiments.realdata_pipeline import _base_initial_state
from inference.em import run_em
from inference.coordinate import refine_likelihood
from models.baseline import BaselineBasis, TemporalBaseline
from models.diagnostics import stability_diagnostics
from models.features import EventFeatures
from models.kernel import ExponentialKernel, HyperedgeAnchor
from simulation.simulator import HawkesSimulator

ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = {"M0": 0, "C-M1": 1, "C-M2": 2, "V-M1": 3, "V-M2": 4}
FINAL_START_STATES = {"qualified", "unqualified", "failed", "timeout", "worker_failed"}


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    replace_saved_file(temporary, path)


def replace_saved_file(temporary, path):
    """Retry a briefly locked destination, while retaining atomic replacement."""
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * 2**attempt)


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def package_version():
    try:
        return read_json(ROOT / "docs/code_manifest.json").get("package_id", "unrecorded")
    except (OSError, ValueError, AttributeError):
        return "unrecorded"


def process_peak_bytes():
    """Peak for this worker process, including imports, not an allocation delta."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "peak", "working", "paged_peak", "paged", "nonpaged_peak",
                "nonpaged", "pagefile", "pagefile_peak")]
    counter = Counters()
    counter.cb = ctypes.sizeof(counter)
    get_current = ctypes.WinDLL("kernel32").GetCurrentProcess
    get_current.restype = wintypes.HANDLE
    query = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    if query(get_current(), ctypes.byref(counter), counter.cb):
        return int(counter.peak)
    return None


def seed_for(cfg, spec, repeat, purpose, model=0, start=0, phase="pilot"):
    phase_id = {"pilot": 1, "formal": 2}[phase]
    entropy = [int(cfg["master_seed"]), spec["experiment_id"], spec["scenario_id"],
               phase_id, int(repeat), int(purpose), int(model), int(start)]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])


def make_plan(cfg, repetitions=None, experiment="both", phase="pilot"):
    if phase not in {"pilot", "formal"}:
        raise ValueError("unknown study phase")
    if phase == "formal" and repetitions is not None:
        raise ValueError("formal repetitions are prespecified per scenario: 100 or 50")
    if phase == "pilot":
        repetitions = 2 if repetitions is None else repetitions
        if repetitions not in (2, 10):
            raise ValueError("pilot repetitions must be 2 or 10")
    from experiments.syn13_overlapping_hyperedges import scenarios as overlap
    from experiments.syn14_time_varying_baseline import scenarios as varying, include_constant_controls
    cases = overlap()
    if experiment != "syn13":
        include_constant_controls(cases)
        if experiment == "syn14":
            cases = [case for case in cases if "shared_with" in case]
        cases += varying()
    plan = []
    for spec in cases:
        count = spec["formal_repetitions"] if phase == "formal" else repetitions
        for repeat in range(1, count + 1):
            plan.append(make_dataset_task(cfg, spec, repeat, phase=phase))
    if phase == "pilot" and repetitions == 10 and experiment != "syn14":
        spec = next(case for case in cases if case["experiment_id"] == 13
                    and case["T"] == 1000 and case["triple_weight"] == 0.4)
        for repeat in range(1, 11):
            task = make_dataset_task(cfg, spec, repeat, double_burn=True)
            plan.append(task)
    return plan


def make_dataset_task(cfg, spec, repeat, double_burn=False, phase="pilot"):
    spec = dict(spec)
    if double_burn:
        spec["models"], spec["warm_from"] = [], {}
    suffix = "/double_burn_in" if double_burn else ""
    relative = f"{spec['experiment']}/{phase}/{spec['name']}{suffix}/rep_{repeat:03d}"
    return {"config": cfg, "scenario": spec, "phase": phase, "repeat": repeat,
            "path": relative, "dataset_id": relative, "double_burn_in": double_burn,
            "burn_in": cfg["burn_in_beta_units"] / cfg["beta"] * (2 if double_burn else 1),
            "generation_seed": seed_for(cfg, spec, repeat, 3 if double_burn else 1, phase=phase),
            "start_seeds": {model: [seed_for(cfg, spec, repeat, 2, MODEL_IDS[model], i, phase=phase)
                                    for i in range(cfg["em"]["n_initialisations"])]
                            for model in spec["models"]}}


def candidate_edges(cfg, model="C-M2"):
    if model == "M0":
        return []
    pairs = list(combinations(range(cfg["n_nodes"]), 2))
    if model.endswith("M1"):
        return pairs
    return pairs + [tuple(e) for e in cfg["true_triples"] + cfg["decoy_triples"]]


def fitted_basis(cfg, spec, model):
    if not model.startswith("V-"):
        return BaselineBasis()
    if spec["background"] == "linear":
        return BaselineBasis("linear", origin=0, end=spec["T"])
    return BaselineBasis("sinusoidal", origin=0, period=cfg["sinusoidal_period"])


def true_parameters(cfg, spec):
    n, rate = cfg["n_nodes"], cfg["baseline_rate"]
    pair = np.zeros((n, n))
    for i in range(n):
        pair[i, (i+1) % n] += cfg["ring_weight"]
    h = {}
    for edge in cfg["true_triples"]:
        for i in edge:
            for j in edge:
                if i != j:
                    pair[i, j] += cfg["within_group_weight"]
        for small in combinations(edge, 2):
            h[tuple(small)] = cfg["pair_completion_weight"]
        h[tuple(edge)] = spec["triple_weight"]
    kind = spec["background"]
    if kind == "constant":
        base = TemporalBaseline(np.full(n, rate))
    elif kind == "sinusoidal":
        coefficients = np.tile([rate*(1+spec["amplitude"])/2,
                                rate*(1-spec["amplitude"])/2], (n, 1))
        base = TemporalBaseline(coefficients, BaselineBasis(kind, period=cfg["sinusoidal_period"]))
    else:
        coefficients = np.tile([rate*(1-spec["amplitude"]), rate*(1+spec["amplitude"])], (n, 1))
        base = TemporalBaseline(coefficients, BaselineBasis(kind, end=spec["T"]))
    return base, pair, h


def parameter_record(background, pair, hyper, edges):
    return {"baseline_basis": background.basis.as_dict(),
            "baseline_coefficients": background.coefficients,
            "alpha_pairwise": pair, "edges": edges,
            "alpha_hyper": [hyper.get(e, 0.0) for e in edges]}


def restore_parameters(record):
    background = TemporalBaseline(record["baseline_coefficients"], BaselineBasis(**record["baseline_basis"]))
    pair = np.asarray(record["alpha_pairwise"], dtype=float)
    hyper = {tuple(e): float(v) for e, v in zip(record["edges"], record["alpha_hyper"])}
    return background, pair, hyper


def load_events(directory):
    with np.load(Path(directory) / "events.npz", allow_pickle=False) as stored:
        return list(zip(stored["times"].tolist(), stored["nodes"].tolist()))


def describe_interval(events, cfg, start, end):
    features = EventFeatures(events, cfg["n_nodes"], candidate_edges(cfg), cfg["beta"],
                             cfg["delta"], start, end, include_start=False)
    counts = np.bincount(features.nodes, minlength=cfg["n_nodes"])
    return {"start": start, "end": end, "n_events": len(features.times),
            "node_counts": counts, "node_rates": counts/(end-start),
            "edges": features.edges, "completion_counts": features.completion_counts,
            "completion_rates": features.completion_counts/(end-start),
            "hyperedge_exposure": features.hyper_exposure,
            "hyperedge_exposure_per_time": features.hyper_exposure/(end-start),
            "pairwise_source_exposure": features.pair_exposure,
            "feature_bytes": features.nbytes}


def generation_worker(path):
    record = read_json(path)
    task, folder = record["task"], Path(path).parent
    cfg, spec = task["config"], task["scenario"]
    background, pair, hyper = true_parameters(cfg, spec)
    record["truth"] = parameter_record(background, pair, hyper, candidate_edges(cfg))
    record["true_stability"] = stability_diagnostics(background, pair, hyper, cfg["beta"])
    record["attempts"] = []
    simulator = HawkesSimulator(background, pair, hyper, ExponentialKernel(cfg["beta"]),
                                 HyperedgeAnchor(cfg["delta"]))
    started = time.perf_counter()
    for cap in cfg["event_caps"]:
        result = simulator.simulate(spec["T"], seed=task["generation_seed"], max_events=cap,
                                    burn_in=task["burn_in"], return_info=True)
        info = result.as_dict()
        info.pop("events")
        info.pop("proposal_trace")
        record["attempts"].append(info)
        np.savez_compressed(folder/"events.npz", times=np.array([t for t, _ in result.events]),
                            nodes=np.array([n for _, n in result.events], dtype=np.int16))
        write_json(path, record)
        if result.completed:
            break
    record["generation_seconds"] = time.perf_counter()-started
    record["status"] = "completed" if result.completed else "event_cap"
    record["events_include_prehistory"] = True
    record["n_events_including_prehistory"] = len(result.events)
    if result.completed:
        diagnostic_start = time.perf_counter()
        split = spec["T"]*cfg["training_fraction"]
        intervals = {"training": (0, split), "test": (split, spec["T"]),
                     "observation": (0, spec["T"])}
        record["intervals"] = {name: describe_interval(result.events, cfg, a, b)
                               for name, (a, b) in intervals.items()}
        record["diagnostic_seconds"] = time.perf_counter()-diagnostic_start
    return record


def initial_parameters(events, cfg, spec, model, seed, index, warm_record=None):
    edges, basis = candidate_edges(cfg, model), fitted_basis(cfg, spec, model)
    split = spec["T"]*cfg["training_fraction"]
    training = [(t, n) for t, n in events if t <= split]
    warm = None
    if warm_record is not None:
        base, pair, hyper = restore_parameters(warm_record)
        if base.basis != basis:
            if base.basis.kind != "constant":
                raise ValueError("only a constant fit may initialise a different temporal basis")
            base = TemporalBaseline(basis.constant_coefficients(base.coefficients[:, 0]), basis)
        warm = SimpleNamespace(mu=base.coefficients, alpha_pairwise=pair, alpha_hyper=hyper)
    mu, pair, hyper = _base_initial_state(training, cfg["n_nodes"], 0.0, split, edges, seed, warm)
    if warm is not None and index == 0:
        mu, pair = warm.mu.copy(), warm.alpha_pairwise.copy()
        hyper = {e: warm.alpha_hyper[e] if warm.alpha_hyper.get(e, 0) > 0 else 0.03 for e in edges}
    mu = basis.constant_coefficients(mu) if np.asarray(mu).ndim == 1 else mu
    mu = np.maximum(mu, basis.lower_bounds(cfg["n_nodes"]))
    mask = ~np.eye(cfg["n_nodes"], dtype=bool)
    pair = np.where(mask, np.maximum(pair, 1e-8), 0.0)
    hyper = {e: max(v, 1e-8) for e, v in hyper.items()}
    return parameter_record(TemporalBaseline(mu, basis), pair, hyper, edges)


def fitting_worker(path):
    record = read_json(path)
    task = record["task"]
    cfg, spec = task["config"], task["scenario"]
    split = cfg["training_fraction"]*spec["T"]
    events = [(t, n) for t, n in load_events(record["dataset_directory"]) if t <= split]
    base, pair, hyper = restore_parameters(record["initial_parameters"])
    settings = dict(cfg["em"])
    settings.pop("n_initialisations")
    result = run_em(events=events, T=spec["T"], n_nodes=cfg["n_nodes"],
                    edge_list=list(hyper), kernel=ExponentialKernel(cfg["beta"]),
                    anchor_calc=HyperedgeAnchor(cfg["delta"]), mu0=base,
                    alpha_pairwise0=pair, alpha_hyper0=hyper,
                    baseline_basis=base.basis, pairwise_mask=~np.eye(cfg["n_nodes"], dtype=bool),
                    hyper_update="free", positive_start=True, include_start=False,
                    target_start=0.0, target_end=split, track_loglik=True,
                    assert_ascent=True, **settings)
    finite = (result.final_objective is not None and np.isfinite(result.final_objective)
              and np.isfinite(result.kkt_residual))
    reasons = []
    if not finite:
        reasons.append("nonfinite_objective_or_gradient")
    if not result.converged:
        reasons.append("stopping_criteria_not_met")
    if result.kkt_residual > cfg["em"]["kkt_tol"]:
        reasons.append("kkt_not_met")
    if result.ascent_violations:
        reasons.append("objective_decreased")
    record.update({"status": "qualified" if not reasons else "unqualified", "reasons": reasons,
                   "final_parameters": parameter_record(result.baseline, result.alpha_pairwise,
                                                         result.alpha_hyper, list(hyper)),
                   "objective": result.final_objective, "n_iter": result.n_iter,
                   "converged": result.converged, "kkt_residual": result.kkt_residual,
                   "absolute_increment": result.final_absolute_increment,
                   "relative_increment": result.final_relative_increment,
                   "ascent_violations": result.ascent_violations,
                   "checkpoints": result.checkpoints + [{"iteration": result.n_iter,
                                                         "objective": result.final_objective,
                                                         "converged": result.converged}],
                   "objective_history": result.objective_history,
                   "training_score": result.score_components,
                   "identifiable_hyperedges": [result.identifiable_hyperedges[e] for e in hyper],
                   "zero_pairwise_source_exposure": result.zero_pairwise_exposure,
                   "stability": result.stability, "feature_bytes": result.feature_bytes,
                   "feature_preparation_seconds": result.preparation_seconds,
                   "fit_seconds": result.elapsed_seconds})
    if cfg.get("refinement"):
        record["em_result"] = dict(record)
        record["refinement_settings"] = cfg["refinement"]
        record["refinement_attempted"] = True
        record["status"] = "running"
        write_json(path, record)
        record = refine_record(record, events)
    return record


def refine_record(record, events=None):
    """Keep the complete EM outcome and refine its endpoint on training data."""
    task = record["task"]
    cfg, spec = task["config"], task["scenario"]
    split = cfg["training_fraction"]*spec["T"]
    if "em_result" not in record:
        keys = ("status", "reasons", "final_parameters", "objective", "n_iter", "converged",
                "kkt_residual", "absolute_increment", "relative_increment", "ascent_violations",
                "checkpoints", "objective_history", "training_score", "identifiable_hyperedges",
                "zero_pairwise_source_exposure", "stability", "feature_bytes", "fit_seconds",
                "feature_preparation_seconds", "wall_seconds", "worker_seconds", "package_version")
        record["em_result"] = {key: record[key] for key in keys if key in record}
    original = record["em_result"]
    if events is None:
        events = load_events(record["dataset_directory"])
    events = [(t, n) for t, n in events if t <= split]
    base, pair, hyper = restore_parameters(original["final_parameters"])
    f = EventFeatures(events, cfg["n_nodes"], list(hyper), cfg["beta"], cfg["delta"],
                      0.0, split, basis=base.basis, include_start=False)
    settings = record["refinement_settings"]
    result = refine_likelihood(f, base.coefficients, pair, hyper, ~np.eye(cfg["n_nodes"], dtype=bool),
                               max_sweeps=settings["max_sweeps"], lambda_l1=cfg["em"]["lambda_l1"],
                               kkt_tol=cfg["em"]["kkt_tol"], convergence_tol=cfg["em"]["convergence_tol"],
                               consecutive_convergence=cfg["em"]["consecutive_convergence"],
                               ascent_tol=cfg["em"]["ascent_tol"])
    if abs(result["initial_objective"]-original["objective"]) > 1e-7*(1+abs(original["objective"])):
        raise AssertionError("saved EM endpoint does not reproduce its training objective")
    new_base = TemporalBaseline(result["baseline_coefficients"], base.basis)
    refinement = {key: value for key, value in result.items()
                  if key not in {"baseline_coefficients", "alpha_pairwise", "alpha_hyper", "training_score"}}
    refinement.update(method="coordinate_maximisation", settings=settings,
                      objective_gain=result["final_objective"]-result["initial_objective"])
    record.update(optimizer="em_then_coordinate_maximisation", refinement=refinement,
                  status="qualified" if result["converged"] else "unqualified",
                  reasons=[] if result["converged"] else ["coordinate_stopping_criteria_not_met"],
                  converged=result["converged"], kkt_residual=result["kkt_residual"],
                  final_parameters=parameter_record(new_base, result["alpha_pairwise"], result["alpha_hyper"], list(hyper)),
                  objective=result["final_objective"], training_score=result["training_score"],
                  absolute_increment=abs(result["objective_history"][-1]-(result["objective_history"][-2]
                      if len(result["objective_history"]) > 1 else result["initial_objective"])),
                  relative_increment=result["relative_increment_history"][-1],
                  objective_history=original["objective_history"]+result["objective_history"],
                  stability=stability_diagnostics(new_base, result["alpha_pairwise"], result["alpha_hyper"], cfg["beta"]))
    record["checkpoints"] = original["checkpoints"] + [{"stage": "coordinate_maximisation",
        "sweep": result["n_sweeps"], "objective": result["final_objective"], "converged": result["converged"]}]
    return record


def worker_main(kind, path):
    started = time.perf_counter()
    record = read_json(path)
    try:
        if kind == "generation":
            record = generation_worker(path)
        elif kind == "fit":
            record = fitting_worker(path)
        else:
            record = refine_record(record)
    except KeyboardInterrupt:
        record = read_json(path)
        record.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as exc:
        record = read_json(path)
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    record.update(worker_seconds=time.perf_counter()-started, peak_process_working_set_bytes=process_peak_bytes(),
                  finished_utc=datetime.now(timezone.utc).isoformat(), package_version=package_version())
    write_json(path, record)
    return 0 if record["status"] in {"qualified", "unqualified", "completed", "event_cap"} else 1


def run_worker(kind, path, timeout):
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    command = [sys.executable, "-X", "utf8", str(ROOT/"scripts/run_synthetic_experiments.py"),
               "--worker", kind, "--record", str(path)]
    started = time.perf_counter()
    with path.with_suffix(".log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT) as child:
            try:
                while True:
                    remaining = timeout-(time.perf_counter()-started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        child.wait(timeout=min(30, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        print(f"    仍在运行：{time.perf_counter()-started:.0f} 秒", flush=True)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                child.kill()
                child.wait()
                record = read_json(path)
                record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "timeout",
                              timeout_seconds=timeout, wall_seconds=time.perf_counter()-started)
                write_json(path, record)
                if isinstance(exc, KeyboardInterrupt):
                    raise
            record = read_json(path)
            if record["status"] == "running":
                unsaved = path.with_suffix(path.suffix+".tmp")
                if unsaved.is_file():
                    preserved = path.parent/"archive"/timestamp()/path.name
                    preserved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(unsaved, preserved)
                    record["unsaved_result_record"] = preserved.relative_to(path.parent).as_posix()
                record.update(status="worker_failed", returncode=child.returncode)
            record["wall_seconds"] = time.perf_counter()-started
            write_json(path, record)
            return record


def choose_start(starts):
    qualified = [(i, r) for i, r in enumerate(starts) if r.get("status") == "qualified"
                 and r.get("objective") is not None and math.isfinite(r["objective"])]
    if not qualified:
        return {"status": "no_qualified_start", "selected_start": None}
    index, best = max(qualified, key=lambda item: item[1]["objective"])
    tolerance = 1e-6*(1+abs(best["objective"]))
    if any(r.get("status") != "qualified" and r.get("objective") is not None
           and r["objective"] > best["objective"] + tolerance for r in starts):
        return {"status": "higher_unqualified_objective", "selected_start": None,
                "best_qualified_start_for_review": index+1}
    similar = [r for _, r in qualified if best["objective"]-r["objective"] <= tolerance]
    vectors = []
    for row in similar:
        p = row["final_parameters"]
        vectors.append(np.concatenate([np.asarray(p["baseline_coefficients"]).ravel(),
                                       np.asarray(p["alpha_pairwise"]).ravel(), p["alpha_hyper"]]))
    spread = np.ptp(np.asarray(vectors), axis=0)
    return {"status": "qualified", "selected_start": index+1,
            "selection_basis": "maximum qualified training objective only",
            "n_qualified_starts": len(qualified), "n_near_optimal_starts": len(similar),
            "near_optimal_max_parameter_range": float(spread.max(initial=0.0))}


def baseline_error(fitted, truth, start, end, points):
    errors = []
    for n in (points, 2*points-1):
        grid = np.linspace(start, end, n)
        difference = fitted.values(grid)-truth.values(grid)
        errors.append(np.trapezoid(difference**2, x=grid, axis=0)/(end-start))
    change = float(np.max(np.abs(errors[1]-errors[0])))
    return {"grid_points": points, "mse_per_node": errors[0], "mse_mean": float(errors[0].mean()),
            "refined_grid_points": 2*points-1, "refined_mse_per_node": errors[1],
            "max_refinement_change": change,
            "grid_stable": change <= 1e-4*(1+float(np.max(np.abs(errors[1]))))}


def assess_model(events, task, model, selected):
    cfg, spec = task["config"], task["scenario"]
    background, pair, hyper = restore_parameters(selected["final_parameters"])
    true_base, true_pair, true_hyper = true_parameters(cfg, spec)
    split = spec["T"]*cfg["training_fraction"]
    started = time.perf_counter()
    features = EventFeatures(events, cfg["n_nodes"], list(hyper), cfg["beta"], cfg["delta"],
                             split, spec["T"], basis=background.basis, include_start=False)
    score = features.score(background.coefficients, pair, hyper)
    score["per_time"] = score["loglik"]/score["duration"]
    score["per_event"] = score["loglik"]/score["n_events"] if score["n_events"] else None
    score["per_event_missing_reason"] = None if score["n_events"] else "zero_test_events"
    scoring_seconds = time.perf_counter()-started
    diagnostic_started = time.perf_counter()
    h_errors = []
    for edge, eligible in zip(hyper, selected["identifiable_hyperedges"]):
        h_errors.append({"edge": edge, "estimate": hyper[edge], "truth": true_hyper.get(edge, 0.0),
                         "identifiable": eligible,
                         "error": hyper[edge]-true_hyper.get(edge, 0.0) if eligible else None})
    pair_eligible = ~np.eye(cfg["n_nodes"], dtype=bool)
    pair_eligible[selected["zero_pairwise_source_exposure"], :] = False
    pair_errors = (pair-true_pair)[pair_eligible]
    h3 = [v for e, v in hyper.items() if len(e) == 3]
    return {"test_score": score, "scoring_seconds": scoring_seconds,
            "parameter_errors": {"hyperedges": h_errors,
                "pairwise_eligible_count": int(pair_eligible.sum()),
                "pairwise_error_matrix": np.where(pair_eligible, pair-true_pair, np.nan),
                "pairwise_mse": float(np.mean(pair_errors**2)) if len(pair_errors) else None,
                "triple_sum_weight": sum(h3) if h3 else None,
                "triple_max_weight": max(h3) if h3 else None},
            "baseline_errors": {label: baseline_error(background, true_base, a, b, cfg["baseline_error_grid"])
                                for label, a, b in (("training", 0, split), ("test", split, spec["T"]))},
            "diagnostic_seconds": time.perf_counter()-diagnostic_started}


def fit_model(directory, task, model, events, fitted):
    cfg, spec = task["config"], task["scenario"]
    model_dir = directory / "models" / model
    final_path = model_dir / "fit.json"
    if final_path.is_file():
        return read_json(final_path)
    warm_name = spec["warm_from"].get(model)
    warm_record = fitted.get(warm_name, {})
    warm = warm_record.get("parameters") if warm_record.get("status") == "qualified" else None
    starts = []
    for index, seed in enumerate(task["start_seeds"][model]):
        path = model_dir / f"start_{index+1}.json"
        current = read_json(path) if path.is_file() else {}
        if current.get("status") not in FINAL_START_STATES:
            if current:
                archive = directory / "archive" / timestamp() / model
                write_json(archive/path.name, current)
            initial = initial_parameters(events, cfg, spec, model, seed, index, warm)
            record = {"status": "running", "task": task, "model": model, "start_index": index+1,
                      "seed": seed, "dataset_directory": str(directory),
                      "warm_source": warm_name if warm is not None else None,
                      "warm_fallback_reason": "source_not_qualified" if warm_name and warm is None else None,
                      "initial_parameters": initial, "pairwise_diagonal_fixed_zero": True,
                      "package_version": package_version()}
            write_json(path, record)
            print(f"  {model} 起点 {index+1}/3", flush=True)
            current = run_worker("fit", path, cfg["execution"]["start_timeout_seconds"])
            print(f"    {current['status']}，{current['wall_seconds']:.1f} 秒，"
                  f"迭代 {current.get('n_iter', '—')}，KKT {current.get('kkt_residual', '—')}", flush=True)
        starts.append(current)
    return finalise_model(directory, task, model, events, starts)


def finalise_model(directory, task, model, events, starts):
    cfg, spec = task["config"], task["scenario"]
    final_path = directory/"models"/model/"fit.json"
    selection = choose_start(starts)
    output = {"model": model, "dataset_id": task["dataset_id"], **selection,
              "start_records": [f"start_{i+1}.json" for i in range(len(starts))],
              "start_statuses": [r["status"] for r in starts],
              "total_start_wall_seconds": sum(r.get("wall_seconds", 0) for r in starts),
              "max_worker_peak_bytes": max((r["peak_process_working_set_bytes"] for r in starts
                                             if r.get("peak_process_working_set_bytes") is not None), default=None),
              "n_parameters": cfg["n_nodes"]*fitted_basis(cfg, spec, model).size
                              + cfg["n_nodes"]*(cfg["n_nodes"]-1)+len(candidate_edges(cfg, model)),
              "penalty": 0.0, "package_version": package_version(),
              "optimizer": "em_then_coordinate_maximisation" if cfg.get("refinement") else "em"}
    if selection["selected_start"] is not None:
        selected = starts[selection["selected_start"]-1]
        output.update(parameters=selected["final_parameters"], training_score=selected["training_score"],
                      kkt_residual=selected["kkt_residual"], stability=selected["stability"])
        try:
            output.update(assess_model(events, task, model, selected))
        except Exception as exc:
            output.update(status="scoring_or_diagnostic_failed", error=str(exc), traceback=traceback.format_exc())
    write_json(final_path, output)
    return output


def comparisons(fits):
    output = {}
    for larger, smaller in (("C-M2", "C-M1"), ("V-M2", "V-M1"), ("C-M1", "M0"), ("C-M2", "M0")):
        if larger not in fits or smaller not in fits:
            continue
        key = larger+"_minus_"+smaller
        if any(fits[name].get("status") != "qualified" for name in (larger, smaller)):
            output[key] = {"status": "missing", "reason": "both_models_must_qualify"}
            continue
        a, b = fits[larger]["test_score"], fits[smaller]["test_score"]
        if a["duration"] != b["duration"] or a["n_events"] != b["n_events"]:
            raise ValueError("paired predictions must use the same test interval and events")
        delta = a["loglik"]-b["loglik"]
        output[key] = {"status": "qualified", "delta_loglik": delta,
                       "delta_per_time": delta/a["duration"],
                       "delta_per_event": delta/a["n_events"] if a["n_events"] else None,
                       "test_events": a["n_events"], "test_duration": a["duration"]}
    return output


def run_dataset(results_root, task):
    directory = results_root / task["path"]
    generation_path = directory / "dataset.json"
    current = read_json(generation_path) if generation_path.is_file() else {}
    if current.get("status") not in {"completed", "event_cap", "failed", "timeout", "worker_failed"}:
        if current:
            write_json(directory/"archive"/timestamp()/"dataset.json", current)
        write_json(generation_path, {"status": "running", "task": task, "package_version": package_version()})
        current = run_worker("generation", generation_path, task["config"]["execution"]["generation_timeout_seconds"])
    print(f"  生成：{current['status']}，事件（含预热）{current.get('n_events_including_prehistory', '—')}", flush=True)
    if current["status"] != "completed" or not task["scenario"]["models"]:
        return
    events = load_events(directory)
    fits = {}
    for model in task["scenario"]["models"]:
        fits[model] = fit_model(directory, task, model, events, fits)
    write_json(directory/"comparisons.json", comparisons(fits))


def refine_dataset(results_root, task):
    """Refine every existing start, including qualified ones; never simulate."""
    directory = results_root/task["path"]
    generation = read_json(directory/"dataset.json")
    if generation["status"] != "completed":
        raise ValueError("refinement requires a complete existing event path")
    if not task["scenario"]["models"]:
        return
    # A completed refinement, including a numerical non-convergence, is final.
    # An interrupted/failed worker with only an EM checkpoint is still pending.
    def done(current):
        return (current.get("status") in {"qualified", "unqualified"}
                and bool(current.get("refinement"))
                and current.get("refinement_settings") == task["config"]["refinement"])
    pending = []
    for model in task["scenario"]["models"]:
        for index in range(task["config"]["em"]["n_initialisations"]):
            path = directory/"models"/model/f"start_{index+1}.json"
            if not done(read_json(path)):
                pending.append(path)
    if not pending and all((directory/"models"/model/"fit.json").is_file()
                           for model in task["scenario"]["models"]) and (directory/"comparisons.json").is_file():
        return
    events = load_events(directory)
    marker = directory/"refinement.json"
    if marker.is_file():
        record = read_json(marker)
        archive = directory/record["original_records"]
    else:
        relative = "archive/"+timestamp()
        archive = directory/relative
        record = {"method": task["config"]["refinement"], "original_records": relative,
                  "policy": "all existing starts; no new data or initialisations"}
        # Copy the original records before replacing any of them.
        for model in task["scenario"]["models"]:
            for source in (directory/"models"/model).glob("*"):
                if source.is_file():
                    target = archive/source.relative_to(directory)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        if (directory/"comparisons.json").is_file():
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(directory/"comparisons.json", archive/"comparisons.json")
        write_json(marker, record)
    # Preserve this particular attempt even if an earlier refinement archive exists.
    if pending:
        attempt_archive = directory/"archive"/timestamp()
        for path in pending:
            for source in (path, path.with_suffix(".log")):
                if source.is_file():
                    target = attempt_archive/source.relative_to(directory)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    fits = {}
    for model in task["scenario"]["models"]:
        starts = []
        for index in range(task["config"]["em"]["n_initialisations"]):
            path = directory/"models"/model/f"start_{index+1}.json"
            current = read_json(path)
            already_done = done(current)
            if not already_done:
                previous_wall = current.get("wall_seconds", 0)
                previous_peak = current.get("peak_process_working_set_bytes")
                if "em_result" not in current:
                    current["em_result"] = dict(current)
                original = current["em_result"]
                if "final_parameters" not in original:
                    raise ValueError("existing start lacks an EM endpoint: "+str(path))
                current.update(status="running", refinement_attempted=True,
                               refinement_settings=task["config"]["refinement"],
                               dataset_directory=str(directory), refinement_only=True)
                current["task"] = task
                for stale_key in ("returncode", "error", "error_type", "traceback", "timeout_seconds"):
                    current.pop(stale_key, None)
                write_json(path, current)
                print(f"  {model} 起点 {index+1}/3：继续优化已保存的EM终点", flush=True)
                current = run_worker("refinement", path, task["config"]["execution"]["start_timeout_seconds"])
                current["refinement_wall_seconds"] = current["wall_seconds"]
                current["wall_seconds"] += previous_wall
                peaks = [value for value in (current.get("peak_process_working_set_bytes"),
                                             previous_peak) if value is not None]
                current["peak_process_working_set_bytes"] = max(peaks, default=None)
                write_json(path, current)
                print(f"    {current['status']}，本次{current['refinement_wall_seconds']:.1f}秒，"
                      f"KKT {current.get('kkt_residual', '—')}", flush=True)
            starts.append(current)
        fits[model] = finalise_model(directory, task, model, events, starts)
    write_json(directory/"comparisons.json", comparisons(fits))
    record["status"] = "completed" if all(
        done(read_json(directory/"models"/model/f"start_{i+1}.json"))
        for model in task["scenario"]["models"] for i in range(task["config"]["em"]["n_initialisations"])) else "incomplete"
    write_json(marker, record)


def descriptive(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    return {"n": n, "mean": float(np.mean(values)) if n else None,
            "median": float(np.median(values)) if n else None,
            "mcse_mean": float(np.std(values, ddof=1)/np.sqrt(n)) if n > 1 else None,
            "p90": float(np.quantile(values, 0.9)) if n else None}


def recovery_summary(rows, planned):
    """Monte Carlo units are independent event paths, never individual edges."""
    def errors(values):
        values = np.asarray([v for v in values if v is not None], dtype=float)
        values = values[np.isfinite(values)]
        bias, squared = descriptive(values), descriptive(values**2)
        return {"n": len(values), "bias": bias["mean"], "mcse_bias": bias["mcse_mean"],
                "mse": squared["mean"], "mcse_mse": squared["mcse_mean"],
                "rmse": math.sqrt(squared["mean"]) if len(values) else None}
    edges = sorted({tuple(e["edge"]) for r in rows for e in r["parameter_errors"]["hyperedges"]})
    per_edge = []
    for edge in edges:
        records = [e for r in rows for e in r["parameter_errors"]["hyperedges"] if tuple(e["edge"]) == edge]
        per_edge.append({"edge": edge, "truth": records[0]["truth"],
                         "unidentifiable": sum(not e["identifiable"] for e in records),
                         **errors([e["error"] for e in records])})
    pair_entries = []
    if rows:
        n = len(rows[0]["parameter_errors"]["pairwise_error_matrix"])
        for source in range(n):
            for target in range(n):
                if source != target:
                    pair_entries.append({"source": source, "target": target,
                        **errors([r["parameter_errors"]["pairwise_error_matrix"][source][target] for r in rows])})
    pair_mse = descriptive([r["parameter_errors"]["pairwise_mse"] for r in rows])
    baseline = {}
    for interval in ("training", "test"):
        mse = descriptive([r["baseline_errors"][interval]["mse_mean"] for r in rows])
        baseline[interval] = {"mse": mse, "rmse": math.sqrt(mse["mean"]) if mse["n"] else None}
    return {"planned": planned, "qualified": len(rows),
            "interpretation": "conditional on numerical qualification; MCSE across paths; no selection or testing claim",
            "hyperedges": per_edge, "pairwise_coefficients": pair_entries,
            "pairwise_mse": pair_mse,
            "pairwise_rmse": math.sqrt(pair_mse["mean"]) if pair_mse["n"] else None,
            "baseline": baseline,
            "triple_sum_weight": descriptive([r["parameter_errors"]["triple_sum_weight"] for r in rows]),
            "triple_max_weight": descriptive([r["parameter_errors"]["triple_max_weight"] for r in rows])}


def summarise(results_root, plan):
    groups, comparisons_by_group, generation = {}, {}, {}
    phase = plan[0]["phase"]
    start_statuses = {}
    missing = []
    for task in plan:
        spec = task["scenario"]
        folder = results_root/task["path"]
        group = spec["experiment"]+"/"+spec["name"]+("/double_burn_in" if task["double_burn_in"] else "")
        gen = read_json(folder/"dataset.json") if (folder/"dataset.json").is_file() else {"status": "not_run"}
        generation.setdefault(group, []).append(gen)
        comparisons_current = True
        for model in spec["models"]:
            fit_path = folder/"models"/model/"fit.json"
            fit = read_json(fit_path) if fit_path.is_file() else {"status": "not_completed"}
            if task["config"].get("refinement") and fit.get("optimizer") != "em_then_coordinate_maximisation":
                fit = {"status": "refinement_pending"}
            key = group+"/"+model
            groups.setdefault(key, []).append(fit)
            for status in fit.get("start_statuses", []):
                start_statuses[status] = start_statuses.get(status, 0)+1
            if fit["status"] in {"not_completed", "refinement_pending"}:
                missing.append(task["dataset_id"]+"/"+model)
                comparisons_current = False
        if comparisons_current and (folder/"comparisons.json").is_file():
            for name, comparison in read_json(folder/"comparisons.json").items():
                comparisons_by_group.setdefault(group+"/"+name, []).append(comparison)
    model_summary = {}
    for key, rows in groups.items():
        qualified = [r for r in rows if r["status"] == "qualified"]
        counts = {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})}
        model_summary[key] = {"planned": len(rows), "qualified": len(qualified), "statuses": counts,
                              "qualification_fraction": len(qualified)/len(rows),
                              "model_seconds": descriptive([r.get("total_start_wall_seconds") for r in rows]),
                              "kkt": descriptive([r.get("kkt_residual") for r in qualified]),
                              "max_worker_peak_bytes": max((r["max_worker_peak_bytes"] for r in rows
                                                             if r.get("max_worker_peak_bytes") is not None), default=None),
                              "formal_repetitions": next(t["scenario"]["formal_repetitions"] for t in plan
                                  if key.startswith(t["scenario"]["experiment"]+"/"+t["scenario"]["name"]+"/"))}
    recovery = {key: recovery_summary([r for r in rows if r["status"] == "qualified"], len(rows))
                for key, rows in groups.items()}
    paired = {}
    for key, rows in comparisons_by_group.items():
        successful = [r for r in rows if r["status"] == "qualified"]
        planned_group = generation[key.rsplit("/", 1)[0]]
        paired[key] = {"planned": len(planned_group),
                       "generation_completed": sum(r["status"] == "completed" for r in planned_group),
                       "recorded": len(rows), "jointly_qualified": len(successful),
                       "delta_loglik": descriptive([r["delta_loglik"] for r in successful]),
                       "delta_per_time": descriptive([r["delta_per_time"] for r in successful]),
                       "interpretation": ("descriptive pilot only; " if phase == "pilot" else "")
                                         + "conditional on joint numerical success; not a hypothesis test"}
    failures = [key for key, row in model_summary.items() if row["qualification_fraction"] < 0.9]
    generation_summary = {}
    for group, rows in generation.items():
        generation_summary[group] = {"planned": len(rows), "completed": sum(r["status"] == "completed" for r in rows),
            "statuses": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
            "generation_seconds": descriptive([r.get("generation_seconds") for r in rows])}
    return {"phase": phase, "purpose": "pilot_cost_convergence_and_exposure_not_formal_evidence" if phase == "pilot"
                                        else "prespecified_repeated_simulation",
            "planned_datasets": len(plan), "planned_models": sum(len(t["scenario"]["models"]) for t in plan),
            "generation": generation_summary, "models": model_summary, "paired_prediction": paired,
            "start_statuses": start_statuses, "parameter_recovery": recovery,
            "not_completed_models": missing, "groups_below_90_percent": failures,
            "cost_extrapolation": "see reviewed pilot estimate in docs/synthetic_experiments.md"}


def pilot_readiness(results_root, cfg, experiment="both"):
    """Check the reviewed full pilot and report unresolved work before formal runs."""
    pilot = make_plan(cfg, 10, experiment)
    config_path = results_root/"synthetic_experiments/pilot/config.json"
    if not config_path.is_file() or read_json(config_path) != cfg:
        return {"ready": False, "reasons": ["complete pilot with the same configuration is required"]}
    problems, groups = [], {}
    for task in pilot:
        directory = results_root/task["path"]
        path = directory/"dataset.json"
        gen = read_json(path) if path.is_file() else {}
        if gen.get("status") != "completed":
            problems.append(task["dataset_id"]+": generation not complete")
        for model in task["scenario"]["models"]:
            folder = directory/"models"/model
            fit_path = folder/"fit.json"
            fit = read_json(fit_path) if fit_path.is_file() else {}
            key = task["scenario"]["experiment"]+"/"+task["scenario"]["name"]+"/"+model
            groups.setdefault(key, []).append(fit.get("status") == "qualified")
            for i in range(cfg["em"]["n_initialisations"]):
                path = folder/f"start_{i+1}.json"
                row = read_json(path) if path.is_file() else {}
                if (row.get("status") not in {"qualified", "unqualified"} or not row.get("refinement")
                        or row.get("refinement_settings") != cfg["refinement"]):
                    problems.append(task["dataset_id"]+f"/{model}/start_{i+1}: refinement unfinished")
    below = [k for k,v in groups.items() if sum(v)/len(v) < 0.9]
    problems += [k+": model qualification below 90%" for k in below]
    return {"ready": not problems, "reasons": problems, "datasets": len(pilot),
            "models": sum(len(v) for v in groups.values()),
            "qualified_models": sum(sum(v) for v in groups.values()),
            "rule": "reviewed full pilot, complete saved refinements, at least 90% qualified models in every cell"}


def snapshot_code(destination, cfg):
    if destination.exists():
        return
    files = []
    for name in ("models", "inference", "simulation"):
        files += list((ROOT/name).glob("*.py"))
    files += [ROOT/"experiments"/name for name in ("realdata_pipeline.py", "synthetic_pipeline.py",
               "syn13_overlapping_hyperedges.py", "syn14_time_varying_baseline.py")]
    files += [ROOT/"scripts/run_synthetic_experiments.py", ROOT/"docs/code_manifest.json", ROOT/"requirements.txt"]
    for path in files:
        if path.is_file():
            target = destination/path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    write_json(destination/"configs/synthetic_experiments.json", cfg)


def package_results(results_root, plan, run_dir, destination):
    destination.mkdir(parents=True, exist_ok=True)
    package = destination/"hypergraph_hawkes_results.zip"
    if package.exists():
        archive = destination/"archive/hypergraph_hawkes"/timestamp()/package.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package, archive)
    paths = list(run_dir.rglob("*"))
    for task in plan:
        directory = results_root/task["path"]
        if directory.exists():
            paths += list(directory.rglob("*"))
    temporary = package.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(paths)):
            if path.is_file() and path.suffix != ".tmp":
                z.write(path, "experiments/results/synthetic/"+path.relative_to(results_root).as_posix())
    replace_saved_file(temporary, package)
    return package


def main(argv=None, experiment="both"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--repetitions", type=int, choices=(2, 10), help="Pilot only; default 2. Formal uses fixed 100/50 repetitions.")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/synthetic_experiments.json")
    parser.add_argument("--results-root", type=Path, default=ROOT/"experiments/results/synthetic")
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true", help="Describe counts and seeds without simulating or fitting.")
    parser.add_argument("--refine-existing", action="store_true", help="Refine all stored EM endpoints; do not generate new paths.")
    parser.add_argument("--worker", choices=("generation", "fit", "refinement"), help=argparse.SUPPRESS)
    parser.add_argument("--record", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.worker:
        if options.record is None:
            parser.error("worker record required")
        return worker_main(options.worker, options.record)
    if options.phase == "formal" and (options.repetitions is not None or options.refine_existing):
        parser.error("formal phase uses fixed scenario repetitions; --repetitions and --refine-existing are pilot only")
    if options.phase == "pilot" and options.repetitions is None:
        options.repetitions = 2
    cfg = read_json(options.config)
    refinement = cfg.get("refinement")
    if refinement and (refinement.get("method") != "coordinate_maximisation"
                       or refinement.get("apply_to_all_starts") is not True):
        parser.error("refinement must use the fixed coordinate method on all starts")
    if options.refine_existing and not refinement:
        parser.error("refinement settings are required for --refine-existing")
    if cfg["em"]["n_initialisations"] != 3 or cfg["em"]["lambda_l1"] != 0:
        parser.error("this study requires three starts and unpenalised fitting")
    if cfg["execution"]["workers"] != 1 or cfg["execution"]["blas_threads"] != 1:
        parser.error("this study uses one worker and one BLAS thread")
    plan = make_plan(cfg, options.repetitions, experiment, options.phase)
    counts = {"datasets": len(plan), "models": sum(len(t["scenario"]["models"]) for t in plan)}
    counts["starts"] = 3*counts["models"]
    if options.plan_only:
        print(json.dumps({"phase": options.phase, "repetitions_per_cell": options.repetitions if options.phase == "pilot" else "100 main / 50 linear",
                          **counts, "scientific_runs_executed": False}, ensure_ascii=False, indent=2))
        return 0
    from scripts.run_model_checks import downloads_directory
    destination = options.download_dir or downloads_directory()
    results_root = options.results_root.resolve()
    readiness = None
    if options.phase == "formal":
        readiness = pilot_readiness(results_root, cfg, experiment)
        if not readiness["ready"]:
            parser.error("formal run not started; resolve pilot records first: "+"; ".join(readiness["reasons"][:5]))
    run_dir = results_root/"synthetic_experiments"/options.phase
    config_path = run_dir/"config.json"
    if config_path.is_file() and read_json(config_path) != cfg:
        old_cfg = read_json(config_path)
        old_core = {k: v for k, v in old_cfg.items() if k != "refinement"}
        new_core = {k: v for k, v in cfg.items() if k != "refinement"}
        if not options.refine_existing or old_core != new_core:
            parser.error("saved scientific configuration differs; preserve this run and use a separate --results-root")
        archive = run_dir/"archive"/timestamp()
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "plan.json", "environment.json", "summary.json"):
            if (run_dir/name).is_file():
                shutil.copy2(run_dir/name, archive/name)
    if options.refine_existing:
        for task in plan:
            directory = results_root/task["path"]
            if not (directory/"dataset.json").is_file():
                parser.error("existing dataset required for refinement: "+str(directory))
            for model in task["scenario"]["models"]:
                for i in range(cfg["em"]["n_initialisations"]):
                    if not (directory/"models"/model/f"start_{i+1}.json").is_file():
                        parser.error("existing start records required: "+str(directory/"models"/model))
    write_json(config_path, cfg)
    if readiness is not None:
        write_json(run_dir/"pilot_readiness.json", readiness)
    snapshot = run_dir/"code"/package_version()
    snapshot_code(snapshot, cfg)
    # The repeat number determines its seed; extending 2 -> 10 preserves repeats 1 and 2.
    write_json(run_dir/"plan.json", {"phase": options.phase,
                                   "repetitions_per_cell": options.repetitions if options.phase == "pilot" else "100 main / 50 linear",
                                   "experiment_scope": experiment, "tasks": plan, **counts})
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "numpy": np.__version__, "rng": type(np.random.default_rng(0).bit_generator).__name__,
                   "package_version": package_version(), "code_snapshot": str(snapshot),
                   "workers": 1, "worker_blas_threads": 1}
    import importlib.metadata
    environment["scipy"] = importlib.metadata.version("scipy")
    write_json(run_dir/"environment.json", environment)
    started = time.perf_counter()
    invocation = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                  "mode": "refine_existing" if options.refine_existing else options.phase, **counts}
    return_code = 0
    try:
        for index, task in enumerate(plan, 1):
            print(f"\n[{index}/{len(plan)}] {task['dataset_id']}", flush=True)
            if options.refine_existing:
                refine_dataset(results_root, task)
            else:
                run_dataset(results_root, task)
            if options.phase == "pilot" or index % 10 == 0:
                write_json(run_dir/"summary.json", summarise(results_root, plan))
        invocation["status"] = "completed"
    except KeyboardInterrupt:
        invocation["status"] = "interrupted"
        return_code = 130
    except Exception as exc:
        invocation.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        return_code = 1
    finally:
        invocation["elapsed_seconds"] = time.perf_counter()-started
        invocation["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir/"runs"/(timestamp()+".json"), invocation)
        try:
            summary = summarise(results_root, plan)
            write_json(run_dir/"summary.json", summary)
            if options.refine_existing:
                readiness = pilot_readiness(results_root, cfg, experiment) if options.repetitions == 10 else None
                if readiness is not None:
                    write_json(run_dir/"readiness.json", readiness)
                    if not readiness["ready"]:
                        return_code = 2
                        print("尚有未完成的pilot记录，暂不进入正式实验。", flush=True)
        finally:
            packaging_start = time.perf_counter()
            package = package_results(results_root, plan, run_dir, destination)
            print(f"\n本次运行状态：{invocation['status']}；实验失败及未完成项见 summary.json。", flush=True)
            print(f"结果打包耗时：{time.perf_counter()-packaging_start:.1f} 秒", flush=True)
            print(f"Results archive: {package}", flush=True)
    return return_code
