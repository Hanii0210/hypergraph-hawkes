"""Independent prediction with shared pair-completion candidates and two backgrounds.

Node screening, candidate ranking, estimation and prediction use distinct
time intervals. All models condition on the available strict event history.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
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

from experiments.realdata_pipeline import _base_initial_state, select_fixed_subset
from experiments.synthetic_pipeline import (
    choose_start, load_events, package_version, parameter_record, plain,
    process_peak_bytes, read_json, replace_saved_file, restore_parameters,
    timestamp, write_json,
)
from inference.em import run_em
from inference.coordinate import refine_likelihood
from models.baseline import BaselineBasis, TemporalBaseline
from models.diagnostics import stability_diagnostics
from models.features import EventFeatures
from models.kernel import ExponentialKernel, HyperedgeAnchor

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ["screening", "C-P", "C-H2-top1", "C-H2-top2", "C-H2-top3",
               "V-P", "V-H2-top1", "V-H2-top2", "V-H2-top3"]
FINAL_START_STATES = {"qualified", "unqualified", "failed", "timeout", "worker_failed"}
CONTINUATION_MAX_SWEEPS = 2000


def validate_config(cfg):
    if (cfg["beta"] != 2 or cfg["delta"] != 0.05 or cfg["top_m"] != [1, 2, 3]
            or cfg["primary_top_m"] != 3 or cfg["backgrounds"] != ["constant", "linear"]
            or cfg["pairwise_diagonal_estimated"] is not True):
        raise ValueError("the prespecified pair-completion/background design has changed")
    if cfg["em"]["n_initialisations"] != 3 or cfg["em"]["lambda_l1"] != 0:
        raise ValueError("three starts and unpenalised fitting are required")
    if cfg["refinement"]["method"] != "coordinate_maximisation" or not cfg["refinement"]["apply_to_all_starts"]:
        raise ValueError("every start must receive the same coordinate refinement")
    if cfg["execution"]["workers"] != 1 or cfg["execution"]["blas_threads"] != 1:
        raise ValueError("use one worker and one BLAS thread")
    if cfg["execution"]["start_timeout_seconds"] <= 0:
        raise ValueError("a positive worker time limit is required")
    if len({s["name"] for s in cfg["datasets"]}) != len(cfg["datasets"]):
        raise ValueError("duplicate dataset name")
    for spec in cfg["datasets"]:
        if (len(spec["starts"]) != 5 or spec["duration"] <= 0 or spec["screen"][0] != 0
                or spec["screen"][1] > min(spec["starts"]) or spec["n_nodes"] < 2
                or spec["starts"] != sorted(set(spec["starts"]))):
            raise ValueError("invalid prespecified windows or independent node-screening span")


def make_plan(cfg, windows=1):
    validate_config(cfg)
    if windows not in (1, 5):
        raise ValueError("use the first fixed window or all five fixed windows")
    tasks = []
    for dataset_index, spec in enumerate(cfg["datasets"], 1):
        for window_index, start in enumerate(spec["starts"][:windows], 1):
            end = start + spec["duration"]
            first, second = start + spec["duration"]/3, start + 2*spec["duration"]/3
            seeds = {model: [int(np.random.SeedSequence([
                cfg["master_seed"], 30, dataset_index, window_index, model_index, i
            ]).generate_state(1, dtype=np.uint64)[0]) for i in range(3)]
                for model_index, model in enumerate(MODEL_NAMES)}
            tasks.append({"config": cfg, "dataset": spec, "dataset_id": spec["name"],
                          "window_index": window_index, "start": start, "end": end,
                          "path": spec["name"]+f"/window_{window_index:02d}",
                          "intervals": {"selection": [start, first], "estimation": [first, second],
                                        "test": [second, end]}, "start_seeds": seeds})
    return tasks


def all_pairwise_mask(n_nodes):
    return np.ones((n_nodes, n_nodes), dtype=bool)


def file_record(path):
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "modified_time_ns": stat.st_mtime_ns}


def read_recording(raw_root, spec):
    """Reuse the three original readers without invoking their legacy mains."""
    if spec["loader"] == "ret1":
        from experiments.real01_ret1 import load_ret1_record
        path, info, shape, orientation, trains, labels = load_ret1_record(
            raw_root, spec["file"], spec["record_index"])
        metadata = {"source_files": [file_record(path)], "spikes_shape": shape,
                    "orientation": orientation, "record_index": spec["record_index"],
                    "condition": "retinal recording with external binary-white-noise stimulation",
                    "time_unit": "seconds", "unit_evidence": "local ret-1 Analysis_example.m",
                    "description": info.get("description") if isinstance(info, dict) else None,
                    "label_key": "original_cell"}
    elif spec["loader"] == "pvc3":
        from experiments.real02_pvc3 import load_area17
        path, trains, labels = load_area17(raw_root)
        # The original reader uses uint64. Require agreement with the supplied
        # little-endian int64 format, rather than silently interpreting negatives.
        for train, name in zip(trains, labels):
            ticks = np.fromfile(path/name, dtype="<i8")
            if np.any(ticks < 0) or not np.array_equal(np.sort(ticks.astype(float)/1e6), train):
                raise ValueError("PVC-3 reader disagrees with the local signed microsecond format")
        info_path = path/"spk_info.inf"
        metadata = {"source_files": [file_record(path/name) for name in labels],
                    "condition": "area17 spontaneous recording", "time_unit": "seconds",
                    "source_time_unit": "microsecond ticks", "source_dtype": "little-endian int64",
                    "conversion": "ticks / 1000000", "label_key": "original_file",
                    "unit_evidence": info_path.read_text(errors="replace") if info_path.is_file() else None}
    elif spec["loader"] == "pvc11":
        from experiments.real03_pvc11 import load_pvc11_spont
        path, trains, labels = load_pvc11_spont(raw_root, spec["monkey"])
        metadata = {"source_files": [file_record(path)], "condition": "monkey2 spontaneous recording",
                    "time_unit": "seconds", "label_key": "original_index",
                    "unit_evidence": "local example_spontaneous_spikes.m and computeSpontCounts.m",
                    "additional_snr_filter": False, "original_reader_removes_empty_trains": True}
    else:
        raise ValueError("unknown dataset reader")
    checked = []
    for train in trains:
        a = np.asarray(train, dtype=float).reshape(-1)
        if not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError("nonfinite or negative spike timestamp")
        checked.append(np.sort(a))
    return checked, labels, metadata


def history_events(trains, selected, end):
    """Keep absolute times and every available event from recording time zero."""
    events = [(float(t), node) for node, source in enumerate(selected)
              for t in trains[source] if 0 <= t <= end]
    return sorted(events, key=lambda event: (event[0], event[1]))


def save_events(folder, events):
    path = folder/"events.npz"
    temporary = path.with_suffix(".npz.tmp")
    folder.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, times=np.asarray([t for t, _ in events], dtype=float),
                            nodes=np.asarray([n for _, n in events], dtype=np.int16))
    replace_saved_file(temporary, path)


def interval_counts(events, n_nodes, start, end):
    rows = [(t, n) for t, n in events if start < t <= end]
    counts = np.bincount([n for _, n in rows], minlength=n_nodes)
    _, multiplicities = np.unique([t for t, _ in rows], return_counts=True)
    return {"start": start, "end": end, "boundary_convention": "(start, end]",
            "n_events": len(rows), "node_counts": counts,
            "n_tied_times": int(np.sum(multiplicities > 1)),
            "n_records_at_tied_times": int(multiplicities[multiplicities > 1].sum()),
            "same_node_same_time_extra_records": len(rows)-len(set(rows))}


def prepare_recording(results_root, raw_root, spec):
    folder = results_root/spec["name"]
    path = folder/"recording.json"
    if path.is_file():
        saved = read_json(path)
        if saved["specification"] != spec:
            raise ValueError("saved recording specification differs; preserve it and use a new results root")
        if not (folder/"events.npz").is_file():
            raise FileNotFoundError("saved recording event file is missing")
        return saved
    trains, labels, metadata = read_recording(raw_root, spec)
    selected, mapping = select_fixed_subset(trains, spec["n_nodes"], *spec["screen"],
                                            labels, metadata["label_key"])
    required_end = max(spec["starts"]) + spec["duration"]
    last_spikes = [float(trains[i][-1]) if len(trains[i]) else None for i in selected]
    covers = all(t is not None and t >= required_end for t in last_spikes)
    events = history_events(trains, selected, required_end)
    record = {"specification": spec, "metadata": metadata, "mapping": mapping,
              "selected_source_indices": selected, "n_available_trains": len(trains),
              "required_observation_end": required_end, "selected_last_spike_times": last_spikes,
              "coverage_check_passed": covers,
              "coverage_interpretation": "selected trains have events beyond all fixed windows; assumes continuous acquisition, last spike is not a recording endpoint",
              "available_history_start": 0.0, "history_before_recording_start": "unobserved",
              "screening_boundary_convention": "[start, end)",
              "tie_handling": "retain original timestamps; all simultaneous records see the same strict past",
              "n_saved_events": len(events), "created_utc": datetime.now(timezone.utc).isoformat()}
    save_events(folder, events)
    write_json(path, record)
    return plain(record)


def prepare_window(results_root, task, recording):
    directory = results_root/task["path"]
    path = directory/"dataset.json"
    if path.is_file():
        saved = read_json(path)
        if saved["task"] != task:
            raise ValueError("saved window task differs")
        if not (directory/"events.npz").is_file():
            raise FileNotFoundError("saved window history is missing")
        return saved
    events = [(t, n) for t, n in load_events(results_root/task["dataset_id"]) if t <= task["end"]]
    save_events(directory, events)
    saved = {"task": task, "status": "prepared", "mapping": recording["mapping"],
             "all_available_history_preserved": True, "absolute_times": True,
             "n_history_events_before_window": sum(t <= task["start"] for t, _ in events),
             "intervals": {name: interval_counts(events, task["dataset"]["n_nodes"], *interval)
                           for name, interval in task["intervals"].items()},
             "package_version": package_version()}
    write_json(path, saved)
    return plain(saved)


def model_interval(task, model):
    return task["intervals"]["selection" if model == "screening" else "estimation"]


def model_basis(task, model):
    return BaselineBasis("linear" if model.startswith("V-") else "constant",
                         origin=task["start"], end=task["end"])


def rank_candidates(events, task, pair, mapping):
    """Rank using the qualified screening fit and selection-block events only."""
    cfg, n = task["config"], task["dataset"]["n_nodes"]
    start, end = task["intervals"]["selection"]
    history = [(t, v) for t, v in events if t <= end]
    edges = list(combinations(range(n), 2))
    f = EventFeatures(history, n, edges, cfg["beta"], cfg["delta"], start, end, include_start=False)
    counts = np.bincount(f.nodes, minlength=n)
    pair = np.asarray(pair, dtype=float)
    ranked = []
    for k, (a, b) in enumerate(edges):
        reasons = []
        if counts[a] == 0 or counts[b] == 0:
            reasons.append("member_without_selection_event")
        if f.hyper_exposure[k] <= 0:
            reasons.append("zero_selection_completion_exposure")
        ranked.append({"edge": [a, b], "member_mapping": [mapping[a], mapping[b]],
                       "ranking_weight": float((pair[a, b]+pair[b, a])/2),
                       "member_selection_counts": [int(counts[a]), int(counts[b])],
                       "selection_completions": int(f.completion_counts[k]),
                       "selection_exposure": float(f.hyper_exposure[k]),
                       "eligible": not reasons, "ineligible_reasons": reasons})
    ranked.sort(key=lambda row: (-row["ranking_weight"], row["edge"]))
    eligible = [r["edge"] for r in ranked if r["eligible"]]
    return {"status": "qualified", "selection_interval": [start, end], "ranked_pairs": ranked,
            "ranking_rule": "descending (Aij+Aji)/2, lexicographic node-index ties",
            "n_eligible": len(eligible), "shared_across_backgrounds": True,
            "prefixes": {str(m): {"edges": eligible[:m], "requested_count": m,
                                   "actual_count": min(m, len(eligible)),
                                   "shortfall_reason": "insufficient_eligible_pairs" if len(eligible) < m else None}
                         for m in cfg["top_m"]}}


def initial_parameters(events, task, model, edges, seed, index, warm_record=None):
    start, end = model_interval(task, model)
    basis, n = model_basis(task, model), task["dataset"]["n_nodes"]
    history = [(t, v) for t, v in events if t <= end]
    warm = None
    if warm_record is not None:
        base, pair, hyper = restore_parameters(warm_record)
        if base.basis != basis:
            if base.basis.kind != "constant":
                raise ValueError("only a constant fit can initialise a different temporal basis")
            base = TemporalBaseline(basis.constant_coefficients(base.coefficients[:, 0]), basis)
        warm = SimpleNamespace(mu=base.coefficients, alpha_pairwise=pair, alpha_hyper=hyper)
    mu, pair, hyper = _base_initial_state(history, n, start, end, edges, seed, warm)
    if warm is not None and index == 0:
        mu, pair = warm.mu.copy(), warm.alpha_pairwise.copy()
        hyper = {edge: max(warm.alpha_hyper.get(edge, 0.03), 1e-8) for edge in edges}
    mu = basis.constant_coefficients(mu) if np.asarray(mu).ndim == 1 else mu
    mu = np.maximum(mu, basis.lower_bounds(n))
    pair = np.maximum(pair, 1e-8)  # All N^2 coefficients, including self-excitation.
    hyper = {edge: max(weight, 1e-8) for edge, weight in hyper.items()}
    return parameter_record(TemporalBaseline(mu, basis), pair, hyper, edges)


def fitting_worker(path):
    record = read_json(path)
    task, model = record["task"], record["model"]
    cfg, n = task["config"], task["dataset"]["n_nodes"]
    start, end = model_interval(task, model)
    events = [(t, v) for t, v in load_events(record["dataset_directory"]) if t <= end]
    base, pair, hyper = restore_parameters(record["initial_parameters"])
    if "em_result" not in record:
        settings = dict(cfg["em"])
        settings.pop("n_initialisations")
        result = run_em(events=events, T=task["end"], n_nodes=n, edge_list=list(hyper),
                        kernel=ExponentialKernel(cfg["beta"]), anchor_calc=HyperedgeAnchor(cfg["delta"]),
                        mu0=base, alpha_pairwise0=pair, alpha_hyper0=hyper,
                        baseline_basis=base.basis, pairwise_mask=all_pairwise_mask(n),
                        hyper_update="free", positive_start=True, include_start=False,
                        target_start=start, target_end=end, track_loglik=True, assert_ascent=True, **settings)
        record["em_result"] = {"status": "qualified" if result.converged else "unqualified",
            "parameters": parameter_record(result.baseline, result.alpha_pairwise, result.alpha_hyper, list(hyper)),
            "objective": result.final_objective, "n_iter": result.n_iter,
            "converged": result.converged, "kkt_residual": result.kkt_residual,
            "absolute_increment": result.final_absolute_increment, "relative_increment": result.final_relative_increment,
            "ascent_violations": result.ascent_violations, "checkpoints": result.checkpoints,
            "objective_history": result.objective_history, "training_score": result.score_components,
            "identifiable_hyperedges": [result.identifiable_hyperedges[e] for e in hyper],
            "zero_pairwise_source_exposure": result.zero_pairwise_exposure,
            "stability": result.stability, "feature_bytes": result.feature_bytes,
            "preparation_seconds": result.preparation_seconds, "fit_seconds": result.elapsed_seconds}
        write_json(path, record)  # Preserve EM before any refinement or interruption.
    original = record["em_result"]
    base, pair, hyper = restore_parameters(original["parameters"])
    f = EventFeatures(events, n, list(hyper), cfg["beta"], cfg["delta"], start, end,
                      basis=base.basis, include_start=False)
    result = refine_likelihood(f, base.coefficients, pair, hyper, all_pairwise_mask(n),
        max_sweeps=cfg["refinement"]["max_sweeps"], lambda_l1=0.0,
        **{k: cfg["em"][k] for k in ("kkt_tol", "convergence_tol", "consecutive_convergence", "ascent_tol")})
    if abs(result["initial_objective"]-original["objective"]) > 1e-7*(1+abs(original["objective"])):
        raise AssertionError("saved EM endpoint does not reproduce its training objective")
    base = TemporalBaseline(result["baseline_coefficients"], base.basis)
    refinement = {k: v for k, v in result.items()
                  if k not in {"baseline_coefficients", "alpha_pairwise", "alpha_hyper", "training_score"}}
    record.update(status="qualified" if result["converged"] else "unqualified",
                  reasons=[] if result["converged"] else ["coordinate_stopping_criteria_not_met"],
                  final_parameters=parameter_record(base, result["alpha_pairwise"], result["alpha_hyper"], list(hyper)),
                  objective=result["final_objective"], training_score=result["training_score"],
                  kkt_residual=result["kkt_residual"], converged=result["converged"],
                  optimizer="em_then_coordinate_maximisation", refinement=refinement,
                  stability=stability_diagnostics(base, result["alpha_pairwise"], result["alpha_hyper"], cfg["beta"]))
    return record


def continue_refinement(path):
    """Continue the identical objective from a saved coordinate endpoint."""
    record = read_json(path)
    task, model = record["task"], record["model"]
    cfg, n = task["config"], task["dataset"]["n_nodes"]
    previous = record["refinement"]
    settings = record["continuation_settings"]
    if settings["max_total_sweeps"] != CONTINUATION_MAX_SWEEPS:
        raise ValueError("continuation must use the documented common total budget")
    remaining = settings["max_total_sweeps"]-previous["n_sweeps"]
    if remaining <= 0 or previous["converged"]:
        raise ValueError("only an unconverged endpoint with remaining budget may continue")
    base, pair, hyper = restore_parameters(record["final_parameters"])
    start, end = model_interval(task, model)
    events = [(t, v) for t, v in load_events(record["dataset_directory"]) if t <= end]
    features = EventFeatures(events, n, list(hyper), cfg["beta"], cfg["delta"], start, end,
                             basis=base.basis, include_start=False)
    saved_score = features.score(base.coefficients, pair, hyper)["loglik"]
    if abs(saved_score-record["objective"]) > 1e-7*(1+abs(record["objective"])):
        raise AssertionError("saved coordinate endpoint does not reproduce its training objective")
    result = refine_likelihood(features, base.coefficients, pair, hyper, all_pairwise_mask(n),
        max_sweeps=remaining, lambda_l1=0.0,
        **{key: cfg["em"][key] for key in ("kkt_tol", "convergence_tol", "consecutive_convergence", "ascent_tol")})
    if abs(result["initial_objective"]-record["objective"]) > 1e-7*(1+abs(record["objective"])):
        raise AssertionError("continuation did not start at the saved coordinate endpoint")
    new_base = TemporalBaseline(result["baseline_coefficients"], base.basis)
    combined = {key: value for key, value in result.items()
                if key not in {"baseline_coefficients", "alpha_pairwise", "alpha_hyper", "training_score"}}
    combined["initial_objective"] = previous["initial_objective"]
    for key in ("n_sweeps", "bound_hits", "bound_reentries", "elapsed_seconds"):
        combined[key] = previous[key]+result[key]
    for key in ("objective_history", "kkt_history", "relative_increment_history"):
        combined[key] = previous[key]+result[key]
    combined["continuation_segments"] = previous.get("continuation_segments", []) + [{
        "starting_sweep": previous["n_sweeps"], "additional_sweeps": result["n_sweeps"],
        "max_total_sweeps": settings["max_total_sweeps"],
        "initial_objective": result["initial_objective"], "final_objective": result["final_objective"],
        "converged": result["converged"], "kkt_residual": result["kkt_residual"],
        "consecutive_counter_restarted": True, "elapsed_seconds": result["elapsed_seconds"]}]
    record.update(status="qualified" if result["converged"] else "unqualified",
                  reasons=[] if result["converged"] else ["coordinate_stopping_criteria_not_met"],
                  final_parameters=parameter_record(new_base, result["alpha_pairwise"], result["alpha_hyper"], list(hyper)),
                  objective=result["final_objective"], training_score=result["training_score"],
                  kkt_residual=result["kkt_residual"], converged=result["converged"], refinement=combined,
                  stability=stability_diagnostics(new_base, result["alpha_pairwise"], result["alpha_hyper"], cfg["beta"]))
    return record


def worker_main(path):
    started = time.perf_counter()
    try:
        saved = read_json(path)
        record = continue_refinement(path) if saved.get("operation") == "continue_unconverged" else fitting_worker(path)
    except KeyboardInterrupt:
        record = read_json(path)
        record.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as exc:
        record = read_json(path)
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    elapsed = time.perf_counter()-started
    previous_seconds = record.get("continuation_current_attempt", {}).get("previous_worker_seconds", 0)
    peak = process_peak_bytes()
    peaks = [x for x in (record.get("peak_process_working_set_bytes"), peak) if x is not None]
    record.update(worker_seconds=previous_seconds+elapsed, last_operation_worker_seconds=elapsed,
                  peak_process_working_set_bytes=max(peaks) if peaks else None,
                  finished_utc=datetime.now(timezone.utc).isoformat(), package_version=package_version())
    write_json(path, record)
    return 0 if record["status"] in {"qualified", "unqualified"} else 1


def run_worker(path, timeout):
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    command = [sys.executable, "-X", "utf8", str(ROOT/"scripts/run_realdata_experiments.py"),
               "--worker", "--record", str(path)]
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
                elapsed = time.perf_counter()-started
                previous_seconds = record.get("continuation_current_attempt", {}).get("previous_wall_seconds", 0)
                record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "timeout",
                              timeout_seconds=timeout, wall_seconds=previous_seconds+elapsed,
                              last_operation_wall_seconds=elapsed)
                write_json(path, record)
                if isinstance(exc, KeyboardInterrupt):
                    raise
            record = read_json(path)
            if record["status"] == "running":
                unsaved = path.with_suffix(path.suffix+".tmp")
                if unsaved.is_file():
                    archived = path.parent/"archive"/timestamp()/path.name
                    archived.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(unsaved, archived)
                    record["unsaved_result_record"] = archived.relative_to(path.parent).as_posix()
                record.update(status="worker_failed", returncode=child.returncode)
            elapsed = time.perf_counter()-started
            previous_seconds = record.get("continuation_current_attempt", {}).get("previous_wall_seconds", 0)
            record.update(wall_seconds=previous_seconds+elapsed, last_operation_wall_seconds=elapsed)
            write_json(path, record)
            return record


def information_diagnostics(f, base, pair, hyper):
    """Observed curvature of the fixed-feature likelihood, not inference or selection."""
    intensity = f.intensity(base.coefficients, pair, hyper)
    columns = [f.background[:, q]*(f.nodes == node)
               for node in range(f.n_nodes) for q in range(f.basis.size)]
    columns += [f.pair[:, source]*(f.nodes == target)
                for source in range(f.n_nodes) for target in range(f.n_nodes)]
    columns += [f.hyper[:, k] for k in range(len(f.edges))]
    weighted = np.column_stack(columns)/intensity[:, None]
    singular = np.linalg.svd(weighted, compute_uv=False)
    tolerance = max(weighted.shape)*np.finfo(float).eps*singular.max(initial=0.0)
    rank = int(np.sum(singular > tolerance))
    dimension = len(columns)
    return {"n_structural_parameters": dimension, "observed_information_rank": rank,
            "rank_tolerance_on_weighted_design": tolerance,
            "weighted_design_singular_values": singular,
            "weighted_design_condition_number": float(singular[0]/singular[-1])
                if rank == dimension and len(singular) else None,
            "interpretation": "joint local curvature diagnostic; not proof of global identifiability, sampling uncertainty, or biological causation"}


def assess_model(events, task, model, parameters):
    base, pair, hyper = restore_parameters(parameters)
    cfg, n = task["config"], task["dataset"]["n_nodes"]
    start, end = model_interval(task, model)
    f = EventFeatures([(t, v) for t, v in events if t <= end], n, list(hyper), cfg["beta"], cfg["delta"],
                      start, end, basis=base.basis, include_start=False)
    train_intensity = f.intensity(base.coefficients, pair, hyper)
    output = {"training_score": f.score(base.coefficients, pair, hyper),
              "training_diagnostics": {"edges": list(hyper), "completion_counts": f.completion_counts,
                "completion_exposure": f.hyper_exposure,
                "identifiable_by_positive_exposure": f.hyper_exposure > 0,
                "zero_exposure_rule": "weight set to zero; retain candidate; exclude weight from parameter-inference claims",
                "pairwise_source_exposure": f.pair_exposure,
                "node_counts": np.bincount(f.nodes, minlength=n),
                "event_intensity_min": float(train_intensity.min()) if len(train_intensity) else None,
                "event_intensity_max": float(train_intensity.max()) if len(train_intensity) else None,
                "information": information_diagnostics(f, base, pair, hyper)},
              "stability": stability_diagnostics(base, pair, hyper, cfg["beta"])}
    if model == "screening":
        return output  # The selector never evaluates the test interval.
    start, end = task["intervals"]["test"]
    test = EventFeatures(events, n, list(hyper), cfg["beta"], cfg["delta"], start, end,
                         basis=base.basis, include_start=False)
    intensity = test.intensity(base.coefficients, pair, hyper)
    knots = [task["start"], *task["intervals"]["estimation"], task["end"]]
    output.update(test_score=test.score(base.coefficients, pair, hyper),
                  test_diagnostics={"event_intensity_min": float(intensity.min()) if len(intensity) else None,
                      "event_intensity_max": float(intensity.max()) if len(intensity) else None,
                      "completion_counts": test.completion_counts, "completion_exposure": test.hyper_exposure,
                      "node_counts": np.bincount(test.nodes, minlength=n)},
                  baseline_extrapolation={"absolute_times": knots, "rates_at_times": base.values(knots),
                      "fit_interval": task["intervals"]["estimation"],
                      "prediction_interval": task["intervals"]["test"],
                      "coefficients_frozen_before_test": True})
    return output


def warm_source(model):
    if model == "screening":
        return None
    if model == "C-P":
        return "screening"
    if model == "V-P":
        return "C-P"
    prefix, top = model[0], int(model[-1])
    return prefix+"-P" if top == 1 else prefix+f"-H2-top{top-1}"


def fit_model(directory, task, model, edges, events, fitted):
    final_path = directory/"models"/model/"fit.json"
    if final_path.is_file():
        return read_json(final_path)
    edges = [tuple(e) for e in edges]
    source = warm_source(model)
    warm = fitted.get(source, {}).get("parameters") if fitted.get(source, {}).get("status") == "qualified" else None
    starts = []
    for index, seed in enumerate(task["start_seeds"][model]):
        path = final_path.parent/f"start_{index+1}.json"
        current = read_json(path) if path.is_file() else {}
        if current.get("status") not in FINAL_START_STATES:
            if current:
                saved = path.parent/"archive"/timestamp()
                write_json(saved/path.name, current)
                if path.with_suffix(".log").is_file():
                    shutil.copy2(path.with_suffix(".log"), saved/path.with_suffix(".log").name)
            if "em_result" in current:
                current.update(status="running", resumed_from="saved_em_endpoint")
            else:
                initial = initial_parameters(events, task, model, edges, seed, index, warm)
                current = {"status": "running", "task": task, "model": model, "seed": seed,
                           "start_index": index+1, "dataset_directory": str(directory),
                           "initial_parameters": initial, "pairwise_diagonal_estimated": True,
                           "warm_source": source if warm is not None else None,
                           "warm_fallback_reason": "source_not_qualified" if source and warm is None else None,
                           "package_version": package_version()}
            write_json(path, current)
            print(f"  {model} / 起点 {index+1}/3", flush=True)
            current = run_worker(path, task["config"]["execution"]["start_timeout_seconds"])
            print(f"    {current['status']}，{current['wall_seconds']:.1f} 秒，"
                  f"KKT {current.get('kkt_residual', '—')}", flush=True)
        starts.append(current)
    return finalise_model(directory, task, model, edges, events, starts)


def finalise_model(directory, task, model, edges, events, starts):
    final_path = directory/"models"/model/"fit.json"
    selection = choose_start(starts)
    n = task["dataset"]["n_nodes"]
    output = {"model": model, **selection, "edges": edges,
              "structural_parameter_count": n*model_basis(task, model).size+n*n+len(edges),
              "pairwise_diagonal_estimated": True, "penalty": 0.0,
              "start_records": [f"start_{i+1}.json" for i in range(3)],
              "start_statuses": [r["status"] for r in starts],
              "total_start_wall_seconds": sum(r.get("wall_seconds", 0) for r in starts),
              "package_version": package_version()}
    if selection["selected_start"] is not None:
        best = starts[selection["selected_start"]-1]
        output.update(parameters=best["final_parameters"], kkt_residual=best["kkt_residual"])
        try:
            output.update(assess_model(events, task, model, best["final_parameters"]))
        except Exception as exc:
            output.update(status="scoring_or_diagnostic_failed", error=str(exc), traceback=traceback.format_exc())
    write_json(final_path, output)
    return plain(output)


def prediction_difference(larger, smaller):
    if larger.get("status") != "qualified" or smaller.get("status") != "qualified":
        return {"status": "missing", "reason": "both_models_must_qualify"}
    a, b = larger["test_score"], smaller["test_score"]
    if a["duration"] != b["duration"] or a["n_events"] != b["n_events"]:
        raise ValueError("paired scores must use identical intervals and test events")
    delta = a["loglik"]-b["loglik"]
    return {"status": "qualified", "delta_loglik": delta, "delta_per_time": delta/a["duration"],
            "delta_per_event": delta/a["n_events"] if a["n_events"] else None,
            "test_events": a["n_events"], "test_duration": a["duration"]}


def comparisons(fits):
    output = {}
    for m in (1, 2, 3):
        constant = prediction_difference(fits.get(f"C-H2-top{m}", {}), fits.get("C-P", {}))
        linear = prediction_difference(fits.get(f"V-H2-top{m}", {}), fits.get("V-P", {}))
        paired = {"status": "missing", "reason": "all_four_models_must_qualify"}
        if constant["status"] == linear["status"] == "qualified":
            if (constant["test_duration"], constant["test_events"]) != (linear["test_duration"], linear["test_events"]):
                raise ValueError("background comparisons must use identical test events")
            paired = {"status": "qualified", "delta_loglik": linear["delta_loglik"]-constant["delta_loglik"],
                      "delta_per_time": linear["delta_per_time"]-constant["delta_per_time"],
                      "delta_per_event": linear["delta_per_event"]-constant["delta_per_event"]
                          if constant["test_events"] else None}
        output[f"top{m}"] = {"primary": m == 3, "constant": constant, "linear": linear,
                              "linear_minus_constant_gain": paired}
    output["V-P_minus_C-P"] = prediction_difference(fits.get("V-P", {}), fits.get("C-P", {}))
    return output


def alias_model(source, model, source_name, prefix):
    result = dict(source)
    result.update(model=model, reused_from=source_name, candidate_prefix=prefix,
                  reuse_reason="identical actual candidate set under the same background and fit interval",
                  start_records=[], start_statuses=[], total_start_wall_seconds=0.0,
                  selected_start_belongs_to=source_name)
    return result


def run_window(results_root, task, recording):
    directory = results_root/task["path"]
    if not recording["coverage_check_passed"]:
        output = {"status": "coverage_failed", "task": task,
                  "reason": "cannot verify every selected train extends beyond all prespecified windows"}
        write_json(directory/"result.json", output)
        return output
    dataset = prepare_window(results_root, task, recording)
    events = load_events(directory)
    fits = {}
    fits["screening"] = fit_model(directory, task, "screening", [], events, fits)
    if fits["screening"]["status"] == "qualified":
        candidates = rank_candidates(events, task, fits["screening"]["parameters"]["alpha_pairwise"], dataset["mapping"])
    else:
        candidates = {"status": "screening_failed", "reason": fits["screening"]["status"]}
    write_json(directory/"candidates.json", candidates)
    for model in MODEL_NAMES[1:]:
        if model.endswith("-P"):
            fits[model] = fit_model(directory, task, model, [], events, fits)
            continue
        if candidates["status"] != "qualified":
            fits[model] = {"model": model, "status": "candidate_screening_failed", "start_statuses": []}
            write_json(directory/"models"/model/"fit.json", fits[model])
            continue
        prefix = candidates["prefixes"][model[-1]]
        edges = prefix["edges"]
        previous = next((name for name, row in fits.items() if name != "screening"
                         and name[0] == model[0] and row.get("edges") == edges), None)
        if previous is not None:
            fits[model] = alias_model(fits[previous], model, previous, prefix)
        else:
            fits[model] = fit_model(directory, task, model, edges, events, fits)
            fits[model]["candidate_prefix"] = prefix
        write_json(directory/"models"/model/"fit.json", fits[model])
    output = {"status": "qualified" if all(f["status"] == "qualified" for f in fits.values()) else "incomplete",
              "task": task, "candidates": candidates, "models": fits, "comparisons": comparisons(fits),
              "interpretation": "conditional one-step event-stream prediction using observed strict past; not an open-loop forecast or evidence of three-node interaction"}
    write_json(directory/"result.json", output)
    return plain(output)


def continuation_plan(results_root, plan):
    """Select work solely from saved training convergence and the common budget."""
    targets = []
    for task in plan:
        directory = results_root/task["path"]
        dataset_path = directory/"dataset.json"
        if not dataset_path.is_file() or read_json(dataset_path)["task"] != task:
            raise ValueError("continuation requires every original saved window with the same scientific configuration")
        if not (directory/"events.npz").is_file() or not (directory/"result.json").is_file():
            raise FileNotFoundError("continuation requires saved window history and results")
        for model in MODEL_NAMES:
            fit = read_json(directory/"models"/model/"fit.json")
            if fit.get("reused_from"):
                continue
            for index in range(1, 4):
                path = directory/"models"/model/f"start_{index}.json"
                record = read_json(path)
                if record["task"] != task or record["seed"] != task["start_seeds"][model][index-1]:
                    raise ValueError("saved initialisation/task differs from the original plan")
                refinement = record.get("refinement", {})
                eligible_status = record.get("status") == "unqualified" or (
                    record.get("operation") == "continue_unconverged"
                    and record.get("status") in {"running", "interrupted", "failed", "timeout", "worker_failed"})
                if (eligible_status and not refinement.get("converged", True)
                        and refinement.get("n_sweeps", CONTINUATION_MAX_SWEEPS) < CONTINUATION_MAX_SWEEPS):
                    targets.append({"window_path": task["path"], "model": model, "start_index": index,
                                    "completed_sweeps": refinement["n_sweeps"],
                                    "remaining_sweeps": CONTINUATION_MAX_SWEEPS-refinement["n_sweeps"],
                                    "saved_training_kkt": record["kkt_residual"]})
    return targets


def continue_window(results_root, task, targets):
    """Archive old records, update only eligible starts, then refresh comparisons."""
    directory = results_root/task["path"]
    models = list(dict.fromkeys(t["model"] for t in targets))
    # A worker may have saved its final endpoint just before an interruption
    # prevented model/window aggregation. Refresh that aggregate without fitting.
    for model in MODEL_NAMES:
        model_directory = directory/"models"/model
        fit = read_json(model_directory/"fit.json")
        if fit.get("reused_from"):
            continue
        starts = [read_json(model_directory/f"start_{i}.json") for i in range(1, 4)]
        selection = choose_start(starts)
        stale = any(fit.get(key) != value for key, value in selection.items())
        if selection.get("selected_start") is not None:
            stale |= fit.get("parameters") != starts[selection["selected_start"]-1]["final_parameters"]
        if stale and model not in models:
            models.append(model)
    if not models:
        # Rebuild only a stale window aggregate, e.g. after its last model saved.
        result = read_json(directory/"result.json")
        stored_fits = {name: read_json(directory/"models"/name/"fit.json") for name in MODEL_NAMES}
        if result.get("models") == stored_fits:
            return result
        saved = directory/"archive"/timestamp()
        saved.mkdir(parents=True, exist_ok=True)
        shutil.copy2(directory/"result.json", saved/"result.json")
        recording = read_json(results_root/task["dataset_id"]/"recording.json")
        return run_window(results_root, task, recording)
    saved = directory/"archive"/timestamp()
    saved.mkdir(parents=True, exist_ok=True)
    shutil.copy2(directory/"result.json", saved/"result.json")
    events = load_events(directory)
    for model in models:
        model_directory = directory/"models"/model
        old_fit = read_json(model_directory/"fit.json")
        backup = saved/"models"/model
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_directory/"fit.json", backup/"fit.json")
        for target in [t for t in targets if t["model"] == model]:
            index = target["start_index"]
            path = model_directory/f"start_{index}.json"
            current = read_json(path)
            shutil.copy2(path, backup/path.name)
            if path.with_suffix(".log").is_file():
                shutil.copy2(path.with_suffix(".log"), backup/path.with_suffix(".log").name)
            attempt = {"archive_record": str((backup/path.name).relative_to(directory)),
                       "previous_wall_seconds": current.get("wall_seconds", 0),
                       "previous_worker_seconds": current.get("worker_seconds", 0),
                       "starting_sweeps": current["refinement"]["n_sweeps"],
                       "starting_objective": current["objective"], "starting_kkt": current["kkt_residual"]}
            current.setdefault("continuation_attempts", []).append(attempt)
            current.update(status="running", operation="continue_unconverged",
                           continuation_settings={"max_total_sweeps": CONTINUATION_MAX_SWEEPS,
                               "original_max_sweeps": task["config"]["refinement"]["max_sweeps"],
                               "stopping_criteria_unchanged": True,
                               "selection_basis": "saved training convergence only; no test scores"},
                           continuation_current_attempt=attempt,
                           dataset_directory=str(directory), continuation_package_version=package_version())
            write_json(path, current)
            print(f"  续算 {model} / 起点 {index}；已完成 {attempt['starting_sweeps']} 轮", flush=True)
            current = run_worker(path, task["config"]["execution"]["start_timeout_seconds"])
            print(f"    {current['status']}；累计 {current['refinement']['n_sweeps']} 轮；"
                  f"KKT {current.get('kkt_residual', '—')}", flush=True)
        starts = [read_json(model_directory/f"start_{i}.json") for i in range(1, 4)]
        finalise_model(directory, task, model, [tuple(e) for e in old_fit["edges"]], events, starts)
    # All remaining fit.json files are reused. Their initialisations and fitted
    # parameters stay as actually run, even if a formerly unavailable warm fit
    # has now converged. Screening and candidate sets are not re-estimated.
    recording = read_json(results_root/task["dataset_id"]/"recording.json")
    return run_window(results_root, task, recording)


def descriptive(values):
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return {"n_available": len(array), "mean": float(array.mean()) if len(array) else None,
            "median": float(np.median(array)) if len(array) else None,
            "min": float(array.min()) if len(array) else None,
            "max": float(array.max()) if len(array) else None}


def summarise(results_root, plan):
    rows, fit_statuses, start_statuses, seconds, memory = [], {}, {}, [], []
    for task in plan:
        directory = results_root/task["path"]
        path = directory/"result.json"
        result = read_json(path) if path.is_file() else {"status": "not_completed", "comparisons": {}}
        rows.append({"dataset": task["dataset_id"], "window_index": task["window_index"],
                     "intervals": task["intervals"], "status": result["status"],
                     "comparisons": result.get("comparisons", {}), "error": result.get("error")})
        for model in MODEL_NAMES:
            final_path = directory/"models"/model/"fit.json"
            fitted = read_json(final_path) if final_path.is_file() else {"status": "not_completed"}
            fit_statuses[fitted["status"]] = fit_statuses.get(fitted["status"], 0)+1
            for index in range(1, 4):
                start_path = final_path.parent/f"start_{index}.json"
                if not start_path.is_file():
                    continue
                record = read_json(start_path)
                status = record["status"]
                start_statuses[status] = start_statuses.get(status, 0)+1
                seconds.append(record.get("wall_seconds", 0))
                if record.get("peak_process_working_set_bytes") is not None:
                    memory.append(record["peak_process_working_set_bytes"])
    aggregates = []
    for dataset in dict.fromkeys(t["dataset_id"] for t in plan):
        subset = [r for r in rows if r["dataset"] == dataset]
        for m in (1, 2, 3):
            for comparison in ("constant", "linear", "linear_minus_constant_gain"):
                available = [r["comparisons"].get(f"top{m}", {}).get(comparison, {}) for r in subset]
                aggregates.append({"dataset": dataset, "top_m": m, "comparison": comparison,
                    "primary": m == 3, "n_planned_windows": len(subset),
                    **{metric: descriptive([x.get(metric) for x in available if x.get("status") == "qualified"])
                       for metric in ("delta_loglik", "delta_per_time", "delta_per_event")}})
    return {"planned_windows": len(plan), "qualified_windows": sum(r["status"] == "qualified" for r in rows),
            "model_status_counts": fit_statuses, "actual_start_status_counts": start_statuses,
            "actual_fit_seconds": sum(seconds), "start_seconds": descriptive(seconds),
            "max_worker_peak_bytes": max(memory) if memory else None,
            "rows": rows, "descriptive_aggregates": aggregates,
            "interpretation": "windows from one recording are dependent descriptive observations, not independent biological or Monte Carlo replicates; no MCSE or significance test",
            "primary_comparison": "top3; full unpenalised held-out log likelihood difference",
            "background_scope": "constant versus one prespecified linear trend; no claim to control all external stimuli or fast nonstationarity"}


def pilot_readiness(results_root, cfg):
    problems = []
    for task in make_plan(cfg, 1):
        directory = results_root/task["path"]
        path = directory/"result.json"
        if not path.is_file():
            problems.append(task["path"]+": first window not completed")
            continue
        result = read_json(path)
        if result.get("task") != task or result.get("status") != "qualified":
            problems.append(task["path"]+": first window not qualified under this configuration")
        for model in MODEL_NAMES:
            fit = result.get("models", {}).get(model, {})
            if fit.get("status") != "qualified":
                problems.append(task["path"]+"/"+model+": model not qualified")
            if not fit.get("reused_from") and fit.get("start_statuses") != ["qualified"]*3:
                problems.append(task["path"]+"/"+model+": not all starts qualified")
    return {"ready": not problems, "reasons": problems,
            "interpretation": "numerical readiness only; review the first-window diagnostics and timing before extending"}


def snapshot_code(destination, cfg):
    if destination.exists():
        return
    files = []
    for folder in ("models", "inference", "simulation"):
        files += list((ROOT/folder).glob("*.py"))
    files += [ROOT/"experiments"/name for name in (
        "realdata_pipeline.py", "synthetic_pipeline.py", "realdata_baseline.py",
        "real01_ret1.py", "real02_pvc3.py", "real03_pvc11.py")]
    files += [ROOT/name for name in ("scripts/run_realdata_experiments.py", "scripts/run_model_checks.py",
        "tests/test_realdata_experiments.py", "docs/realdata_experiments.md", "docs/code_manifest.json", "requirements.txt")]
    for path in files:
        if path.is_file():
            target = destination/path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    write_json(destination/"configs/realdata_experiments.json", cfg)


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
        paths += list(directory.rglob("*")) if directory.exists() else []
        paths += [results_root/task["dataset_id"]/name for name in ("recording.json", "events.npz", "error.json")]
    temporary = package.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            if path.is_file() and path.suffix != ".tmp":
                archive.write(path, "experiments/results/realdata/baseline/"+path.relative_to(results_root).as_posix())
    replace_saved_file(temporary, package)
    return package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, choices=(1, 5), default=1)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/realdata_experiments.json")
    parser.add_argument("--raw-root", type=Path, default=ROOT/"data/raw")
    parser.add_argument("--results-root", type=Path, default=ROOT/"experiments/results/realdata/baseline")
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--continue-unconverged", action="store_true",
                        help="Continue saved unconverged coordinate endpoints to 2000 total sweeps; never rerun EM or qualified starts.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--record", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.worker:
        if options.record is None:
            parser.error("worker record required")
        return worker_main(options.record)
    if options.continue_unconverged and options.windows != 5:
        parser.error("continuation reviews the complete five-window study; pass --windows 5")
    cfg = read_json(options.config)
    plan = make_plan(cfg, options.windows)
    counts = {"windows": len(plan), "planned_models": len(plan)*len(MODEL_NAMES),
              "planned_starts": len(plan)*len(MODEL_NAMES)*3}
    if options.plan_only:
        preview = {**counts, "scientific_fits_executed": False,
                   "note": "identical actual candidate sets reuse one fit under the same background"}
        if options.continue_unconverged:
            preview["continuation_targets"] = continuation_plan(options.results_root.resolve(), plan)
            preview["starts_to_continue"] = len(preview["continuation_targets"])
            preview["max_total_coordinate_sweeps"] = CONTINUATION_MAX_SWEEPS
        print(json.dumps(preview, indent=2))
        return 0
    from scripts.run_model_checks import downloads_directory
    results_root = options.results_root.resolve()
    destination = (options.download_dir or downloads_directory()).resolve()
    config_path = results_root/"config.json"
    if config_path.is_file() and read_json(config_path) != cfg:
        parser.error("saved study configuration differs; preserve it and use a separate --results-root")
    targets = continuation_plan(results_root, plan) if options.continue_unconverged else []
    readiness = pilot_readiness(results_root, cfg) if options.windows == 5 else None
    if readiness is not None and not readiness["ready"]:
        parser.error("complete and review the first fixed window of every dataset before all five: "+"; ".join(readiness["reasons"][:3]))
    write_json(config_path, cfg)
    run_dir = results_root/"realdata_experiments"/("pilot" if options.windows == 1 else "full")
    write_json(run_dir/"config.json", cfg)
    write_json(run_dir/"plan.json", {**counts, "tasks": plan})
    if options.continue_unconverged:
        amendment = {"scope": "all saved unqualified coordinate endpoints, regardless of dataset or model",
                     "original_max_sweeps": cfg["refinement"]["max_sweeps"],
                     "max_total_sweeps": CONTINUATION_MAX_SWEEPS,
                     "scientific_configuration_unchanged": True, "stopping_criteria_unchanged": True,
                     "reason": "original coordinate budget exhausted while training KKT still exceeds the fixed threshold",
                     "targets_at_first_continuation": targets}
        if not (run_dir/"execution_amendment.json").exists():
            write_json(run_dir/"execution_amendment.json", amendment)
        print(f"本次从保存终点续算 {len(targets)} 个起点；已合格结果直接复用。", flush=True)
    if readiness is not None:
        write_json(run_dir/"pilot_readiness.json", readiness)
    snapshot = run_dir/"code"/package_version()
    snapshot_code(snapshot, cfg)
    import importlib.metadata
    write_json(run_dir/"environment.json", {"python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "numpy": np.__version__, "scipy": importlib.metadata.version("scipy"),
        "workers": 1, "blas_threads": 1, "package_version": package_version(), "code_snapshot": str(snapshot),
        "raw_root": str(options.raw_root.resolve()), "rng": type(np.random.default_rng(0).bit_generator).__name__})
    started = time.perf_counter()
    invocation = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", **counts,
                  "mode": "continue_unconverged" if options.continue_unconverged else "initial_or_resume",
                  "continuation_targets": targets}
    recordings = {}
    return_code = 0
    try:
        for index, task in enumerate(plan, 1):
            print(f"\n[{index}/{len(plan)}] {task['dataset_id']} / 窗口 {task['window_index']}", flush=True)
            try:
                name = task["dataset_id"]
                if options.continue_unconverged:
                    continue_window(results_root, task, [t for t in targets if t["window_path"] == task["path"]])
                else:
                    if name not in recordings:
                        recordings[name] = prepare_recording(results_root, options.raw_root.resolve(), task["dataset"])
                    run_window(results_root, task, recordings[name])
            except Exception as exc:
                failure = {"status": "failed", "task": task, "error": str(exc), "traceback": traceback.format_exc()}
                write_json(results_root/task["path"]/"result.json", failure)
                print("  此窗口未完成，保留错误后继续下一预定窗口："+str(exc), flush=True)
            write_json(run_dir/"summary.json", summarise(results_root, plan))
        invocation["status"] = "completed"
    except KeyboardInterrupt:
        invocation["status"] = "interrupted"
        return_code = 130
    except Exception as exc:
        invocation.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        return_code = 1
    finally:
        invocation.update(elapsed_seconds=time.perf_counter()-started,
                          finished_utc=datetime.now(timezone.utc).isoformat())
        try:
            summary = summarise(results_root, plan)
            write_json(run_dir/"summary.json", summary)
            incomplete = (summary["qualified_windows"] != len(plan)
                          or any(k != "qualified" and v for k, v in summary["actual_start_status_counts"].items()))
            if incomplete and return_code == 0:
                invocation["status"] = "completed_with_issues"
                return_code = 2
            print(f"\n窗口合格：{summary['qualified_windows']}/{len(plan)}；"
                  f"起点状态：{summary['actual_start_status_counts']}", flush=True)
        finally:
            write_json(run_dir/"runs"/(timestamp()+".json"), invocation)
            print("保存项目结果并打包；下载目录同名旧结果包会先归档。", flush=True)
            try:
                package = package_results(results_root, plan, run_dir, destination)
                print("结果包："+str(package), flush=True)
            except Exception as exc:
                write_json(run_dir/"packaging_error.json", {"error": str(exc), "traceback": traceback.format_exc()})
                print("打包未完成，项目结果仍已保存："+str(exc), flush=True)
                return_code = 1
    return return_code
