"""Check the packaged saved records without simulation, fitting, or raw data.

The checks compare original fit records, derived start summaries, and saved
paper inputs. They do not recompute event-level likelihoods or gradients.
By default the report is printed as JSON and no file is written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np


class Verification:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.checks = 0
        self.failure_count = 0
        self.failures = []
        self.results = {}
        self.files_read = set()

    def check(self, condition, location, message):
        self.checks += 1
        if not condition:
            self.failure_count += 1
            if len(self.failures) < 40:
                self.failures.append({"location": str(location), "message": message})

    def path(self, relative):
        relative = str(relative)
        p = (self.root / relative).resolve()
        if Path(relative).is_absolute() or not p.is_relative_to(self.root):
            raise ValueError("Record path must stay within the package: " + relative)
        return p

    def read(self, relative):
        p = self.path(relative)
        self.files_read.add(str(relative))
        return json.loads(p.read_text(encoding="utf-8-sig"))

    def lines(self, relative):
        p = self.path(relative)
        self.files_read.add(str(relative))
        with p.open(encoding="utf-8-sig") as f:
            return [json.loads(line) for line in f if line.strip()]

    def section(self, name, method):
        before = self.failure_count
        try:
            detail = method()
        except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration,
                ArithmeticError, np.linalg.LinAlgError) as exc:
            self.check(False, name, f"{type(exc).__name__}: {exc}")
            detail = {}
        self.results[name] = {"passed": self.failure_count == before, **detail}

    @staticmethod
    def close(a, b, atol=1e-10, rtol=1e-12):
        return (isinstance(a, (int, float)) and isinstance(b, (int, float))
                and math.isfinite(a) and math.isfinite(b)
                and math.isclose(a, b, abs_tol=atol, rel_tol=rtol))

    def start_summaries(self, name, expected, study, role):
        rows = self.lines("records/derived/" + name + ".jsonl")
        self.check(len(rows) == expected, name, f"Expected {expected} rows, found {len(rows)}")
        groups = defaultdict(list)
        seen = set()
        for row in rows:
            key = (row["dataset_id"], row["model"], row["start_index"])
            location = "/".join(map(str, key))
            self.check(key not in seen, location, "Duplicate derived start summary")
            seen.add(key)
            self.check(row["schema_version"] == 1 and
                       row["record_kind"] == "derived_start_endpoint_summary" and
                       row["study"] == study and row["record_role"] == role,
                       location, "Unexpected derived-record schema or role")
            source = row["original_record"]
            self.check(bool(source["path"]) and bool(source["source_archive"]) and
                       source["included_in_package"] is False,
                       location, "Invalid provenance for the omitted original start record")
            h = row["history"]
            self.check(h["em_length"] == row["em_iterations"] and
                       h["coordinate_length"] == row["coordinate_sweeps"] and
                       h["combined_length"] == h["em_length"] + h["coordinate_length"],
                       location, "Recorded history lengths disagree with iteration counts")
            for field in ("training_objective", "training_loglik", "kkt_residual"):
                self.check(isinstance(row[field], (int, float)) and math.isfinite(row[field]),
                           location, "Nonfinite endpoint field: " + field)
            self.check(row["kkt_residual"] >= 0, location, "Negative KKT residual")
            for field in ("em_min_adjacent_increment", "coordinate_min_increment_including_initial"):
                value = h[field]
                self.check(value is None or (math.isfinite(value) and value >= -1e-6),
                           location, "Recorded objective decrease exceeds tolerance: " + field)
            for field in ("em_to_coordinate_objective_gap", "final_objective_history_gap"):
                value = h[field]
                self.check(value is None or self.close(value, 0, atol=1e-7),
                           location, "Inconsistent objective history join or endpoint: " + field)
            self.check(self.close(row["training_objective"] - row["training_loglik"],
                                  h["objective_minus_training_loglik"], atol=1e-7),
                       location, "Objective/log-likelihood summary mismatch")
            if role == "current":
                tail = row["last_five_relative_increments"]
                self.check(row["final_status"] == "qualified" and row["final_converged"] is True,
                           location, "Current start is not marked qualified and converged")
                self.check(row["kkt_residual"] <= 1e-4, location, "Final KKT exceeds 1e-4")
                self.check(len(tail) == 5 and all(math.isfinite(x) and 0 <= x < 1e-7 for x in tail),
                           location, "Final five increments do not meet the recorded stopping rule")
                self.check(bool(tail) and row["final_relative_increment"] == tail[-1],
                           location, "Final relative increment disagrees with the recorded tail")
            groups[key[:2]].append(row)
        return rows, groups

    def selected_fit(self, fit, starts, location, seeds):
        starts = sorted(starts, key=lambda s: s["start_index"])
        self.check([s["start_index"] for s in starts] == [1, 2, 3], location,
                   "Expected exactly three distinct start summaries")
        if len(starts) != 3:
            return
        self.check([str(s["seed"]) for s in starts] == [str(s) for s in seeds],
                   location, "Start seeds differ from the original plan")
        self.check(fit["status"] == "qualified" and fit["n_qualified_starts"] == 3 and
                   fit["start_statuses"] == ["qualified"] * 3,
                   location, "Selected-fit qualification fields disagree")
        best = max(starts, key=lambda s: s["training_objective"])
        self.check(fit["selected_start"] == best["start_index"], location,
                   "Selected start is not the maximum qualified training objective")
        marked = [s for s in starts if s["selected_for_current_fit"]]
        self.check(len(marked) == 1 and marked[0]["start_index"] == fit["selected_start"],
                   location, "Derived selected-start marker disagrees with the original fit")
        selected = next((s for s in starts if s["start_index"] == fit["selected_start"]), None)
        if selected:
            self.check(self.close(selected["training_loglik"], fit["training_score"]["loglik"]),
                       location, "Selected start training score differs from the fit")
            self.check(selected["kkt_residual"] == fit["kkt_residual"],
                       location, "Selected start KKT differs from the fit")

    def synthetic(self):
        cfg = self.read("records/synthetic/study/formal/config.json")
        plan = self.read("records/synthetic/study/formal/plan.json")["tasks"]
        summary = self.read("records/synthetic/study/formal/summary.json")
        self.check(cfg == self.read("configs/synthetic_experiments.json"), "synthetic/config", "Scientific configurations differ")
        self.check(len(plan) == 1100 and len({t["dataset_id"] for t in plan}) == 1100,
                   "synthetic/plan", "Expected 1100 distinct planned paths")
        starts, groups = self.start_summaries("synthetic_starts", 12600, "synthetic", "current")
        paper_rows = self.read("paper/data/synthetic_replicates.json")
        paper = {(r["dataset_id"], r["model"]): r for r in paper_rows}
        self.check(len(paper_rows) == len(paper) == 4200, "synthetic/paper", "Expected 4200 distinct paper rows")
        expected_keys = set()
        fit_count = 0
        exact_direct_rows = 0
        max_rederived_difference = 0.0
        n = cfg["n_nodes"]
        true_pair = np.zeros((n, n))
        true_pairs = {}
        for i in range(n):
            true_pair[i, (i + 1) % n] += cfg["ring_weight"]
        for edge in cfg["true_triples"]:
            for i in edge:
                for j in edge:
                    if i != j:
                        true_pair[i, j] += cfg["within_group_weight"]
            for e in combinations(edge, 2):
                true_pairs[tuple(e)] = cfg["pair_completion_weight"]
        for task in plan:
            location = task["path"]
            spec = task["scenario"]
            self.check(task["config"] == cfg and task["phase"] == "formal", location,
                       "Task uses a different configuration or is not a formal repetition")
            prefix = "records/synthetic/" + location
            dataset = self.read(prefix + "/dataset.json")
            self.check(dataset["status"] == "completed" and dataset["task"] == task,
                       location, "Dataset completion or plan metadata mismatch")
            fits = {}
            for model in spec["models"]:
                key = (task["dataset_id"], model)
                expected_keys.add(key)
                fit = self.read(prefix + "/models/" + model + "/fit.json")
                fits[model] = fit
                fit_count += 1
                self.check(fit["model"] == model and fit["dataset_id"] == task["dataset_id"],
                           location + "/" + model, "Fit identity mismatch")
                self.selected_fit(fit, groups.get(key, []), location + "/" + model,
                                  task["start_seeds"][model])
                p = fit["parameters"]
                hyper = {tuple(e): v for e, v in zip(p["edges"], p["alpha_hyper"])}
                row = dict(dataset_id=task["dataset_id"], experiment=spec["experiment"],
                           scenario=spec["name"], repeat=task["repeat"], n=spec["formal_repetitions"],
                           T=spec["T"], truth=spec["triple_weight"], background=spec["background"],
                           model=model, test_loglik=fit["test_score"]["loglik"],
                           test_duration=fit["test_score"]["duration"],
                           baseline_train_mse=fit["baseline_errors"]["training"]["mse_mean"],
                           baseline_test_mse=fit["baseline_errors"]["test"]["mse_mean"],
                           triple_estimates=[hyper.get(tuple(e)) for e in cfg["true_triples"]],
                           decoy_estimates=[hyper.get(tuple(e)) for e in cfg["decoy_triples"]])
                original = paper[key]
                direct_match = all(original[k] == v for k, v in row.items())
                self.check(direct_match, location + "/" + model, "Original stored fit fields differ from paper row")
                exact_direct_rows += int(direct_match)
                pairs = [v for e, v in hyper.items() if len(e) == 2]
                derived = {
                    "pairwise_mse": float(np.mean((np.asarray(p["alpha_pairwise"]) - true_pair)[~np.eye(n, dtype=bool)] ** 2)),
                    "triple_sum": sum(v for e, v in hyper.items() if len(e) == 3) if model.endswith("M2") else None,
                    "pair_completion_mse": float(np.mean([(v - true_pairs.get(e, 0)) ** 2 for e, v in hyper.items() if len(e) == 2])) if pairs else None,
                }
                for k, v in derived.items():
                    ok = original[k] is None if v is None else self.close(original[k], v, atol=1e-12)
                    self.check(ok, location + "/" + model, "Reaggregated parameter statistic differs: " + k)
                    if v is not None and math.isfinite(v) and math.isfinite(original[k]):
                        max_rederived_difference = max(max_rederived_difference, abs(original[k] - v))
            comparisons = self.read(prefix + "/comparisons.json")
            rebuilt = {}
            for larger, smaller in (("C-M2", "C-M1"), ("V-M2", "V-M1"), ("C-M1", "M0"), ("C-M2", "M0")):
                if larger in fits and smaller in fits:
                    a, b = fits[larger]["test_score"], fits[smaller]["test_score"]
                    d = a["loglik"] - b["loglik"]
                    self.check(a["duration"] == b["duration"] and a["n_events"] == b["n_events"], location, "Unpaired prediction records")
                    rebuilt[larger + "_minus_" + smaller] = dict(status="qualified", delta_loglik=d,
                        delta_per_time=d/a["duration"], delta_per_event=d/a["n_events"] if a["n_events"] else None,
                        test_events=a["n_events"], test_duration=a["duration"])
            self.check(comparisons == rebuilt, location, "Saved model comparisons do not match fitted test scores")
        self.check(fit_count == 4200 and set(groups) == expected_keys == set(paper), "synthetic", "Fit/start/paper identity coverage differs")
        self.check(summary["planned_datasets"] == 1100 and summary["planned_models"] == 4200 and
                   summary["start_statuses"] == {"qualified": 12600} and
                   not summary["not_completed_models"] and not summary["groups_below_90_percent"],
                   "synthetic/summary", "Saved study counts or qualification disagree")
        self.check(sum(v["qualified"] for v in summary["models"].values()) == 4200 and
                   sum(v["completed"] for v in summary["generation"].values()) == 1100,
                   "synthetic/summary", "Saved grouped model/dataset counts disagree")
        return {"datasets": len(plan), "fits": fit_count, "start_summaries": len(starts),
                "paper_rows_direct_fields_exact": exact_direct_rows,
                "maximum_reaggregated_parameter_statistic_difference": max_rederived_difference}

    def realdata(self):
        current, groups = self.start_summaries("realdata_starts", 405, "realdata", "current")
        archived, archived_groups = self.start_summaries("realdata_archived_starts", 9, "realdata", "archived_precontinuation")
        windows = self.read("paper/data/realdata_fits.json")
        plan = self.read("records/realdata/realdata_experiments/full/plan.json")["tasks"]
        planned = {t["path"]: t for t in plan}
        self.check(len(windows) == len(plan) == 15, "realdata", "Expected 15 windows")
        keys = set()
        count = 0
        for window in windows:
            task = window["task"]
            path = task["path"]
            self.check(task == planned[path] and window["status"] == "qualified", path, "Neural window plan/status mismatch")
            dataset = self.read("records/realdata/" + path + "/dataset.json")
            self.check(dataset["task"] == task, path, "Neural dataset metadata differs from plan")
            for model, fit in window["models"].items():
                key = (path, model)
                keys.add(key)
                self.selected_fit(fit, groups.get(key, []), path + "/" + model, task["start_seeds"][model])
                count += 1
        self.check(count == 135 and set(groups) == keys, "realdata", "Expected 135 selected models and their start summaries")
        for key, previous in archived_groups.items():
            now = {s["start_index"]: s for s in groups.get(key, [])}
            for old in previous:
                index = old["start_index"]
                self.check(index in now and old["seed"] == now[index]["seed"], str(key), "Archived/current start seed linkage mismatch")
                self.check(not old["selected_for_current_fit"], str(key), "Archived start must not be marked as a current selected start")
        return {"windows": len(windows), "selected_models": count, "current_start_summaries": len(current),
                "archived_start_summaries": len(archived)}

    def benchmark(self):
        prefix = "records/benchmark/"
        cfg = self.read(prefix + "config.json")
        plan = self.read(prefix + "plan.json")
        summary = self.read(prefix + "summary.json")
        paper = self.read("paper/data/timing_analysis.json")
        self.check(len(plan) == 9 and cfg["repetitions"] == 10, "benchmark", "Expected nine settings and ten repetitions")
        self.check(summary == self.read("paper/data/timing_summary.json"), "benchmark/summary", "Original and paper timing summaries differ")
        rebuilt = []
        measured = warmups = qualified = raw_em = 0
        for task in plan:
            dataset = self.read(prefix + task["path"] + "/dataset.json")
            self.check(dataset["task"] == task and dataset["config"] == cfg, task["path"], "Benchmark dataset metadata mismatch")
            for phase in ("features", "iteration", "fit"):
                rel = task["path"] + "/" + phase
                request = self.read(prefix + rel + "/request.json")
                self.check(request == {"task": task, "config": cfg, "phase": phase}, rel, "Benchmark phase request mismatch")
                records = [self.read(prefix + rel + f"/repeat_{i:02d}.json") for i in range(1, 11)]
                wp = list(self.path(prefix + rel + "/warmups").glob("*.json"))
                self.check(len(wp) == 1, rel, "Expected one excluded warmup")
                warmups += len(wp)
                for p in wp:
                    w = self.read(p.relative_to(self.root))
                    self.check(w["excluded_from_timing_summary"] is True and w["status"] == "completed", rel, "Warmup exclusion or completion mismatch")
                for i, r in enumerate(records, 1):
                    measured += 1
                    self.check(r["task"] == task and r["phase"] == phase and r["repetition"] == i and r["n_events"] == dataset["n_events"], rel, "Benchmark record identity mismatch")
                    self.check(r["status"] == "completed" and math.isfinite(r["seconds"]) and r["seconds"] > 0, rel, "Invalid benchmark timing")
                    if phase == "fit":
                        qualified += int(r["fit_qualified"])
                        raw_em += int(r["em_converged"])
                        self.check(r["fit_qualified"] and 0 <= r["final_kkt"] <= cfg["em"]["kkt_tol"] and
                                   r["minimum_objective_increment"] >= -cfg["em"]["ascent_tol"] and
                                   len(r["last_relative_increments"]) == 5 and
                                   all(0 <= x < cfg["em"]["convergence_tol"] for x in r["last_relative_increments"]),
                                   rel, "Benchmark fit endpoint diagnostics fail recorded thresholds")
                values = [r["seconds"] for r in records]
                row = dict(**task, phase=phase, n_events=dataset["n_events"], n_measured=10,
                           median_seconds=float(np.median(values)), q25_seconds=float(np.quantile(values, .25)),
                           q75_seconds=float(np.quantile(values, .75)), min_seconds=min(values), max_seconds=max(values),
                           planned_repetitions=10, qualified_fits=10 if phase == "fit" else None,
                           max_process_peak_bytes=max(r["peak_process_working_set_bytes"] for r in records),
                           status_counts={"completed": 10}, source_records=[rel+f"/repeat_{i:02d}.json" for i in range(1, 11)])
                target = next(r for r in summary["rows"] if r["path"] == task["path"] and r["phase"] == phase)
                self.check(row == target, rel, "Benchmark timing summary does not reaggregate exactly")
                target = next(r for r in paper["rows"] if r["path"] == task["path"] and r["phase"] == phase)
                with_ratio = {**row, "iqr_over_median": (row["q75_seconds"]-row["q25_seconds"])/row["median_seconds"]}
                self.check(target == with_ratio, rel, "Paper timing row differs from the recorded timings")
                rebuilt.append(row)
        max_slope_difference = 0.0
        for phase in ("features", "iteration", "fit"):
            use = sorted([r for r in rebuilt if r["sweep"] == "event_count" and r["phase"] == phase], key=lambda r:r["n_events"])
            x = np.log([r["n_events"] for r in use]); y = np.log([r["median_seconds"] for r in use])
            slope, intercept = np.polyfit(x, y, 1)
            expected = dict(main_slope=float(slope), last_three_sensitivity_slope=float(np.polyfit(x[-3:], y[-3:], 1)[0]),
                            r_squared_log_scale=1-float(np.sum((y-(slope*x+intercept))**2)/np.sum((y-y.mean())**2)))
            for key, value in expected.items():
                stored = paper["slopes"][phase][key]
                self.check(self.close(stored, value, atol=1e-12), phase, "Timing slope or R-squared mismatch: " + key)
                if math.isfinite(stored) and math.isfinite(value):
                    max_slope_difference = max(max_slope_difference, abs(stored-value))
        self.check((measured, warmups, qualified, raw_em) == (270, 27, 90, 80), "benchmark", "Benchmark count mismatch")
        return {"settings": len(plan), "measurements": measured, "excluded_warmups": warmups,
                "qualified_fits": qualified, "raw_em_converged": raw_em,
                "maximum_reaggregated_slope_difference": max_slope_difference}

    def provenance(self):
        manifest = self.read("records/manifest.json")
        expected_counts = {"synthetic_comparisons": 1100, "original_record_copies": 6793,
                           "synthetic_datasets": 1100, "synthetic_fits": 4200,
                           "synthetic_formal_metadata": 6, "benchmark_json_records": 340,
                           "realdata_root_config": 1, "realdata_recordings": 3,
                           "realdata_candidate_records": 15, "realdata_windows": 15,
                           "realdata_study_metadata": 13, "synthetic_starts": 12600,
                           "realdata_starts": 405, "realdata_archived_starts": 9}
        self.check(manifest["schema_version"] == 1 and manifest["counts"] == expected_counts,
                   "records/manifest.json", "Manifest record counts differ from the fixed study archive")
        expected_derived = {"records/derived/synthetic_starts.jsonl": 12600,
                            "records/derived/realdata_starts.jsonl": 405,
                            "records/derived/realdata_archived_starts.jsonl": 9}
        self.check({r["path"]: r["rows"] for r in manifest["derived_summaries"]} == expected_derived and
                   all(r["original_start_records_included"] is False for r in manifest["derived_summaries"]),
                   "records/manifest.json", "Derived-summary inventory or interpretation mismatch")
        recorded_paths = [r["path"] for r in manifest["files"]]
        packaged_paths = {p.relative_to(self.root).as_posix()
                          for p in self.path("records").rglob("*") if p.is_file()}
        packaged_paths.discard("records/manifest.json")
        self.check(len(recorded_paths) == len(set(recorded_paths)) and
                   set(recorded_paths) == packaged_paths, "records/manifest.json",
                   "Record inventory differs from manifest")
        rows = self.lines("records/provenance.jsonl")
        originals = {r["path"] for r in manifest["files"] if r["kind"] == "original_saved_record_copy"}
        self.check(len(rows) == len(originals) == 6793 and
                   {r["packaged_path"] for r in rows} == originals,
                   "records/provenance.jsonl", "Original record inventory differs from source list")
        return {"original_records": len(rows)}

    def run(self):
        for name, method in (("provenance", self.provenance), ("synthetic", self.synthetic),
                             ("realdata", self.realdata), ("benchmark", self.benchmark)):
            self.section(name, method)
        return {"schema_version": 1, "status": "passed" if not self.failure_count else "failed",
                "scope": "Consistency of packaged original records, derived endpoint summaries, and paper inputs; no simulation, fitting, or event-level likelihood/gradient recomputation.",
                "checks": self.checks, "failure_count": self.failure_count,
                "failures": self.failures, "failure_details_truncated": self.failure_count > len(self.failures),
                "sections": self.results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Root of the extracted reproduction package")
    parser.add_argument("--output", type=Path, help="Optional NEW JSON report path outside the package; existing files are never overwritten")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.output:
        out = args.output.resolve()
        if out.is_relative_to(root) or args.output.exists() or args.output.is_symlink():
            print(json.dumps({"status": "failed", "failure_count": 1,
                              "failures": [{"location": str(args.output), "message": "Output must be a new path outside the package; no files were changed."}]}))
            return 2
    result = Verification(root).run()
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as f:
                f.write(text)
        except OSError as exc:
            print(json.dumps({"status": "failed", "failure_count": 1,
                              "failures": [{"location": str(args.output), "message": f"Could not create report: {exc}"}]}))
            return 2
    print(text, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
