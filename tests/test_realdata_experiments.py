"""Deterministic checks of data isolation, prediction and resumable real-data jobs."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from experiments import realdata_baseline as study
from models.features import EventFeatures


class RealdataExperimentsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = study.read_json(study.ROOT/"configs/realdata_experiments.json")
        self.task = copy.deepcopy(study.make_plan(self.cfg, 1)[0])
        self.task.update(start=3.0, end=6.0,
                         intervals={"selection": [3.0, 4.0], "estimation": [4.0, 5.0], "test": [5.0, 6.0]})
        self.task["dataset"]["n_nodes"] = 3
        self.mapping = [{"new_node": i, "original_cell": i+10} for i in range(3)]

    def test_prespecified_counts_seeds_and_first_window_reuse(self):
        pilot, full = study.make_plan(self.cfg, 1), study.make_plan(self.cfg, 5)
        self.assertEqual((len(pilot), len(full)), (3, 15))
        self.assertEqual(len(full)*len(study.MODEL_NAMES)*3, 405)
        indexed = {t["path"]: t for t in full}
        for task in pilot:
            self.assertEqual(task, indexed[task["path"]])
        seeds = [s for t in full for values in t["start_seeds"].values() for s in values]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual([t["start"] for t in pilot], [20, 20, 40])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(study.main(["--plan-only"]), 0)
        self.assertEqual(json.loads(output.getvalue())["planned_starts"], 81)

    def test_node_selection_uses_only_independent_screen(self):
        trains = [np.array([0.1, 0.2, 3.1]), np.array([0.3, 0.4]), np.array([0.5])]
        first = study.select_fixed_subset(trains, 2, 0, 1)
        later = [x.copy() for x in trains]
        later[2] = np.r_[later[2], np.linspace(1, 100, 200)]
        self.assertEqual(first, study.select_fixed_subset(later, 2, 0, 1))
        self.assertEqual(first[0], [0, 1])

    def test_absolute_history_boundaries_and_strict_ties(self):
        trains = [np.array([0.1, 3.0, 4.0, 4.5, 6.0, 7.0]), np.array([4.5]), np.array([])]
        events = study.history_events(trains, [0, 1, 2], 6)
        self.assertEqual(events[0], (0.1, 0))
        self.assertIn((6.0, 0), events)
        self.assertNotIn((7.0, 0), events)
        counts = study.interval_counts(events, 3, 4, 5)
        self.assertEqual((counts["n_events"], counts["n_tied_times"]), (2, 1))
        f = EventFeatures(events, 3, [], 2, 0.05, 4, 5, include_start=False)
        np.testing.assert_array_equal(f.pair[0], f.pair[1])
        expected = sum(np.exp(-2*(4.5-t)) for t in (0.1, 3.0, 4.0))
        self.assertAlmostEqual(f.pair[0, 0], expected)
        self.assertEqual(f.pair[0, 1], 0.0)
        cropped = EventFeatures([(t, n) for t, n in events if t > 4], 3, [], 2, 0.05, 4, 5)
        self.assertGreater(f.pair_exposure[0], cropped.pair_exposure[0])

    def test_shared_candidate_ranking_nesting_and_future_isolation(self):
        events = [(3.10, 0), (3.11, 1), (3.12, 2), (3.2, 0), (3.21, 1), (3.22, 2)]
        pair = np.zeros((3, 3))
        first = study.rank_candidates(events, self.task, pair, self.mapping)
        second = study.rank_candidates(events+[(4.2, 0)]*50+[(5.2, 2)]*100, self.task, pair, self.mapping)
        self.assertEqual(first, second)
        self.assertEqual(first["prefixes"]["3"]["edges"], [[0, 1], [0, 2], [1, 2]])
        self.assertEqual(first["prefixes"]["1"]["edges"], first["prefixes"]["3"]["edges"][:1])
        self.assertTrue(first["shared_across_backgrounds"])

    def test_ineligible_candidates_and_actual_shortfall_are_retained(self):
        events = [(2.90, 2), (3.10, 0), (3.11, 1)]
        ranked = study.rank_candidates(events, self.task, np.ones((3, 3)), self.mapping)
        self.assertEqual(ranked["n_eligible"], 1)
        self.assertEqual(ranked["prefixes"]["3"]["edges"], [[0, 1]])
        self.assertEqual(ranked["prefixes"]["3"]["actual_count"], 1)
        self.assertEqual(len(ranked["ranked_pairs"]), 3)
        self.assertIn("member_without_selection_event", ranked["ranked_pairs"][1]["ineligible_reasons"])

    def test_initialisations_all_pair_coefficients_and_frozen_basis(self):
        events = [(0.5, 0), (4.1, 0), (4.2, 1)]
        a = study.initial_parameters(events, self.task, "V-H2-top1", [(0, 1)], 91, 0)
        b = study.initial_parameters(events+[(5.5, 2)]*40, self.task, "V-H2-top1", [(0, 1)], 91, 0)
        self.assertEqual(study.plain(a), study.plain(b))
        base, pair, hyper = study.restore_parameters(a)
        self.assertTrue(np.all(pair > 0))
        self.assertTrue(study.all_pairwise_mask(3).all())
        self.assertEqual((base.basis.origin, base.basis.end, base.basis.kind), (3, 6, "linear"))
        warm_base = study.TemporalBaseline([0.2, 0.3, 0.4], study.model_basis(self.task, "C-P"))
        warm = study.parameter_record(warm_base, np.zeros((3, 3)), {}, [])
        initial = study.initial_parameters(events, self.task, "V-P", [], 91, 0, warm)
        base, pair, _ = study.restore_parameters(initial)
        np.testing.assert_allclose(base.values([3, 4, 5, 6]), np.tile([0.2, 0.3, 0.4], (4, 1)))
        self.assertTrue(np.all(np.diag(pair) == 1e-8))

    def test_unpenalised_predictive_score_and_test_compensator(self):
        basis = study.model_basis(self.task, "V-P")
        base = study.TemporalBaseline([[1, 4], [2, 5], [3, 6]], basis)
        parameters = study.parameter_record(base, np.zeros((3, 3)), {}, [])
        events = [(4.1, 0), (5.2, 0), (5.4, 1), (6.0, 2)]
        result = study.assess_model(events, self.task, "V-P", parameters)
        expected_integral = sum((base.values([5])[0]+base.values([6])[0])/2)
        expected_event = np.log([base.values([t])[0, n] for t, n in events if t > 5]).sum()
        self.assertAlmostEqual(result["test_score"]["loglik"], expected_event-expected_integral)
        self.assertEqual(result["test_score"]["n_events"], 3)
        nofuture = study.assess_model(events[:1], self.task, "V-P", parameters)
        self.assertEqual(study.plain(result["training_score"]), study.plain(nofuture["training_score"]))
        self.assertEqual(result["training_diagnostics"]["information"]["n_structural_parameters"], 15)
        selector = study.assess_model(events, self.task, "screening",
            study.parameter_record(study.TemporalBaseline([1, 1, 1], study.model_basis(self.task, "screening")), np.zeros((3, 3)), {}, []))
        self.assertNotIn("test_score", selector)

    def test_no_test_event_still_has_integral_and_missing_per_event_gain(self):
        base = study.TemporalBaseline([1, 1, 1], study.model_basis(self.task, "C-P"))
        parameters = study.parameter_record(base, np.zeros((3, 3)), {}, [])
        output = study.assess_model([], self.task, "C-P", parameters)
        self.assertEqual(output["test_score"]["loglik"], -3)
        delta = study.prediction_difference({"status": "qualified", **output}, {"status": "qualified", **output})
        self.assertEqual(delta["delta_loglik"], 0)
        self.assertIsNone(delta["delta_per_event"])

    def test_shared_history_conditional_prediction_uses_only_strict_past(self):
        base = study.TemporalBaseline([1, 1, 1], study.model_basis(self.task, "C-P"))
        pair = np.zeros((3, 3)); pair[0, 1] = 0.4
        parameters = study.parameter_record(base, pair, {}, [])
        events = [(4.5, 0), (5.2, 0), (5.4, 1)]
        output = study.assess_model(events, self.task, "C-P", parameters)["test_score"]
        lam = 1+0.4*(np.exp(-2*0.9)+np.exp(-2*0.2))
        self.assertAlmostEqual(output["event_log_intensity"], np.log(lam))
        expected_pair = 0.4*((np.exp(-1)-np.exp(-3))/2+(1-np.exp(-1.6))/2)
        self.assertAlmostEqual(output["pairwise_integral"], expected_pair)

    def test_hand_event_worker_preserves_em_and_zero_exposure_candidate(self):
        task = copy.deepcopy(self.task)
        task["config"]["em"].update(max_iter=40, max_total_iter=80)
        events = [(0.1, 0), (3.1, 0), (4.1, 0), (4.3, 1), (4.6, 0), (4.8, 1), (5.5, 2)]
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            study.save_events(directory, events)
            initial = study.initial_parameters(events, task, "C-H2-top1", [(0, 2)], 9, 0)
            path = directory/"start_1.json"
            study.write_json(path, {"status": "running", "task": task, "model": "C-H2-top1",
                                   "dataset_directory": str(directory), "initial_parameters": initial})
            record = study.run_worker(path, 30)
            self.assertIn(record["status"], {"qualified", "unqualified"})
            self.assertIn("em_result", record)
            self.assertGreaterEqual(record["objective"]+1e-6, record["em_result"]["objective"])
            self.assertEqual(record["final_parameters"]["alpha_hyper"], [0.0])
            self.assertEqual(record["em_result"]["identifiable_hyperedges"], [False])
            self.assertEqual(record["training_score"]["n_events"], 4)
            self.assertNotIn("test_score", record)

    def test_interrupted_refinement_resumes_saved_em_without_refitting(self):
        events = [(4.1, 0), (4.3, 1)]
        initial = study.initial_parameters(events, self.task, "C-P", [], 3, 0)
        base, pair, hyper = study.restore_parameters(initial)
        f = EventFeatures(events, 3, [], 2, 0.05, 4, 5, basis=base.basis, include_start=False)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            study.save_events(directory, events)
            path = directory/"start_1.json"
            original = {"parameters": initial, "objective": f.score(base.coefficients, pair, hyper)["loglik"]}
            study.write_json(path, {"status": "running", "task": self.task, "model": "C-P",
                "dataset_directory": str(directory), "initial_parameters": initial, "em_result": original})
            with patch.object(study, "run_em", side_effect=AssertionError("must not refit EM")):
                record = study.fitting_worker(path)
            self.assertEqual(record["em_result"], study.plain(original))
            self.assertIn("refinement", record)

    def test_shortfall_aliases_same_fit_and_background_difference(self):
        score = {"loglik": 10, "duration": 2, "n_events": 4}
        fitted = {"status": "qualified", "test_score": score, "edges": [[0, 1]],
                  "start_statuses": ["qualified"]*3, "total_start_wall_seconds": 10}
        alias = study.alias_model(fitted, "C-H2-top3", "C-H2-top1", {"actual_count": 1})
        self.assertEqual(alias["test_score"], score)
        self.assertEqual(alias["start_records"], [])
        fits = {"C-P": {**fitted, "test_score": {**score, "loglik": 7}}, "C-H2-top3": fitted,
                "V-P": {**fitted, "test_score": {**score, "loglik": 9}},
                "V-H2-top3": {**fitted, "test_score": {**score, "loglik": 15}}}
        result = study.comparisons(fits)["top3"]
        self.assertEqual(result["linear_minus_constant_gain"]["delta_loglik"], 3)
        self.assertTrue(result["primary"])

    def test_failed_screening_retains_window_and_blocks_only_completion_models(self):
        recording = {"coverage_check_passed": True}
        called = []
        def fit(directory, task, model, edges, events, fitted):
            called.append(model)
            return {"status": "no_qualified_start", "edges": [], "start_statuses": []}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root/self.task["path"]
            study.save_events(directory, [])
            with patch.object(study, "prepare_window", return_value={"mapping": self.mapping}), patch.object(study, "fit_model", side_effect=fit):
                result = study.run_window(root, self.task, recording)
            self.assertEqual(called, ["screening", "C-P", "V-P"])
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["models"]["C-H2-top3"]["status"], "candidate_screening_failed")
            self.assertEqual(result["comparisons"]["top3"]["constant"]["status"], "missing")

    def test_full_run_requires_pilot_and_reports_no_independence_mcse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertFalse(study.pilot_readiness(root, self.cfg)["ready"])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                study.main(["--windows", "5", "--results-root", str(root)])
            summary = study.summarise(root, study.make_plan(self.cfg, 1))
            self.assertEqual(summary["qualified_windows"], 0)
            self.assertTrue(all(r["n_planned_windows"] == 1 for r in summary["descriptive_aggregates"]))
            self.assertNotIn("mcse", study.descriptive([1, 2, 3]))

    def test_interrupted_main_packages_and_preserves_previous_package(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["datasets"] = cfg["datasets"][:1]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root/"config.json"
            study.write_json(config, cfg)
            destination = root/"downloads"
            destination.mkdir()
            package = destination/"hypergraph_hawkes_results.zip"
            package.write_bytes(b"previous scientific result package")
            with patch.object(study, "prepare_recording", side_effect=KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
                code = study.main(["--config", str(config), "--results-root", str(root/"results"),
                                   "--download-dir", str(destination)])
            self.assertEqual(code, 130)
            old = list((destination/"archive").rglob("*.zip"))
            self.assertEqual(len(old), 1)
            self.assertEqual(old[0].read_bytes(), b"previous scientific result package")
            with zipfile.ZipFile(package) as archive:
                runs = [name for name in archive.namelist() if "/runs/" in name]
                self.assertEqual(json.loads(archive.read(runs[0]))["status"], "interrupted")
                self.assertFalse(any("/synthetic/" in name for name in archive.namelist()))

    def saved_coordinate_endpoint(self, directory):
        """One sweep on fixed hand events supplies a genuine resumable endpoint."""
        events = [(0.1, 0), (3.1, 1), (4.1, 0), (4.3, 1), (4.6, 0), (4.8, 1), (5.5, 2)]
        study.save_events(directory, events)
        initial = study.initial_parameters(events, self.task, "C-P", [], 9, 0)
        base, pair, hyper = study.restore_parameters(initial)
        f = EventFeatures(events, 3, [], 2, 0.05, 4, 5, basis=base.basis, include_start=False)
        first = study.refine_likelihood(f, base.coefficients, pair, hyper, study.all_pairwise_mask(3), max_sweeps=1)
        endpoint = study.parameter_record(study.TemporalBaseline(first["baseline_coefficients"], base.basis),
                                          first["alpha_pairwise"], first["alpha_hyper"], [])
        refinement = {k: v for k, v in first.items()
                      if k not in {"baseline_coefficients", "alpha_pairwise", "alpha_hyper", "training_score"}}
        return {"status": "unqualified", "task": self.task, "model": "C-P", "start_index": 1,
                "seed": self.task["start_seeds"]["C-P"][0], "dataset_directory": str(directory),
                "initial_parameters": initial, "final_parameters": endpoint, "objective": first["final_objective"],
                "training_score": first["training_score"], "kkt_residual": first["kkt_residual"],
                "converged": False, "refinement": refinement,
                "em_result": {"parameters": initial, "objective": f.score(base.coefficients, pair, hyper)["loglik"],
                              "test_fixture": True}, "wall_seconds": 2.0, "worker_seconds": 1.5}

    def test_continuation_uses_coordinate_endpoint_preserves_history_and_never_refits_em(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            record = self.saved_coordinate_endpoint(directory)
            old = study.plain(record)
            record["continuation_settings"] = {"max_total_sweeps": study.CONTINUATION_MAX_SWEEPS}
            path = directory/"start_1.json"
            study.write_json(path, record)
            solve = study.refine_likelihood
            def checked(features, coefficients, pair, hyper, mask, **kwargs):
                base, a, h = study.restore_parameters(old["final_parameters"])
                np.testing.assert_array_equal(coefficients, base.coefficients)
                np.testing.assert_array_equal(pair, a)
                self.assertEqual(hyper, h)
                self.assertTrue(mask.all())
                self.assertTrue(all(t <= 5 for t in features.times))
                self.assertEqual(kwargs["max_sweeps"], study.CONTINUATION_MAX_SWEEPS-1)
                self.assertEqual(kwargs["kkt_tol"], self.cfg["em"]["kkt_tol"])
                self.assertEqual(kwargs["convergence_tol"], self.cfg["em"]["convergence_tol"])
                return solve(features, coefficients, pair, hyper, mask, **kwargs)
            with patch.object(study, "run_em", side_effect=AssertionError("EM must not run")), patch.object(study, "refine_likelihood", side_effect=checked):
                result = study.continue_refinement(path)
            self.assertEqual(result["em_result"], old["em_result"])
            self.assertEqual(result["initial_parameters"], old["initial_parameters"])
            self.assertEqual(result["refinement"]["objective_history"][:1], old["refinement"]["objective_history"])
            self.assertGreaterEqual(result["objective"]+1e-6, old["objective"])
            self.assertEqual(result["status"], "qualified")
            self.assertLessEqual(result["refinement"]["n_sweeps"], study.CONTINUATION_MAX_SWEEPS)
            self.assertEqual(result["refinement"]["continuation_segments"][0]["starting_sweep"], 1)
            self.assertNotIn("test_score", result)

    def continuation_fixture(self, root):
        directory = root/self.task["path"]
        directory.mkdir(parents=True)
        record = self.saved_coordinate_endpoint(directory)
        study.write_json(directory/"dataset.json", {"task": self.task})
        study.write_json(directory/"result.json", {"status": "incomplete", "test_score": -100})
        study.write_json(root/self.task["dataset_id"]/"recording.json", {"coverage_check_passed": True})
        for model in study.MODEL_NAMES:
            study.write_json(directory/"models"/model/"fit.json", {"reused_from": "C-P"} if model != "C-P" else
                             {"model": "C-P", "status": "no_qualified_start", "selected_start": None, "edges": []})
        for i in (1, 2, 3):
            current = copy.deepcopy(record)
            current.update(start_index=i, seed=self.task["start_seeds"]["C-P"][i-1])
            if i != 1:
                current.update(status="qualified", converged=True)
                current["refinement"]["converged"] = True
            study.write_json(directory/"models/C-P"/f"start_{i}.json", current)
        return directory

    def test_continuation_targets_depend_only_on_training_status_and_remaining_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = self.continuation_fixture(root)
            targets = study.continuation_plan(root, [self.task])
            self.assertEqual([t["start_index"] for t in targets], [1])
            study.write_json(directory/"result.json", {"status": "incomplete", "test_score": 1e12})
            self.assertEqual(study.continuation_plan(root, [self.task]), targets)
            path = directory/"models/C-P/start_1.json"
            record = study.read_json(path)
            record["refinement"]["n_sweeps"] = study.CONTINUATION_MAX_SWEEPS
            study.write_json(path, record)
            self.assertEqual(study.continuation_plan(root, [self.task]), [])
            self.assertEqual(study.read_json(path)["status"], "unqualified")

    def test_continuation_archives_failure_and_leaves_qualified_starts_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = self.continuation_fixture(root)
            model_dir = directory/"models/C-P"
            before = {i: (model_dir/f"start_{i}.json").read_bytes() for i in (1, 2, 3)}
            targets = study.continuation_plan(root, [self.task])
            called = []
            def worker(path, timeout):
                called.append(path.name)
                result = study.continue_refinement(path)
                result["wall_seconds"] = result["continuation_current_attempt"]["previous_wall_seconds"]+0.1
                study.write_json(path, result)
                return study.read_json(path)
            with patch.object(study, "run_worker", side_effect=worker), patch.object(study, "run_window", return_value={"status":"qualified"}), contextlib.redirect_stdout(io.StringIO()):
                study.continue_window(root, self.task, targets)
            self.assertEqual(called, ["start_1.json"])
            for i in (2, 3):
                self.assertEqual((model_dir/f"start_{i}.json").read_bytes(), before[i])
            archived = list((directory/"archive").rglob("start_1.json"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_bytes(), before[1])
            current = study.read_json(model_dir/"start_1.json")
            self.assertEqual(current["status"], "qualified")
            self.assertEqual(study.continuation_plan(root, [self.task]), [])
            self.assertTrue(study.read_json(model_dir/"fit.json")["status"] == "qualified")

    def test_resume_refreshes_stale_model_without_running_a_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = self.continuation_fixture(root)
            path = directory/"models/C-P/start_1.json"
            record = study.read_json(path)
            record.update(status="qualified", converged=True)
            record["refinement"]["converged"] = True
            study.write_json(path, record)
            self.assertEqual(study.continuation_plan(root, [self.task]), [])
            with patch.object(study, "run_worker", side_effect=AssertionError("no worker should run")), patch.object(study, "run_window", return_value={"status":"qualified"}):
                result = study.continue_window(root, self.task, [])
            self.assertEqual(result["status"], "qualified")
            self.assertEqual(study.read_json(directory/"models/C-P/fit.json")["status"], "qualified")
            archived = list((directory/"archive").rglob("fit.json"))
            self.assertEqual(study.read_json(archived[0])["status"], "no_qualified_start")

    def test_continuation_budget_cannot_be_reset_or_exhausted_endpoint_marked_successful(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            record = self.saved_coordinate_endpoint(directory)
            record["continuation_settings"] = {"max_total_sweeps": 200}
            path = directory/"start_1.json"
            study.write_json(path, record)
            with self.assertRaises(ValueError):
                study.continue_refinement(path)
            record["continuation_settings"]["max_total_sweeps"] = study.CONTINUATION_MAX_SWEEPS
            record["refinement"]["n_sweeps"] = study.CONTINUATION_MAX_SWEEPS
            study.write_json(path, record)
            with patch.object(study, "refine_likelihood", side_effect=AssertionError("budget exhausted")), self.assertRaises(ValueError):
                study.continue_refinement(path)
            self.assertEqual(study.read_json(path)["status"], "unqualified")


if __name__ == "__main__":
    unittest.main()
