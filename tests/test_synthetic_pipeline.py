"""Deterministic checks of the scientific design and pilot orchestration."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from experiments import synthetic_pipeline as study


class SyntheticPipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = study.read_json(study.ROOT / "configs/synthetic_experiments.json")

    def test_design_counts_seeds_and_reused_constant_paths(self):
        small = study.make_plan(self.cfg, 2)
        full = study.make_plan(self.cfg, 10)
        self.assertEqual(len(small), 24)
        self.assertEqual(sum(len(t["scenario"]["models"]) for t in small), 92)
        self.assertEqual(len(full), 130)  # 120 fitted paths plus ten independent 2B paths
        self.assertEqual(sum(len(t["scenario"]["models"]) for t in full), 460)
        indexed = {t["dataset_id"]: t for t in full}
        for task in small:
            self.assertEqual(task, indexed[task["dataset_id"]])
        seeds = [t["generation_seed"] for t in full]
        seeds += [s for t in full for model in t["start_seeds"].values() for s in model]
        self.assertEqual(len(seeds), len(set(seeds)))
        shared = [t for t in small if "shared_with" in t["scenario"]]
        self.assertEqual(len(shared), 4)
        self.assertTrue(all(t["scenario"]["experiment_id"] == 13 for t in shared))
        self.assertTrue(all(len(t["scenario"]["models"]) == 5 for t in shared))
        self.assertEqual(len(study.candidate_edges(self.cfg)), 21)
        base, pair, hyper = study.true_parameters(self.cfg, small[0]["scenario"])
        self.assertAlmostEqual(float(pair.sum(axis=1).max()), 0.35)
        self.assertTrue(np.all(np.diag(pair) == 0))
        self.assertEqual(sum(len(e) == 2 and w > 0 for e, w in hyper.items()), 6)

    def test_initialisation_uses_training_only_and_preserves_warm_constant(self):
        task = study.make_plan(self.cfg, 2)[0]
        spec = task["scenario"]
        past = [(1.0, 0), (2.0, 1), (3.0, 2), (4.0, 3), (5.0, 4), (6.0, 5)]
        seed = 123
        a = study.initial_parameters(past, self.cfg, spec, "C-M2", seed, 0)
        b = study.initial_parameters(past + [(499.0, 1)]*10, self.cfg, spec, "C-M2", seed, 0)
        self.assertEqual(study.plain(a), study.plain(b))
        warm_base = study.TemporalBaseline(np.full(6, 0.4))
        warm = study.parameter_record(warm_base, np.zeros((6, 6)), {}, study.candidate_edges(self.cfg, "C-M1"))
        init = study.initial_parameters(past, self.cfg, spec, "V-M2", seed, 0, warm)
        background, pair, hyper = study.restore_parameters(init)
        np.testing.assert_allclose(background.values([0.0, 12.5, 30.0]), 0.4)
        self.assertTrue(np.all(pair[~np.eye(6, dtype=bool)] > 0))
        self.assertTrue(np.all(np.diag(pair) == 0))
        self.assertTrue(all(v > 0 for v in hyper.values()))

    def test_formal_counts_seed_separation_and_fixed_linear_repetitions(self):
        pilot = study.make_plan(self.cfg, 10)
        formal = study.make_plan(self.cfg, phase="formal")
        self.assertEqual(len(formal), 1100)
        self.assertEqual(sum(len(t["scenario"]["models"]) for t in formal), 4200)
        self.assertEqual(sum(t["scenario"]["background"] == "linear" for t in formal), 100)
        self.assertEqual(sum("shared_with" in t["scenario"] for t in formal), 200)
        self.assertTrue(all(not t["double_burn_in"] and '/formal/' in t["path"] for t in formal))
        def all_seeds(plan):
            return [t["generation_seed"] for t in plan]+[v for t in plan for values in t["start_seeds"].values() for v in values]
        self.assertFalse(set(all_seeds(pilot)) & set(all_seeds(formal)))
        self.assertEqual(len(all_seeds(formal)), len(set(all_seeds(formal))))
        old = next(t for t in pilot if t['scenario']['name']=='moderate_T500' and t['repeat']==5)
        self.assertEqual(old['generation_seed'], 6524390598782008268)
        for task in formal:
            spec = task['scenario']
            self.assertEqual(task['generation_seed'], study.seed_for(self.cfg, spec, task['repeat'], 1, phase='formal'))
        with self.assertRaises(ValueError):
            study.make_plan(self.cfg, 100, phase='formal')
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(study.main(['--phase','formal','--plan-only']),0)
        self.assertEqual(json.loads(output.getvalue())['starts'],12600)

    def test_atomic_save_retries_lock_but_preserves_old_record_on_persistent_failure(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'record.json'
            study.write_json(path,{'status':'old'})
            original=Path.replace
            attempts=[]
            def transient(temp,target):
                attempts.append(1)
                if len(attempts)<3:raise PermissionError('Windows destination locked')
                return original(temp,target)
            with patch.object(Path,'replace',transient), patch.object(study.time,'sleep'):
                study.write_json(path,{'status':'new'})
            self.assertEqual(study.read_json(path),{'status':'new'})
            self.assertEqual(len(attempts),3)
            with patch.object(Path,'replace',side_effect=PermissionError('persistent')), patch.object(study.time,'sleep'):
                with self.assertRaises(PermissionError):
                    study.write_json(path,{'status':'unsaved'})
            self.assertEqual(study.read_json(path),{'status':'new'})
            self.assertEqual(study.read_json(path.with_suffix('.json.tmp')),{'status':'unsaved'})

    def test_recovery_mcse_uses_independent_paths_and_excludes_unidentifiable_errors(self):
        rows=[]
        for e in (1.0,3.0):
            rows.append({'parameter_errors':{'hyperedges':[
                {'edge':[0,1,2],'truth':0.2,'estimate':0.2+e,'error':e,'identifiable':True},
                {'edge':[1,2,3],'truth':0.0,'estimate':e,'error':None,'identifiable':False}],
                'pairwise_error_matrix':[[None,e],[e,None]],'pairwise_mse':e*e,
                'triple_sum_weight':e,'triple_max_weight':e},
                'baseline_errors':{k:{'mse_mean':e*e} for k in ('training','test')}})
        result=study.recovery_summary(rows,3)
        self.assertEqual(result['planned'],3)
        self.assertEqual(result['qualified'],2)
        h=result['hyperedges'][0]
        self.assertEqual(h['bias'],2)
        self.assertEqual(h['mcse_bias'],1)
        self.assertEqual(h['mse'],5)
        self.assertEqual(h['mcse_mse'],4)
        self.assertAlmostEqual(h['rmse'],np.sqrt(5))
        self.assertEqual(result['hyperedges'][1]['n'],0)
        self.assertIsNone(result['hyperedges'][1]['mse'])

    def test_worker_save_failure_preserves_unsaved_result_without_calling_it_success(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'start_1.json'
            study.write_json(path,{'status':'running','objective':-10})
            pending={'status':'qualified','objective':-9}
            path.with_suffix('.json.tmp').write_text(json.dumps(pending),encoding='utf-8')
            with patch.object(study.subprocess,'Popen') as process:
                child=process.return_value.__enter__.return_value
                child.returncode=1
                child.wait.return_value=1
                result=study.run_worker('fit',path,60)
            self.assertEqual(result['status'],'worker_failed')
            self.assertEqual(result['objective'],-10)
            self.assertEqual(study.read_json(path.parent/result['unsaved_result_record']),pending)

    def test_selection_never_uses_test_scores_or_hides_better_unqualified_fit(self):
        p = {"baseline_coefficients": [[0.4]], "alpha_pairwise": [[0.0]], "alpha_hyper": []}
        rows = [{"status": "qualified", "objective": -10.0, "final_parameters": p, "test_score": 100},
                {"status": "qualified", "objective": -9.0, "final_parameters": p, "test_score": -100},
                {"status": "failed"}]
        self.assertEqual(study.choose_start(rows)["selected_start"], 2)
        rows.append({"status": "unqualified", "objective": -8.0})
        self.assertEqual(study.choose_start(rows)["status"], "higher_unqualified_objective")
        self.assertEqual(study.choose_start(rows[:0])["status"], "no_qualified_start")
        fits = {m: {"status": "qualified", "test_score": {"loglik": v, "duration": 10, "n_events": 0}}
                for m, v in (("C-M1", -2.0), ("C-M2", -3.0))}
        delta = study.comparisons(fits)["C-M2_minus_C-M1"]
        self.assertEqual(delta["delta_loglik"], -1.0)
        self.assertIsNone(delta["delta_per_event"])

    def test_fit_worker_cannot_receive_test_events(self):
        task = study.make_plan(self.cfg, 2)[0]
        initial = study.initial_parameters([(1, 0), (2, 1)], self.cfg, task["scenario"], "M0", 1, 0)
        record = {"task": task, "initial_parameters": study.plain(initial), "dataset_directory": "unused"}
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop before any fitting")
        with patch.object(study, "read_json", return_value=record), \
             patch.object(study, "load_events", return_value=[(-1, 0), (1, 1), (450, 5)]), \
             patch.object(study, "run_em", side_effect=capture):
            with self.assertRaisesRegex(RuntimeError, "stop before"):
                study.fitting_worker("unused")
        self.assertEqual(captured["events"], [(-1, 0), (1, 1)])
        self.assertFalse(captured["include_start"])
        self.assertEqual(captured["lambda_l1"], 0.0)
        self.assertEqual(captured["max_total_iter"], 5000)

    def test_interrupted_run_preserves_record_and_packages_results(self):
        def interrupt(root, task):
            study.write_json(root/task["path"]/"dataset.json", {"task": task, "status": "interrupted"})
            raise KeyboardInterrupt
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            download = folder/"downloads"
            download.mkdir()
            (download/"hypergraph_hawkes_results.zip").write_bytes(b"prior result")
            with patch.object(study, "run_dataset", side_effect=interrupt), contextlib.redirect_stdout(io.StringIO()):
                result = study.main(["--results-root", str(folder/"results"), "--download-dir", str(download)])
            self.assertEqual(result, 130)
            with zipfile.ZipFile(download/"hypergraph_hawkes_results.zip") as archive:
                self.assertTrue(any(name.endswith("dataset.json") for name in archive.namelist()))
                name = next(name for name in archive.namelist() if name.endswith("summary.json"))
                summary = json.loads(archive.read(name))
                self.assertEqual(summary["planned_models"], 92)
                self.assertEqual(len(summary["not_completed_models"]), 92)
            archived = list((download/"archive").rglob("hypergraph_hawkes_results.zip"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_bytes(), b"prior result")

    def test_worker_and_scoring_on_fixed_hand_constructed_events(self):
        task = study.make_plan(self.cfg, 2)[0]
        task["scenario"]["T"] = 9
        events = [(float(t), n) for t in [-1, *range(1, 10)] for n in range(6)]
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            np.savez_compressed(folder/"events.npz", times=np.array([t for t, _ in events]),
                                nodes=np.array([n for _, n in events]))
            path = folder/"start_1.json"
            study.write_json(path, {"status": "running", "purpose": "fixed_implementation_fixture",
                "task": task, "dataset_directory": str(folder),
                "initial_parameters": study.initial_parameters(events, self.cfg, task["scenario"], "M0", 1, 0)})
            result = study.run_worker("fit", path, 60)
            self.assertIn(result["status"], ("qualified", "unqualified"), result.get("error"))
            self.assertEqual(result["training_score"]["n_events"], 36)
            assessment = study.assess_model(events, task, "M0", result)
            self.assertEqual(assessment["test_score"]["n_events"], 18)
            self.assertEqual(assessment["test_score"]["duration"], 3)
            self.assertTrue(np.isfinite(assessment["test_score"]["loglik"]))

    def test_refinement_preserves_all_em_records_and_resumes_without_refitting(self):
        old_cfg = dict(self.cfg)
        old_cfg.pop("refinement")
        task = study.make_plan(old_cfg, 2)[0]
        task["scenario"].update(T=9, models=["M0"], warm_from={})
        events = [(float(t), n) for t in [-1, *range(1, 10)] for n in range(6)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            directory = root/task["path"]
            directory.mkdir(parents=True)
            np.savez_compressed(directory/"events.npz", times=np.array([t for t, _ in events]),
                                nodes=np.array([n for _, n in events]))
            study.write_json(directory/"dataset.json", {"status": "completed", "task": task})
            first = directory/"models/M0/start_1.json"
            study.write_json(first, {"status": "running", "model": "M0", "task": task,
                "purpose": "fixed_implementation_fixture", "seed": task["start_seeds"]["M0"][0],
                "dataset_directory": str(directory),
                "initial_parameters": study.initial_parameters(events, old_cfg, task["scenario"], "M0", 1, 0)})
            before = study.run_worker("fit", first, 60)
            originals = []
            for index, seed in enumerate(task["start_seeds"]["M0"]):
                sample = dict(before, seed=seed, start_index=index+1)
                originals.append(sample)
                study.write_json(first.with_name(f"start_{index+1}.json"), sample)
            task["config"] = self.cfg
            with contextlib.redirect_stdout(io.StringIO()):
                study.refine_dataset(root, task)
            for index, original in enumerate(originals):
                after = study.read_json(first.with_name(f"start_{index+1}.json"))
                self.assertEqual(after["em_result"], original)
                self.assertEqual(after["initial_parameters"], original["initial_parameters"])
                self.assertEqual(after["seed"], original["seed"])
                self.assertEqual(after["status"], "qualified")
                self.assertLessEqual(after["kkt_residual"], 1e-4)
                self.assertGreaterEqual(after["objective"], original["objective"]-1e-6)
            backup = study.read_json(directory/"refinement.json")["original_records"]
            self.assertEqual(study.read_json(directory/backup/"models/M0/start_1.json"), originals[0])
            with patch.object(study, "run_worker", side_effect=AssertionError("must reuse completed refinement")):
                study.refine_dataset(root, task)
            # Reproduce the returned pilot's final-save failure. Retry only that
            # saved EM endpoint and preserve both the failed JSON and its log.
            failed_path=first.with_name('start_3.json')
            failed=study.read_json(failed_path)
            failed.update(status='worker_failed',returncode=1)
            failed.pop('refinement')
            study.write_json(failed_path,failed)
            failed_path.with_suffix('.log').write_text('PermissionError: WinError 5',encoding='utf-8')
            study.write_json(root/'synthetic_experiments/pilot/config.json',self.cfg)
            with patch.object(study,'make_plan',return_value=[task]):
                self.assertFalse(study.pilot_readiness(root,self.cfg)['ready'])
            with patch.object(study,'run_worker',wraps=study.run_worker) as worker, contextlib.redirect_stdout(io.StringIO()):
                study.refine_dataset(root,task)
            self.assertEqual(worker.call_count,1)
            self.assertEqual(worker.call_args.args[1],failed_path)
            recovered=study.read_json(failed_path)
            self.assertEqual(recovered['status'],'qualified')
            self.assertEqual(recovered['em_result'],originals[2])
            self.assertNotIn('returncode',recovered)
            self.assertTrue(any(study.read_json(p)==failed for p in (directory/'archive').rglob('start_3.json')))
            self.assertTrue(any(p.read_text(encoding='utf-8')=='PermissionError: WinError 5'
                                for p in (directory/'archive').rglob('start_3.log')))
            with patch.object(study,'make_plan',return_value=[task]):
                self.assertTrue(study.pilot_readiness(root,self.cfg)['ready'])

    def test_formal_refuses_to_generate_without_completed_pilot(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(study,'run_dataset',side_effect=AssertionError('must not simulate')), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    study.main(['--phase','formal','--results-root',d])
            self.assertEqual(raised.exception.code,2)
            self.assertFalse((Path(d)/'synthetic_experiments/formal').exists())


if __name__ == "__main__":
    unittest.main()
