"""Deterministic checks for timing scope, shared inputs and saved failures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from experiments import syn07_scalability as study
from inference.em import run_em
from models.kernel import ExponentialKernel, HyperedgeAnchor

EVENTS=[(.1,0),(.12,1),(.3,2),(.5,0),(.65,1),(.8,2),(1.1,0),(1.12,1),(1.4,2),(1.7,0),(1.8,1),(1.95,2)]

class ScalabilityTests(unittest.TestCase):
    def setUp(self):
        self.cfg=study.read_json(study.CONFIG)
        self.task={'sweep':'node_count','n_nodes':3,'T':2,'path':'node_count/N_3'}

    def test_original_grid_and_fixed_repetitions(self):
        plan=study.make_plan(self.cfg)
        self.assertEqual(len(plan),9)
        self.assertEqual([p['T'] for p in plan[:5]],[100,200,400,800,1600])
        self.assertEqual([p['n_nodes'] for p in plan[5:]],[3,5,8,12])
        self.assertEqual(self.cfg['repetitions'],10)
        self.assertEqual(self.cfg['warmup_repetitions'],1)
        self.assertEqual(len({p['path'] for p in plan}),9)

    def test_cached_iteration_matches_production_without_mutating_initials(self):
        f=study.build_features(EVENTS,self.task,self.cfg)
        b,a,h,mask=study.initial_parameters(self.task,self.cfg)
        original=(b.copy(),a.copy(),h.copy())
        bb,aa,hh,obj=study.cached_iteration(f,b,a,h,mask,self.cfg['lambda_l1'])
        result=run_em(EVENTS,2,3,list(h),ExponentialKernel(1),HyperedgeAnchor(.5),b[:,0],a,h,
            n_iter=1,lambda_l1=self.cfg['lambda_l1'],hyper_update='free',positive_start=True,
            pairwise_mask=mask,include_start=False,track_loglik=True)
        np.testing.assert_array_equal(bb,result.baseline_coefficients)
        np.testing.assert_array_equal(aa,result.alpha_pairwise)
        self.assertEqual(hh,result.alpha_hyper);self.assertAlmostEqual(obj,result.final_objective,places=12)
        np.testing.assert_array_equal(b,original[0]);np.testing.assert_array_equal(a,original[1]);self.assertEqual(h,original[2])
        self.assertTrue(np.all(a[mask]>0));self.assertTrue(np.all(a[~mask]==0))

    def test_full_fit_records_numerical_criteria(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['em'].update(max_iter=10,max_total_iter=20,min_iter=5)
        cfg['refinement_max_sweeps']=30
        row=study.measure_phase('fit',EVENTS,self.task,cfg)
        self.assertGreater(row['seconds'],0)
        self.assertIn('em_iterations',row);self.assertIn('coordinate_sweeps',row)
        self.assertEqual(row['fit_qualified'],row['final_kkt']<=cfg['em']['kkt_tol'] and all(
            x<cfg['em']['convergence_tol'] for x in row['last_relative_increments']) and len(row['last_relative_increments'])==5)
        self.assertEqual(np.diag(row['final_parameters']['alpha_pairwise']).tolist(),[0,0,0])

    def test_features_and_iteration_are_separate_measurements(self):
        f=study.build_features(EVENTS,self.task,self.cfg)
        row=study.measure_phase('features',EVENTS,self.task,self.cfg)
        self.assertEqual(row['feature_bytes'],f.nbytes)
        with patch.object(study,'build_features',side_effect=AssertionError('iteration rebuilt features')):
            row=study.measure_phase('iteration',EVENTS,self.task,self.cfg,f)
        self.assertIn('objective_after_one_update',row)

    def test_warmup_excluded_and_completed_repeats_reused(self):
        cfg=copy.deepcopy(self.cfg);cfg['repetitions']=2
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)/'iteration';folder.mkdir()
            np.savez_compressed(Path(td)/'events.npz',times=np.array(EVENTS)[:,0],nodes=np.array(EVENTS)[:,1].astype(int))
            study.write_json(folder/'request.json',{'phase':'iteration','task':self.task,'config':cfg})
            with patch.object(study,'measure_phase',return_value={'status':'completed','seconds':.1}) as measure:
                self.assertEqual(study.worker(folder/'request.json'),0)
                self.assertEqual(measure.call_count,3)
            before=(folder/'repeat_01.json').read_bytes()
            self.assertTrue(study.phase_finished(folder,2))
            with patch.object(study,'measure_phase',return_value={'status':'completed','seconds':.2}) as measure:
                self.assertEqual(study.worker(folder/'request.json'),0)
                self.assertEqual(measure.call_count,1)
            self.assertEqual((folder/'repeat_01.json').read_bytes(),before)
            self.assertEqual(len(list((folder/'warmups').glob('*.json'))),2)

    def test_error_status_is_not_a_zero_time_or_success(self):
        result=study.describe_times([{'status':'completed','seconds':2},{'status':'failed','seconds':None},
                                    {'status':'completed','seconds':4}])
        self.assertEqual(result['median_seconds'],3)
        self.assertEqual(result['n_measured'],2)
        self.assertIsNone(study.describe_times([{'status':'failed'}])['median_seconds'])

    def test_summary_excludes_warmups_and_requires_full_grid_for_slope(self):
        cfg=copy.deepcopy(self.cfg);cfg['repetitions']=2
        plan=study.make_plan(cfg)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for task in plan:
                d=root/task['path'];n=task['T']*task['n_nodes']
                study.write_json(d/'dataset.json',{'n_events':n})
                for phase in study.PHASES:
                    for rep in [1,2]:
                        study.write_json(d/phase/f'repeat_{rep:02d}.json',{'status':'completed','seconds':n*.0001,
                          'fit_qualified':True,'phase':phase})
                    study.write_json(d/phase/'warmups'/'warm.json',{'status':'completed','seconds':100000})
            summary=study.summarise(root,plan,cfg)
            self.assertEqual(summary['status'],'completed')
            self.assertEqual(summary['status_counts']['completed'],54)
            for phase in study.PHASES:self.assertAlmostEqual(summary['event_count_slopes'][phase]['main_slope'],1)
            study.write_json(root/plan[0]['path']/'fit/repeat_01.json',{'status':'completed','seconds':.01,'fit_qualified':False})
            changed=study.summarise(root,plan,cfg)
            self.assertIsNone(changed['event_count_slopes']['fit']['main_slope'])
            self.assertEqual(changed['status'],'completed_with_issues')

    def test_package_preserves_existing_return_and_only_benchmark_files(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'results';dest=Path(td)/'delivery';root.mkdir();dest.mkdir()
            study.write_json(root/'summary.json',{'status':'test'})
            old=dest/'hypergraph_hawkes_results.zip';old.write_bytes(b'previous-return')
            output=study.package_results(root,dest)
            archives=list((dest/'archive/hypergraph_hawkes').rglob('*.zip'))
            self.assertEqual(len(archives),1);self.assertEqual(archives[0].read_bytes(),b'previous-return')
            with zipfile.ZipFile(output) as z:
                self.assertEqual(z.namelist(),['experiments/results/synthetic/syn07_scalability/summary.json'])

if __name__=='__main__':unittest.main()
