"""Syn07: repeated timing of the revised implementation on the original grid.

The saved inputs are reused by three separate worker processes per scale.
Times exclude simulation, imports and result serialization. Full fits include
parameter initialization, both feature constructions, EM and refinement.
Old syn07_scalability.pkl remains untouched.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.synthetic_pipeline import read_json, write_json, timestamp, package_version, process_peak_bytes, parameter_record
from inference.e_step import EStep
from inference.em import run_em
from inference.coordinate import refine_likelihood
from models.baseline import BaselineBasis, TemporalBaseline
from models.features import EventFeatures
from models.kernel import ExponentialKernel, HyperedgeAnchor
from simulation.simulator import HawkesSimulator

PHASES=('features','iteration','fit')
TERMINAL={'completed','failed','timeout','worker_failed'}
CONFIG=ROOT/'configs/syn07_scalability.json'
RESULTS=ROOT/'experiments/results/synthetic/syn07_scalability'

def make_plan(cfg):
    rows=[]
    for T in cfg['event_count_sweep']['durations']:
        rows.append({'sweep':'event_count','n_nodes':cfg['event_count_sweep']['n_nodes'],'T':T,'path':f'event_count/T_{T}'})
    for n in cfg['node_count_sweep']['n_nodes']:
        rows.append({'sweep':'node_count','n_nodes':n,'T':cfg['node_count_sweep']['duration'],'path':f'node_count/N_{n}'})
    return rows

def initial_parameters(task,cfg):
    rng=np.random.default_rng(cfg['initialization_seed']);n=task['n_nodes']
    b=rng.uniform(.1,.5,n)[:,None]
    a=rng.uniform(0,.2,(n,n));mask=~np.eye(n,dtype=bool)
    a=np.where(mask,np.maximum(a,1e-8),0)
    h={tuple(cfg['hyperedge']):float(rng.uniform(.05,.4))}
    return b,a,h,mask

def build_features(events,task,cfg):
    return EventFeatures(events,task['n_nodes'],[tuple(cfg['hyperedge'])],cfg['beta'],cfg['delta'],
                         0,task['T'],BaselineBasis(),include_start=False)

def cached_iteration(f,b,a,h,mask,penalty):
    """The production E/M update plus objective; checked against run_em(n_iter=1).

    Does not include feature preparation or adaptive stopping checks. Each timed
    call receives the same starting state; it is not a sequential fit trajectory.
    """
    stats=EStep.compute_sufficient(f,b,a,h)
    db,da,dh=f.background_exposure,f.pair_exposure[:,None],f.hyper_exposure+penalty
    new_b=np.maximum(np.divide(stats['background_counts'],db,out=np.zeros_like(b),where=db>0),f.basis.lower_bounds(f.n_nodes))
    new_a=np.divide(stats['pairwise_counts'],da,out=np.zeros_like(a),where=da>0);new_a[~mask]=0
    weights=np.divide(stats['hyperedge_counts'],dh,out=np.zeros_like(dh),where=dh>1e-12)
    weights[f.hyper_exposure<=1e-12]=0
    new_h=dict(zip(f.edges,map(float,weights)))
    score=f.score(new_b,new_a,new_h)
    return new_b,new_a,new_h,score['loglik']-penalty*float(weights.sum())

def complete_fit(events,task,cfg):
    b,a,h,mask=initial_parameters(task,cfg)
    em=run_em(events,task['T'],task['n_nodes'],list(h),ExponentialKernel(cfg['beta']),HyperedgeAnchor(cfg['delta']),
        b[:,0],a,h,hyper_update='free',lambda_l1=cfg['lambda_l1'],pairwise_mask=mask,positive_start=True,
        include_start=False,track_loglik=True,assert_ascent=True,**cfg['em'])
    f=build_features(events,task,cfg)
    ref=refine_likelihood(f,em.baseline_coefficients,em.alpha_pairwise,em.alpha_hyper,mask,
        lambda_l1=cfg['lambda_l1'],max_sweeps=cfg['refinement_max_sweeps'],
        **{k:cfg['em'][k] for k in ['kkt_tol','convergence_tol','consecutive_convergence','ascent_tol']})
    return em,ref

def fit_diagnostics(em,ref,cfg):
    h=ref['alpha_hyper'];basis=BaselineBasis()
    all_objectives=np.r_[em.objective_history,ref['objective_history']]
    minimum=float(np.diff(all_objectives).min(initial=0))
    qualified=bool(ref['converged'] and ref['kkt_residual']<=cfg['em']['kkt_tol'] and minimum>=-cfg['em']['ascent_tol'])
    return {'fit_qualified':qualified,'em_converged':em.converged,'em_iterations':em.n_iter,
        'em_kkt':em.kkt_residual,'coordinate_sweeps':ref['n_sweeps'],'final_kkt':ref['kkt_residual'],
        'final_objective':ref['final_objective'],'last_relative_increments':ref['relative_increment_history'][-5:],
        'minimum_objective_increment':minimum,'em_ascent_violations':em.ascent_violations,
        'final_parameters':parameter_record(TemporalBaseline(ref['baseline_coefficients'],basis),ref['alpha_pairwise'],h,list(h)),
        'feature_bytes':em.feature_bytes,'coordinate_design_bytes':ref['design_bytes']}

def measure_phase(phase,events,task,cfg,features=None):
    # Explicit garbage collection is outside the timer. Imports, event loading
    # and record formatting are excluded from all three measurements.
    b,a,h,mask=initial_parameters(task,cfg)
    gc.collect()
    started=time.perf_counter_ns()
    if phase=='features':
        result=build_features(events,task,cfg)
    elif phase=='iteration':
        result=cached_iteration(features,b,a,h,mask,cfg['lambda_l1'])
    elif phase=='fit':
        result=complete_fit(events,task,cfg)
    else:raise ValueError('unknown timing phase')
    elapsed=(time.perf_counter_ns()-started)/1e9
    if not np.isfinite(elapsed) or elapsed<=0:raise RuntimeError('invalid elapsed time')
    row={'status':'completed','seconds':elapsed,'peak_process_working_set_bytes':process_peak_bytes()}
    if phase=='features':row['feature_bytes']=result.nbytes
    elif phase=='iteration':
        row['objective_after_one_update']=result[3];row['feature_bytes']=features.nbytes
        row['parameters_after_one_update']=parameter_record(TemporalBaseline(result[0],BaselineBasis()),result[1],result[2],list(result[2]))
    else:row.update(fit_diagnostics(*result,cfg))
    return row

def repetition_paths(directory,count):
    return [directory/f'repeat_{i:02d}.json' for i in range(1,count+1)]

def phase_finished(directory,count):
    return all(p.exists() and read_json(p).get('status') in TERMINAL for p in repetition_paths(directory,count))

def worker(record_path):
    request=read_json(record_path);directory=record_path.parent;cfg=request['config'];task=request['task'];phase=request['phase']
    with np.load(directory.parent/'events.npz',allow_pickle=False) as saved:
        events=list(zip(saved['times'].tolist(),saved['nodes'].tolist()))
    features=build_features(events,task,cfg) if phase=='iteration' else None
    stamp=timestamp()
    try:
        warm=measure_phase(phase,events,task,cfg,features)
        warm.update(excluded_from_timing_summary=True,phase=phase)
        write_json(directory/'warmups'/f'{stamp}.json',warm)
        for index,path in enumerate(repetition_paths(directory,cfg['repetitions']),1):
            if path.exists() and read_json(path).get('status') in TERMINAL:continue
            base={'phase':phase,'task':task,'repetition':index,'n_events':len(events),'package_version':package_version()}
            write_json(path,{**base,'status':'running'})
            try:result=measure_phase(phase,events,task,cfg,features)
            except Exception as exc:
                result={'status':'failed','error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
            write_json(path,{**base,**result})
    except Exception as exc:
        write_json(directory/'worker_error.json',{'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        return 1
    return 0

def ensure_dataset(directory,task,cfg):
    directory.mkdir(parents=True,exist_ok=True)
    meta=directory/'dataset.json';path=directory/'events.npz'
    if meta.exists():
        saved=read_json(meta)
        if saved['task']!=task or saved['config']!=cfg:raise ValueError('saved dataset configuration differs')
        if not saved['simulation']['completed']:return False
        if not path.exists():raise FileNotFoundError(path)
        return True
    n=task['n_nodes'];a=np.zeros((n,n));a[cfg['pairwise_source'],cfg['pairwise_target']]=cfg['pairwise_weight']
    sim=HawkesSimulator(np.full(n,cfg['baseline_rate']),a,{tuple(cfg['hyperedge']):cfg['hyperedge_weight']},
                        ExponentialKernel(cfg['beta']),HyperedgeAnchor(cfg['delta']))
    generated=sim.simulate(task['T'],seed=cfg['simulation_seed'],max_events=cfg['max_events'],return_info=True)
    arr=np.asarray(generated.events).reshape(-1,2)
    np.savez_compressed(path,times=arr[:,0],nodes=arr[:,1].astype(int))
    info=generated.as_dict();info.pop('events');info.pop('proposal_trace')
    b,p,h,_=initial_parameters(task,cfg)
    write_json(meta,{'task':task,'config':cfg,'n_events':len(arr),'simulation':info,
        'initial_parameters':parameter_record(TemporalBaseline(b,BaselineBasis()),p,h,list(h))})
    return generated.completed

def run_phase(directory,task,cfg,phase):
    directory.mkdir(parents=True,exist_ok=True)
    if phase_finished(directory,cfg['repetitions']):return
    record=directory/'request.json'
    write_json(record,{'task':task,'config':cfg,'phase':phase})
    with (directory/'worker.log').open('a',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--worker',str(record)],
            cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=cfg['phase_timeout_seconds'])
        except subprocess.TimeoutExpired:
            child.kill();child.wait();code='timeout'
        except KeyboardInterrupt:
            child.terminate();child.wait();raise
    if not phase_finished(directory,cfg['repetitions']):
        for index,p in enumerate(repetition_paths(directory,cfg['repetitions']),1):
            if p.exists() and read_json(p).get('status') in TERMINAL:continue
            write_json(p,{'task':task,'phase':phase,'repetition':index,'status':'timeout' if code=='timeout' else 'worker_failed',
                          'returncode':code,'seconds':None})

def describe_times(rows):
    values=[r['seconds'] for r in rows if r.get('status')=='completed' and r.get('seconds') is not None]
    if not values:return {'n_measured':0,'median_seconds':None,'q25_seconds':None,'q75_seconds':None}
    return {'n_measured':len(values),'median_seconds':float(np.median(values)),
            'q25_seconds':float(np.quantile(values,.25)),'q75_seconds':float(np.quantile(values,.75)),
            'min_seconds':min(values),'max_seconds':max(values)}

def summarise(root,plan,cfg):
    summary=[];counts={};all_rows=[]
    for task in plan:
        folder=root/task['path']
        dataset=read_json(folder/'dataset.json') if (folder/'dataset.json').exists() else None
        for phase in PHASES:
            rows=[read_json(p) for p in repetition_paths(folder/phase,cfg['repetitions']) if p.exists()]
            all_rows+=rows
            values=describe_times(rows)
            peak=[r.get('peak_process_working_set_bytes') for r in rows if r.get('peak_process_working_set_bytes') is not None]
            summary.append({**task,'phase':phase,'n_events':dataset['n_events'] if dataset else None,**values,
                'planned_repetitions':cfg['repetitions'],'qualified_fits':sum(r.get('fit_qualified',False) for r in rows) if phase=='fit' else None,
                'max_process_peak_bytes':max(peak) if peak else None,'status_counts':{status:sum(r.get('status')==status for r in rows) for status in sorted({r.get('status') for r in rows})},
                'source_records':[p.relative_to(root).as_posix() for p in repetition_paths(folder/phase,cfg['repetitions']) if p.exists()]})
    for row in all_rows:counts[row['status']]=counts.get(row['status'],0)+1
    slopes={}
    for phase in PHASES:
        use=[r for r in summary if r['sweep']=='event_count' and r['phase']==phase and r['n_measured']==cfg['repetitions']
             and r['n_events'] and (phase!='fit' or r['qualified_fits']==cfg['repetitions'])]
        use.sort(key=lambda r:r['n_events'])
        slopes[phase]={'eligible_settings':len(use),'main_slope':None,'last_three_sensitivity_slope':None,
                       'scope':'full event-count grid required; full-fit slope also requires every fit qualified'}
        if len(use)==len(cfg['event_count_sweep']['durations']):
            x=np.log([r['n_events'] for r in use]);y=np.log([r['median_seconds'] for r in use])
            slopes[phase].update(main_slope=float(np.polyfit(x,y,1)[0]),last_three_sensitivity_slope=float(np.polyfit(x[-3:],y[-3:],1)[0]))
    all_completed=len(all_rows)==len(plan)*len(PHASES)*cfg['repetitions'] and all(r['status']=='completed' for r in all_rows)
    all_qualified=all_completed and all(r.get('fit_qualified',True) for r in all_rows)
    return {'status':'completed' if all_qualified else 'completed_with_issues','all_records_present':len(all_rows)==len(plan)*3*cfg['repetitions'],
        'all_fit_attempts_qualified':all_qualified,'status_counts':counts,'planned_measurements':len(plan)*3*cfg['repetitions'],
        'rows':summary,'event_count_slopes':slopes,'repetition_interpretation':'Ten timing repeats of the same data and starting state; IQR measures runtime variability, not statistical estimation uncertainty.'}

def snapshot(root):
    dest=root/'code'
    if dest.exists():return
    for folder in ['models','inference','simulation']:
        for src in (ROOT/folder).glob('*.py'):
            target=dest/src.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    for name in ['experiments/syn07_scalability.py','experiments/syn07_plot_scalability.py','experiments/synthetic_pipeline.py',
                 'experiments/realdata_pipeline.py','configs/syn07_scalability.json','docs/code_manifest.json','docs/scalability.md']:
        src=ROOT/name
        if src.exists():
            target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)

def package_results(root,destination):
    destination.mkdir(parents=True,exist_ok=True)
    path=destination/'hypergraph_hawkes_results.zip'
    if path.exists():
        archived=destination/'archive/hypergraph_hawkes'/timestamp()/path.name
        archived.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,archived)
    temporary=path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for src in sorted(root.rglob('*')):
            if src.is_file() and src.suffix!='.tmp':z.write(src,'experiments/results/synthetic/syn07_scalability/'+src.relative_to(root).as_posix())
    temporary.replace(path)
    return path

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--results-root',type=Path,default=RESULTS)
    parser.add_argument('--output-dir',type=Path,default=Path.home()/'Downloads')
    args=parser.parse_args(argv)
    if args.worker:return worker(args.worker)
    cfg=read_json(CONFIG);plan=make_plan(cfg)
    if args.plan_only:
        print(json.dumps({'tasks':plan,'settings':cfg,'phases':PHASES,'measured_runs':len(plan)*3*cfg['repetitions'],
                          'full_fit_attempts':len(plan)*cfg['repetitions'],'excluded_full_fit_warmups':len(plan)},ensure_ascii=False,indent=2));return 0
    root=args.results_root;root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists() and read_json(root/'config.json')!=cfg:raise ValueError('saved benchmark configuration differs')
    write_json(root/'config.json',cfg);write_json(root/'plan.json',plan);snapshot(root)
    import scipy
    environment={'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,
        'platform':platform.platform(),'processor':platform.processor(),'package_version':package_version(),
        'thread_environment':{k:os.environ[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},
        'memory_scope':'Peak worker process working set, including imports, fixed cached data and warmup; not a stage-only allocation delta.'}
    write_json(root/'environment.json',environment)
    started=time.perf_counter();stamp=timestamp();run={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running'}
    try:
        for i,task in enumerate(plan,1):
            print(f"[{i}/{len(plan)}] {task['path']}：预计算、单轮更新、完整拟合各预热1次并计时10次。",flush=True)
            directory=root/task['path']
            if not ensure_dataset(directory,task,cfg):continue
            for phase in PHASES:
                run_phase(directory/phase,task,cfg,phase)
                write_json(root/'summary.json',summarise(root,plan,cfg))
        run['status']=summarise(root,plan,cfg)['status']
    except KeyboardInterrupt:
        run['status']='interrupted';print('已中断，保留已完成记录并打包。',flush=True)
    except Exception as exc:
        run.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        run.update(elapsed_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(root/'runs'/f'{stamp}.json',run)
        write_json(root/'summary.json',summarise(root,plan,cfg))
        package=package_results(root,args.output_dir)
        print(f"状态：{run['status']}；本次耗时 {run['elapsed_seconds']:.1f} 秒。\n结果包：{package}",flush=True)
    return 0 if run['status']=='completed' else 2

if __name__=='__main__':raise SystemExit(main())
