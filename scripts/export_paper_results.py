"""Export completed study results to the manuscript plotting/table format.

Input: result directories or the corresponding study ZIP files.
No simulation, fitting, filtering by predictive performance or raw-data access.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.synthetic_pipeline import restore_parameters, true_parameters


class Results:
    def __init__(self, path, prefix):
        self.path = Path(path)
        self.zip = ZipFile(self.path) if self.path.is_file() else None
        self.prefix = prefix.rstrip('/')+'/'

    def read(self, relative):
        if self.zip:
            return json.loads(self.zip.read(self.prefix+relative))
        return json.loads((self.path/relative).read_text(encoding='utf-8'))

    def close(self):
        if self.zip:
            self.zip.close()


def qualified(record, label):
    if record.get('status') != 'qualified':
        raise ValueError(f'{label}: expected a qualified result, got {record.get("status")}')


def synthetic(source):
    meta = 'synthetic_experiments/formal/'
    plan = source.read(meta+'plan.json')['tasks']
    cfg = source.read(meta+'config.json')
    rows = []
    if len(plan) != 1100:
        raise ValueError('The paper export requires the full 1100-path formal design.')
    for task in plan:
        spec = task['scenario']
        _, true_pair, true_hyper = true_parameters(cfg, spec)
        for model in spec['models']:
            fit = source.read(task['path']+'/models/'+model+'/fit.json')
            qualified(fit, task['path']+'/'+model)
            _, pair, hyper = restore_parameters(fit['parameters'])
            pair_mse = float(np.mean((pair-true_pair)[~np.eye(cfg['n_nodes'], dtype=bool)]**2))
            rows.append(dict(dataset_id=task['dataset_id'], experiment=spec['experiment'],
                scenario=spec['name'], repeat=task['repeat'], n=spec['formal_repetitions'],
                T=spec['T'], truth=spec['triple_weight'], background=spec['background'],
                model=model, test_loglik=fit['test_score']['loglik'],
                test_duration=fit['test_score']['duration'], pairwise_mse=pair_mse,
                baseline_train_mse=fit['baseline_errors']['training']['mse_mean'],
                baseline_test_mse=fit['baseline_errors']['test']['mse_mean'],
                triple_estimates=[hyper.get(tuple(e)) for e in cfg['true_triples']],
                decoy_estimates=[hyper.get(tuple(e)) for e in cfg['decoy_triples']],
                triple_sum=sum(v for e,v in hyper.items() if len(e)==3) if model.endswith('M2') else None,
                pair_completion_mse=float(np.mean([(v-true_hyper.get(e,0))**2
                    for e,v in hyper.items() if len(e)==2])) if hyper else None))
    if len(rows) != 4200:
        raise ValueError('The paper export requires 4200 formal model fits.')
    return {'synthetic_replicates.json':rows}


def neural(source):
    plan = source.read('realdata_experiments/full/plan.json')['tasks']
    if len(plan) != 15:
        raise ValueError('The paper export requires all 15 neural windows.')
    windows, fits = [], []
    short_names = {'ret1':'ret-1', 'pvc3':'PVC-3', 'pvc11':'PVC-11'}
    for task in plan:
        result = source.read(task['path']+'/result.json')
        dataset = source.read(task['path']+'/dataset.json')
        qualified(result, task['path'])
        fits.append(result)
        models = result['models']
        windows.append(dict(dataset=task['dataset_id'], window_index=task['window_index'],
            short_name=short_names[task['dataset']['loader']], status=result['status'],
            intervals=dataset['intervals'], selected_mapping=dataset['mapping'],
            candidate_prefixes=result['candidates']['prefixes'],
            primary_comparison=result['comparisons']['top3'],
            all_comparisons=result['comparisons'],
            V_H2_top3_minus_C_H2_top3=models['V-H2-top3']['test_score']['loglik']
                                     -models['C-H2-top3']['test_score']['loglik']))
    return {'realdata_analysis.json':{'windows':windows}, 'realdata_fits.json':fits}


def timing(source):
    summary = source.read('summary.json')
    if not summary['all_records_present'] or not summary['all_fit_attempts_qualified']:
        raise ValueError('The paper export requires a complete qualified timing grid.')
    rows, settings, slopes = summary['rows'], [], {}
    seen = set()
    for row in rows:
        if row['path'] not in seen:
            settings.append({k:row[k] for k in ('path','sweep','n_nodes','T','n_events')})
            seen.add(row['path'])
    for phase in ('features','iteration','fit'):
        subset = sorted([r for r in rows if r['sweep']=='event_count' and r['phase']==phase],
                        key=lambda r:r['n_events'])
        x = np.log([r['n_events'] for r in subset])
        y = np.log([r['median_seconds'] for r in subset])
        slope, intercept = np.polyfit(x,y,1)
        slopes[phase] = dict(event_counts=[r['n_events'] for r in subset],
            median_seconds=[r['median_seconds'] for r in subset], main_slope=float(slope),
            r_squared_log_scale=1-float(np.sum((y-(slope*x+intercept))**2)/np.sum((y-y.mean())**2)),
            last_three_sensitivity_slope=float(np.polyfit(x[-3:],y[-3:],1)[0]))
    environment = source.read('environment.json')
    environment.pop('executable',None)
    return {'timing_summary.json':summary,
            'timing_analysis.json':dict(settings=settings, rows=rows, slopes=slopes, environment=environment)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic',type=Path,default=ROOT/'experiments/results/synthetic')
    parser.add_argument('--realdata',type=Path,default=ROOT/'experiments/results/realdata/baseline')
    parser.add_argument('--timing',type=Path,default=ROOT/'experiments/results/synthetic/syn07_scalability')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reproduced_paper')
    args = parser.parse_args()
    sources = [(synthetic,args.synthetic,'experiments/results/synthetic'),
               (neural,args.realdata,'experiments/results/realdata/baseline'),
               (timing,args.timing,'experiments/results/synthetic/syn07_scalability')]
    outputs = {}
    for export,path,prefix in sources:
        source = Results(path,prefix)
        try:
            outputs.update(export(source))
        finally:
            source.close()
    directory = args.output_dir.resolve()
    (directory/'data').mkdir(parents=True, exist_ok=True)
    for name,data in outputs.items():
        (directory/'data'/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    for name in ('make_figures.py','make_tables.py'):
        shutil.copyfile(ROOT/'paper'/name,directory/name)
    print(f'Exported 4200 simulation fits, 15 neural windows and 9 timing settings to {directory}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
