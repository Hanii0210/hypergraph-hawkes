"""Run bounded implementation checks and always package their report."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib.metadata
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

ROOT=Path(__file__).resolve().parents[1]


def downloads_directory():
    if os.name=='nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders') as key:
                value=winreg.QueryValueEx(key,'{374DE290-123F-4565-9164-39C4925E467B}')[0]
            return Path(os.path.expandvars(value))
        except OSError:
            pass
    return Path.home()/'Downloads'


def peak_memory(process):
    """Windows child peak working set; unavailable measurements stay null."""
    if os.name!='nt':return None
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
                (name,ctypes.c_size_t) for name in ['PeakWorkingSetSize','WorkingSetSize',
                'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
        counter=Counters()
        counter.cb=ctypes.sizeof(counter)
        api=ctypes.WinDLL('psapi').GetProcessMemoryInfo
        api.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        api.restype=wintypes.BOOL
        if api(wintypes.HANDLE(int(process._handle)),ctypes.byref(counter),counter.cb):
            return int(counter.PeakWorkingSetSize)
    except (AttributeError,OSError,ValueError):
        pass
    return None


def run_one(args,timeout,log):
    env=os.environ.copy()
    env['PYTHONIOENCODING']='utf-8'
    env['PYTHONDONTWRITEBYTECODE']='1'
    started=time.perf_counter()
    command=[sys.executable,'-X','utf8',*args]
    with subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace') as process:
        try:
            output,_=process.communicate(timeout=timeout)
            status='passed' if process.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:
            process.kill()
            output,_=process.communicate()
            status='timeout'
        except KeyboardInterrupt:
            process.kill()
            output,_=process.communicate()
            log.write(output)
            log.flush()
            raise
        peak=peak_memory(process)
        log.write('\nCOMMAND: '+' '.join(command)+'\n'+output+'\n')
        log.flush()
        return {'command':args,'status':status,'returncode':process.returncode,
                'elapsed_seconds':time.perf_counter()-started,'peak_working_set_bytes':peak}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deterministic-only',action='store_true',help='Skip stochastic simulator unit checks; this is not the complete user check.')
    parser.add_argument('--timeout-seconds',type=float,default=180)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'experiments/results/checks')
    parser.add_argument('--download-dir',type=Path,default=None)
    options=parser.parse_args()
    if options.timeout_seconds<=0:parser.error('timeout must be positive')
    destination=downloads_directory() if options.download_dir is None else options.download_dir
    destination.mkdir(parents=True,exist_ok=True)
    out=options.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report_path,log_path=out/'model_checks.json',out/'model_checks.log'
    for old in [report_path,log_path]:
        if old.exists():
            archived=ROOT/'archive/model_checks'/stamp/old.name
            archived.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(old,archived)
    report={'schema_version':1,'purpose':'implementation_checks_not_scientific_results',
            'started_utc':datetime.now(timezone.utc).isoformat(),'project':str(ROOT),
            'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'cpu_count':os.cpu_count(),'deterministic_only':options.deterministic_only,
            'checks':[],'status':'running'}
    started=time.perf_counter()
    manifest_path=ROOT/'docs/code_manifest.json'
    exit_code=1
    with log_path.open('w',encoding='utf-8') as log:
        try:
            report['package_id']='unrecorded'
            if manifest_path.is_file():
                try:
                    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
                    report['package_id']=manifest.get('package_id','unrecorded')
                except (OSError,ValueError,AttributeError) as exc:
                    report['version_record_note']=str(exc)
            report['dependencies']={name:importlib.metadata.version(name) for name in ['numpy','scipy']}
            commands=[['-m','unittest','discover','-s','tests','-p',name] for name in [
                'test_temporal_baseline.py','test_history_features.py','test_em_free.py','test_coordinate.py',
                'test_synthetic_pipeline.py','test_realdata_experiments.py','test_scalability.py']]
            commands += [['tests/'+name] for name in ['test_em_interval.py','test_likelihood_interval.py',
                'test_estep.py','test_mstep.py','test_kernel.py','test_tensor.py']]
            if not options.deterministic_only:
                commands.append(['-m','unittest','discover','-s','tests','-p','test_simulator_status.py'])
            for index,args in enumerate(commands,1):
                print(f'[{index}/{len(commands)}] {args[-1]}',flush=True)
                result=run_one(args,options.timeout_seconds,log)
                report['checks'].append(result)
                print(f"  {result['status']} ({result['elapsed_seconds']:.2f} s)",flush=True)
                report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            passed=all(row['status']=='passed' for row in report['checks'])
            report['status']='passed' if passed else 'failed'
            report['complete_check_passed']=passed and not options.deterministic_only
            exit_code=0 if passed else 1
        except KeyboardInterrupt:
            report['status']='interrupted'
            report['complete_check_passed']=False
            exit_code=130
            log.write(traceback.format_exc())
        except Exception as exc:
            report['status']='failed'
            report['complete_check_passed']=False
            report['exception']={'type':type(exc).__name__,'message':str(exc)}
            log.write(traceback.format_exc())
        finally:
            report['elapsed_seconds']=time.perf_counter()-started
            report['finished_utc']=datetime.now(timezone.utc).isoformat()
            report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            log.flush()
    package=destination/'hypergraph_hawkes_results.zip'
    if package.exists():
        archived=destination/'archive/hypergraph_hawkes'/stamp/package.name
        archived.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(package,archived)
    temporary=destination/'hypergraph_hawkes_results.zip.tmp'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path,'experiments/results/checks/model_checks.json')
        archive.write(log_path,'experiments/results/checks/model_checks.log')
        if manifest_path.is_file():archive.write(manifest_path,'docs/code_manifest.json')
        if (ROOT/'requirements.txt').is_file():archive.write(ROOT/'requirements.txt','requirements.txt')
    temporary.replace(package)
    print(f"\n状态：{report['status']}；完整检查通过：{report['complete_check_passed']}")
    print(f'项目报告：{report_path}')
    print(f'Results archive: {package}')
    return exit_code


if __name__=='__main__':raise SystemExit(main())
