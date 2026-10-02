"""Run user-defined E/U test cases from a JSON config.

Replications are parallelized by worker *processes* inside onebit_bai.experiment.
Each case checkpoints raw trials.csv after every completed replication and can
be resumed safely with --resume.
"""
from __future__ import annotations
import argparse, json, math, subprocess, sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _validate(c):
    means=[float(x) for x in c['means']]
    if len(means)<2: raise ValueError('Need K>=2 means.')
    if max(means)==sorted(means)[-2]: raise ValueError('Need a unique best arm.')
    if not all(0<x<1 for x in means):
        raise ValueError('This combined E/U runner expects shared means in (0,1).')
    if not (0<float(c.get('beta',.5))<1): raise ValueError('beta must be in (0,1).')
    for key in ['Q1','Q3','correctness','nonasymptotic']:
        if key not in c.get('experiments',{}): raise ValueError(f'Missing experiments.{key}')
    return means


def _common(means): return ['--means',*map(str,means)]


def _case_flags(c, model, kind):
    means=[float(x) for x in c['means']]
    beta=str(c.get('beta',.5))
    models=c.get('models',{})
    num=c.get('numerics',{})
    e=models.get('E',{}); u=models.get('U',{})
    en=num.get('E',{}); un=num.get('U',{})
    if model=='E':
        base=['--model','E','--e-family',e.get('family','gaussian'),*_common(means),'--sigma',str(e.get('sigma',.2)),
              '--beta',beta,'--qpoints',str(en.get('qpoints',31)),'--xgrid',str(en.get('xgrid',81)),
              '--oracle-refresh',str(en.get('oracle_refresh',25)),'--refresh',str(en.get('refresh',1)),
              '--decide',str(en.get('decide',1))]
    else:
        base=['--model','U','--u-generator',u.get('generator','beta'),*_common(means),
              '--concentration',str(u.get('concentration',12.0)),'--beta',beta,
              '--xgrid',str(un.get('xgrid',61)),'--qquad',str(un.get('qquad',301)),
              '--refresh',str(un.get('refresh',5)),'--decide',str(un.get('decide',5))]
    exp=c['experiments'][kind]
    flags=[*base,'--deltas',*map(str,exp['deltas'])]
    if kind=='Q1':
        if model=='E': flags += ['--no-companion','--e-boundary',exp.get('E_boundary','certified'),'--trace-every',str(exp.get('trace_every',1000))]
        else: flags += ['--skip-safe-bit','--u-boundary',exp.get('U_boundary','practical'),'--trace-every',str(exp.get('trace_every',1000))]
    elif kind=='Q3':
        flags += ['--full']
        if model=='E': flags += ['--no-companion','--e-boundary',exp.get('E_boundary','practical'),'--full-boundary',exp.get('full_boundary','practical')]
        else: flags += ['--skip-safe-bit','--u-boundary',exp.get('U_boundary','practical'),'--full-boundary',exp.get('full_boundary','practical')]
    elif kind=='correctness':
        if model=='E':
            flags += ['--no-companion','--e-boundary',exp.get('E_boundary','certified')]
        else:
            flags += ['--u-boundary',exp.get('U_boundary','practical'),
                      '--u-methods',exp.get('U_methods','raw')]
    elif kind=='nonasymptotic':
        if model=='E': flags += ['--e-boundary',exp.get('E_boundary','certified')]
        else: flags += ['--skip-safe-bit','--u-boundary',exp.get('U_boundary','certified')]
    return flags


def _enabled(c, model): return bool(c.get('models',{}).get(model,{}).get('enabled',True))


def _run_case(c, case_name, model, kind, cases_root, workers, resume, do_run):
    exp=c['experiments'][kind]
    reps=int(exp.get(f'{model}_reps',exp.get('reps',100)))
    cap=int(exp.get(f'{model}_cap',exp.get('cap',100000)))
    out=cases_root/case_name
    flags=_case_flags(c,model,kind)
    cmd=[sys.executable,'-m','onebit_bai.experiment',*flags,'--reps',str(reps),'--cap',str(cap),
         '--workers',str(workers),'--output',str(out),'--verbose']
    if resume: cmd.append('--resume')
    eta=float(c['experiments']['nonasymptotic'].get('eta',.05))
    report=[sys.executable,'-m','onebit_bai.focused_report','--input',str(out),'--eta',str(eta)]
    print(f'\n[{case_name}] model={model} kind={kind} reps={reps} cap={cap:,} workers={workers}')
    print('EXPERIMENT:',' '.join(map(str,cmd)))
    if not do_run: return 0,cmd
    out.mkdir(parents=True,exist_ok=True)
    rc=subprocess.call(cmd,cwd=ROOT)
    if rc:return rc,cmd
    rc=subprocess.call(report,cwd=ROOT)
    return rc,cmd


def _report_only(suite,eta):
    cases=suite/'cases'
    for child in sorted(cases.iterdir()) if cases.exists() else []:
        if (child/'trials.csv').exists():
            rc=subprocess.call([sys.executable,'-m','onebit_bai.focused_report','--input',str(child),'--eta',str(eta)],cwd=ROOT)
            if rc:return rc
    return subprocess.call([sys.executable,'-m','onebit_bai.combined_report','--cases-root',str(cases),'--output',str(suite),'--eta',str(eta)],cwd=ROOT)



def _physical_cores():
    try:
        pairs=set()
        phys=core=None
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('physical id'):
                phys=int(line.split(':',1)[1])
            elif line.startswith('core id'):
                core=int(line.split(':',1)[1])
            elif not line.strip() and phys is not None and core is not None:
                pairs.add((phys,core)); phys=core=None
        if phys is not None and core is not None: pairs.add((phys,core))
        if pairs: return len(pairs)
    except Exception:
        pass
    import os
    return max(1,(os.cpu_count() or 2)//2)

def main():
    p=argparse.ArgumentParser(description='User-defined E/U experiment suite with multicore replication parallelism.')
    p.add_argument('--config',required=True)
    p.add_argument('--workers',type=int,default=None,help='Worker processes. Override config workers.')
    p.add_argument('--run',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--report-only',action='store_true',help='Regenerate CSV summaries/figures from existing trials.csv without Monte Carlo.')
    p.add_argument('--output-root',default=None)
    p.add_argument('--only',nargs='*',choices=['Q1','Q3','correctness','nonasymptotic'],default=None)
    a=p.parse_args()
    c=_load(a.config); means=_validate(c)
    name=c.get('name','custom')
    workers=int(a.workers if a.workers is not None else c.get('workers',1))
    if workers==0:
        workers=max(1,_physical_cores())
        print(f'AUTO WORKERS: using {workers} physical-core worker processes')
    if workers<1: raise ValueError('workers must be >=0 (0 = auto physical cores)')
    eta=float(c['experiments']['nonasymptotic'].get('eta',.05))
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    suite=(ROOT/(a.output_root or f'custom_runs/{stamp}_{name}')).resolve()
    cases_root=suite/'cases'; cases_root.mkdir(parents=True,exist_ok=True)
    (suite/'CUSTOM_CASE.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    meta={'name':name,'means':means,'K':len(means),'workers':workers,'eta':eta,
          'Q1_L':[math.log(1/float(d)) for d in c['experiments']['Q1']['deltas']],
          'data_policy':'Each case atomically checkpoints cases/<CASE>/trials.csv after every completed replication; use --resume with the same output-root.',
          'parallelism':'Independent replications use worker processes; cases run sequentially to avoid oversubscription.'}
    (suite/'CUSTOM_RUN_META.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('SUITE:',suite)
    print('CUSTOM CASE:',name,'K=',len(means),'means=',means)
    print('WORKERS:',workers)
    print('Q1 L GRID:',', '.join(f'{x:.2f}' for x in meta['Q1_L']))
    if a.report_only:
        return _report_only(suite,eta)
    selected=set(a.only or ['Q1','Q3','correctness','nonasymptotic'])
    commands=[]
    plan=[('Q1','E_Q1','U_Q1'),('Q3','E_Q3','U_Q3'),('correctness','E_CORRECTNESS','U_CORRECTNESS'),('nonasymptotic','E_NONASYM','U_NONASYM_CERTIFIED')]
    for kind,ecase,ucase in plan:
        if kind not in selected or not c['experiments'][kind].get('enabled',True): continue
        for model,case in [('E',ecase),('U',ucase)]:
            if not _enabled(c,model): continue
            rc,cmd=_run_case(c,case,model,kind,cases_root,workers,a.resume,a.run)
            commands.append(' '.join(map(str,cmd)))
            if rc:
                print('FAILED:',case,'Completed raw data remain checkpointed in',cases_root)
                return rc
    (suite/'commands.txt').write_text('\n'.join(commands)+'\n',encoding='utf-8')
    report=[sys.executable,'-m','onebit_bai.combined_report','--cases-root',str(cases_root),'--output',str(suite),'--eta',str(eta)]
    print('\nCOMBINED REPORT:',' '.join(map(str,report)))
    if not a.run:
        print('\nPREVIEW ONLY. Add --run to execute.')
        return 0
    rc=subprocess.call(report,cwd=ROOT)
    if rc:return rc
    print('\nDONE. Reusable data:')
    print(' ',suite/'all_trials_combined.csv')
    print(' ',cases_root/'<CASE>'/'trials.csv')
    return 0

if __name__=='__main__': raise SystemExit(main())
