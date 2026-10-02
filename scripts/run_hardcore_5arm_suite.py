"""Paper-scale/hardcore K=5 experiment suite for the AISTATS one-bit project.

The primary instance mirrors the CLUSTER-CLOSE5 mean vector used in the
Bernoulli non-asymptotic reference, but runs it in both:
  * Model E: Gaussian one-parameter exponential family;
  * Model U: Beta laws supported on [0,1].

Raw data are checkpointed after every complete replication. Re-running with
--resume and the same --output-root continues from the existing trials.csv.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

INSTANCES={
    'sep5':[.70,.50,.40,.30,.20],
    'cluster5':[.70,.64,.63,.62,.61],
    # K=5 extension of the equal-close stress idea: unique best, tied challengers.
    'tie5':[.60,.56,.56,.56,.56],
}

# Replication counts deliberately mirror the scale of the Bernoulli reference:
# 200 per calibration setting and 600 for the stress run in the paper profile.
PROFILES={
    'pilot':{
        'E_Q1':10,'U_Q1':8,'E_Q3':10,'U_Q3':8,
        'E_CORRECTNESS':40,'U_CORRECTNESS':40,
        'E_NONASYM':20,'U_NONASYM_CERTIFIED':8,'U_NONASYM_SHAPE':12,
        'U_CORRECTNESS_CERTIFIED_RAW':8,
    },
    'hardcore':{
        'E_Q1':100,'U_Q1':80,'E_Q3':100,'U_Q3':60,
        'E_CORRECTNESS':300,'U_CORRECTNESS':300,
        'E_NONASYM':150,'U_NONASYM_CERTIFIED':30,'U_NONASYM_SHAPE':100,
        'U_CORRECTNESS_CERTIFIED_RAW':30,
    },
    'paper':{
        'E_Q1':200,'U_Q1':200,'E_Q3':200,'U_Q3':200,
        'E_CORRECTNESS':600,'U_CORRECTNESS':600,
        'E_NONASYM':400,'U_NONASYM_CERTIFIED':80,'U_NONASYM_SHAPE':200,
        'U_CORRECTNESS_CERTIFIED_RAW':80,
    },
}

CAPS={
    'pilot':{
        'E_Q1':80000,'U_Q1':80000,'E_Q3':80000,'U_Q3':80000,
        'E_CORRECTNESS':50000,'U_CORRECTNESS':50000,
        'E_NONASYM':100000,'U_NONASYM_CERTIFIED':120000,'U_NONASYM_SHAPE':100000,
        'U_CORRECTNESS_CERTIFIED_RAW':120000,
    },
    'hardcore':{
        'E_Q1':450000,'U_Q1':450000,'E_Q3':450000,'U_Q3':450000,
        'E_CORRECTNESS':250000,'U_CORRECTNESS':250000,
        'E_NONASYM':450000,'U_NONASYM_CERTIFIED':500000,'U_NONASYM_SHAPE':400000,
        'U_CORRECTNESS_CERTIFIED_RAW':500000,
    },
    'paper':{
        'E_Q1':700000,'U_Q1':700000,'E_Q3':700000,'U_Q3':700000,
        'E_CORRECTNESS':400000,'U_CORRECTNESS':400000,
        'E_NONASYM':700000,'U_NONASYM_CERTIFIED':700000,'U_NONASYM_SHAPE':700000,
        'U_CORRECTNESS_CERTIFIED_RAW':700000,
    },
}

# Deep asymptotic grid. L=log(1/delta) ranges from 9.21 to 23.03.
ASYM_DELTAS=['1e-4','1e-5','1e-6','1e-7','1e-8','1e-10']
RATIO_DELTAS=['1e-4','1e-6','1e-8','1e-10']
# Correctness cannot be statistically validated at 1e-8 with a few hundred runs;
# use moderate deltas and many replications instead.
COR_DELTAS=['.1','.05','.02','.01']
FINITE_DELTAS=['.01','.001','.0001']


def common_means(means):
    return ['--means',*map(str,means)]


def cases_for(means, strict=False):
    # refresh/decide=1 is closest to the exact policy but raw-profile U can be
    # dramatically slower. The default keeps U at 5 while E is per-round.
    u_step='1' if strict else '5'
    return {
      'E_Q1': ['--model','E','--e-family','gaussian',*common_means(means),'--sigma','.2','--beta','.5',
               '--deltas',*ASYM_DELTAS,'--no-companion','--e-boundary','certified',
               '--qpoints','31','--xgrid','81','--oracle-refresh','25','--refresh','1','--decide','1',
               '--trace-every','1000'],
      'U_Q1': ['--model','U','--u-generator','beta',*common_means(means),'--concentration','12','--beta','.5',
               '--deltas',*ASYM_DELTAS,'--skip-safe-bit','--u-boundary','practical',
               '--xgrid','61','--qquad','301','--refresh',u_step,'--decide',u_step,
               '--trace-every','1000'],
      'E_Q3': ['--model','E','--e-family','gaussian',*common_means(means),'--sigma','.2','--beta','.5',
               '--deltas',*RATIO_DELTAS,'--no-companion','--full','--e-boundary','practical','--full-boundary','practical',
               '--qpoints','31','--xgrid','81','--oracle-refresh','25','--refresh','1','--decide','1'],
      'U_Q3': ['--model','U','--u-generator','beta',*common_means(means),'--concentration','12','--beta','.5',
               '--deltas',*RATIO_DELTAS,'--skip-safe-bit','--full','--u-boundary','practical','--full-boundary','practical',
               '--xgrid','61','--qquad','301','--refresh',u_step,'--decide',u_step],
      'E_CORRECTNESS': ['--model','E','--e-family','gaussian',*common_means(means),'--sigma','.2','--beta','.5',
               '--deltas',*COR_DELTAS,'--no-companion','--e-boundary','certified',
               '--qpoints','25','--xgrid','61','--oracle-refresh','25','--refresh','1','--decide','1'],
      # This case includes u_marginal, which focused_report uses for a readily runnable certified correctness check.
      'U_CORRECTNESS': ['--model','U','--u-generator','beta',*common_means(means),'--concentration','12','--beta','.5',
               '--deltas',*COR_DELTAS,'--u-boundary','practical','--xgrid','51','--qquad','241',
               '--refresh',u_step,'--decide',u_step],
      'E_NONASYM': ['--model','E','--e-family','gaussian',*common_means(means),'--sigma','.2','--beta','.5',
               '--deltas',*FINITE_DELTAS,'--e-boundary','certified','--qpoints','25','--xgrid','61',
               '--oracle-refresh','10','--refresh','1','--decide','1'],
      'U_NONASYM_CERTIFIED': ['--model','U','--u-generator','beta',*common_means(means),'--concentration','12','--beta','.5',
               '--deltas',*FINITE_DELTAS,'--skip-safe-bit','--u-boundary','certified','--xgrid','51','--qquad','241',
               '--refresh',u_step,'--decide',u_step],
      # Descriptive finite-shape run; kept separate from the theorem-certified case.
      'U_NONASYM_SHAPE': ['--model','U','--u-generator','beta',*common_means(means),'--concentration','12','--beta','.5',
               '--deltas',*FINITE_DELTAS,'--skip-safe-bit','--u-boundary','practical','--xgrid','61','--qquad','301',
               '--refresh',u_step,'--decide',u_step],
      # Exact raw-profile correctness stress: likely censored, but scientifically useful.
      'U_CORRECTNESS_CERTIFIED_RAW': ['--model','U','--u-generator','beta',*common_means(means),'--concentration','12','--beta','.5',
               '--deltas','.1','.05','--skip-safe-bit','--u-boundary','certified','--xgrid','51','--qquad','241',
               '--refresh',u_step,'--decide',u_step],
    }


def run_case(case, flags, reps, cap, cases_root, eta, do_run, resume, workers):
    out=cases_root/case
    cmd=[sys.executable,'-m','onebit_bai.experiment',*flags,'--reps',str(reps),'--cap',str(cap),
         '--output',str(out),'--verbose','--workers',str(workers)]
    if resume: cmd.append('--resume')
    report=[sys.executable,'-m','onebit_bai.focused_report','--input',str(out),'--eta',str(eta)]
    print(f'\n[{case}] reps={reps} cap={cap:,}')
    print('EXPERIMENT:',' '.join(map(str,cmd)))
    if not do_run: return 0, ' '.join(map(str,cmd))
    out.mkdir(parents=True,exist_ok=True)
    rc=subprocess.call(cmd,cwd=ROOT)
    if rc: return rc, ' '.join(map(str,cmd))
    rc=subprocess.call(report,cwd=ROOT)
    return rc, ' '.join(map(str,cmd))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',choices=sorted(PROFILES),default='paper')
    p.add_argument('--instance',choices=sorted(INSTANCES),default='cluster5')
    p.add_argument('--eta',type=float,default=.05)
    p.add_argument('--run',action='store_true')
    p.add_argument('--strict',action='store_true',help='Use refresh=decide=1 also for Model U; much slower.')
    p.add_argument('--workers',type=int,default=1,help='Parallel worker processes for independent replications.')
    p.add_argument('--resume',action='store_true',help='Resume checkpointed trials.csv in an existing suite directory.')
    p.add_argument('--output-root',default=None,help='Suite directory. Default: hardcore_5arm_runs/<timestamp>')
    p.add_argument('--include-u-shape',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--include-u-raw-certified-correctness',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args()

    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    suite=(ROOT/(a.output_root or f'hardcore_5arm_runs/{stamp}_{a.instance}_{a.profile}')).resolve()
    cases_root=suite/'cases'; cases_root.mkdir(parents=True,exist_ok=True)
    cfg=cases_for(INSTANCES[a.instance],a.strict)
    order=['E_Q1','U_Q1','E_Q3','U_Q3','E_CORRECTNESS','U_CORRECTNESS','E_NONASYM','U_NONASYM_CERTIFIED']
    if a.include_u_shape: order.append('U_NONASYM_SHAPE')
    if a.include_u_raw_certified_correctness: order.append('U_CORRECTNESS_CERTIFIED_RAW')
    reps=PROFILES[a.profile]; caps=CAPS[a.profile]
    commands=[]

    meta={
      'profile':a.profile,'instance':a.instance,'means':INSTANCES[a.instance], 'K':5,
      'eta':a.eta,'strict_U_refresh':a.strict,
      'asymptotic_deltas':[float(x) for x in ASYM_DELTAS],
      'asymptotic_L':[float(__import__('math').log(1/float(x))) for x in ASYM_DELTAS],
      'ratio_deltas':[float(x) for x in RATIO_DELTAS],
      'correctness_deltas':[float(x) for x in COR_DELTAS],
      'finite_deltas':[float(x) for x in FINITE_DELTAS],
      'data_policy':'Every case checkpoints trials.csv after each full replication. Use --resume to continue; the suite path is never reused unless explicitly requested.',
      'important_U_note':'U_Q1/U_Q3/U_NONASYM_SHAPE use practical raw-profile boundaries. U_NONASYM_CERTIFIED and U_CORRECTNESS_CERTIFIED_RAW use the theorem-certified sieve boundary and may be heavily censored.',
    }
    (suite/'HARDCORE_SUITE_CONFIG.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('SUITE:',suite)
    print('INSTANCE:',a.instance,INSTANCES[a.instance])
    print('PROFILE:',a.profile,'strict=',a.strict,'eta=',a.eta,'workers=',a.workers)
    print('ASYMPTOTIC L GRID:',', '.join(f'{x:.2f}' for x in meta['asymptotic_L']))
    if a.profile=='paper':
        print('NOTE: paper-scale profile uses 200 replications for Q1/Q3 and 600 for correctness.')

    for i,case in enumerate(order,1):
        print(f'\n=== CASE {i}/{len(order)} ===')
        rc,cmd=run_case(case,cfg[case],reps[case],caps[case],cases_root,a.eta,a.run,a.resume,a.workers)
        commands.append(cmd)
        if rc:
            print('FAILED:',case,'rc=',rc)
            print('All completed case data remain in',cases_root)
            return rc
    (suite/'commands.txt').write_text('\n'.join(commands)+'\n',encoding='utf-8')

    # The four-paper-figure report intentionally uses the certified U finite case.
    report=[sys.executable,'-m','onebit_bai.combined_report','--cases-root',str(cases_root),
            '--output',str(suite),'--eta',str(a.eta)]
    print('\nCOMBINED REPORT:',' '.join(map(str,report)))
    if not a.run:
        print('\nPREVIEW ONLY. Add --run to execute.')
        return 0
    rc=subprocess.call(report,cwd=ROOT)
    if rc:return rc
    print('\nDONE. Reusable raw data:')
    print(' ',suite/'all_trials_combined.csv')
    print(' ',cases_root/'<CASE>'/'trials.csv')
    print('Combined figures and CSV summaries are in',suite)
    return 0

if __name__=='__main__': raise SystemExit(main())
