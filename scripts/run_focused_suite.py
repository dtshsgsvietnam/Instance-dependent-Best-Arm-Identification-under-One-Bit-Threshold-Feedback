"""Run the complete E/U focused suite and create exactly four combined figures.

Every case keeps its own config.json, oracles.json and checkpointed trials.csv.
The suite then creates four E/U combined figures plus combined CSV summaries.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CASES=['E_Q1','U_Q1','E_Q3','U_Q3','E_CORRECTNESS','U_CORRECTNESS','E_NONASYM','U_NONASYM_CERTIFIED']
QUICK={'E_Q1':30,'U_Q1':20,'E_Q3':24,'U_Q3':12,'E_CORRECTNESS':100,'U_CORRECTNESS':100,'E_NONASYM':60,'U_NONASYM_CERTIFIED':12}
STANDARD={'E_Q1':100,'U_Q1':60,'E_Q3':80,'U_Q3':40,'E_CORRECTNESS':300,'U_CORRECTNESS':300,'E_NONASYM':120,'U_NONASYM_CERTIFIED':24}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',action='store_true')
    p.add_argument('--workers',type=int,default=1,help='Parallel worker processes for independent replications.')
    p.add_argument('--profile',choices=['quick','standard'],default='quick')
    p.add_argument('--eta',type=float,default=.05)
    p.add_argument('--output-root',default=None,help='Suite directory. Default: focused_suite_runs/<timestamp>')
    p.add_argument('--cases',nargs='*',choices=CASES,default=None,help='Optional subset; default runs all eight cases')
    a=p.parse_args()
    chosen=a.cases or CASES
    reps=QUICK if a.profile=='quick' else STANDARD
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    suite=(ROOT/(a.output_root or f'focused_suite_runs/{stamp}')).resolve()
    cases_root=suite/'cases'; cases_root.mkdir(parents=True,exist_ok=True)
    try:
        rel_cases=cases_root.relative_to(ROOT)
    except ValueError:
        rel_cases=cases_root
    commands=[]
    print('SUITE:',suite)
    print('PROFILE:',a.profile,'eta=',a.eta)
    for i,case in enumerate(chosen,1):
        cmd=[sys.executable,str(ROOT/'scripts'/'run_focused.py'),'--case',case,'--reps',str(reps[case]),
             '--eta',str(a.eta),'--output-root',str(rel_cases)]
        if a.run: cmd.append('--run')
        commands.append(' '.join(map(str,cmd)))
        print(f'\n[{i}/{len(chosen)}] {case} | reps={reps[case]}')
        print(' '.join(map(str,cmd)))
        if a.run:
            rc=subprocess.call(cmd,cwd=ROOT)
            if rc:
                print('CASE FAILED:',case,'rc=',rc)
                print('Completed case data remain saved under',cases_root)
                return rc
    (suite/'commands.txt').write_text('\n'.join(commands)+'\n',encoding='utf-8')
    (suite/'SUITE_CONFIG.json').write_text(json.dumps({'profile':a.profile,'eta':a.eta,'cases':chosen,'reps':{c:reps[c] for c in chosen}},indent=2),encoding='utf-8')
    report=[sys.executable,'-m','onebit_bai.combined_report','--cases-root',str(cases_root),'--output',str(suite),'--eta',str(a.eta)]
    print('\nCOMBINED REPORT:',' '.join(report))
    if not a.run:
        print('\nPREVIEW ONLY. Add --run to execute Monte Carlo.')
        return 0
    return subprocess.call(report,cwd=ROOT)

if __name__=='__main__': raise SystemExit(main())
