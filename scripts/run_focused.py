"""Run focused cases for Q1, Q3, correctness, or non-asymptotic coverage.

Examples:
  python scripts/run_focused.py --case E_Q1 --run
  python scripts/run_focused.py --case E_Q3 --run
  python scripts/run_focused.py --case E_CORRECTNESS --run
  python scripts/run_focused.py --case E_NONASYM --run
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

CASES={
'E_Q1': dict(flags=['--model','E','--e-family','gaussian','--means','.75','.45','--sigma','.2','--beta','.5','--deltas','.2','.1','.05','.02','.01','--no-companion','--e-boundary','certified','--qpoints','11','--xgrid','31','--oracle-refresh','20','--refresh','5','--decide','5'],reps=40,cap=20000),
'U_Q1': dict(flags=['--model','U','--u-generator','beta','--means','.8','.2','--concentration','12','--beta','.5','--deltas','.2','.1','.05','.02','.01','--skip-safe-bit','--u-boundary','practical','--xgrid','41','--qquad','181','--refresh','5','--decide','5'],reps=30,cap=12000),
'E_Q3': dict(flags=['--model','E','--e-family','gaussian','--means','.75','.45','--sigma','.2','--beta','.5','--deltas','.2','.1','.05','.02','--no-companion','--full','--e-boundary','practical','--full-boundary','practical','--qpoints','11','--xgrid','31','--oracle-refresh','20','--refresh','5','--decide','5'],reps=40,cap=16000),
'U_Q3': dict(flags=['--model','U','--u-generator','beta','--means','.8','.2','--concentration','12','--beta','.5','--deltas','.2','.1','.05','--skip-safe-bit','--full','--u-boundary','practical','--full-boundary','practical','--xgrid','31','--qquad','121','--refresh','10','--decide','10'],reps=24,cap=9000),
'E_CORRECTNESS': dict(flags=['--model','E','--e-family','gaussian','--means','.75','.45','--sigma','.2','--beta','.5','--deltas','.1','.05','--no-companion','--e-boundary','certified','--qpoints','11','--xgrid','31','--refresh','5','--decide','5'],reps=200,cap=16000),
'U_CORRECTNESS': dict(flags=['--model','U','--u-generator','beta','--means','.8','.2','--concentration','12','--beta','.5','--deltas','.1','.05','--u-boundary','practical','--xgrid','31','--qquad','121','--refresh','10','--decide','10'],reps=200,cap=9000),
'E_NONASYM': dict(flags=['--model','E','--e-family','gaussian','--means','.75','.45','--sigma','.2','--beta','.5','--deltas','.2','.1','.05','.02','--e-boundary','certified','--qpoints','11','--xgrid','31','--oracle-refresh','10','--refresh','5','--decide','5'],reps=120,cap=22000),
'U_NONASYM_SHAPE': dict(flags=['--model','U','--u-generator','beta','--means','.8','.2','--concentration','12','--beta','.5','--deltas','.2','.1','.05','.02','--skip-safe-bit','--u-boundary','practical','--xgrid','41','--qquad','181','--refresh','5','--decide','5'],reps=100,cap=14000),
'U_NONASYM_CERTIFIED': dict(flags=['--model','U','--u-generator','beta','--means','.8','.2','--concentration','12','--beta','.5','--deltas','.2','.1','.05','--skip-safe-bit','--u-boundary','certified','--xgrid','31','--qquad','121','--refresh','5','--decide','5'],reps=20,cap=30000),
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',choices=sorted(CASES),required=True)
    p.add_argument('--run',action='store_true')
    p.add_argument('--reps',type=int)
    p.add_argument('--cap',type=int)
    p.add_argument('--eta',type=float,default=.05)
    p.add_argument('--output-root',default='focused_results')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();c=CASES[a.case]
    reps=a.reps or c['reps'];cap=a.cap or c['cap']
    out=ROOT/a.output_root/a.case
    cmd=[sys.executable,'-m','onebit_bai.experiment',*c['flags'],'--reps',str(reps),'--cap',str(cap),'--output',str(out),'--verbose','--workers',str(a.workers)]
    if a.resume: cmd.append('--resume')
    report=[sys.executable,'-m','onebit_bai.focused_report','--input',str(out),'--eta',str(a.eta)]
    print('CASE:',a.case)
    print('EXPERIMENT:',' '.join(cmd))
    print('REPORT:',' '.join(report))
    if a.case.startswith('U_Q1') or a.case=='U_NONASYM_SHAPE':
        print('WARNING: U practical raw-profile boundary is descriptive, not the theorem-certified PAC boundary.')
    if a.case=='U_NONASYM_CERTIFIED':
        print('WARNING: certified U sieve boundary is extremely conservative; censoring is expected at ordinary caps.')
    if not a.run:
        print('PREVIEW ONLY. Add --run to execute.')
        return 0
    out.mkdir(parents=True,exist_ok=True)
    rc=subprocess.call(cmd,cwd=ROOT)
    if rc:return rc
    return subprocess.call(report,cwd=ROOT)
if __name__=='__main__':raise SystemExit(main())
