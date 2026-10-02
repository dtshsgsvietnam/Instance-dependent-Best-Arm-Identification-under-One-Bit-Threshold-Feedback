"""Reusable opt-in launcher; nothing is executed on import."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def run_allowed(allowed:dict[str,str], description:str):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--case',required=True,choices=sorted(allowed),help='Select ONE study case; prints available names below')
    p.add_argument('--profile',choices=['pilot','paper'],default='pilot')
    p.add_argument('--run',action='store_true',help='EXECUTE selected experiments; omit to preview commands')
    p.add_argument('--reps',type=int)
    p.add_argument('--cap',type=int)
    p.add_argument('--refresh',type=int)
    p.add_argument('--decide',type=int)
    p.add_argument('--output-root',default='results_v36')
    p.add_argument('--refine',action='store_true',help='Run an extra population oracle resolution comparison')
    p.epilog='\n'.join(f'{name}: {desc}' for name,desc in allowed.items())
    args=p.parse_args()
    cmd=[sys.executable,str(ROOT/'run_test_cases.py'),'--case',args.case,'--profile',args.profile,
         '--output-root',args.output_root]
    for x in ('reps','cap','refresh','decide'):
        if getattr(args,x) is not None:cmd.extend(['--'+x,str(getattr(args,x))])
    if args.refine:cmd.append('--refine')
    if not args.run:cmd.append('--dry-run')
    print('\nSELECTED:',args.case,'\nPURPOSE:',allowed[args.case],flush=True)
    print('ACTION:', 'EXECUTE' if args.run else 'PREVIEW (no experiments)', flush=True)
    return subprocess.call(cmd,cwd=ROOT)
