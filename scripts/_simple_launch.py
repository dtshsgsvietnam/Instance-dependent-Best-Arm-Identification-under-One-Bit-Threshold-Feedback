from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def common_parser(description:str):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--model',choices=['E','U'],required=True)
    p.add_argument('--run',action='store_true',help='Actually run; without this flag only print commands')
    p.add_argument('--reps',type=int,default=None)
    p.add_argument('--cap',type=int,default=None)
    p.add_argument('--strict',action='store_true',help='Use refresh=1, decide=1 and finer grids; much slower, closest numerical policy')
    p.add_argument('--output-root',default='results_simple')
    return p

def run_experiment(args, stem:str, flags:list[str], default_reps:int, default_cap:int, eta:float=.05):
    reps=args.reps or default_reps; cap=args.cap or default_cap
    out=ROOT/args.output_root/f'{stem}_{args.model}'
    cmd=[sys.executable,'-m','onebit_bai.experiment',*flags,'--reps',str(reps),'--cap',str(cap),'--output',str(out),'--verbose']
    print('EXPERIMENT:',' '.join(cmd))
    print('REPORT:',sys.executable,'-m onebit_bai.report --input',out,'--eta',eta)
    if not args.run:
        print('\nPREVIEW ONLY. Add --run to execute.')
        return 0
    rc=subprocess.call(cmd,cwd=ROOT)
    if rc:return rc
    return subprocess.call([sys.executable,'-m','onebit_bai.report','--input',str(out),'--eta',str(eta)],cwd=ROOT)
