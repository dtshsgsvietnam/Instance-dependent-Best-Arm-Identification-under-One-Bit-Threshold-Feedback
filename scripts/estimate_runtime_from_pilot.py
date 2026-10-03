#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd


def estimate(path,reps,workers):
    p=Path(path)/'trials.csv'
    if not p.exists():return None
    d=pd.read_csv(p)
    # run_time_s is repeated across delta rows; use one row per replication/method.
    x=d.sort_values(['rep','delta']).groupby(['rep','method'],as_index=False).first().run_time_s
    mean=float(x.mean());med=float(x.median())
    ideal=reps*mean/max(1,workers)/3600
    return {'pilot_mean_s_per_rep':mean,'pilot_median_s_per_rep':med,'ideal_final_hours':ideal,
            'planning_hours_1p35x':ideal*1.35,'workers':workers,'final_reps':reps}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--final-config',required=True);ap.add_argument('--workers-e',type=int,required=True);ap.add_argument('--workers-u',type=int,required=True);a=ap.parse_args()
    cfg=json.loads(Path(a.final_config).read_text());root=Path(a.root)
    E=estimate(root/'E_COMPANION',cfg['E']['reps'],a.workers_e)
    U=estimate(root/'U_PRACTICAL',cfg['U_practical']['reps'],a.workers_u)
    print('\nRuntime estimate from THIS pilot (more reliable than historical guesses):')
    for name,v in [('E companion',E),('U practical',U)]:
        if v:print(f"{name}: mean={v['pilot_mean_s_per_rep']:.1f}s/rep, workers={v['workers']}, ideal={v['ideal_final_hours']:.2f}h, planning≈{v['planning_hours_1p35x']:.2f}h")
    if E and U:
        total=E['planning_hours_1p35x']+U['planning_hours_1p35x']
        print(f"Sequential planning total ≈ {total:.2f} hours = {total/24:.2f} days")

if __name__=='__main__':main()
