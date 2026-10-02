"""Keep already-completed u_raw rows from an older U_CORRECTNESS run.

Creates a backup before modifying trials.csv. Useful when v39.4 ran both u_raw
and u_marginal but the user wants to resume raw-only under v39.5.
"""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--case-dir',required=True,help='Path to .../cases/U_CORRECTNESS')
a=p.parse_args()
case=Path(a.case_dir).expanduser().resolve()
f=case/'trials.csv'
if not f.exists(): raise FileNotFoundError(f)
df=pd.read_csv(f)
if 'method' not in df.columns: raise ValueError('trials.csv has no method column')
raw=df[df.method=='u_raw'].copy()
if raw.empty: raise ValueError('No u_raw rows found; refusing to modify file.')
# A replication is complete iff it has exactly one row per delta.
nd=raw.delta.nunique()
counts=raw.groupby('rep').size()
complete=set(int(r) for r,c in counts.items() if int(c)==nd)
raw=raw[raw.rep.isin(complete)].copy().sort_values(['rep','delta'])
backup=case/'trials.before_raw_only.bak.csv'
if backup.exists():
    raise FileExistsError(f'{backup} already exists; refusing to overwrite backup.')
shutil.copy2(f,backup)
tmp=case/'trials.csv.tmp'
raw.to_csv(tmp,index=False)
tmp.replace(f)
# v39.4 has no signature. If a partial/new signature exists, remove it so the
# first v39.5 resume can establish the raw-only signature explicitly.
sig=case/'statistical_signature.json'
if sig.exists(): sig.unlink()
print(f'BACKUP: {backup}')
print(f'KEPT: {len(complete)} complete raw-only replications, {len(raw)} rows, {nd} deltas/rep')
print(f'WROTE: {f}')
