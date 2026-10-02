"""Validate result integrity; refuses asymmetric censoring and false PAC labels.

Only reads previously generated files. Does NOT run Monte Carlo.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args()
    root=Path(a.results);t=pd.read_csv(root/'trials.csv');cfg=json.loads((root/'config.json').read_text())
    o=json.loads((root/'oracles.json').read_text());errors=[];warnings=[]
    if t.duplicated(['rep','method','delta']).any():errors.append('DUPLICATE rep/method/delta keys')
    if (t.stopped.isin([0,1])==False).any():errors.append('Stopped indicator outside {0,1}')
    if ((t.stopped==0)&t.tau.notna()).any():errors.append('Censored paths have fake tau values')
    if ((t.stopped==1)&t.tau.isna()).any():errors.append('Stopped paths with missing tau')
    if ((t.error==1)&(t.stopped!=1)).any():errors.append('Wrong event but run did not stop')
    if (t[t.stopped==1].tau>cfg['cap']).any():errors.append('Stopped after configured cap')
    if not np.isfinite(o['onebit_gamma']) or o['onebit_gamma']<=0:errors.append('Invalid one-bit population Gamma')
    if o['full_gamma'] is not None and o['onebit_gamma']>o['full_gamma']*(1+2e-3):
        warnings.append('Gamma_bit > Gamma_full: refine oracle and check numerical optimization; violates information ordering if exact.')
    if cfg['model']=='U' and cfg.get('u_generator')=='beta':
        if abs(o.get('u_beta_concentration',12.)-cfg.get('concentration',12.))>1e-12:
            errors.append('SIMULATION/POPULATION beta concentration mismatch')
    for (rep,method),g in t.groupby(['rep','method']):
        h=g.sort_values('delta',ascending=False)
        # With a common delta-independent trajectory, a stricter (smaller)
        # confidence must NEVER stop sooner than a looser confidence.
        d=h[h.stopped==1]
        if len(d)>1 and np.any(np.diff(d.tau.to_numpy())<0):
            errors.append(f'Non-nested stopping at rep={rep}, method={method}')
        stopped=h.stopped.to_numpy()
        if np.any(np.diff(stopped)>0):
            errors.append(f'Smaller delta stopped while larger delta censored ({rep},{method})')
    if (t.groupby(['method','delta']).stopped.mean()<1).any():
        warnings.append('SOME CENSORED CELLS: complete-data means, quantiles, and bit/full ratios must NOT be reported for those cells.')
    if cfg.get('model')=='U' and cfg.get('u_boundary')=='practical':
        warnings.append('U RAW PRACTICAL boundary has no theorem PAC certificate, even if empirical error is low.')
    if cfg.get('model')=='E' and cfg.get('oracle_refresh',1)>1:
        warnings.append('Periodic/grid E numerical oracle is not the exact continuum regularized theorem oracle.')
    print(f'VALIDATED: {len(t)} trial rows; errors={len(errors)} warnings={len(warnings)}')
    for msg in errors:print('ERROR:',msg)
    for msg in warnings:print('WARNING:',msg)
    out=root/'VALIDATION.json';out.write_text(json.dumps(dict(errors=errors,warnings=warnings),indent=2))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
