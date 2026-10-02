from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description='Compare lambda_hat_n=min_j W_n/n with population Gamma')
p.add_argument('--results',required=True)
p.add_argument('--method',required=True,choices=['e_tt','e_companion','u_raw','u_marginal','e_full','u_full'])
a=p.parse_args(); root=Path(a.results)
o=json.loads((root/'oracles.json').read_text())
gamma=o['full_gamma'] if a.method.endswith('_full') else o['onebit_gamma']
f=root/f'trace_rep0_{a.method}.csv'
if not f.exists(): raise SystemExit(f'Missing {f}; rerun with --trace-every > 0')
d=pd.read_csv(f)
d['lambda_over_gamma']=d.lambda_hat/gamma
print('population Gamma =',gamma)
print(d[['n','leader','lambda_hat','lambda_over_gamma']].tail(12).to_string(index=False))
if len(d)>=5:
    tail=d.tail(max(5,len(d)//5))
    print('tail mean(lambda_hat/Gamma) =',float(tail.lambda_over_gamma.mean()))
fig,ax=plt.subplots(figsize=(7,4.5));ax.plot(d.n,d.lambda_over_gamma)
ax.axhline(1,linestyle='--');ax.set_xlabel('n');ax.set_ylabel('lambda_hat_n / Gamma')
ax.set_title('Evidence-rate diagnostic');fig.tight_layout();fig.savefig(root/f'rate_trace_{a.method}.png',dpi=160)
print('wrote',root/f'rate_trace_{a.method}.png')
