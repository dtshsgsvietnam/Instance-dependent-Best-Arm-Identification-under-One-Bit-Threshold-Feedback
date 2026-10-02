"""Plot exact paper U sieve penalty vs practical threshold without sampling data."""
import argparse,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from onebit_bai.algorithms import U_cert_boundary,practical_boundary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--delta',type=float,default=.05)
    p.add_argument('--K',type=int,default=2);p.add_argument('--n-max',type=int,default=5000)
    p.add_argument('--output',default='u_boundaries.png');a=p.parse_args()
    n=np.unique(np.geomspace(2,a.n_max,80).astype(int))
    certified=np.array([U_cert_boundary(int(t),a.delta,a.K) for t in n])
    practical=np.array([practical_boundary(int(t),a.delta,a.K) for t in n])
    fig,ax=plt.subplots(figsize=(8,5));ax.plot(n,certified,label='paper-certified U sieve GLR boundary')
    ax.plot(n,practical,label='practical U (NO delta certificate)')
    ax.set_xlabel('Calendar n');ax.set_ylabel('Evidence stopping threshold')
    ax.set_xscale('log');ax.set_yscale('log');ax.legend();fig.tight_layout();fig.savefig(a.output,dpi=150)
    print('Wrote',a.output,'; no experiments were executed.')
