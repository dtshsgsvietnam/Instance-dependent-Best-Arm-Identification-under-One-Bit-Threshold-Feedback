#!/usr/bin/env python3
"""Create paper-facing non-asymptotic diagnostics from E companion and U practical runs."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT_CODE))
from onebit_bai.report import e_remainder, u_remainder


def load_case(case: Path):
    if not (case/'trials.csv').exists(): return None
    return (pd.read_csv(case/'trials.csv'),
            json.loads((case/'config.json').read_text()),
            json.loads((case/'oracles.json').read_text()))


def bootstrap_mean(x, seed=314159, B=2000):
    x=np.asarray(x,float)
    if len(x)<2:return float(np.mean(x)),np.nan,np.nan
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(x),size=(B,len(x)))
    z=x[idx].mean(axis=1)
    return float(x.mean()),*map(float,np.quantile(z,[.025,.975]))


def summarize(case_name, loaded, eta):
    trials,conf,oracle=loaded
    gamma=float(oracle['onebit_gamma']); T=1/gamma; K=len(conf['mus'])
    rows=[]
    for d,g in trials.groupby('delta'):
        L=float(np.log(1/d)); stop=g[g.stopped==1]
        complete=len(stop)==len(g)
        mean=lo=hi=np.nan
        qnom=q95=np.nan
        if complete:
            mean,lo,hi=bootstrap_mean(stop.tau.to_numpy(float), seed=314159+int(round(L*10)))
            qnom=float(np.quantile(stop.tau,1-3*eta));q95=float(np.quantile(stop.tau,.95))
        if case_name=='E':
            R=e_remainder(L,K,eta)
        else:
            # Shape from Theorem-U, used only as a diagnostic because this run
            # uses the practical boundary rather than the certified boundary.
            R=u_remainder(L,T,eta,CT=1)
        def ne(x): return (x-T*L)/R if np.isfinite(x) else np.nan
        rows.append(dict(case=case_name,delta=d,L=L,reps=len(g),stop_rate=float(g.stopped.mean()),
                         Gamma=gamma,Tstar=T,mean_tau=mean,mean_CI_lo=lo,mean_CI_hi=hi,
                         q_1_minus_3eta=qnom,q95=q95,remainder_basis=R,
                         normalized_mean_excess=ne(mean),
                         normalized_mean_excess_lo=ne(lo),normalized_mean_excess_hi=ne(hi),
                         normalized_nominal_quantile_excess=ne(qnom),
                         interpretation=('theorem-facing E companion (numerical discretization/refresh remains)'
                           if case_name=='E' else
                           'practical-U theorem-motivated shape diagnostic; NOT Theorem-12 validation')))
    return pd.DataFrame(rows).sort_values('L')


def plot_one(df,title,out):
    h=df.dropna(subset=['normalized_mean_excess'])
    if h.empty:return
    fig=plt.figure(figsize=(5.0,3.55));ax=plt.axes()
    y=h.normalized_mean_excess.to_numpy()
    lo=h.normalized_mean_excess_lo.to_numpy();hi=h.normalized_mean_excess_hi.to_numpy()
    ax.errorbar(h.L,y,yerr=np.vstack([y-lo,hi-y]),marker='o',capsize=3,label='Mean excess (95% bootstrap CI)')
    ax.plot(h.L,h.normalized_nominal_quantile_excess,marker='s',linestyle='--',label='Empirical (1-3η)-quantile excess')
    ax.set_xlabel(r'$L_\delta=\log(1/\delta)$')
    ax.set_ylabel('Excess / theorem-shaped remainder')
    ax.set_title(title);ax.grid(alpha=.25);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(out.with_suffix('.png'),dpi=300,bbox_inches='tight');fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--eta',type=float,default=.05);a=ap.parse_args()
    root=Path(a.root);frames=[]
    e=load_case(root/'E_COMPANION');u=load_case(root/'U_PRACTICAL')
    if e is not None:
        de=summarize('E',e,a.eta);frames.append(de);plot_one(de,'Model E companion: finite-confidence correction scale',root/'E_nonasym_correction_scale')
    if u is not None:
        du=summarize('U',u,a.eta);frames.append(du);plot_one(du,'Model U practical Top-Two: theorem-motivated correction shape',root/'U_practical_nonasym_shape')
    if frames:
        all_=pd.concat(frames,ignore_index=True);all_.to_csv(root/'nonasym_summary.csv',index=False)
        print(all_.to_string(index=False))
        print('\nIMPORTANT: E is the companion experiment; U practical is not a direct numerical validation of the certified Theorem-12 stopping rule.')

if __name__=='__main__':main()
