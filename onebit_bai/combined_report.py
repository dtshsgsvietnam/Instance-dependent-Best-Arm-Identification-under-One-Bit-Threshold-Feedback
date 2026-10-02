"""Combine the focused E/U experiments into exactly four paper-facing figures.

The four questions are:
  Q1  one-bit first-order ratio -> 1;
  Q3  one-bit/full empirical stopping-time ratio vs population prediction;
  COR correctness vs 1-delta;
  FIN holdout coverage of the theorem-shaped non-asymptotic bound.

Raw per-replication data remain in cases/<CASE>/trials.csv.  This module only
aggregates those files; it never discards or overwrites the raw trials.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _case(cases: Path, name: str) -> Path:
    return cases / name


def _model_label(model: str) -> str:
    if model == 'E':
        return 'Model E (Gaussian member of exponential family)'
    return 'Model U (Beta law on [0,1])'


def _collect_q1(cases: Path) -> pd.DataFrame:
    rows=[]
    for case,model,method in [('E_Q1','E','e_tt'),('U_Q1','U','u_raw')]:
        d=_read_csv(_case(cases,case)/'summary_focused.csv')
        if d.empty: continue
        h=d[(d.method==method)&d.first_order_ratio.notna()].copy()
        if h.empty: continue
        h['case']=case; h['model']=model; h['model_label']=_model_label(model)
        rows.append(h)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def _collect_q3(cases: Path) -> pd.DataFrame:
    rows=[]
    for case,model,one,full in [('E_Q3','E','e_tt','e_full'),('U_Q3','U','u_raw','u_full')]:
        d=_read_csv(_case(cases,case)/'paired_full_comparison_focused.csv')
        if d.empty: continue
        need={'onebit','full','empirical_mean_ratio'}
        if not need.issubset(d.columns): continue
        h=d[(d.onebit==one)&(d.full==full)].copy()
        if h.empty: continue
        h['case']=case; h['model']=model; h['model_label']=_model_label(model)
        h['log1d']=np.log(1/h.delta.astype(float))
        h['empirical_over_theory']=h.empirical_mean_ratio/h.population_predicted_ratio
        rows.append(h)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def _collect_correctness(cases: Path) -> pd.DataFrame:
    rows=[]
    for case,model in [('E_CORRECTNESS','E'),('U_CORRECTNESS','U')]:
        d=_read_csv(_case(cases,case)/'correctness_focused.csv')
        if d.empty: continue
        d=d.copy(); d['case']=case; d['model']=model; d['model_label']=_model_label(model)
        # recover the actually reported method from raw data/config when possible
        raw=_read_csv(_case(cases,case)/'trials.csv')
        if model=='U' and not raw.empty and 'u_marginal' in set(raw.method): method='u_marginal'
        else: method='e_tt' if model=='E' else 'u_raw'
        d['method']=method
        rows.append(d)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def _collect_finite(cases: Path) -> pd.DataFrame:
    rows=[]
    for case,model in [('E_NONASYM','E'),('U_NONASYM_CERTIFIED','U')]:
        d=_read_csv(_case(cases,case)/'nonasymptotic_coverage_focused.csv')
        if d.empty: continue
        d=d.copy(); d['case']=case; d['model']=model; d['model_label']=_model_label(model)
        d['log1d']=np.log(1/d.delta.astype(float))
        rows.append(d)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def _save_q1(d: pd.DataFrame, out: Path):
    fig,ax=plt.subplots(figsize=(7.8,4.8))
    if d.empty:
        ax.text(.5,.5,'No Q1 data found',ha='center',va='center',transform=ax.transAxes)
    else:
        for label,g in d.groupby('model_label'):
            g=g.sort_values('log1d')
            yerr=np.vstack([g.first_order_ratio-g.first_order_ratio_CI_lo,
                            g.first_order_ratio_CI_hi-g.first_order_ratio])
            ax.errorbar(g.log1d,g.first_order_ratio,yerr=yerr,marker='o',capsize=3,label=label)
        ax.axhline(1,linestyle='--',linewidth=1,label='first-order limit = 1')
    ax.set_xlabel('log(1/delta)')
    ax.set_ylabel('mean(tau_1bit) * Gamma_1bit / log(1/delta)')
    ax.set_title('Q1: one-bit first-order convergence')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'01_Q1_onebit_asymptotic_E_U.png',dpi=180); plt.close(fig)


def _save_q3(d: pd.DataFrame, out: Path):
    fig,ax=plt.subplots(figsize=(7.8,4.8))
    if d.empty:
        ax.text(.5,.5,'No complete Q3 paired data found',ha='center',va='center',transform=ax.transAxes)
    else:
        for label,g in d.groupby('model_label'):
            h=g[g.empirical_mean_ratio.notna()].sort_values('log1d')
            if h.empty: continue
            yerr=np.vstack([h.empirical_mean_ratio-h.bootstrap_CI_lo,
                            h.bootstrap_CI_hi-h.empirical_mean_ratio])
            ax.errorbar(h.log1d,h.empirical_mean_ratio,yerr=yerr,marker='o',capsize=3,
                        label=f'{label}: empirical')
            theory=float(h.population_predicted_ratio.iloc[0])
            ax.hlines(theory,float(h.log1d.min()),float(h.log1d.max()),linestyles='--',linewidth=1,
                      label=f'{label}: theory = {theory:.3g}')
    ax.set_xlabel('log(1/delta)')
    ax.set_ylabel('mean(tau_1bit) / mean(tau_full)')
    ax.set_title('Q3: one-bit/full sample-cost ratio — empirical vs theory')
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'02_Q3_onebit_vs_full_E_U.png',dpi=180); plt.close(fig)


def _save_correctness(d: pd.DataFrame, out: Path):
    fig,ax=plt.subplots(figsize=(7.8,4.8))
    if d.empty:
        ax.text(.5,.5,'No correctness data found',ha='center',va='center',transform=ax.transAxes)
    else:
        for label,g in d.groupby('model_label'):
            g=g.sort_values('delta')
            complete=g[g.complete.astype(bool)]
            if not complete.empty:
                yerr=np.vstack([complete.correctness-complete.CP95_lo,
                                complete.CP95_hi-complete.correctness])
                ax.errorbar(complete.delta,complete.correctness,yerr=yerr,marker='o',capsize=3,
                            label=f'{label}: empirical (95% CP)')
            cens=g[~g.complete.astype(bool)]
            if not cens.empty:
                ax.scatter(cens.delta,cens.correctness,marker='x',
                           label=f'{label}: correct-stop lower bound (censored)')
        x=np.sort(d.delta.unique())
        ax.plot(x,1-x,linestyle='--',label='target 1-delta')
    ax.set_xscale('log'); ax.set_ylim(0,1.02)
    ax.set_xlabel('delta'); ax.set_ylabel('probability of correct recommendation')
    ax.set_title('Correctness: empirical correct-stop rate vs target')
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'03_correctness_E_U.png',dpi=180); plt.close(fig)


def _save_finite(d: pd.DataFrame, out: Path, eta: float):
    fig,ax=plt.subplots(figsize=(7.8,4.8))
    plotted=False
    if not d.empty:
        for label,g in d.groupby('model_label'):
            h=g[g.coverage.notna()].sort_values('log1d') if 'coverage' in g else pd.DataFrame()
            if not h.empty:
                yerr=np.vstack([h.coverage-h.coverage_CP95_lower,
                                h.coverage_CP95_upper-h.coverage])
                ax.errorbar(h.log1d,h.coverage,yerr=yerr,marker='o',capsize=3,
                            label=f'{label}: holdout coverage (95% CP)')
                plotted=True
            # If censoring leaves coverage unknown, show only the identified interval.
            if {'coverage_raw_lower','coverage_raw_upper'}.issubset(g.columns):
                u=g[g.coverage.isna() & g.coverage_raw_lower.notna() & g.coverage_raw_upper.notna()].copy()
                if not u.empty:
                    u=u.sort_values('log1d')
                    mid=(u.coverage_raw_lower+u.coverage_raw_upper)/2
                    yerr=np.vstack([mid-u.coverage_raw_lower,u.coverage_raw_upper-mid])
                    ax.errorbar(u.log1d,mid,yerr=yerr,marker='x',capsize=3,linestyle='none',
                                label=f'{label}: censor-compatible interval (not CI)')
                    plotted=True
    if not plotted:
        ax.text(.5,.5,'No evaluable finite-bound holdout cells.\nFor certified Model U this often means censoring at the simulation cap.',
                ha='center',va='center',transform=ax.transAxes)
    ax.axhline(1-3*eta,linestyle='--',linewidth=1,label=f'nominal 1-3 eta = {1-3*eta:.2f}')
    ax.set_ylim(0,1.02); ax.set_xlabel('log(1/delta)'); ax.set_ylabel('holdout coverage')
    ax.set_title('Non-asymptotic probability-bound coverage')
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'04_nonasymptotic_coverage_E_U.png',dpi=180); plt.close(fig)


def _combine_raw_trials(cases: Path, out: Path):
    rows=[]
    for child in sorted(cases.iterdir()) if cases.exists() else []:
        if not child.is_dir(): continue
        d=_read_csv(child/'trials.csv')
        if d.empty: continue
        d=d.copy(); d.insert(0,'case',child.name); rows.append(d)
    if rows:
        pd.concat(rows,ignore_index=True).to_csv(out/'all_trials_combined.csv',index=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases-root',required=True,help='Directory containing E_Q1, U_Q1, ... case folders')
    p.add_argument('--output',required=True)
    p.add_argument('--eta',type=float,default=.05)
    a=p.parse_args()
    cases=Path(a.cases_root); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    q1=_collect_q1(cases); q3=_collect_q3(cases); cor=_collect_correctness(cases); fin=_collect_finite(cases)
    q1.to_csv(out/'combined_Q1.csv',index=False)
    q3.to_csv(out/'combined_Q3.csv',index=False)
    cor.to_csv(out/'combined_correctness.csv',index=False)
    fin.to_csv(out/'combined_nonasymptotic.csv',index=False)
    _combine_raw_trials(cases,out)
    _save_q1(q1,out); _save_q3(q3,out); _save_correctness(cor,out); _save_finite(fin,out,a.eta)
    manifest={
      'four_figures':[
        '01_Q1_onebit_asymptotic_E_U.png','02_Q3_onebit_vs_full_E_U.png',
        '03_correctness_E_U.png','04_nonasymptotic_coverage_E_U.png'],
      'models':{
        'E':'Gaussian test instance; Gaussian is a member of the regular one-parameter exponential-family Model E.',
        'U':'Beta test instance supported on [0,1], used for the arbitrary bounded-support Model U.'},
      'finite_note':'Model-U certified finite boundary may be heavily censored; censor-compatible intervals are not confidence intervals.',
      'eta':a.eta
    }
    (out/'COMBINED_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\nCOMBINED REPORT WRITTEN TO',out)
    for fn in manifest['four_figures']:
        print('  -',out/fn)
    print('  -',out/'all_trials_combined.csv','(all raw rows copied, source trials.csv files are untouched)')

if __name__=='__main__': main()
