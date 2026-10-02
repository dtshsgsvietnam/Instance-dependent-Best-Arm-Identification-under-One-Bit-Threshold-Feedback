"""Focused reports for the four questions requested by the paper experiments.

Only reports:
Q1  one-bit first-order ratio E[tau] Gamma_1b / log(1/delta) -> 1;
Q3  empirical mean stopping-time ratio one-bit/full vs Gamma_full/Gamma_1b;
COR correctness / correct-stop proportion vs 1-delta;
FIN finite-confidence holdout coverage of a theorem-shaped bound.

For FIN, the paper constants are existential.  We therefore fit C on a training
split and evaluate coverage on an independent holdout split.  This is an
empirical calibration diagnostic, not a proof of the theorem's unknown C.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from .report import summarize, paired_full_comparison, e_remainder, u_remainder


def cp_interval(successes:int, n:int, alpha:float=.05):
    if n <= 0:
        return np.nan, np.nan
    a = alpha/2
    lo = 0.0 if successes == 0 else float(beta_dist.ppf(a, successes, n-successes+1))
    hi = 1.0 if successes == n else float(beta_dist.ppf(1-a, successes+1, n-successes))
    return lo, hi


def onebit_method(model:str):
    return 'e_tt' if model == 'E' else 'u_raw'


def nonasym_method(model:str):
    return 'e_companion' if model == 'E' else 'u_raw'


def correctness_method(model:str, trials:pd.DataFrame):
    # For U, the marginal-bit control is the readily runnable certified check.
    if model == 'U' and 'u_marginal' in set(trials.method):
        return 'u_marginal'
    return onebit_method(model)


def finite_holdout_focused(trials:pd.DataFrame, oracle:dict, model:str, eta:float=.05):
    method = nonasym_method(model)
    gg = trials[trials.method == method].copy()
    if gg.empty:
        return pd.DataFrame(), np.nan
    reps = sorted(gg.rep.unique())
    train = set(reps[::2]); test = set(reps[1::2])
    if not train or not test:
        return pd.DataFrame(), np.nan
    gamma = float(oracle['onebit_gamma'])
    T = 1.0/gamma
    K = len(oracle['mus'])
    Cs=[]
    for delta,g in gg[gg.rep.isin(train)].groupby('delta'):
        if not bool((g.stopped == 1).all()):
            continue
        L=np.log(1/delta)
        R=e_remainder(L,K,eta) if model=='E' else u_remainder(L,T,eta)
        q=float(np.quantile(g.tau.to_numpy(float),1-3*eta))
        Cs.append(max(0.0,(q-T*L)/R))
    if not Cs:
        return pd.DataFrame(), np.nan
    Cfit=max(Cs)
    rows=[]
    for delta,g in gg[gg.rep.isin(test)].groupby('delta'):
        L=np.log(1/delta)
        R=e_remainder(L,K,eta) if model=='E' else u_remainder(L,T,eta)
        bound=T*L+Cfit*R
        success=(g.stopped==1)&(g.tau<=bound)
        # If cap is below the proposed bound and the run is censored, its status
        # relative to tau<=bound is genuinely unknown.
        unknown=(g.stopped==0)&(g.censor_at<bound)
        n=len(g); s=int(success.sum()); u=int(unknown.sum())
        raw_lo=s/n; raw_hi=(s+u)/n
        if u==0:
            cp_lo,cp_hi=cp_interval(s,n)
            observed=s/n
        else:
            cp_lo=cp_hi=observed=np.nan
        rows.append({
            'method':method,'delta':float(delta),'eta':eta,
            'nominal_probability':1-3*eta,'C_fitted_train':Cfit,
            'holdout_reps':n,'successes':s,'unknown_censored':u,
            'bound':bound,'coverage':observed,
            'coverage_raw_lower':raw_lo,'coverage_raw_upper':raw_hi,
            'coverage_CP95_lower':cp_lo,'coverage_CP95_upper':cp_hi,
            'empirical_95pct_supports_nominal':bool(cp_lo >= 1-3*eta) if np.isfinite(cp_lo) else False,
            'NOTE':'C is fitted on independent training reps; this checks theorem-shape calibration, not the unknown theorem constant.'
        })
    return pd.DataFrame(rows).sort_values('delta',ascending=False), Cfit


def _save_asymptotic(summary, root, model):
    method=onebit_method(model)
    h=summary[(summary.method==method)&summary.first_order_ratio.notna()].sort_values('log1d')
    if h.empty:return
    fig,ax=plt.subplots(figsize=(7.4,4.6))
    yerr=np.vstack([h.first_order_ratio-h.first_order_ratio_CI_lo,
                    h.first_order_ratio_CI_hi-h.first_order_ratio])
    ax.errorbar(h.log1d,h.first_order_ratio,yerr=yerr,marker='o',capsize=3,label=method)
    ax.axhline(1,linestyle='--',linewidth=1,label='theory limit = 1')
    ax.set_xlabel('log(1/delta)')
    ax.set_ylabel('mean(tau_1bit) * Gamma_1bit / log(1/delta)')
    ax.set_title('Q1: one-bit first-order ratio')
    ax.legend();fig.tight_layout();fig.savefig(root/'01_Q1_onebit_asymptotic.png',dpi=170);plt.close(fig)


def _save_full_ratio(paired, root, oracle, model):
    # Q1/correctness-only cases may still have a population full-feedback oracle
    # even though no full-feedback Monte Carlo method was run. In that case
    # paired_full_comparison() legitimately returns an empty DataFrame (with no
    # columns), so there is nothing to plot.
    required={'onebit','full','empirical_mean_ratio','delta','bootstrap_CI_lo','bootstrap_CI_hi'}
    if paired is None or paired.empty or not required.issubset(set(paired.columns)):
        return
    one=onebit_method(model); full='e_full' if model=='E' else 'u_full'
    h=paired[(paired['onebit']==one)&(paired['full']==full)&paired['empirical_mean_ratio'].notna()].copy()
    if h.empty:return
    h['L']=np.log(1/h.delta)
    h=h.sort_values('L')
    th=float(oracle['full_gamma'])/float(oracle['onebit_gamma'])
    fig,ax=plt.subplots(figsize=(7.4,4.6))
    yerr=np.vstack([h.empirical_mean_ratio-h.bootstrap_CI_lo,
                    h.bootstrap_CI_hi-h.empirical_mean_ratio])
    ax.errorbar(h.L,h.empirical_mean_ratio,yerr=yerr,marker='o',capsize=3,label='empirical mean tau ratio')
    ax.axhline(th,linestyle='--',linewidth=1,label=f'theory Gamma_full/Gamma_1bit = {th:.3g}')
    ax.set_xlabel('log(1/delta)')
    ax.set_ylabel('mean(tau_1bit) / mean(tau_full)')
    ax.set_title('Q3: one-bit / full sample-cost ratio')
    ax.legend();fig.tight_layout();fig.savefig(root/'02_Q3_onebit_vs_full.png',dpi=170);plt.close(fig)


def _save_correctness(trials,root,model):
    method=correctness_method(model,trials)
    rows=[]
    for delta,g in trials[trials.method==method].groupby('delta'):
        n=len(g); stopped=int(g.stopped.sum()); correct=int(((g.stopped==1)&(g.error==0)).sum())
        complete=stopped==n
        if complete:
            p=correct/n; lo,hi=cp_interval(correct,n)
        else:
            p=correct/n; lo=hi=np.nan
        rows.append({'delta':delta,'n':n,'stopped':stopped,'correct_by_cap':correct,
                     'correctness':p,'CP95_lo':lo,'CP95_hi':hi,'complete':complete})
    d=pd.DataFrame(rows).sort_values('delta',ascending=False)
    d.to_csv(root/'correctness_focused.csv',index=False)
    if d.empty:return d,method
    fig,ax=plt.subplots(figsize=(7.4,4.6))
    good=d[d.complete]
    if not good.empty:
        yerr=np.vstack([good.correctness-good.CP95_lo,good.CP95_hi-good.correctness])
        ax.errorbar(good.delta,good.correctness,yerr=yerr,marker='o',capsize=3,label=f'{method}: empirical correctness (95% CP)')
    bad=d[~d.complete]
    if not bad.empty:
        ax.scatter(bad.delta,bad.correctness,marker='x',label='correct-stop by cap (censored)')
    x=np.sort(d.delta.unique())
    ax.plot(x,1-x,linestyle='--',label='target 1-delta')
    ax.set_xscale('log');ax.set_ylim(0,1.02)
    ax.set_xlabel('delta');ax.set_ylabel('probability of correct recommendation')
    ax.set_title('Correctness: correct-stop rate vs 1-delta')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(root/'03_correctness.png',dpi=170);plt.close(fig)
    return d,method


def _save_finite_coverage(finite,root):
    if finite.empty:return
    fig,ax=plt.subplots(figsize=(7.4,4.6))
    h=finite[finite.coverage.notna()].copy()
    if not h.empty:
        h['L']=np.log(1/h.delta);h=h.sort_values('L')
        yerr=np.vstack([h.coverage-h.coverage_CP95_lower,h.coverage_CP95_upper-h.coverage])
        ax.errorbar(h.L,h.coverage,yerr=yerr,marker='o',capsize=3,label='holdout coverage (95% CP)')
        nominal=float(h.nominal_probability.iloc[0])
        ax.axhline(nominal,linestyle='--',linewidth=1,label=f'nominal 1-3eta = {nominal:.2f}')
        ax.set_ylim(0,1.02);ax.set_xlabel('log(1/delta)');ax.set_ylabel('P_holdout(tau <= fitted bound)')
        ax.set_title('Non-asymptotic probability-bound coverage')
        ax.legend();fig.tight_layout();fig.savefig(root/'04_nonasymptotic_coverage.png',dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description='Focused report: Q1, Q3, correctness, finite-probability coverage')
    p.add_argument('--input',required=True)
    p.add_argument('--eta',type=float,default=.05)
    a=p.parse_args();root=Path(a.input)
    trials=pd.read_csv(root/'trials.csv')
    conf=json.loads((root/'config.json').read_text())
    oracle=json.loads((root/'oracles.json').read_text())
    model=oracle['model']
    summary=summarize(trials,conf,oracle,a.eta)
    summary.to_csv(root/'summary_focused.csv',index=False)
    paired=paired_full_comparison(trials,oracle)
    paired.to_csv(root/'paired_full_comparison_focused.csv',index=False)
    finite,Cfit=finite_holdout_focused(trials,oracle,model,a.eta)
    finite.to_csv(root/'nonasymptotic_coverage_focused.csv',index=False)
    corr,corr_method=_save_correctness(trials,root,model)
    _save_asymptotic(summary,root,model)
    if oracle.get('full_gamma') is not None:_save_full_ratio(paired,root,oracle,model)
    _save_finite_coverage(finite,root)

    # Concise terminal report -- intentionally only the user's four questions.
    print('\n'+'='*74)
    print('FOCUSED RESULTS')
    print('='*74)
    m=onebit_method(model)
    a1=summary[(summary.method==m)&summary.first_order_ratio.notna()][['delta','reps','stopped','first_order_ratio','first_order_ratio_CI_lo','first_order_ratio_CI_hi']]
    print('\n[Q1] One-bit first-order ratio: mean(tau)*Gamma_1bit/log(1/delta) -> 1')
    print(a1.to_string(index=False) if not a1.empty else 'No complete Q1 cells.')
    if not paired.empty and oracle.get('full_gamma') is not None:
        p3=paired[(paired.onebit==m)&paired.empirical_mean_ratio.notna()][['delta','paired_reps','empirical_mean_ratio','bootstrap_CI_lo','bootstrap_CI_hi','population_predicted_ratio']]
        print('\n[Q3] one-bit/full ratio: empirical vs Gamma_full/Gamma_1bit')
        print(p3.to_string(index=False) if not p3.empty else 'No complete paired cells.')
    print(f'\n[CORRECTNESS] method={corr_method}; target is >= 1-delta')
    print(corr.to_string(index=False) if not corr.empty else 'No correctness rows.')
    print(f'\n[NON-ASYMPTOTIC] eta={a.eta:g}, nominal probability=1-3eta={1-3*a.eta:.3f}')
    if np.isfinite(Cfit):print(f'Fitted C on TRAIN split only: {Cfit:.6g}')
    print(finite.to_string(index=False) if not finite.empty else 'No evaluable finite-bound holdout cells.')
    print('\nFiles kept intentionally small:')
    for fn in ['01_Q1_onebit_asymptotic.png','02_Q3_onebit_vs_full.png','03_correctness.png','04_nonasymptotic_coverage.png','summary_focused.csv','paired_full_comparison_focused.csv','correctness_focused.csv','nonasymptotic_coverage_focused.csv']:
        if (root/fn).exists(): print('  -',root/fn)
    print('='*74)

if __name__=='__main__':main()
