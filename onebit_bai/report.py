"""Censor-aware asymptotic, quantile, empirical finite-bound, and PAC diagnostics.

Crucial distinction: paper's E/U finite-bound constants are *existential*,
therefore plots of fitted coefficients are empirical diagnostics, NOT
numerically evaluated certified finite-confidence inequalities.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from .algorithms import (practical_boundary,E_cert_boundary,U_cert_boundary,
                         U_marginal_cert_boundary,full_bounded_cert_boundary)


def clopper_pearson_upper(events:int, n:int, alpha:float=0.05):
    if n<=0:return np.nan
    if events>=n:return 1.0
    return float(beta_dist.ppf(1-alpha,events+1,n-events))


def ci_bootstrap_mean(data, reps=1000,seed=246):
    data=np.asarray(data,dtype=float)
    if data.size<2:return (np.nan,np.nan)
    rng=np.random.default_rng(seed)
    bmeans=np.mean(rng.choice(data,(reps,len(data)),replace=True),axis=1)
    return tuple(np.quantile(bmeans,[.025,.975]))


def e_remainder(L,K,eta):
    h=1+K*np.log(np.e+L)
    A=h+1+np.log(1/eta)
    return L**(2/3)*np.sqrt(A)+np.sqrt(L*A)+A+(1+np.log(1/eta))**1.5+1


def u_remainder(L,T,eta,CT=1):
    # CT is existential in the paper: default CT=1 is an illustrative basis!
    q=CT*eta**(-.25)
    Q=q**1.5
    z=1+T*L+Q
    return Q+z**.75*np.log(np.e*z)**1.5+1


def solve_predictor(Gamma, boundary,cap=10_000_000):
    if Gamma is None or Gamma<=0:return np.nan
    hi=8
    while hi*Gamma < boundary(hi) and hi<cap:hi*=2
    if hi>=cap:return np.nan
    lo=1
    while lo<hi:
        mid=(lo+hi)//2
        if mid*Gamma>=boundary(mid):hi=mid
        else:lo=mid+1
    return int(lo)


def rate_for(method,oracle):
    if method in ('e_tt','e_companion','u_raw'):return oracle['onebit_gamma']
    if method in ('e_full','u_full'):return oracle['full_gamma']
    return None  # u_marginal is a validity-only control, not a Gamma_U optimizer


def predictor_boundary(method,n,delta,conf,K):
    if method in ('e_tt','e_companion'):
        return (E_cert_boundary(n,delta,K,conf['mu_lo'],conf['mu_hi'],
                               conf['sigma'],np.linspace((.1 if conf['e_family']=='bernoulli' else conf['qmin']),
                                           (.9 if conf['e_family']=='bernoulli' else conf['qmax']),
                                           conf['qpoints']),conf['e_family'])
                if conf['e_boundary']=='certified' else practical_boundary(n,delta,K))
    if method=='u_raw':
        return U_cert_boundary(n,delta,K) if conf['u_boundary']=='certified' else practical_boundary(n,delta,K)
    if method=='u_full':
        return (full_bounded_cert_boundary(n,delta,K) if conf['full_boundary']=='certified'
                else practical_boundary(n,delta,K))
    if method=='e_full':
        return (full_bounded_cert_boundary(n,delta,K) if conf['full_boundary']=='certified' and
                conf['e_family']=='bernoulli' else practical_boundary(n,delta,K))
    return U_marginal_cert_boundary(n,delta,K)


def summarize(trials,conf,oracle,eta=.05,alpha=.05):
    K=len(conf['mus'])
    rows=[]
    for (method,delta),g in trials.groupby(['method','delta']):
        g=g.sort_values('rep')
        n=len(g);stopped=int(g.stopped.sum());wrong=int(g.error.sum())
        x=g.loc[g.stopped==1,'tau'].to_numpy(dtype=float)
        gamma=rate_for(method,oracle)
        L=np.log(1/delta)
        complete=(stopped==n)
        mean=float(x.mean()) if complete else np.nan
        se_tau=(float(x.std(ddof=1))/np.sqrt(len(x))) if complete and len(x)>1 else np.nan
        q_targets=[.1,.5,.9,.95,.99,1-3*eta]
        quantiles={f'q{int(round(100*q)):02}':float(np.quantile(x,q)) if complete else np.nan
                   for q in q_targets}
        lo,hi=ci_bootstrap_mean(x) if complete else (np.nan,np.nan)
        r=(mean*gamma/L) if complete and gamma else np.nan
        stopped_rows=g[g.stopped==1]
        mean_lambda_stop=float(stopped_rows.lambda_at_stop.mean()) if ('lambda_at_stop' in stopped_rows and len(stopped_rows)) else np.nan
        mean_stop_over_boundary=float(stopped_rows.stop_stat_over_boundary.mean()) if ('stop_stat_over_boundary' in stopped_rows and len(stopped_rows)) else np.nan
        mean_boundary_over_L=float(stopped_rows.boundary_over_L.mean()) if ('boundary_over_L' in stopped_rows and len(stopped_rows)) else np.nan
        lambda_stop_over_gamma=(mean_lambda_stop/gamma) if gamma and np.isfinite(mean_lambda_stop) else np.nan
        pred=solve_predictor(gamma,lambda t:predictor_boundary(method,t,delta,conf,K)) if gamma else np.nan
        targetquant=quantiles[f'q{int(round(100*(1-3*eta))):02}']
        if method=='e_companion' and complete:
            remainder=e_remainder(L,K,eta)
        elif method=='u_raw' and complete:
            remainder=u_remainder(L,1/gamma,eta) if gamma else np.nan
        else:remainder=np.nan
        implied_C=max(0,(targetquant-L/gamma)/remainder) if (
            gamma and complete and np.isfinite(remainder) and remainder>0) else np.nan
        rows.append({'method':method,'delta':delta,'log1d':L,'reps':n,'stopped':stopped,
          'stop_rate':stopped/n,'cap_hit_rate':1-stopped/n,'wrong':wrong,'error_observed':wrong/n,
          'error_per_delta':wrong/(n*delta),
          'wrong_event_CP95_upper':clopper_pearson_upper(wrong,n,alpha),
          'eventual_correctness_CP_upper_if_complete':clopper_pearson_upper(wrong,n,alpha) if complete else np.nan,
          'correctness_inference_status':'COMPLETE' if complete else 'CENSORED_LOWER_BOUND_ON_EVENTUAL_ERROR',
          'zero_error_n_needed_at_this_delta':int(np.ceil(np.log(alpha)/np.log1p(-delta))),
          'mean_tau_uncensored_only':mean,'se_tau':se_tau,'mean_CI_lo':lo,'mean_CI_hi':hi,
          'gamma_population_approx':gamma,'first_order_ratio':r,
          'first_order_ratio_CI_lo':lo*gamma/L if gamma else np.nan,
          'first_order_ratio_CI_hi':hi*gamma/L if gamma else np.nan,
          'fixed_point_predictor':pred,'mean_to_predictor':mean/pred if complete and pred else np.nan,
          'implied_C_quantile':implied_C,'finite_remainder_basis':remainder,
          'quantile_ratio_to_TL':targetquant*gamma/L if complete and gamma else np.nan,
          'mean_lambda_at_stop':mean_lambda_stop,'lambda_at_stop_over_gamma':lambda_stop_over_gamma,
          'mean_stop_stat_over_boundary':mean_stop_over_boundary,'mean_boundary_over_log1d':mean_boundary_over_L,
          **quantiles})
    result=pd.DataFrame(rows).sort_values(['method','delta'],ascending=[True,False])
    result['adjacent_incremental_slope_ratio']=np.nan
    for method,group in result.groupby('method'):
        group=group.sort_values('log1d')
        prev=None
        for idx,row in group.iterrows():
            if prev is not None and pd.notna(row.mean_tau_uncensored_only) and pd.notna(prev.mean_tau_uncensored_only) and pd.notna(row.gamma_population_approx):
                dx=row.log1d-prev.log1d
                if dx>0:result.loc[idx,'adjacent_incremental_slope_ratio']=(row.mean_tau_uncensored_only-prev.mean_tau_uncensored_only)*row.gamma_population_approx/dx
            prev=row
    return result


def fitted_constant_holdout(trials, summary,oracle,eta=.05):
    """Exploratory train/test validation of a *data-fitted* finite overhead.

    This is never a proof of paper's unknown instance constants. Pairing is
    preserved via replication IDs, and only complete train/test paths count.
    """
    rows=[]
    for method in ('e_companion','u_raw'):
        if method not in set(trials.method):continue
        gg=trials[trials.method==method]
        reps=sorted(gg.rep.unique());cut=len(reps)//2
        train=set(reps[:cut]);test=set(reps[cut:])
        if not train or not test:continue
        rate=rate_for(method,oracle)
        if not rate:continue
        # Coefficient trained across all delta; predict holdout at each delta.
        Cs=[]
        for delta, g in gg[gg.rep.isin(train)].groupby('delta'):
            if not np.all(g.stopped):continue
            L=np.log(1/delta);T=1/rate
            R=(e_remainder(L,len(oracle['mus']),eta) if method=='e_companion' else
               u_remainder(L,T,eta))
            q=np.quantile(g.tau,1-3*eta)
            Cs.append(max(0,(q-T*L)/R))
        if not Cs:continue
        C=max(Cs)
        for delta,g in gg[gg.rep.isin(test)].groupby('delta'):
            L=np.log(1/delta);T=1/rate
            R=e_remainder(L,len(oracle['mus']),eta) if method=='e_companion' else u_remainder(L,T,eta)
            thr=T*L+C*R
            # A cap earlier than the proposed threshold makes this replicate
            # UNKNOWABLE if it has not stopped; never condition on the fast
            # stopped subset, as doing so produces optimistic coverage.
            success=(g.stopped==1)&(g.tau<=thr)
            unknown=(g.stopped==0)&(g.censor_at<thr)
            known=int(len(g)-unknown.sum())
            lo=float(success.sum()/len(g));hi=float((success.sum()+unknown.sum())/len(g))
            coverage=lo if not bool(unknown.any()) else np.nan
            rows.append({'method':method,'delta':delta,'training_reps':len(train),
                         'holdout_reps':len(test),'C_fitted_train_max':C,
                         'holdout_threshold':thr,'holdout_evaluable':known,
                         'holdout_P_tau_le_threshold':coverage,
                         'holdout_coverage_lower':lo,'holdout_coverage_upper':hi,
                         'holdout_unknown_censored':int(unknown.sum()),
                         'nominal_1_minus_3eta':1-3*eta,
                         'WARNING':'empirical coefficient, conservative censor intervals; NOT a proved bound'})
    return pd.DataFrame(rows)


def plot_report(data,out,oracle,conf):
    # No prescribed colors: let matplotlib manage distinct method cycles.
    fig,ax=plt.subplots(figsize=(8,5))
    has=False
    for method,g in data.groupby('method'):
        h=g.dropna(subset=['first_order_ratio']).sort_values('log1d')
        if h.empty:continue
        has=True
        ax.errorbar(h.log1d,h.first_order_ratio,
               yerr=[h.first_order_ratio-h.first_order_ratio_CI_lo,
                     h.first_order_ratio_CI_hi-h.first_order_ratio],
               label=method,marker='o',capsize=3)
    if has:
        ax.axhline(1,linestyle='--',linewidth=1,label='first-order reference = 1')
        ax.set_xlabel('log(1/delta)')
        ax.set_ylabel('mean(tau) * population Gamma / log(1/delta)')
        ax.set_title('First-order convergence diagnostic (complete cells only)')
        ax.legend();fig.tight_layout();fig.savefig(out/'asymptotic_ratio.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));has=False
    for method,g in data.groupby('method'):
        h=g.dropna(subset=['adjacent_incremental_slope_ratio']).sort_values('log1d')
        if h.empty:continue
        has=True;ax.plot(h.log1d,h.adjacent_incremental_slope_ratio,marker='o',label=method)
    if has:
        ax.axhline(1,linestyle='--',linewidth=1)
        ax.set_xlabel('log(1/delta) (right endpoint of adjacent pair)')
        ax.set_ylabel('Gamma * Delta mean(tau) / Delta log(1/delta)')
        ax.set_title('Incremental slope diagnostic (high noise with few reps)')
        ax.legend();fig.tight_layout();fig.savefig(out/'incremental_slope.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));has=False
    for method,g in data.groupby('method'):
        h=g.dropna(subset=['mean_to_predictor']).sort_values('log1d')
        if h.empty:continue
        has=True;ax.plot(h.log1d,h.mean_to_predictor,marker='o',label=method)
    if has:
        ax.axhline(1,linestyle='--',linewidth=1)
        ax.set_xlabel('log(1/delta)');ax.set_ylabel('mean(tau) / numerical n_pred')
        ax.set_title('Empirical agreement with finite fixed-point predictor')
        ax.legend();fig.tight_layout();fig.savefig(out/'finite_predictor.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));has=False
    for method,g in data.groupby('method'):
        h=g.sort_values('delta',ascending=False)
        if h.empty:continue
        has=True
        ax.plot(h.delta,h.error_observed,marker='o',label=f'{method} empirical')
        ax.plot(h.delta,h.eventual_correctness_CP_upper_if_complete,linestyle=':',label=f'{method} complete-case CP upper')
    if has:
        x=np.sort(data.delta.unique())
        ax.plot(x,x,linestyle='--',label='target delta (only for certified policies)')
        ax.set_xscale('log');ax.set_yscale('symlog',linthresh=.001)
        ax.set_xlabel('delta');ax.set_ylabel('error probability / exact CP95 upper')
        ax.set_title('Correctness: underpowered for very small delta')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'correctness.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));has=False
    for method,g in data.groupby('method'):
        h=g.dropna(subset=['quantile_ratio_to_TL']).sort_values('log1d')
        if h.empty:continue
        has=True;ax.plot(h.log1d,h.quantile_ratio_to_TL,marker='o',label=method)
    if has:
        ax.axhline(1,linestyle='--',linewidth=1)
        ax.set_xlabel('log(1/delta)');ax.set_ylabel('empirical q_(1-3 eta) / (T*log(1/delta))')
        ax.set_title('Finite-confidence quantiles (NOT theorem upper bounds)')
        ax.legend();fig.tight_layout();fig.savefig(out/'finite_quantiles.png',dpi=160)
    plt.close(fig)


def paired_full_comparison(trials, oracle):
    """Bootstrap paired ratio of means, never ratio of per-replication means."""
    pairs=[('e_tt','e_full'),('u_raw','u_full')]
    rows=[]
    for one,full in pairs:
        if not {one,full}.issubset(set(trials.method)) or oracle['full_gamma'] is None:continue
        for delta in sorted(trials.delta.unique(),reverse=True):
            a=trials[(trials.method==one)&(trials.delta==delta)][['rep','tau','stopped']]
            b=trials[(trials.method==full)&(trials.delta==delta)][['rep','tau','stopped']]
            m=a.merge(b,on='rep',suffixes=('_bit','_full'))
            # Do not silently drop censored pairs: complete-case selection
            # biases the full-vs-bit mean ratio toward fast trajectories.
            if len(m)<2 or not bool(((m.stopped_bit==1)&(m.stopped_full==1)).all()):
                rows.append({'onebit':one,'full':full,'delta':delta,
                  'paired_reps':len(m),'empirical_mean_ratio':np.nan,
                  'bootstrap_CI_lo':np.nan,'bootstrap_CI_hi':np.nan,
                  'population_predicted_ratio':oracle['full_gamma']/oracle['onebit_gamma'],
                  'comparison_status':'INCOMPLETE_CENSORED_OR_TOO_FEW'})
                continue
            num=m.tau_bit.to_numpy();den=m.tau_full.to_numpy()
            ratio=num.mean()/den.mean()
            rng=np.random.default_rng(27000+int(delta*1e6))
            picks=rng.integers(0,len(m),(1000,len(m)))
            boot=num[picks].mean(axis=1)/den[picks].mean(axis=1)
            lo,hi=np.quantile(boot,[.025,.975])
            rows.append({'onebit':one,'full':full,'delta':delta,'paired_reps':len(m),
                         'empirical_mean_ratio':ratio,'bootstrap_CI_lo':lo,'bootstrap_CI_hi':hi,
                         'population_predicted_ratio':oracle['full_gamma']/oracle['onebit_gamma'],
                         'comparison_status':'COMPLETE_PAIRED'})
    columns=['onebit','full','delta','paired_reps','empirical_mean_ratio',
             'bootstrap_CI_lo','bootstrap_CI_hi','population_predicted_ratio',
             'comparison_status']
    return pd.DataFrame(rows, columns=columns)


def write_report(summary,holdout,paired,oracle,conf,out,eta):
    model=oracle['model']
    def fmt(x):return 'NA' if pd.isna(x) else f'{x:.5g}'
    lines=[f'# AISTATS one-bit {model} Monte Carlo audit', '',
       '## Population numerical oracle (not a computer-assisted proof)', '',
       f"- Gamma_onebit = {fmt(oracle['onebit_gamma'])}",
       f"- Gamma_full = {fmt(oracle['full_gamma']) if oracle['full_gamma'] is not None else 'not computed'}",
       f"- Predicted full / one-bit sample cost = {fmt(oracle['full_gamma']/oracle['onebit_gamma']) if oracle['full_gamma'] is not None else 'not computed'}",
       f"- {oracle['oracle_note']}",'',
       '## What a first-order ratio means','',
       r'The plotted ratio is E[tau_delta] * Gamma / log(1/delta). A theorem predicts a limit of one for its EXACT policy and CERTIFIED boundary as delta -> 0. This finite-grid/periodic implementation is a numerical approximation; visible downward/upward movements at moderate delta do not confirm or refute that limit.', '',
       '## Quantiles and empirically implied constants','',
       f'For eta={eta}, the table uses the empirical (1-3eta)={1-3*eta:.2f} quantile. For E companion and U raw, `implied_C_quantile` is fitted to the THEOREM\'S SHAPE only. E/U published constants are unspecified, so the coefficient is NOT a certified finite-sample bound.', '',
       '| Method | delta | stopped / n | observed errors | one-sided CP95 | ratio mean | lambda_stop/Gamma | stat/boundary | boundary/L | q50 | q90 | q95 | q99 | implied C |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _,row in summary.iterrows():
        lines.append(f"| {row.method} | {fmt(row.delta)} | {int(row.stopped)}/{int(row.reps)} | {int(row.wrong)} | {fmt(row.wrong_event_CP95_upper)} | {fmt(row.first_order_ratio)} | {fmt(row.lambda_at_stop_over_gamma)} | {fmt(row.mean_stop_stat_over_boundary)} | {fmt(row.mean_boundary_over_log1d)} | {fmt(row.q50)} | {fmt(row.q90)} | {fmt(row.q95)} | {fmt(row.q99)} | {fmt(row.implied_C_quantile)} |")
    lines += ['', '## Correctness interpretation', '',
       'The empirical wrong-event rate counts runs which STOP AND ERR by the simulation cap. For censored cells, wrong_event_CP95_upper is a confidence bound only for errors observed BY THE CAP, not for eventual error; eventual_correctness_CP_upper_if_complete is intentionally NA. If any runs are censored, that statistic cannot upper-bound the eventual error probability or establish almost-sure stopping. Even with zero errors, 200 replications yield a one-sided 95% CP upper bound about 1.5%; validating delta=1e-4 at the same confidence with zero errors requires about 29,956 independent replications.',
       'Only E with its parametric certified boundary, U with its profile-certified boundary, and the marginal-bit conservative control claim a mathematical finite-time correctness certificate (assuming the exact model/GLR). `u_raw` with `--u-boundary practical`, and full Gaussian with practical boundary, have no such certificate from this code.',
       '', '## Holdout check of *empirically fitted* finite-bound shapes', '',
       'Training-set constants are chosen after seeing data. When a trajectory is censored before the proposed quantile threshold, exact holdout coverage is UNKNOWN. We report conservative lower/upper fractions over ALL test replications, never conditional complete-case coverage. This analysis does not make the unknown theorem constants explicit.', '']
    if holdout.empty:lines.append('No eligible complete holdout cells.')
    else:
        lines += ['|Method|delta|fitted C|holdout coverage lower--upper|unknown/test|', '|---|---:|---:|---:|---:|']
        for _,r in holdout.iterrows():
            lines.append(f"|{r.method}|{fmt(r.delta)}|{fmt(r.C_fitted_train_max)}|[{fmt(r.holdout_coverage_lower)}, {fmt(r.holdout_coverage_upper)}]|{int(r.holdout_unknown_censored)}/{int(r.holdout_reps)}|")
    lines+=['','## Paired full-feedback comparison','',
          'Predicted ratio Gamma_full/Gamma_onebit assumes a matching fixed-beta policy for each *true* observation model and asymptotically vanishing oracle error; paired finite-delta sample means need not equal it.', '']
    if paired.empty:lines.append('No eligible full-feedback paired comparison.')
    else:
        lines+=['|methods|delta|paired n|mean-tau ratio (bit/full)|paired bootstrap 95%|predicted Gamma ratio|status|', '|---|---:|---:|---:|---:|---:|---|']
        for _,r in paired.iterrows():
            lines.append(f'|{r.onebit}/{r.full}|{fmt(r.delta)}|{int(r.paired_reps)}|{fmt(r.empirical_mean_ratio)}|[{fmt(r.bootstrap_CI_lo)}, {fmt(r.bootstrap_CI_hi)}]|{fmt(r.population_predicted_ratio)}|{r.comparison_status}|')
    lines+=['','## Protocol notes','',
         'One replication shares per-arm LOCAL potential-reward streams across algorithms. Thresholds in E are independently generated by each arm-threshold policy. Multiple delta stopping levels share a policy trajectory, so method-by-delta rows are paired and not independent observations. Bootstrap a replication as a block across methods and deltas for final cross-method comparisons.',
         f"Algorithm update every {conf['refresh']} steps, stop check every {conf['decide']}; continuum shared E oracle and exact certified U sieve boundary are not equivalent to numerical approximations unless explicitly selected and verified."]
    (out/'ANALYSIS.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True,help='Directory containing trials.csv/config.json/oracles.json')
    parser.add_argument('--eta',type=float,default=.05)
    args=parser.parse_args();root=Path(args.input)
    trials=pd.read_csv(root/'trials.csv')
    conf=json.loads((root/'config.json').read_text())
    oracle=json.loads((root/'oracles.json').read_text())
    summ=summarize(trials,conf,oracle,args.eta)
    summ.to_csv(root/'summary.csv',index=False)
    holdout=fitted_constant_holdout(trials,summ,oracle,args.eta)
    holdout.to_csv(root/'finite_holdout.csv',index=False)
    paired=paired_full_comparison(trials,oracle)
    paired.to_csv(root/'paired_full_comparison.csv',index=False)
    plot_report(summ,root,oracle,conf)
    write_report(summ,holdout,paired,oracle,conf,root,args.eta)
    print(f'Reports: {root}/ANALYSIS.md and summary.csv, finite_holdout.csv, 5 PNG charts')

if __name__=='__main__':main()
