"""Independent *deterministic* numerical checks for key U finite-proof algebra.

These are regression tests/counterexample probes, NOT formal proofs,
NOT Monte Carlo experiments, and NOT a PAC validation.
"""
import itertools
import numpy as np
from onebit_bai.u_profile import sieve_penalty
from onebit_bai.experiment import compute_oracles


def test_monotone_sieve_downward_rounding_loss_with_boundary_probabilities():
    # Exhaustively check all bit patterns for an elementary 5-threshold profile.
    for n in (1,2,5):
        eta=.25*(n+1)**(-1/3); xi=(1/16)*(n+1)**(-2/3)
        J=int(np.ceil((1-eta/2)/xi)); Delta=(1-eta/2)/J
        grid=eta/4 + Delta*np.arange(J+1)
        raw=np.array([1.,.9,.51,.1,0.])[:n]
        smooth=(1-eta)*raw+eta/2
        rounded=np.array([grid[np.searchsorted(grid,u,side='right')-1] for u in smooth])
        assert np.all(np.diff(rounded)<=1e-15)
        a=-np.log1p(-eta)+4*xi/eta
        for labels in itertools.product((0.,1.),repeat=n):
            y=np.asarray(labels)
            def logscore(values):
                pos=values[y==1];neg=values[y==0]
                return float(np.log(pos).sum()+np.log1p(-neg).sum())
            # Impossible raw labels have score -inf, making inequality trivial.
            if all((y==0) | (raw>0)) and all((y==1) | (raw<1)):
                assert logscore(rounded)>=logscore(raw)-n*a-1e-10


def test_positive_deficit_does_not_increase_when_nominated_deficit_exceeds_s():
    # The pathwise aggregation inequality is purely deterministic once p sums <=1.
    s=.5
    for p in ((.2,.4,.3),(.25,.25,.25),(.001,.998,0.0)):
        pp=np.array(p,dtype=float)
        assert pp.sum()<=1+1e-10
        for D in itertools.product((-2.,-.7,-.1,.1,.7,2.),repeat=3):
            d=np.asarray(D,dtype=float)
            for a in range(3):
                if d[a] <= s:continue
                dn=d+s*(pp-np.eye(3)[a])
                old=np.maximum(d,0).sum();new=np.maximum(dn,0).sum()
                assert new<=old+1e-11


def test_critical_level_chords_allow_flat_h():
    # h(w)=min(a*w,plateau) is concave, monotone and *not* strictly increasing.
    s=.5; plateau=.8; a=4.; wstar=.25
    gamma_star=plateau
    eps=.1; gamma=gamma_star-eps
    c=gamma/a
    U=gamma/c
    for w in np.linspace(0,s,101):
        h=min(a*w,plateau)
        if w<=c:
            assert h<=gamma-(eps/s)*(c-w)+1e-12
        assert h>=gamma-U*max(c-w,0)-1e-12


def test_bv_right_jump_uses_closed_empirical_subcdf():
    # Left-continuous g jumps RIGHT at a; evaluating D(a)=Delta((0,a]) matters.
    a,b=.4,.7
    def g(q):return 2. if q<=a else 1.
    direct=g(a)-g(b)  # delta_a - delta_b
    D_B=0.; D_a=1.; dg_at_a=-1.  # g(a+)-g(a)
    via_formula=g(1.)*D_B-D_a*dg_at_a
    assert direct==via_formula==1.


def test_theory_certified_u_sieve_boundary_is_sublinear_in_limit_numerically():
    # NOT an asymptotic proof; tests numeric formula for the paper's sieve penalty.
    n=np.array([10_000,100_000,1_000_000],dtype=int)
    scaled=np.array([sieve_penalty(int(z))/z for z in n])
    assert np.isfinite(scaled).all() and np.all(np.diff(scaled)<0)


def test_population_beta_concentration_parameter_is_not_silently_ignored():
    mus=np.array([.8,.2])
    q=np.linspace(0,1,5)
    fast=compute_oracles('U',mus,.5,.2,q,9,11,False,'gaussian','beta',4.0)
    slow=compute_oracles('U',mus,.5,.2,q,9,11,False,'gaussian','beta',25.0)
    assert abs(fast['onebit_gamma']-slow['onebit_gamma'])>1e-4


def test_full_feedback_report_refuses_complete_case_selection_bias():
    import pandas as pd
    from onebit_bai.report import paired_full_comparison
    rows=[]
    # In the first pair, bit stopped but full is censored. Second pair is complete.
    for rep in (0,1):
        rows.append(dict(rep=rep,method='u_raw',delta=.05,stopped=1,tau=120+rep*5))
        rows.append(dict(rep=rep,method='u_full',delta=.05,stopped=int(rep==1),
                         tau=(80 if rep==1 else np.nan)))
    o=dict(onebit_gamma=.1,full_gamma=.15)
    result=paired_full_comparison(pd.DataFrame(rows),o)
    assert len(result)==1
    assert result.iloc[0].comparison_status=='INCOMPLETE_CENSORED_OR_TOO_FEW'
    assert np.isnan(result.iloc[0].empirical_mean_ratio)


def test_holdout_censored_before_threshold_is_unknown_not_success():
    import pandas as pd
    from onebit_bai.report import fitted_constant_holdout
    # Two complete training reps, then two validation reps: one is unknown
    # because its cap is earlier than the predicted quantile threshold.
    t=pd.DataFrame([
        dict(rep=0,method='u_raw',delta=.05,stopped=1,tau=20.,censor_at=100.),
        dict(rep=1,method='u_raw',delta=.05,stopped=1,tau=20.,censor_at=100.),
        dict(rep=2,method='u_raw',delta=.05,stopped=1,tau=1.,censor_at=100.),
        dict(rep=3,method='u_raw',delta=.05,stopped=0,tau=np.nan,censor_at=1.),
    ])
    h=fitted_constant_holdout(t,pd.DataFrame(),dict(onebit_gamma=.1,mus=[.8,.2]))
    assert len(h)==1
    row=h.iloc[0]
    assert row.holdout_unknown_censored==1
    assert row.holdout_coverage_lower==.5
    assert row.holdout_coverage_upper==1.
    assert np.isnan(row.holdout_P_tau_le_threshold)


def test_censored_correctness_has_no_eventual_upper_certification():
    import pandas as pd
    from onebit_bai.report import summarize
    t=pd.DataFrame([
        dict(rep=0,method='u_raw',delta=.05,stopped=1,tau=100.,error=0),
        dict(rep=1,method='u_raw',delta=.05,stopped=0,tau=np.nan,error=0),
    ])
    conf=dict(mus=[.8,.2],u_boundary='practical')
    r=summarize(t,conf,dict(onebit_gamma=.1,full_gamma=None),eta=.05)
    assert r.iloc[0].correctness_inference_status=='CENSORED_LOWER_BOUND_ON_EVENTUAL_ERROR'
    assert np.isnan(r.iloc[0].eventual_correctness_CP_upper_if_complete)
    assert np.isfinite(r.iloc[0].wrong_event_CP95_upper)  # ONLY through the cap.
