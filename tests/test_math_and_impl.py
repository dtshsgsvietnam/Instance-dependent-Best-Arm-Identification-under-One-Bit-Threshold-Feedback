"""Tests are numerical/unit checks, NOT verification of the paper theorems."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_legendre
from onebit_bai.u_profile import RawProfile,pair_glr_raw,sieve_penalty
from onebit_bai.oracles import (BetaLaw,BernoulliLaw,binary_kl,_Tlambda,
    gaussian_shared_grid_oracle,gaussian_full_oracle,bernoulli_shared_grid_oracle,
    bernoulli_full_oracle, u_population_oracle,u_full_population_oracle)
from onebit_bai.algorithms import U_marginal_cert_boundary,practical_boundary
from onebit_bai.report import clopper_pearson_upper


def test_raw_profile_dual_against_direct_constrained_optimization():
    q=np.array([.2,.4,.6,.8])
    pi=RawProfile(q,np.ones(4,dtype=int))
    pj=RawProfile(q,np.zeros(4,dtype=int))
    exact=pair_glr_raw(pi,pj)
    def neg_ll(s):
        si=s[:4];sj=s[4:]
        return -np.sum(np.log(np.maximum(si,1e-14)))-np.sum(np.log(np.maximum(1-sj,1e-14)))
    inequalities=[{'type':'ineq','fun':lambda s,i=i:s[i]-s[i+1]} for i in [0,1,2,4,5,6]]
    inequalities.append({'type':'ineq','fun':lambda s:pj.mean_max(s[4:])-pi.mean_min(s[:4])})
    sol=minimize(neg_ll,np.array([.6]*4+[.4]*4),method='SLSQP',
         constraints=inequalities,bounds=[(1e-9,1-1e-9)]*8,
         options={'maxiter':2000,'ftol':1e-10})
    assert sol.success
    np.testing.assert_allclose(exact,sol.fun,rtol=0,atol=1e-5)


def test_raw_glr_zero_when_unrestricted_fits_admit_null():
    pi=RawProfile(np.array([.2,.4,.8]),np.array([0,0,0]))
    pj=RawProfile(np.array([.2,.4,.8]),np.array([1,1,1]))
    assert pair_glr_raw(pi,pj)==0


def test_isotonic_profile_permutation_invariance():
    q=np.array([.1,.3,.5,.75]);y=np.array([1,0,1,0])
    a=RawProfile(q,y);b=RawProfile(q[[2,0,3,1]],y[[2,0,3,1]])
    np.testing.assert_allclose(a.s0,b.s0,atol=1e-12)
    np.testing.assert_allclose(a.M,b.M,atol=1e-12)


def test_transport_scalar_monotonicity():
    p=np.array([0.,.3,.7,1.]);lam=2
    a=_Tlambda(p,lam);b=_Tlambda(p,-lam)
    assert np.all(np.diff(a)>=0) and np.all(np.diff(b)>=0)
    assert np.all(a<=p+1e-12) and np.all(b>=p-1e-12)


def test_bernoulli_e_grid_oracle_equals_full():
    mus=[.75,.45,.25]
    _,ge=bernoulli_shared_grid_oracle(mus,.5,np.linspace(.1,.9,3),101)
    _,gf=bernoulli_full_oracle(mus,.5,101)
    np.testing.assert_allclose(ge,gf,rtol=1e-10,atol=1e-10)


def test_bernoulli_u_population_equals_bounded_full_information():
    laws=[BernoulliLaw(.8),BernoulliLaw(.2)]
    _,gu=u_population_oracle(laws,.5,qnodes=17,xcount=41)
    _,gf=u_full_population_oracle(laws,.5,xcount=41,nodes=17)
    np.testing.assert_allclose(gu,gf,atol=5e-5)


def test_current_status_observed_kl_is_richer_than_marginal_bit_kl():
    a,b=BetaLaw(.7,12),BetaLaw(.45,12)
    z,w=roots_legendre(141);q=(z+1)/2;w=w/w.sum()
    d=float(w@binary_kl(a.survival(q),b.survival(q)))
    marginal=float(binary_kl(a.mean,b.mean))
    assert d>=marginal-1e-9


def test_two_bounded_certificates_positive_and_CP_power():
    assert sieve_penalty(50)>0
    assert U_marginal_cert_boundary(100,.1,3)>practical_boundary(100,.1,3)*.5
    # 200 error-free replicates cannot empirically validate delta=1e-4.
    assert clopper_pearson_upper(0,200)>.01


def test_gaussian_onebit_cannot_outperform_full_at_pop_level():
    mus=[.7,.6,.45]
    _,g=gaussian_shared_grid_oracle(mus,.5,np.linspace(0,1,5),31,sigma=.2)
    _,gf=gaussian_full_oracle(mus,.5,sigma=.2)
    assert 0<g<=gf+1e-8


def test_closed_form_block_optimizer_matches_stationary_equation():
    from onebit_bai.u_profile import _block_opt
    rng=np.random.default_rng(1234)
    for _ in range(200):
        a=float(rng.integers(0,30)); b=float(rng.integers(0,30))
        if a+b==0: continue
        c=float(rng.normal()*50)
        s=float(_block_opt(a,b,c))
        assert -1e-12 <= s <= 1+1e-12
        if 1e-10 < s < 1-1e-10:
            grad=a/s - b/(1-s) + c
            assert abs(grad) < 1e-7
        elif s <= 1e-10:
            # right derivative at zero is non-positive only when a=0
            assert a==0
            assert c <= b + 1e-9
        else:
            assert b==0
            assert a+c >= -1e-9
