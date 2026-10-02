"""Executable, δ-independent sampling policies for the two AISTATS models.

E: Gaussian canonical one-parameter exponential family, discretized shared
query oracle; its approximation is *not* the exact regularized continuum oracle.
U: arbitrary bounded generator (Beta used in CLI), true unbinned raw-profile
GLR with a numerical generalized-PAVA constrained fit; periodic update is
optional computational approximation (set --refresh 1 for per-round updates).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from .oracles import gaussian_shared_grid_oracle,bernoulli_shared_grid_oracle, binary_kl, gaussian_full_oracle
from .u_profile import RawProfile, pair_glr_raw, sieve_penalty


def practical_boundary(n, delta, K):
    return np.log(K-1)+2*np.log(np.pi)+2*np.log(max(1,n))-np.log(6)-np.log(delta)


def E_cert_boundary(n,delta,K,mu_lo,mu_hi,sigma,qgrid,family="gaussian"):
    # Theorem Ecal uses the NATURAL exponential-family parameter θ = μ/σ².
    # Directly covering μ yields the same form, Gμ=max |∂μ log Ber(pq(μ))|.
    from scipy.stats import norm
    if family=="bernoulli":
        assert 0<mu_lo<mu_hi<1
        G=max(1/mu_lo,1/(1-mu_hi))
    else:
        # PROVABLE global log-likelihood Lipschitz bound on compact
        # (mu,q) region, not a possibly UNDERestimating numerical grid.
        # Inverse Mills ratio phi(z)/Phi(z) decreases in z; the
        # complementary ratio phi(z)/Phi(-z) increases in z.
        from scipy.special import log_ndtr
        q=np.asarray(qgrid)
        zmin=(mu_lo-float(q.max()))/sigma
        zmax=(mu_hi-float(q.min()))/sigma
        G=max(np.exp(norm.logpdf(zmin)-log_ndtr(zmin)),
              np.exp(norm.logpdf(zmax)-log_ndtr(-zmax)))/sigma
    R=mu_hi-mu_lo
    return 1+2*np.log(np.pi)-np.log(6)-np.log(delta)+K*np.log(max(1,n))+K*np.log(2+G*R*n)


def U_cert_boundary(n,delta,K):
    return sieve_penalty(n)+practical_boundary(n,delta,K)


def U_marginal_cert_boundary(n,delta,K):
    # Jourdan et al. bounded-observation GLR calibration (2022, Lemma 2).
    # This is δ-correct for the MARGINAL bit-GLR only, not raw-profile GLR.
    return np.log(1/delta)+2*np.log1p(n/2)+2+np.log(K-1)


def full_bounded_cert_boundary(n,delta,K):
    return U_marginal_cert_boundary(n,delta,K)


def gaussian_mle(q,y,lo,hi,sigma):
    q=np.asarray(q)
    y=np.asarray(y)
    if len(y)==0:return .5*(lo+hi)
    def ll(mu):
        p=np.clip(ndtr((mu-q)/sigma),1e-14,1-1e-14)
        return float(np.sum(y*np.log(p)+(1-y)*np.log1p(-p)))
    opt=minimize_scalar(lambda mu:-ll(mu),bounds=(lo,hi),method='bounded',options={'xatol':1e-7})
    return float(opt.x)


def gaussian_mle_counts(qlevels,n1,n0,lo,hi,sigma):
    """Exact finite-grid sufficient statistics: no O(local sample count) MLE calls."""
    qlevels=np.asarray(qlevels,dtype=float)
    n1=np.asarray(n1,dtype=float);n0=np.asarray(n0,dtype=float)
    def ll(mu):
        p=np.clip(ndtr((mu-qlevels)/sigma),1e-14,1-1e-14)
        return float(n1@np.log(p)+n0@np.log1p(-p))
    opt=minimize_scalar(lambda mu:-ll(mu),bounds=(lo,hi),method='bounded',
                        options={'xatol':1e-7})
    return float(opt.x)


def gaussian_E_glr_counts(qlevels,n1_i,n0_i,n1_j,n0_j,mi,mj,lo,hi,sigma):
    if mi<=mj+1e-8:return 0.
    qlevels=np.asarray(qlevels,dtype=float)
    def ll(n1,n0,mu):
        p=np.clip(ndtr((mu-qlevels)/sigma),1e-14,1-1e-14)
        return float(n1@np.log(p)+n0@np.log1p(-p))
    unres=ll(n1_i,n0_i,mi)+ll(n1_j,n0_j,mj)
    opt=minimize_scalar(lambda x:-ll(n1_i,n0_i,x)-ll(n1_j,n0_j,x),
          bounds=(lo,hi),method='bounded',options={'xatol':1e-7})
    return max(0.,unres+opt.fun)


def gaussian_E_glr(qi,yi,qj,yj,mi,mj,lo,hi,sigma):
    if mi <= mj+1e-8:return 0.0
    qi,yi,qj,yj=map(np.asarray,(qi,yi,qj,yj))
    def ll(q,y,mu):
        p=np.clip(ndtr((mu-q)/sigma),1e-14,1-1e-14)
        return float(np.sum(y*np.log(p)+(1-y)*np.log1p(-p)))
    unrestricted=ll(qi,yi,mi)+ll(qj,yj,mj)
    optimum=minimize_scalar(lambda x:-ll(qi,yi,x)-ll(qj,yj,x),
                             bounds=(lo,hi),method='bounded',options={'xatol':1e-6})
    return max(0.,unrestricted+optimum.fun)


def full_gaussian_glr(mean_i,mean_j,n_i,n_j,sigma):
    if mean_i<=mean_j:return 0.0
    return (mean_i-mean_j)**2/(2*sigma*sigma*(1/n_i+1/n_j))


def empirical_bounded_kinf(xdata,target,upward):
    from scipy.optimize import minimize_scalar
    mu=float(np.mean(xdata))
    if upward and target<=mu:return 0.0
    if not upward and target>=mu:return 0.0
    if target<=0 or target>=1:return float('inf')
    delta=(target-xdata) if upward else (xdata-target)
    ub=(1/(1-target) if upward else 1/target)*(1-1e-9)
    def cost(lam):
        if np.min(1+lam*delta)<=0:return -1e99
        return np.log1p(lam*delta).sum()
    opt=minimize_scalar(lambda lam:-cost(lam),bounds=(0,ub),method='bounded',options={'xatol':1e-7})
    return max(0.,cost(opt.x),cost(ub))


def full_U_glr(xi,xj):
    mi,mj=np.mean(xi),np.mean(xj)
    if mi<=mj:return 0.0
    def cost(x):
        return empirical_bounded_kinf(xi,x,False)+empirical_bounded_kinf(xj,x,True)
    opt=minimize_scalar(cost,bounds=(mj,mi),method='bounded',options={'xatol':1e-5})
    return max(0.,min(cost(mj+1e-6),cost(mi-1e-6),opt.fun))


def bit_glr(y_i,y_j):
    ni,nj=len(y_i),len(y_j)
    pi,pj=np.mean(y_i),np.mean(y_j)
    if pi<=pj:return 0.0
    pool=(ni*pi+nj*pj)/(ni+nj)
    return float(ni*binary_kl(pi,pool)+nj*binary_kl(pj,pool))


@dataclass
class Configuration:
    model:str
    algorithm:str
    deltas:tuple[float,...]
    e_family:str='gaussian'
    beta:float=.5
    sigma:float=.2
    mu_lo:float=-.2
    mu_hi:float=1.2
    qgrid:tuple[float,...]=tuple(np.linspace(0,1,9))
    oracle_xcount:int=19
    oracle_refresh:int=80
    challenger_refresh:int=10
    decision_every:int=10
    max_rounds:int=12000
    guard_exponent:float=.65
    guard_scale:float=1.0
    e_boundary:str='certified'
    u_boundary:str='practical'
    full_boundary:str='practical'
    companion_a0:float=.12
    trace_every:int=0
    continue_after_all_stops:bool=False


def run_trajectory(cfg:Configuration, X:np.ndarray, Q:np.ndarray,
                   internal_seed:int, oracle_trace=None):
    """X,Q have shape (K,max_rounds); local streams shared across methods.

    Returns one first crossing per δ (or a censored result). Sampling path is
    δ-independent and therefore legitimately reused across stopping levels.
    `oracle_trace` optional diagnostic output for debugging.
    """
    rng=np.random.default_rng(internal_seed)
    K=X.shape[0]
    assert K>=2 and X.shape[1]>=cfg.max_rounds and Q.shape==X.shape
    deltas=np.asarray(cfg.deltas)
    counts=np.zeros(K,dtype=int)
    ys=[[] for _ in range(K)];qs=[[] for _ in range(K)];xs=[[] for _ in range(K)]
    forced=np.zeros(K,dtype=int)
    times=np.full(len(deltas),-1,dtype=int)
    recs=np.full(len(deltas),-1,dtype=int)
    cached=np.zeros(K)
    cached_best=0
    qgrid=np.asarray(cfg.qgrid)
    use_grid_stats=(cfg.model=='E' and cfg.algorithm in ('e_tt','e_companion'))
    if use_grid_stats:
        qlevels=np.unique(np.r_[qgrid,.5] if cfg.algorithm=='e_companion' else qgrid)
        qmap={round(float(q),12):idx for idx,q in enumerate(qlevels)}
        n1=np.zeros((K,len(qlevels)),dtype=float)
        n0=np.zeros_like(n1)
        mle_cache=np.full(K,np.nan)
        mle_cached_count=np.full(K,-1,dtype=int)
    design=None
    # Exact U-profile cache: a RawProfile is rebuilt only when that arm's local
    # sample count changed. This changes no statistic; it only avoids repeated
    # sorting/PAVA work for untouched arms between challenger refreshes.
    u_profile_cache=[None]*K if cfg.model=='U' and cfg.algorithm in ('u_raw','u_marginal') else None
    u_profile_cached_count=np.full(K,-1,dtype=int) if u_profile_cache is not None else None
    oracle_history=[]
    trace=[]
    stop_stats=np.full(len(deltas),np.nan,dtype=float)
    stop_boundaries=np.full(len(deltas),np.nan,dtype=float)
    last_update=-1
    # Any method can use this same stream for nested deltas.
    def draw(arm,qt=None,mark_forced=False):
        arm=int(arm)
        k=counts[arm]
        if qt is None:
            qt=float(Q[arm,k])
            # U assumes Q in the open interval. np.random.random can, in principle,
            # return exactly 0. Clip only machine-endpoint accidents; this does not
            # bin or otherwise alter the raw threshold stream.
            if cfg.model=='U':
                qt=float(np.clip(qt,np.nextafter(0.0,1.0),np.nextafter(1.0,0.0)))
        yy=int(X[arm,k]>=qt)
        qs[arm].append(float(qt))
        ys[arm].append(yy)
        xs[arm].append(float(X[arm,k]))
        if use_grid_stats:
            key=round(float(qt),12)
            qidx=qmap[key]
            (n1 if yy else n0)[arm,qidx]+=1
        counts[arm]+=1
        if mark_forced:forced[arm]+=1
    for arm in range(K):
        qt=float(qgrid[len(qgrid)//2]) if cfg.model=='E' and cfg.algorithm=='e_companion' else (
            float(qgrid[arm%len(qgrid)]) if cfg.model=='E' and cfg.algorithm=='e_tt' else None)
        draw(arm,qt)
    n=K
    while n<=cfg.max_rounds:
        due=(last_update<0 or n-last_update>=cfg.challenger_refresh or n%cfg.decision_every==0)
        if due:
            if cfg.model=='E':
                if cfg.algorithm=='e_full':
                    mle=np.array([np.mean(v) for v in xs])
                elif cfg.e_family=='bernoulli':
                    mle=np.clip(n1.sum(axis=1)/counts,cfg.mu_lo,cfg.mu_hi)
                else:
                    for k in range(K):
                        if mle_cached_count[k]!=counts[k]:
                            mle_cache[k]=gaussian_mle_counts(qlevels,n1[k],n0[k],
                                              cfg.mu_lo,cfg.mu_hi,cfg.sigma)
                            mle_cached_count[k]=counts[k]
                    mle=mle_cache.copy()
            else:
                mle=np.array([np.mean(v) if cfg.algorithm=='u_full' else np.mean(ys[k]) for k,v in enumerate(xs)])
            cached_best=int(np.argmax(mle))
            cached.fill(0.)
            if cfg.algorithm=='u_raw' or cfg.algorithm=='u_marginal':
                profiles=u_profile_cache
                for k in range(K):
                    if u_profile_cached_count[k] != counts[k]:
                        profiles[k]=RawProfile(qs[k],ys[k])
                        u_profile_cached_count[k]=counts[k]
            for j in range(K):
                if j==cached_best:continue
                if cfg.algorithm in ('e_tt','e_companion') and cfg.e_family=='bernoulli':
                    cached[j]=bit_glr(ys[cached_best],ys[j])
                elif cfg.algorithm in ('e_tt','e_companion'):
                    cached[j]=gaussian_E_glr_counts(qlevels,n1[cached_best],n0[cached_best],
                       n1[j],n0[j],mle[cached_best],mle[j],cfg.mu_lo,cfg.mu_hi,cfg.sigma)
                elif cfg.algorithm=='e_full' and cfg.e_family=='bernoulli':
                    cached[j]=bit_glr(xs[cached_best],xs[j])
                elif cfg.algorithm=='e_full':
                    cached[j]=full_gaussian_glr(mle[cached_best],mle[j],counts[cached_best],counts[j],cfg.sigma)
                elif cfg.algorithm in ('u_raw','u_marginal'):
                    cached[j]=pair_glr_raw(profiles[cached_best],profiles[j])
                elif cfg.algorithm=='u_full':
                    cached[j]=full_U_glr(np.asarray(xs[cached_best]),np.asarray(xs[j]))
                else:raise ValueError(cfg.algorithm)
            last_update=n
            if cfg.model=='E' and cfg.algorithm in ('e_tt','e_companion') and (
                  design is None or n%cfg.oracle_refresh==0):
                design,gamma=(bernoulli_shared_grid_oracle(mle,cfg.beta,qgrid,cfg.oracle_xcount)
                              if cfg.e_family=="bernoulli" else
                              gaussian_shared_grid_oracle(mle,cfg.beta,qgrid,cfg.oracle_xcount,cfg.sigma))
                oracle_history.append((n,gamma,mle.tolist()))
        if n%cfg.decision_every==0 or n==cfg.max_rounds:
            if cfg.algorithm=='u_marginal':
                statistic=min(bit_glr(ys[cached_best],ys[j]) for j in range(K) if j!=cached_best)
            else:
                statistic=min(cached[j] for j in range(K) if j!=cached_best)
            for di,d in enumerate(deltas):
                if times[di]>=0:continue
                if cfg.model=='E' and cfg.algorithm in ('e_tt','e_companion'):
                    boundary=(E_cert_boundary(n,d,K,cfg.mu_lo,cfg.mu_hi,cfg.sigma,(qlevels if use_grid_stats else qgrid),cfg.e_family)
                               if cfg.e_boundary=='certified' else practical_boundary(n,d,K))
                elif cfg.model=='U' and cfg.algorithm=='u_raw':
                    boundary=(U_cert_boundary(n,d,K) if cfg.u_boundary=='certified' else
                              practical_boundary(n,d,K))
                elif cfg.model=='U' and cfg.algorithm=='u_marginal':
                    boundary=U_marginal_cert_boundary(n,d,K)
                elif cfg.algorithm=='e_full' and cfg.e_family=='bernoulli' and cfg.full_boundary=='certified':
                    boundary=full_bounded_cert_boundary(n,d,K)
                elif cfg.algorithm=='u_full' and cfg.full_boundary=='certified':
                    boundary=full_bounded_cert_boundary(n,d,K)
                else:
                    boundary=practical_boundary(n,d,K)
                if statistic>=boundary:
                    times[di]=n;recs[di]=cached_best
                    stop_stats[di]=float(statistic);stop_boundaries[di]=float(boundary)
            if cfg.trace_every and (n%cfg.trace_every==0 or n==cfg.max_rounds):
                trace.append({'n':int(n),'leader':int(cached_best),'statistic':float(statistic),
                              'lambda_hat':float(statistic/n),
                              'counts':counts.astype(int).tolist(),
                              'fractions':(counts/max(1,n)).astype(float).tolist()})
            if (np.all(times>=0) and not cfg.continue_after_all_stops) or n>=cfg.max_rounds:break
        if n>=cfg.max_rounds:break
        next_idx=n+1
        if cfg.algorithm=='e_companion':
            unmet=np.where(forced<cfg.companion_a0*next_idx**(2/3))[0]
            if unmet.size:
                arm=int(unmet[np.argmin(forced[unmet])]);qt=.5
                draw(arm,qt,mark_forced=True)
            else:
                # Regular E companion samples the ENTIRE joint action distribution.
                probs=design.ravel()/max(1e-15,design.sum())
                index=int(rng.choice(probs.size,p=probs))
                draw(index//len(qgrid),float(qgrid[index%len(qgrid)]))
        elif np.min(counts)<(next_idx**(.5 if cfg.model=='U' else cfg.guard_exponent))*cfg.guard_scale/(K**.5 if cfg.model=='U' else 1):
            arm=int(np.argmin(counts))
            if cfg.model=='E' and cfg.algorithm=='e_tt':
                d=design[arm]
                qt=float(rng.choice(qgrid,p=d/d.sum())) if d.sum()>1e-14 else float(rng.choice(qgrid))
                draw(arm,qt)
            else:draw(arm)
        else:
            cands=[j for j in range(K) if j!=cached_best]
            challenger=min(cands,key=lambda j: (cached[j]+np.log(max(counts[j],1)),j))
            arm=cached_best if rng.random()<cfg.beta else challenger
            if cfg.algorithm=='e_tt':
                d=design[arm]
                qt=float(rng.choice(qgrid,p=d/d.sum())) if d.sum()>1e-14 else float(rng.choice(qgrid))
                draw(arm,qt)
            else:draw(arm)
        n+=1
    if oracle_trace is not None:oracle_trace.extend(oracle_history)
    return {'times':times.tolist(),'recommendations':recs.tolist(),'counts':counts.tolist(),
            'rounds_run':int(n),'forced':forced.tolist(),
            'stop_stats':stop_stats.tolist(),'stop_boundaries':stop_boundaries.tolist(),
            'trace':trace}
