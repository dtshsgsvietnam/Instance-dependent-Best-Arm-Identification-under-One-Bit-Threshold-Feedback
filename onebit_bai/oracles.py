"""Discretized *shared* Model-E oracle and Model-U/full-feedback population values.

All reported population rates are numerical approximations unless a closed
form is explicitly available. A shared best-arm threshold measure is always
represented by ONE vector of action weights: no pairwise relaxation is used.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog, brentq, minimize_scalar
from scipy.special import ndtr, xlogy, roots_legendre, roots_jacobi
from scipy.stats import beta as beta_dist


def binary_kl(p, r):
    p = np.asarray(p)
    r = np.clip(r, 1e-14, 1-1e-14)
    return xlogy(p, p/r) + xlogy(1-p, (1-p)/(1-r))


def gaussian_p(mu, q, sigma=.2):
    return ndtr((np.asarray(mu)-np.asarray(q))/sigma)


def gaussian_qkl(a, b, q, sigma=.2):
    return binary_kl(gaussian_p(a,q,sigma), gaussian_p(b,q,sigma))


def gaussian_full_kl(a, b, sigma=.2):
    return (np.asarray(a)-np.asarray(b))**2/(2*sigma*sigma)


def _shared_lp(mus, best, beta, qgrid, xcount, divergences):
    """Exact LP for a finite q/x grid; x-grid inner min is an approximation."""
    K = len(mus)
    M = len(qgrid)
    challengers = [j for j in range(K) if j != best]
    rows = []
    for j in challengers:
        xgrid = np.unique(np.r_[np.linspace(min(mus[j],mus[best]), max(mus[j],mus[best]), xcount), mus[j], mus[best]])
        for x in xgrid:
            row = np.zeros(K*M + 1)
            row[best*M:(best+1)*M] = -divergences(mus[best], x, qgrid)
            row[j*M:(j+1)*M] = -divergences(mus[j], x, qgrid)
            row[-1] = 1.0
            rows.append(row)
    eq = np.zeros((2,K*M+1))
    eq[0,:-1] = 1
    eq[1,best*M:(best+1)*M] = 1
    target = np.array([1.0,beta])
    c = np.zeros(K*M+1)
    c[-1] = -1
    sol = linprog(c, A_ub=np.asarray(rows), b_ub=np.zeros(len(rows)), A_eq=eq,
                  b_eq=target, bounds=[(0,None)]*(K*M)+[(None,None)], method='highs')
    if not sol.success:
        raise ArithmeticError('Shared finite-design LP failed: '+sol.message)
    return np.maximum(0.0,sol.x[:-1]).reshape(K,M), max(0.0,sol.x[-1])


def gaussian_shared_grid_oracle(mus, beta=.5, qgrid=None, xcount=25, sigma=.2):
    mus=np.asarray(mus,dtype=float)
    best=int(np.argmax(mus))
    if qgrid is None:
        qgrid=np.linspace(0,1,11)
    qgrid=np.asarray(qgrid)
    if np.sort(mus)[-1]-np.sort(mus)[-2]<1e-4:
        w=np.full((len(mus),len(qgrid)), (1-beta)/((len(mus)-1)*len(qgrid)))
        w[best,:] = beta/len(qgrid)
        return w, 0.0
    return _shared_lp(mus,best,beta,qgrid,xcount,lambda a,x,q:gaussian_qkl(a,x,q,sigma))


def _lp_scalar_rates(bmus, best, beta, xcount, cost_fn):
    K=len(bmus)
    others=[j for j in range(K) if j!=best]
    # Small LP: variables w_j, Gamma. Each active comparator has its own x.
    rows=[]
    for j_idx,j in enumerate(others):
        for x in np.linspace(bmus[j], bmus[best],xcount):
            row=np.zeros(len(others)+1)
            row[j_idx]=-cost_fn(j,x,False)
            row[-1]=1.0
            rows.append((row,beta*cost_fn(best,x,True)))
    eq=np.zeros((1,len(others)+1));eq[0,:len(others)]=1
    c=np.zeros(len(others)+1);c[-1]=-1
    A=np.array([v[0] for v in rows]);upper=np.array([v[1] for v in rows])
    sol=linprog(c,A_ub=A,b_ub=upper,A_eq=eq,b_eq=[1-beta],
                bounds=[(0,None)]*len(others)+[(None,None)],method='highs')
    if not sol.success:
        raise ArithmeticError('Scalar oracle LP failed: '+sol.message)
    w=np.zeros(K);w[best]=beta;w[others]=sol.x[:len(others)]
    return w,max(0.0,float(sol.x[-1]))


def gaussian_full_oracle(mus,beta=.5,sigma=.2):
    """Closed form full-feedback Gaussian fixed-beta Chernoff value."""
    mus=np.asarray(mus)
    b=int(np.argmax(mus));others=[j for j in range(len(mus)) if j!=b]
    cs=np.array([(mus[b]-mus[j])**2/(2*sigma**2) for j in others])
    assert cs.min()>0
    def masses(g):
        return g*beta/(beta*cs-g)
    rate=brentq(lambda g: masses(g).sum()-(1-beta), 1e-12,
                min(beta*cs)*(.9999999999))
    w=np.zeros(len(mus));w[b]=beta;w[others]=masses(rate)
    return w,float(rate)


@dataclass
class BetaLaw:
    """Beta(a,b) rewards on [0,1] for *generation*, not assumed by U algorithm."""
    mean:float
    concentration:float=12.0
    def sample(self,rng,n):
        return rng.beta(self.mean*self.concentration,(1-self.mean)*self.concentration,n)
    def survival(self,q):
        return beta_dist.sf(q,self.mean*self.concentration,(1-self.mean)*self.concentration)
    def full_expectation_quadrature(self,N=96):
        aa=self.mean*self.concentration
        bb=(1-self.mean)*self.concentration
        # [−1,1] weighted by (1−z)^(bb−1) (1+z)^(aa−1)
        z,w=roots_jacobi(N,bb-1,aa-1)
        return (z+1)/2,w/w.sum()


def _Tlambda(p, lam):
    p=np.asarray(p)
    if abs(lam)<1e-12:return p.copy()
    # Stable quadratic branch. Interior formula exact except for rounding
    disc=np.maximum(0.0,(1+lam)**2-4*lam*p)
    return np.clip(2*p / (1+lam+np.sqrt(disc)),0,1) if lam>-1 else np.clip((1+lam-np.sqrt(disc))/(2*lam),0,1)


def scalar_u_cost(survival,wq,target,upward=False):
    """Exact scalar variational expression via q quadrature for B=1.

    `upward=False` computes K_U^-(F,x), true mean -> smaller x.
    Quad/survival represent one bounded law; no raw-sample discretization.
    """
    current=float(wq@survival)
    if (target>=current and not upward) or (target<=current and upward):return 0.0
    if target<=0 or target>=1:return np.inf
    sign=-1 if upward else 1
    def m(mult):
        return float(wq@_Tlambda(survival,sign*mult))
    hi=1.0
    for _ in range(70):
        if (m(hi)>=target if upward else m(hi)<=target):break
        hi*=2
    else:raise ArithmeticError('Scalar U multiplier root not bracketed')
    val=brentq(lambda L:m(L)-target,0,hi,xtol=1e-9)
    profile=_Tlambda(survival,sign*val)
    # For an interior target, the exact optimizer is interior wherever the true
    # Bernoulli parameter is interior.  Floating-point evaluation of T_lambda
    # can nevertheless round values such as 1-1e-16 to exactly 1 (or 0), which
    # spuriously turns a finite quadrature cost into +inf and can poison the LP.
    # The theoretical endpoint cases target in {0,1} were already handled above;
    # here we only stabilize the numerical quadrature at machine precision.
    profile=np.clip(profile,1e-14,1-1e-14)
    return float(wq@binary_kl(survival,profile))


def u_population_oracle(laws, beta=.5, qnodes=81,xcount=31):
    mus=np.asarray([g.mean for g in laws]);best=int(np.argmax(mus))
    z,w=roots_legendre(qnodes)
    q=(z+1)/2;w=w/w.sum()
    surv=[g.survival(q) for g in laws]
    cache={}
    def cost(j,x,minus):
        key=(j,round(float(x),9),minus)
        if key not in cache:
            cache[key]=scalar_u_cost(surv[j],w,x,upward=not minus)
        return cache[key]
    return _lp_scalar_rates(mus,best,beta,xcount,cost)


def full_bounded_Kinf(law, x, downward, nodes=96):
    """K_inf(F,x) via the standard bounded-distribution reverse-KL dual.

    downward: K^- for mu>x, with log(1+lambda (X-x)) / lambda<=1/x.
    upward: K^+ for mu<x, with log(1+lambda (x-X)) / lambda<=1/(1-x).
    """
    if downward and x>=law.mean: return 0.0
    if not downward and x<=law.mean: return 0.0
    if x<=0 or x>=1:return np.inf
    xx, ww=law.full_expectation_quadrature(nodes)
    diff=(xx-x) if downward else (x-xx)
    upper=(1/x if downward else 1/(1-x))*(1-1e-9)
    def objective(lam):
        if np.any(1+lam*diff<=0):return -1e100
        return float(ww@np.log1p(lam*diff))
    sol=minimize_scalar(lambda v:-objective(v),bounds=(0,upper),method='bounded',options={'xatol':1e-9})
    return max(0.0,objective(sol.x),objective(upper))


def u_full_population_oracle(laws,beta=.5,xcount=31,nodes=96):
    mus=np.array([g.mean for g in laws]);best=int(np.argmax(mus));cache={}
    def cost(j,x,minus):
        key=(j,round(float(x),9),minus)
        if key not in cache:cache[key]=full_bounded_Kinf(laws[j],x,minus,nodes)
        return cache[key]
    return _lp_scalar_rates(mus,best,beta,xcount,cost)

@dataclass
class BernoulliLaw:
    """Endpoint law on {0,1}: useful for E/U/full information sanity checks."""
    mean:float
    def sample(self,rng,n):
        return rng.binomial(1,self.mean,n).astype(float)
    def survival(self,q):
        q=np.asarray(q)
        return np.where(q<=0.,1.,np.where(q<=1.,self.mean,0.))
    def full_expectation_quadrature(self,N=96):
        return np.array([0.,1.]),np.array([1.-self.mean,self.mean])


def bernoulli_shared_grid_oracle(mus,beta=.5,qgrid=None,xcount=25):
    mus=np.asarray(mus)
    if qgrid is None:qgrid=np.linspace(.1,.9,9)
    return _shared_lp(mus,int(np.argmax(mus)),beta,np.asarray(qgrid),xcount,
                      lambda a,x,q:np.ones_like(q)*binary_kl(a,x))


def bernoulli_full_oracle(mus,beta=.5,xcount=81):
    mus=np.asarray(mus);best=int(np.argmax(mus))
    def cost(j,x,minus):
        return float(binary_kl(mus[j],x)) if ((minus and x<mus[j]) or
                                        (not minus and x>mus[j])) else 0.
    return _lp_scalar_rates(mus,best,beta,xcount,cost)
