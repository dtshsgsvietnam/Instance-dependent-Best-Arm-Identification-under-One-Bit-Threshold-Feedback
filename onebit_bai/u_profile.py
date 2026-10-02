"""Raw (unbinned) current-status profile likelihood and order-restricted GLR.

The supremum over arbitrary distributions is reduced to a finite isotonic
likelihood on the *observed, unrounded* thresholds. The exact (in supremum
sense) mean-order constraint for fitted survival values s at ordered q is
    min_mean(s_i; q_i) <= max_mean(s_j; q_j).
No fictitious binning of thresholds is used. A 1-D Lagrange multiplier and
GENERALIZED PAVA maximize the concave likelihood subject to this constraint.
Solver accuracy is numerical; this does not constitute an exact symbolic oracle.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args)==1 and not kwargs:
            return args[0]
        def deco(fn): return fn
        return deco


@njit(cache=True)
def _block_opt(a: float, b: float, c: float) -> float:
    """Exact maximizer of a log(s)+b log(1-s)+c*s on [0,1].

    This replaces the former 48-step bisection by the closed-form root of
        c s^2 + (a+b-c) s - a = 0.
    The objective is concave; when a,b>0 the unique maximizer is interior.
    Endpoint cases are handled explicitly.
    """
    if a == 0.0:
        if c <= b:
            return 0.0
        # c>b>=0: unique interior solution of -b/(1-s)+c=0.
        return 1.0 - b / c
    if b == 0.0:
        if a + c >= 0.0:
            return 1.0
        # c<-a<0: unique interior solution of a/s+c=0.
        return -a / c
    if abs(c) < 1e-14:
        return a / (a + b)
    Bq = a + b - c
    disc = Bq * Bq + 4.0 * c * a
    if disc < 0.0 and disc > -1e-12:
        disc = 0.0
    root = np.sqrt(disc)
    # Stable quadratic roots via q-formula.
    qv = -0.5 * (Bq + (root if Bq >= 0.0 else -root))
    r1 = qv / c
    r2 = (-a / qv) if qv != 0.0 else np.nan
    if 0.0 < r1 < 1.0:
        return r1
    if 0.0 < r2 < 1.0:
        return r2
    # Numerical fallback near an endpoint. The concave score identifies the
    # admissible endpoint uniquely in the only remaining cases.
    if r1 <= 0.0 and r2 <= 0.0:
        return 0.0
    if r1 >= 1.0 and r2 >= 1.0:
        return 1.0
    cand = r1 if 0.0 <= r1 <= 1.0 else r2
    if cand < 0.0:
        cand = 0.0
    elif cand > 1.0:
        cand = 1.0
    return cand


@njit(cache=True)
def _pava_fit(y: np.ndarray, linear: np.ndarray) -> np.ndarray:
    """Monotone nonincreasing fitted survival; a general linear perturbation."""
    n = len(y)
    aa = np.empty(n)
    bb = np.empty(n)
    cc = np.empty(n)
    vals = np.empty(n)
    starts = np.empty(n, dtype=np.int64)
    ends = np.empty(n, dtype=np.int64)
    nb = 0
    for idx in range(n):
        starts[nb] = idx
        ends[nb] = idx + 1
        aa[nb] = float(y[idx])
        bb[nb] = 1.0 - float(y[idx])
        cc[nb] = linear[idx]
        vals[nb] = _block_opt(aa[nb], bb[nb], cc[nb])
        nb += 1
        # For non-increasing survival, adjacent increase violates isotonicity.
        while nb >= 2 and vals[nb-2] < vals[nb-1] - 1e-12:
            aa[nb-2] += aa[nb-1]
            bb[nb-2] += bb[nb-1]
            cc[nb-2] += cc[nb-1]
            ends[nb-2] = ends[nb-1]
            vals[nb-2] = _block_opt(aa[nb-2], bb[nb-2], cc[nb-2])
            nb -= 1
    s = np.empty(n)
    for block in range(nb):
        for k in range(starts[block], ends[block]):
            s[k] = vals[block]
    return s


@njit(cache=True)
def _ll(y: np.ndarray, s: np.ndarray) -> float:
    total = 0.0
    for i in range(len(y)):
        if y[i] == 1:
            if s[i] <= 0:
                return -1e200
            total += np.log(s[i])
        else:
            if s[i] >= 1:
                return -1e200
            total += np.log(1.0-s[i])
    return total


class RawProfile:
    def __init__(self, q: np.ndarray, y: np.ndarray, B: float=1.0):
        assert B > 0
        q, y = np.asarray(q, dtype=float), np.asarray(y, dtype=np.int8)
        assert q.size == y.size and q.size > 0
        assert np.all(q > 0) and np.all(q < B)
        assert np.all((y == 0) | (y == 1))
        order = np.argsort(q, kind='stable')
        self.q, self.y, self.B = q[order], y[order], B
        # Raw Q ties have probability zero. Stable sorting is a numerical convention.
        self.n = q.size
        self.min_coef = np.diff(np.r_[0.0, self.q])
        self.max_coef = np.r_[np.diff(self.q), B-self.q[-1]]
        self.max_intercept = self.q[0]
        self.s0 = _pava_fit(self.y, np.zeros(self.n))
        self.M = float(_ll(self.y, self.s0))

    def fitted(self, linear: np.ndarray) -> tuple[np.ndarray, float]:
        s = _pava_fit(self.y, np.asarray(linear, dtype=float))
        return s, float(_ll(self.y, s))

    def mean_min(self, s: np.ndarray) -> float:
        return float(self.min_coef @ s)

    def mean_max(self, s: np.ndarray) -> float:
        return float(self.max_intercept + self.max_coef @ s)


def pair_glr_raw(pi: RawProfile, pj: RawProfile, tol: float=1e-6) -> float:
    """Supremum-based raw two-arm likelihood-ratio for mean(i) <= mean(j).

    Numerically solves the generalized-PAVA dual. Returns zero when the
    unrestricted fits admit a mean-ordered realization. Sorted original q
    values are *never rounded or binned*. Pairwise dual residual ~ tol*B.
    """
    assert abs(pi.B-pj.B) < 1e-12
    if pi.mean_min(pi.s0) <= pj.mean_max(pj.s0) + tol*pi.B:
        return 0.0
    a, b = pi.min_coef, pj.max_coef
    last = [None]
    def evaluate(lam: float) -> float:
        si, lli = pi.fitted(-lam*a)
        sj, llj = pj.fitted(+lam*b)
        last[0] = (lli, llj)
        return pi.mean_min(si) - pj.mean_max(sj)
    hi = 1.0
    for _ in range(55):
        if evaluate(hi) < 0.0:
            break
        hi *= 2.0
    else:
        raise ArithmeticError('Profile dual could not bracket mean-order root')
    lam = brentq(evaluate, 0.0, hi, xtol=1e-7, rtol=1e-8)
    _ = evaluate(lam)
    lli, llj = last[0]
    return max(0.0, pi.M + pj.M - lli - llj)


def sieve_penalty(n: int) -> float:
    """Paper-v34 *certified* U proof penalty c_n (NOT practical boundary)."""
    from scipy.special import gammaln
    eta = .25*(n+1)**(-1/3)
    xi = (1/16)*(n+1)**(-2/3)
    J = int(np.ceil((1-eta/2)/xi))
    a = -np.log1p(-eta) + 4*xi/eta
    logM = gammaln(n+J+1)-gammaln(n+1)-gammaln(J+1)
    return float(n*a+2*logM)
