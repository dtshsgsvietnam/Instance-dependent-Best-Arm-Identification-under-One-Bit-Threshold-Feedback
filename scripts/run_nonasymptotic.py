"""Finite-confidence quantiles, tail shape, held-out empirical correction."""
from _launch import run_allowed
if __name__=='__main__':
    raise SystemExit(run_allowed({
        'E_NONASYM_COMPANION':'E global-grid companion, finite quantiles; only approximate theorem policy',
        'U_NONASYM_PRACTICAL':'U practical raw-profile: exploratory quantiles ONLY',
        'U_CERTIFIED_BOUNDARY':'U raw exact-sieve analytic boundary: feasibility/censoring probe; may not stop',
    }, 'Finite-confidence quantiles and EMPIRICAL, not proved, upper-bound fits'))
