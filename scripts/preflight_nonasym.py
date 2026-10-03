#!/usr/bin/env python3
"""Population-level feasibility preflight for the non-asymptotic experiment suite.

This script does no Monte Carlo.  It computes population information rates and
solves n*Gamma >= implemented_boundary(n, delta).  For U-certified this is only
an optimistic population crossing predictor; actual stopping can be later.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from onebit_bai.experiment import compute_oracles
from onebit_bai.algorithms import E_cert_boundary, U_cert_boundary, practical_boundary


def solve_crossing(gamma, boundary, max_search=10**12):
    if not (gamma and gamma > 0):
        return None
    lo, hi = 1, 8
    while hi < max_search and hi * gamma < boundary(hi):
        lo, hi = hi, min(max_search, hi * 2)
    if hi * gamma < boundary(hi):
        return None
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid * gamma >= boundary(mid):
            hi = mid
        else:
            lo = mid
    return int(hi)


def fmt_n(n):
    if n is None:
        return "> search limit"
    if n >= 1e9:
        return f"{n/1e9:.3g}e9"
    if n >= 1e6:
        return f"{n/1e6:.3g}e6"
    if n >= 1e3:
        return f"{n/1e3:.3g}k"
    return str(n)


def e_section(c):
    means = np.asarray(c['means'], float)
    qgrid = np.linspace(0.0, 1.0, int(c['qpoints']))
    oracle = compute_oracles('E', means, c['beta'], c['sigma'], qgrid,
                             c['xgrid'], 81, include_full=False,
                             e_family='gaussian')
    gamma = float(oracle['onebit_gamma'])
    print("\n[E companion / certified boundary]")
    print(f"Gamma_pop = {gamma:.12g}; T* = {1/gamma:.4f}; cap = {c['cap']:,}")
    out = []
    for d in c['deltas']:
        b = lambda n, d=d: E_cert_boundary(n, d, len(means), -0.2, 1.2,
                                            c['sigma'], qgrid, 'gaussian')
        n = solve_crossing(gamma, b)
        out.append((d, n))
        flag = "OK" if n is not None and n < c['cap'] else "CAP-RISK"
        print(f"delta={d:.1e}  population crossing={fmt_n(n):>10}  [{flag}]")
    return gamma, out


def u_practical_section(c):
    means = np.asarray(c['means'], float)
    qgrid = np.linspace(0.0, 1.0, 9)
    oracle = compute_oracles('U', means, c['beta'], .2, qgrid,
                             c['xgrid'], c['qquad'], include_full=False,
                             u_generator='beta', concentration=c['concentration'])
    gamma = float(oracle['onebit_gamma'])
    print("\n[U raw / practical boundary -- empirical diagnostic only]")
    print(f"Gamma_pop = {gamma:.12g}; T* = {1/gamma:.4f}; cap = {c['cap']:,}")
    out = []
    for d in c['deltas']:
        b = lambda n, d=d: practical_boundary(n, d, len(means))
        n = solve_crossing(gamma, b)
        out.append((d, n))
        flag = "OK" if n is not None and n < c['cap'] else "CAP-RISK"
        print(f"delta={d:.1e}  population crossing={fmt_n(n):>10}  [{flag}]")
    return gamma, out


def u_cert_section(c):
    means = np.asarray(c['means'], float)
    qgrid = np.linspace(0.0, 1.0, 9)
    oracle = compute_oracles('U', means, c['beta'], .2, qgrid,
                             c['xgrid'], c['qquad'], include_full=False,
                             u_generator='beta', concentration=c['concentration'])
    gamma = float(oracle['onebit_gamma'])
    print("\n[U raw / CERTIFIED boundary -- feasibility only]")
    print(f"easy preflight means={means.tolist()}; Gamma_pop={gamma:.12g}; T*={1/gamma:.4f}")
    out = []
    for d in c['deltas']:
        b = lambda n, d=d: U_cert_boundary(n, d, len(means))
        n = solve_crossing(gamma, b, int(c.get('max_search', 10**12)))
        out.append((d, n))
        print(f"delta={d:.2g}  optimistic population crossing={fmt_n(n):>10}")
    print("NOTE: if these crossings are millions/billions, do NOT launch a 200-rep certified-U Monte Carlo.")
    return gamma, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT/'configs_nonasym_final.json'))
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    e_section(cfg['E'])
    u_practical_section(cfg['U_practical'])
    u_cert_section(cfg['U_certified_preflight'])

if __name__ == '__main__':
    main()
