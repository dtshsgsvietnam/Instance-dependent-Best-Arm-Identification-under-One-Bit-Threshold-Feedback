#!/usr/bin/env python3
"""Smoke/pilot/final runner for the non-asymptotic experiment plan.

E: theorem-facing companion, certified E boundary, companion only.
U: practical raw-current-status Top-Two, used only as a theorem-motivated
   empirical correction-shape diagnostic.  The certified U rule is preflighted
   but is intentionally NOT launched here because its sieve boundary is
   computationally prohibitive at paper-scale horizons.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DEFAULT = ROOT / 'configs_nonasym_final.json'


def run(cmd, cwd=ROOT):
    print('\n$', ' '.join(map(str, cmd)), flush=True)
    return subprocess.check_call(list(map(str, cmd)), cwd=cwd)


def choose_profile(cfg, profile):
    E = dict(cfg['E']); U = dict(cfg['U_practical'])
    if profile == 'smoke':
        E.update(reps=3, deltas=[1e-4], cap=min(E['cap'], 60000))
        U.update(reps=3, deltas=[1e-4], cap=min(U['cap'], 30000))
    elif profile == 'pilot':
        E['reps'] = 20; U['reps'] = 20
    elif profile == 'final':
        pass
    else:
        raise ValueError(profile)
    return E, U


def e_cmd(c, out, workers, resume):
    cmd = [sys.executable, '-m', 'onebit_bai.experiment',
        '--model','E','--e-family','gaussian','--e-methods','companion',
        '--means', *map(str,c['means']), '--sigma',str(c['sigma']), '--beta',str(c['beta']),
        '--qpoints',str(c['qpoints']), '--xgrid',str(c['xgrid']),
        '--oracle-refresh',str(c['oracle_refresh']), '--refresh',str(c['refresh']),
        '--decide',str(c['decide']), '--companion-a0',str(c['companion_a0']),
        '--deltas', *map(str,c['deltas']), '--e-boundary',c['boundary'],
        '--reps',str(c['reps']), '--cap',str(c['cap']), '--seed',str(c['seed']),
        '--workers',str(workers), '--mp-start-method','auto',
        '--skip-full-oracle','--output',str(out),'--verbose']
    if resume: cmd.append('--resume')
    return cmd


def u_cmd(c, out, workers, resume):
    cmd = [sys.executable, '-m', 'onebit_bai.experiment',
        '--model','U','--u-generator','beta','--u-methods','raw','--skip-safe-bit',
        '--means', *map(str,c['means']), '--concentration',str(c['concentration']),
        '--beta',str(c['beta']), '--xgrid',str(c['xgrid']), '--qquad',str(c['qquad']),
        '--refresh',str(c['refresh']), '--decide',str(c['decide']),
        '--deltas', *map(str,c['deltas']), '--u-boundary',c['boundary'],
        '--reps',str(c['reps']), '--cap',str(c['cap']), '--seed',str(c['seed']),
        '--workers',str(workers), '--mp-start-method','auto',
        '--skip-full-oracle','--output',str(out),'--verbose']
    if resume: cmd.append('--resume')
    return cmd


def report(out, eta):
    run([sys.executable,'-m','onebit_bai.focused_report','--input',str(out),'--eta',str(eta)])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('profile', choices=['smoke','pilot','final','report'])
    ap.add_argument('--config', default=str(CONFIG_DEFAULT))
    ap.add_argument('--output-root', default=None)
    ap.add_argument('--only', choices=['E','U','both'], default='both')
    ap.add_argument('--workers-e', type=int, default=None)
    ap.add_argument('--workers-u', type=int, default=None)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--skip-preflight', action='store_true')
    a=ap.parse_args()

    cfg=json.loads(Path(a.config).read_text())
    eta=float(cfg.get('eta',.05))
    root=Path(a.output_root or (ROOT/'runs'/f'nonasym_{a.profile}')).resolve()
    eout=root/'E_COMPANION'
    uout=root/'U_PRACTICAL'
    root.mkdir(parents=True,exist_ok=True)

    # Prevent nested BLAS threading on multicore servers.
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key,'1')

    if a.profile != 'report' and not a.skip_preflight:
        run([sys.executable, str(ROOT/'scripts'/'preflight_nonasym.py'), '--config', a.config])

    if a.profile == 'report':
        if eout.joinpath('trials.csv').exists(): report(eout,eta)
        if uout.joinpath('trials.csv').exists(): report(uout,eta)
        run([sys.executable, str(ROOT/'scripts'/'analyze_nonasym_results.py'),
             '--root',str(root),'--eta',str(eta)])
        return 0

    E,U=choose_profile(cfg,a.profile)
    we=int(a.workers_e if a.workers_e is not None else E['workers'])
    wu=int(a.workers_u if a.workers_u is not None else U['workers'])
    # Smoke should not reserve many processes.
    if a.profile=='smoke':
        we=min(we,4); wu=min(wu,4)

    if a.only in ('E','both'):
        run(e_cmd(E,eout,we,a.resume))
        report(eout,eta)
    if a.only in ('U','both'):
        run(u_cmd(U,uout,wu,a.resume))
        report(uout,eta)

    run([sys.executable, str(ROOT/'scripts'/'analyze_nonasym_results.py'),
         '--root',str(root),'--eta',str(eta)])
    if a.profile=='pilot':
        run([sys.executable, str(ROOT/'scripts'/'estimate_runtime_from_pilot.py'),
             '--root',str(root),'--final-config',a.config,
             '--workers-e',str(we),'--workers-u',str(wu)])
    return 0

if __name__=='__main__':
    raise SystemExit(main())
