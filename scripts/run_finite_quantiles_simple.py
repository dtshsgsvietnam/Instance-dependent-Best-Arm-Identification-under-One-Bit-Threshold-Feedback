"""Finite-confidence quantiles and theorem-shape remainder diagnostics."""
from _simple_launch import common_parser,run_experiment
if __name__=='__main__':
    p=common_parser('Non-asymptotic quantiles and probability-bound shape')
    a=p.parse_args()
    if a.model=='E':
        flags=['--model','E','--e-family','gaussian','--means','.75','.45','--sigma','.2','--beta','.5',
               '--deltas','.2','.1','.05','.02','--e-boundary','certified','--qpoints',('21' if a.strict else '11'),
               '--xgrid',('61' if a.strict else '31'),'--oracle-refresh',('1' if a.strict else '10'),
               '--refresh',('1' if a.strict else '5'),'--decide',('1' if a.strict else '5')]
        raise SystemExit(run_experiment(a,'FINITE',flags,80,20000,eta=.05))
    # U raw practical gives observable finite quantiles. The exact certified U
    # bound is separately diagnosed by u_cert_boundary_curve.py because c_n is
    # too conservative to cross at ordinary Monte Carlo horizons.
    flags=['--model','U','--u-generator','beta','--means','.8','.2','--concentration','12','--beta','.5',
           '--deltas','.2','.1','.05','.02','--skip-safe-bit','--u-boundary','practical','--xgrid','41','--qquad','181',
           '--refresh',('1' if a.strict else '5'),'--decide',('1' if a.strict else '5')]
    raise SystemExit(run_experiment(a,'FINITE',flags,60,12000,eta=.05))
