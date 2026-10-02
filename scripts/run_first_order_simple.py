"""Check tau*Gamma/log(1/delta) and empirical error as confidence -> 1."""
from _simple_launch import common_parser,run_experiment
if __name__=='__main__':
    p=common_parser('First-order asymptotic check: mean(tau)*Gamma/L -> 1')
    a=p.parse_args()
    if a.model=='E':
        flags=['--model','E','--e-family','gaussian','--means','.75','.45','--sigma','.2','--beta','.5',
               '--deltas','.2','.1','.05','.02','.01','--no-companion','--e-boundary','certified',
               '--qpoints',('21' if a.strict else '11'),'--xgrid',('61' if a.strict else '31'),
               '--oracle-refresh',('1' if a.strict else '20'),'--refresh',('1' if a.strict else '5'),
               '--decide',('1' if a.strict else '5')]
        raise SystemExit(run_experiment(a,'FIRST_ORDER',flags,40,20000))
    # For U, the theorem-certified raw sieve boundary is extremely conservative.
    # This first-order stopping-time run uses the practical o(log n) boundary only
    # as a descriptive tau-ratio diagnostic; run_stopping_rate_simple checks the
    # theorem's core evidence rate independently of this boundary.
    flags=['--model','U','--u-generator','beta','--means','.8','.2','--concentration','12','--beta','.5',
           '--deltas','.2','.1','.05','.02','.01','--skip-safe-bit','--u-boundary','practical',
           '--xgrid','41','--qquad','181','--refresh',('1' if a.strict else '5'),'--decide',('1' if a.strict else '5')]
    raise SystemExit(run_experiment(a,'FIRST_ORDER',flags,30,12000))
