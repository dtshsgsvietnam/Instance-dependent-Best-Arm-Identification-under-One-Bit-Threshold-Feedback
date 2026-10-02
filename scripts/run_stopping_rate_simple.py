"""Long-path diagnostic for lambda_hat_n := min_j W_n/n versus Gamma=1/T*."""
from _simple_launch import common_parser,run_experiment
if __name__=='__main__':
    p=common_parser('Evidence-rate check: lambda_hat_n=min W_n/n should approach Gamma')
    a=p.parse_args()
    if a.model=='E':
        flags=['--model','E','--e-family','gaussian','--means','.75','.45','--sigma','.2','--beta','.5',
               '--deltas','1e-30','--no-companion','--e-boundary','certified','--trace-every','25','--continue-after-stop',
               '--qpoints',('21' if a.strict else '11'),'--xgrid',('61' if a.strict else '31'),
               '--oracle-refresh',('1' if a.strict else '20'),'--refresh',('1' if a.strict else '5'),'--decide',('1' if a.strict else '5')]
        raise SystemExit(run_experiment(a,'RATE',flags,1,10000))
    flags=['--model','U','--u-generator','beta','--means','.8','.2','--concentration','12','--beta','.5',
           '--deltas','1e-30','--skip-safe-bit','--u-boundary','certified','--trace-every','25','--continue-after-stop',
           '--xgrid','41','--qquad','181','--refresh',('1' if a.strict else '5'),'--decide',('1' if a.strict else '5')]
    raise SystemExit(run_experiment(a,'RATE',flags,1,6000))
