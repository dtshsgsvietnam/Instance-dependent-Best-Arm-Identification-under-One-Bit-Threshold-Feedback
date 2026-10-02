"""Optional later: paired full-feedback comparators."""
from _simple_launch import common_parser,run_experiment
if __name__=='__main__':
    p=common_parser('Optional full-feedback comparison')
    a=p.parse_args()
    if a.model=='E':
        flags=['--model','E','--e-family','gaussian','--means','.75','.45','--sigma','.2','--beta','.5',
               '--deltas','.2','.1','.05','--no-companion','--full','--e-boundary','practical','--full-boundary','practical',
               '--qpoints','11','--xgrid','31','--oracle-refresh','20','--refresh','5','--decide','5']
        raise SystemExit(run_experiment(a,'FULL',flags,40,12000))
    flags=['--model','U','--u-generator','beta','--means','.8','.2','--concentration','12','--beta','.5',
           '--deltas','.2','.1','.05','--full','--u-boundary','practical','--full-boundary','practical',
           '--xgrid','31','--qquad','121','--refresh','10','--decide','10']
    raise SystemExit(run_experiment(a,'FULL',flags,20,7000))
