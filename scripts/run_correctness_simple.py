"""Independent-replication error-rate diagnostic. PAC means <=delta, not equality."""
from _simple_launch import common_parser,run_experiment
if __name__=='__main__':
    p=common_parser('Correctness check using certified, computationally usable stopping rules')
    a=p.parse_args()
    if a.model=='E':
        flags=['--model','E','--e-family','gaussian','--means','.75','.45','--sigma','.2','--beta','.5',
               '--deltas','.2','.1','.05','--no-companion','--e-boundary','certified','--qpoints','11','--xgrid','31',
               '--oracle-refresh','20','--refresh','5','--decide','5']
        raise SystemExit(run_experiment(a,'CORRECTNESS',flags,120,12000))
    # Same U raw-TCI sampling path, but a marginal-bit GLR with the published
    # bounded-observation certified boundary. This is a correctness control.
    # The raw-profile theorem-certified c_n boundary is available but usually
    # does not stop at practical horizons.
    flags=['--model','U','--u-generator','beta','--means','.8','.2','--concentration','12','--beta','.5',
           '--deltas','.2','.1','.05','--u-boundary','practical','--xgrid','31','--qquad','121','--refresh','5','--decide','5']
    raise SystemExit(run_experiment(a,'CORRECTNESS',flags,120,12000))
