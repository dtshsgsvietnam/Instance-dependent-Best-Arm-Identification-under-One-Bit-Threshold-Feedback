import numpy as np
from onebit_bai.experiment import compute_oracles, make_parser


def test_close5_u_population_oracle_is_finite():
    mus=np.array([.70,.64,.63,.62,.61])
    oracle=compute_oracles('U',mus,.5,.2,np.linspace(0,1,9),31,121,True,'gaussian','beta',12.)
    assert np.isfinite(oracle['onebit_gamma']) and oracle['onebit_gamma']>0
    assert np.isfinite(oracle['full_gamma']) and oracle['full_gamma']>0
    assert np.isclose(sum(oracle['onebit_mass']),1.0,atol=1e-7)


def test_workers_cli_is_available():
    p=make_parser()
    args=p.parse_args(['--model','E','--workers','3'])
    assert args.workers==3
