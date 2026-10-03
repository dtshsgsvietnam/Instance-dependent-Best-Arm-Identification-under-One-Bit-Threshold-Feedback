from onebit_bai.experiment import methods_for
from onebit_bai.algorithms import E_cert_boundary, U_cert_boundary, practical_boundary


def test_e_companion_only_mode():
    assert methods_for('E', e_methods='companion') == ['e_companion']
    assert methods_for('E', e_methods='tt') == ['e_tt']
    assert methods_for('E', e_methods='both') == ['e_tt','e_companion']


def test_certified_boundaries_dominate_practical_examples():
    n=1000;delta=.01;K=5
    # U certified is practical + a positive sieve penalty.
    assert U_cert_boundary(n,delta,K) > practical_boundary(n,delta,K)
    # E certified boundary must be finite and positive on the paper grid.
    import numpy as np
    b=E_cert_boundary(n,delta,K,-.2,1.2,.2,np.linspace(0,1,31),'gaussian')
    assert b > 0
