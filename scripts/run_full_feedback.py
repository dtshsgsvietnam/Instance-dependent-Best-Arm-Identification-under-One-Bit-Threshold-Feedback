"""Information and finite-sample cost of full versus one-bit observations."""
from _launch import run_allowed
if __name__=='__main__':
    raise SystemExit(run_allowed({
        'FULL_E_MATCHED_PRACTICAL':'Gaussian full vs E one-bit; both descriptive practical boundaries',
        'FULL_U_MATCHED_PRACTICAL':'Nonparametric bounded full vs U raw profile; both descriptive',
        'E_BERNOULLI_IDENTITY':'E Bernoulli q in (0,1): population Gamma one-bit==full',
        'U_BERNOULLI_IDENTITY':'U Bernoulli endpoint law: population Gamma one-bit==full',
    }, 'Paired full-feedback comparison; incomplete pairs are now NOT silently dropped'))
