"""Explicit independent replication PAC diagnostic, with censor warnings."""
from _launch import run_allowed
if __name__=='__main__':
    raise SystemExit(run_allowed({
        'E_CORRECTNESS':'Gaussian E certified GLR (assuming exact real-arithmetic evaluation)',
        'U_CORRECTNESS_MARGINAL':'U marginal bit GLR has a known safe boundary; raw method in same run is PRACTICAL',
        'U_CERTIFIED_BOUNDARY':'U original raw-profile boundary: genuine theorem but very conservative',
        'E_BERNOULLI_IDENTITY':'E Bernoulli information-preservation and error sanity',
        'U_BERNOULLI_IDENTITY':'U endpoint Bernoulli information-preservation and error sanity',
    }, 'Wrong-stop error / one-sided exact Clopper-Pearson diagnostic'))
