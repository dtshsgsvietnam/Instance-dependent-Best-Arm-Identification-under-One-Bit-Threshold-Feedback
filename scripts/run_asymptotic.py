"""First-order rate / Gamma diagnostics for both experiments."""
from _launch import run_allowed
if __name__=='__main__':
    raise SystemExit(run_allowed({
        'E_ASYM_2':'Gaussian E, K=2; shared grid approximates the continuum oracle',
        'E_ASYM_3':'Gaussian E, K=3; tests the SHARED best-arm threshold law',
        'U_ASYM_2':'Beta U, K=2; practical raw-profile boundary is NOT PAC certified',
        'U_ASYM_3':'Beta U, K=3; practical raw-profile boundary is NOT PAC certified',
    }, 'Asymptotic first-order ratio mean(tau)*Gamma/log(1/delta)'))
