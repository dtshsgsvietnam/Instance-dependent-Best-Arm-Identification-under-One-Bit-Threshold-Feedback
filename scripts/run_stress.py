"""Tied suboptimals and hard multi-arm instance stress tests."""
from _launch import run_allowed
if __name__=='__main__':
    raise SystemExit(run_allowed({
        'E_TIED_SUBOPTIMAL':'E K=4 with tied suboptimal means',
        'U_HARD_FIVE':'U K=5 Beta with small adjacent gaps; very expensive',
    }, 'Stress tests; inspect cap-censoring and oracle refinement before any inference'))
