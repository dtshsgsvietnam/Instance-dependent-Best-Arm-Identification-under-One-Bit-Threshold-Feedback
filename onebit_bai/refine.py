"""Numerical grid/quadrature refinement diagnostic for population Gamma."""
import argparse,json
from pathlib import Path
import numpy as np
from .experiment import compute_oracles


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,help='Experiment output directory')
    args=p.parse_args();root=Path(args.input)
    conf=json.loads((root/'config.json').read_text())
    original=json.loads((root/'oracles.json').read_text())
    qmin=.1 if conf['model']=='E' and conf.get('e_family','gaussian')=='bernoulli' else conf['qmin']
    qmax=.9 if conf['model']=='E' and conf.get('e_family','gaussian')=='bernoulli' else conf['qmax']
    oldq=np.linspace(qmin,qmax,conf['qpoints'])
    newq=np.linspace(qmin,qmax,2*conf['qpoints']-1)
    refined=compute_oracles(conf['model'],np.asarray(conf['mus']),conf['beta'],
            conf['sigma'],newq,2*conf['xgrid']-1,2*conf['qquad']-1,
            not conf['skip_full_oracle'],conf.get('e_family','gaussian'),conf.get('u_generator','beta'),conf.get('concentration',12.0))
    result={'old':{'gamma_bit':original['onebit_gamma'],'gamma_full':original['full_gamma'],
                   'q_points':len(oldq),'x_points':conf['xgrid'],'quad_nodes':conf['qquad']},
            'refined':{'gamma_bit':refined['onebit_gamma'],'gamma_full':refined['full_gamma'],
                   'q_points':len(newq),'x_points':2*conf['xgrid']-1,'quad_nodes':2*conf['qquad']-1},
            'relative_change_onebit':abs(refined['onebit_gamma']-original['onebit_gamma'])/
                    max(1e-30,abs(refined['onebit_gamma'])),
            'warning':'Stability under one refinement is necessary, not proof of global continuum optimality.'}
    if refined['full_gamma'] is not None and original['full_gamma'] is not None:
        result['relative_change_full']=abs(refined['full_gamma']-original['full_gamma'])/max(1e-30,abs(refined['full_gamma']))
    (root/'oracle_refinement.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
