"""Separate finite-grid E q-refinement and adversarial separator x-refinement.

No simulation replications. Useful because refining q AND x together can
conceal offsetting numerical biases in Gamma_E.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from onebit_bai.experiment import compute_oracles

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True)
    a=p.parse_args();r=Path(a.results)
    cfg=json.loads((r/'config.json').read_text())
    model=cfg['model'];means=np.asarray(cfg['mus']);qnum=cfg['qpoints'];xnum=cfg['xgrid'];quad=cfg['qquad']
    lo=.1 if model=='E' and cfg.get('e_family')=='bernoulli' else cfg['qmin']
    hi=.9 if model=='E' and cfg.get('e_family')=='bernoulli' else cfg['qmax']
    q=np.linspace(lo,hi,qnum)
    def do(qv,xv,quadn):
        z=compute_oracles(model,means,cfg['beta'],cfg['sigma'],qv,xv,quadn,
                not cfg['skip_full_oracle'],cfg.get('e_family','gaussian'),
                cfg.get('u_generator','beta'),cfg.get('concentration',12.0))
        return dict(gamma_bit=float(z['onebit_gamma']),gamma_full=z['full_gamma'])
    base=do(q,xnum,quad)
    x_only=do(q,2*xnum-1,quad)
    output=dict(model=model,base=base,x_only=x_only,
        x_refinement_delta=x_only['gamma_bit']-base['gamma_bit'],
        meaning='For LP over FINITE separators, adding x constraints should not increase Gamma except numerical tolerance.')
    if model=='E':
        q_more=np.linspace(lo,hi,2*qnum-1)
        q_only=do(q_more,xnum,quad)
        output['q_only']=q_only
        output['q_refinement_delta']=q_only['gamma_bit']-base['gamma_bit']
        output['expected_q_direction']='nonnegative: nested q action set enlarges design choices'
    else:
        output['quadrature_only']=do(q,xnum,2*quad-1)
        output['expected_q_direction']='No monotonic direction guaranteed for quadrature approximations'
    dest=r/'oracle_refinement_separate.json';dest.write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
