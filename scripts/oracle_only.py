"""Only compute numerical population characteristic values; NO experiment replications.

Run from the package root:
    python scripts/oracle_only.py --model U --means .8 .2 --concentration 12 --quick
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from onebit_bai.experiment import compute_oracles

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['E','U'],required=True)
    p.add_argument('--means',nargs='+',type=float,required=True)
    p.add_argument('--e-family',choices=['gaussian','bernoulli'],default='gaussian')
    p.add_argument('--u-generator',choices=['beta','bernoulli'],default='beta')
    p.add_argument('--concentration',type=float,default=12.0)
    p.add_argument('--sigma',type=float,default=.2)
    p.add_argument('--beta',type=float,default=.5)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--output',help='Optional JSON output path')
    args=p.parse_args()
    assert len(args.means)>=2 and sorted(args.means)[-1]>sorted(args.means)[-2]
    q=np.linspace(.1,.9,5) if args.e_family=='bernoulli' else np.linspace(0,1,7 if args.quick else 17)
    x=11 if args.quick else 53
    quad=21 if args.quick else 181
    v=compute_oracles(args.model,np.array(args.means),args.beta,args.sigma,q,x,quad,True,
                      args.e_family,args.u_generator,args.concentration)
    v.update(model=args.model,means=args.means,beta=args.beta,
             u_beta_concentration=args.concentration,e_family=args.e_family)
    v['ratio_full_to_bit_gamma']=v['full_gamma']/v['onebit_gamma']
    v['population_data_processing_check']=(v['onebit_gamma']<=v['full_gamma']+max(1e-7,0.001*v['full_gamma']))
    v['warning']='Finite-grid/separator numerical population rates; solve again at higher resolution.'
    print(json.dumps(v,indent=2))
    if args.output:Path(args.output).write_text(json.dumps(v,indent=2))
if __name__=='__main__':main()
