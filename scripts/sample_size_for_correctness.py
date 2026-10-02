"""Plan independent replications needed for exact one-sided zero-error bounds."""
from __future__ import annotations
import argparse,math
p=argparse.ArgumentParser()
p.add_argument('--deltas',type=float,nargs='+',default=[.1,.05,.02,.01,.001])
p.add_argument('--alpha',type=float,default=.05)
p.add_argument('--simultaneous-cells',type=int,default=1)
a=p.parse_args()
per=a.alpha/a.simultaneous_cells
print(f'One-sided confidence 1-alpha={1-a.alpha}, cells={a.simultaneous_cells}')
for d in a.deltas:
    n=math.ceil(math.log(per)/math.log1p(-d))
    print(f'delta={d:9g} -> n>={n} independent replications if ZERO wrong stops AND all stop')
