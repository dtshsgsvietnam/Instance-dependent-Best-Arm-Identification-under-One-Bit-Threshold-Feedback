"""Interactive terminal wizard that writes a JSON test case for run_custom_suite.py."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def ask(prompt, default=None):
    tail=f' [{default}]' if default is not None else ''
    x=input(prompt+tail+': ').strip()
    return x if x else (str(default) if default is not None else '')

def yn(prompt, default=True):
    d='Y' if default else 'N'
    x=ask(prompt+' (y/n)',d).lower()
    return x.startswith('y')

def floats(prompt, default):
    x=ask(prompt,' '.join(map(str,default))).replace(',',' ')
    return [float(v) for v in x.split()]

def ints(prompt, default): return int(ask(prompt,default))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='my_custom_case.json');a=p.parse_args()
    print('=== ONEBIT custom-case wizard ===')
    name=ask('Case name','my_cluster5')
    means=floats('Arm means (space/comma separated)',[.70,.64,.63,.62,.61])
    beta=float(ask('beta',.5)); workers=ints('Worker PROCESSES (0 = auto physical cores)',0)
    useE=yn('Run Model E?',True)
    efam=ask('E generator/family (gaussian or bernoulli)','gaussian').lower()
    if efam not in {'gaussian','bernoulli'}: raise ValueError('E family must be gaussian or bernoulli in this focused engine.')
    useU=yn('Run Model U?',True)
    ugen=ask('U data generator (beta or bernoulli)','beta').lower()
    if ugen not in {'beta','bernoulli'}: raise ValueError('U generator must be beta or bernoulli.')
    sigma=float(ask('E Gaussian sigma (ignored for Bernoulli)',.2)); conc=float(ask('U Beta concentration (ignored for Bernoulli)',12))
    q1=floats('Q1 deltas (large L recommended)',[1e-4,1e-5,1e-6,1e-7,1e-8,1e-10])
    q3=floats('Q3 deltas',[1e-4,1e-6,1e-8,1e-10])
    cor=floats('Correctness deltas',[.1,.05,.02,.01])
    fin=floats('Non-asymptotic deltas',[.01,.001,.0001])
    q1reps=ints('Q1 reps',200);q3reps=ints('Q3 reps',200);correps=ints('Correctness reps',600)
    ereps=ints('E non-asymptotic reps',400);ureps=ints('U certified non-asymptotic reps',80)
    cap=ints('General cap',700000);eta=float(ask('eta for finite coverage',.05))
    c={
      'name':name,'means':means,'beta':beta,'workers':workers,
      'models':{
        'E':{'enabled':useE,'family':efam,'sigma':sigma},
        'U':{'enabled':useU,'generator':ugen,'concentration':conc}},
      'numerics':{
        'E':{'qpoints':31,'xgrid':81,'oracle_refresh':25,'refresh':1,'decide':1},
        'U':{'xgrid':61,'qquad':301,'refresh':5,'decide':5}},
      'experiments':{
        'Q1':{'enabled':True,'deltas':q1,'reps':q1reps,'cap':cap,'E_boundary':'certified','U_boundary':'practical','trace_every':1000},
        'Q3':{'enabled':True,'deltas':q3,'reps':q3reps,'cap':cap,'E_boundary':'practical','U_boundary':'practical','full_boundary':'practical'},
        'correctness':{'enabled':True,'deltas':cor,'reps':correps,'cap':cap,'E_boundary':'certified','U_boundary':'practical','U_methods':'raw'},
        'nonasymptotic':{'enabled':True,'deltas':fin,'E_reps':ereps,'U_reps':ureps,'cap':cap,'eta':eta,'E_boundary':'certified','U_boundary':'certified'}
      }
    }
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(c,indent=2),encoding='utf-8')
    print('\nWROTE',path)
    print('Preview: python scripts/run_custom_suite.py --config',path)
    print('Run:     python scripts/run_custom_suite.py --config',path,'--run')

if __name__=='__main__': main()
