"""Paired, nested-δ Monte Carlo and population information calculations."""
from __future__ import annotations
import os
# Prevent BLAS/OpenMP oversubscription when many replication processes run.
# setdefault preserves an explicit user choice. These must be set before NumPy/SciPy import.
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import argparse, json, time, socket, platform, sys, multiprocessing as mp, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from .oracles import (BetaLaw,BernoulliLaw,bernoulli_shared_grid_oracle,bernoulli_full_oracle,gaussian_shared_grid_oracle,gaussian_full_oracle,
                      u_population_oracle,u_full_population_oracle)
from .algorithms import Configuration,run_trajectory


PRESETS={
  'E':{
     'e_sep3':[.70,.60,.45],
     'e_tie4':[.70,.64,.64,.60],
     'e_easy2':[.75,.45]
  },
  'U':{
     'u_easy3':[.75,.45,.25],
     'u_hard5':[.70,.60,.50,.40,.30],
     'u_tie5':[.70,.60,.60,.40,.30],
     'u_easy2':[.8,.2]
  }
}


def methods_for(model,full=False,companion=True,safe_bit=True,u_methods="both"):
    if model=='E':
        m=['e_tt'];
        if companion:m.append('e_companion')
        if full:m.append('e_full')
        return m
    # Backward compatibility: --skip-safe-bit forces raw-only.
    if not safe_bit:
        u_methods="raw"
    if u_methods=="raw":
        m=['u_raw']
    elif u_methods=="marginal":
        m=['u_marginal']
    elif u_methods=="both":
        m=['u_raw','u_marginal']
    else:
        raise ValueError(f"Unknown u_methods={u_methods}")
    if full:m.append('u_full')
    return m


def compute_oracles(model,mus,beta,sigma,qgrid,xcount,quadrature,include_full=True,e_family="gaussian",u_generator="beta",concentration=12.0):
    if model=='E':
        if e_family=="bernoulli":
            design,gam=bernoulli_shared_grid_oracle(mus,beta,qgrid,xcount)
            wfull,gfull=bernoulli_full_oracle(mus,beta,max(41,xcount))
        else:
            design,gam=gaussian_shared_grid_oracle(mus,beta,qgrid,xcount,sigma)
            wfull,gfull=gaussian_full_oracle(mus,beta,sigma)
        return {'onebit_gamma':gam,'full_gamma':gfull,
                'onebit_mass':design.sum(axis=1).tolist(),'full_mass':wfull.tolist(),
                'query_design':design.tolist(), 'oracle_note':
                'E numerical shared LP; Gaussian full analytic, Bernoulli full numerical scalar LP.'}
    laws=[(BetaLaw(float(mu),float(concentration)) if u_generator=="beta" else BernoulliLaw(float(mu))) for mu in mus]
    w,gam=u_population_oracle(laws,beta,max(9,quadrature),max(9,xcount))
    if include_full:
        wf,gf=u_full_population_oracle(laws,beta,max(9,xcount),max(48,quadrature))
    else:wf,gf=None,None
    return {'onebit_gamma':gam,'full_gamma':gf,
            'onebit_mass':w.tolist(),'full_mass':wf.tolist() if wf is not None else None,
            'oracle_note':f'U generator={u_generator}; beta concentration={concentration} when applicable; model-free algorithm; x/q quadrature approximations.'}


def generate_local_streams(model,mus,max_rounds,sigma,concentration,seed,e_family="gaussian",u_generator="beta"):
    K=len(mus)
    X=np.empty((K,max_rounds),dtype=float)
    Q=np.empty_like(X)
    seeds=np.random.SeedSequence(seed).spawn(K)
    for k,(mu,s) in enumerate(zip(mus,seeds)):
        xrng,qrng=[np.random.default_rng(v) for v in s.spawn(2)]
        if model=='E':
            X[k]=xrng.binomial(1,mu,max_rounds) if e_family=='bernoulli' else xrng.normal(mu,sigma,max_rounds)
        else:
            X[k]=xrng.binomial(1,mu,max_rounds) if u_generator=='bernoulli' else xrng.beta(mu*concentration,(1-mu)*concentration,max_rounds)
        Q[k]=qrng.random(max_rounds)
    return X,Q



def _atomic_write_csv(rows, path: Path):
    """Checkpoint raw Monte Carlo rows atomically from the parent process."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _run_replication_task(payload):
    """One independent replication. Safe to execute in a separate process.

    The worker owns X/Q and runs all requested methods sequentially so paired
    one-bit/full comparisons use exactly the same local streams.
    """
    (rep, model, mus_list, cap, sigma, concentration, base_seed, e_family,
     u_generator, methods, configs, deltas) = payload
    mus=np.asarray(mus_list,dtype=float)
    X,Q=generate_local_streams(model,mus,cap,sigma,concentration,
                               base_seed+rep*999983,e_family,u_generator)
    true_best=int(np.argmax(mus))
    rep_rows=[]
    traces={}
    for idx,name in enumerate(methods):
        start=time.monotonic()
        result=run_trajectory(configs[name],X,Q,base_seed+rep*7907+idx*3331)
        elapsed=time.monotonic()-start
        if rep==0 and configs[name].trace_every>0:
            traces[name]=result.get('trace',[])
        for di,d in enumerate(deltas):
            stopped=result['times'][di]>=0
            recommendation=result['recommendations'][di]
            rep_rows.append({'rep':rep,'model':model,'method':name,'delta':d,'tau':(
                    result['times'][di] if stopped else np.nan),
                'censor_at':result['rounds_run'],'stopped':int(stopped),
                'error':int(stopped and recommendation!=true_best),
                'recommended':int(recommendation),'true_best':true_best,
                'stop_stat':(result['stop_stats'][di] if stopped else np.nan),
                'stop_boundary':(result['stop_boundaries'][di] if stopped else np.nan),
                'stop_stat_over_boundary':(result['stop_stats'][di]/result['stop_boundaries'][di]
                    if stopped and result['stop_boundaries'][di]>0 else np.nan),
                'lambda_at_stop':(result['stop_stats'][di]/result['times'][di]
                    if stopped and result['times'][di]>0 else np.nan),
                'boundary_over_L':(result['stop_boundaries'][di]/np.log(1/d) if stopped else np.nan),
                'run_time_s':elapsed,'seed':base_seed+rep*999983})
    return rep, rep_rows, traces


def _prewarm_numba_for_u():
    """Compile the tiny PAVA kernels before fork on Linux to reduce worker JIT storms."""
    try:
        from .u_profile import RawProfile, pair_glr_raw
        q=np.array([.2,.4,.6,.8],dtype=float)
        p1=RawProfile(q,np.array([1,1,0,0],dtype=np.int8))
        p2=RawProfile(q,np.array([1,0,0,0],dtype=np.int8))
        _=pair_glr_raw(p1,p2)
    except Exception:
        # Prewarming is an optimization only; worker execution remains authoritative.
        pass

def execute(args):
    model=args.model
    if model=="E" and args.e_family=="bernoulli" and (args.mu_lo<=0 or args.mu_hi>=1):
        args.mu_lo=.01;args.mu_hi=.99
    mus=np.array(args.means if args.means else PRESETS[model][args.preset])
    assert np.max(mus)>np.partition(mus,-2)[-2] and 0<np.min(mus) and np.max(mus)<1
    qgrid=np.linspace((.1 if args.e_family=="bernoulli" else args.qmin),
                     (.9 if args.e_family=="bernoulli" else args.qmax),args.qpoints)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    trials_path=out/'trials.csv'
    if trials_path.exists() and not args.resume:
        raise FileExistsError(
            f'{trials_path} already exists. Refusing to overwrite raw Monte Carlo data. '
            'Use --resume to continue this run, or choose a new --output directory.'
        )
    oracle=compute_oracles(model,mus,args.beta,args.sigma,qgrid,
                           args.xgrid,args.qquad,not args.skip_full_oracle,args.e_family,args.u_generator,args.concentration)
    oracle.update({'model':model,'mus':mus.tolist(),'beta':args.beta,
                   'delta_levels':args.deltas,'u_beta_concentration':args.concentration})
    (out/'oracles.json').write_text(json.dumps(oracle,indent=2))
    methods=methods_for(model,args.full,args.companion,not args.skip_safe_bit,args.u_methods)
    configs={name:Configuration(model=model,algorithm=name,deltas=tuple(args.deltas),e_family=args.e_family,
                        beta=args.beta,sigma=args.sigma,mu_lo=args.mu_lo,
                        mu_hi=args.mu_hi,qgrid=tuple(qgrid),oracle_xcount=args.xgrid,
                        oracle_refresh=args.oracle_refresh,challenger_refresh=args.refresh,
                        decision_every=args.decide,max_rounds=args.cap,
                        e_boundary=args.e_boundary,u_boundary=args.u_boundary,
                        full_boundary=args.full_boundary,
                        companion_a0=args.companion_a0,trace_every=args.trace_every,
                        continue_after_all_stops=args.continue_after_stop) for name in methods}
    config={**vars(args),'mus':mus.tolist(),'methods':methods,'oracle':oracle}
    # Resume guard: workers/output/verbosity may change, statistical settings may not.
    sig_payload={k:v for k,v in config.items() if k not in {
        'workers','resume','verbose','output','mp_start_method','oracle'}}
    sig_text=json.dumps(sig_payload,sort_keys=True,default=str,separators=(',',':'))
    sig=hashlib.sha256(sig_text.encode('utf-8')).hexdigest()
    sig_path=out/'statistical_signature.json'
    if args.resume and sig_path.exists():
        old_sig=json.loads(sig_path.read_text()).get('sha256')
        if old_sig != sig:
            raise RuntimeError('Refusing --resume: statistical configuration changed. '
                               'Only worker count/start method/output verbosity may change. '
                               'Use a new output directory for a different algorithm/config.')
    sig_path.write_text(json.dumps({'sha256':sig,'payload':sig_payload},indent=2,default=str))
    (out/'config.json').write_text(json.dumps(config,indent=2,default=str))
    expected_rows_per_rep=len(methods)*len(args.deltas)
    rows=[]
    completed=set()
    if args.resume and trials_path.exists():
        old_trials=pd.read_csv(trials_path)
        if not old_trials.empty:
            # Only complete replications are eligible for resume. The experiment
            # checkpoints after each full replication, so this also guards against
            # manually truncated CSV files.
            counts=old_trials.groupby('rep').size()
            completed=set(int(r) for r,c in counts.items() if int(c)==expected_rows_per_rep)
            old_trials=old_trials[old_trials.rep.isin(completed)].copy()
            rows=old_trials.to_dict('records')
            print(f'RESUME: found {len(completed)} complete replications in {trials_path}',flush=True)
    run_meta={
        'hostname':socket.gethostname(),
        'platform':platform.platform(),
        'python':sys.version,
        'logical_cpus':os.cpu_count(),
        'workers':int(args.workers),
        'mp_start_method':args.mp_start_method,
        'checkpoint':'atomic parent-process trials.csv after every completed replication',
    }
    (out/'run_environment.json').write_text(json.dumps(run_meta,indent=2),encoding='utf-8')
    t0=time.monotonic()
    pending=[rep for rep in range(args.reps) if rep not in completed]
    total_done=len(completed)
    if not pending:
        print(f'RESUME: all {args.reps} replications are already complete.',flush=True)
    payload_common=(model,mus.tolist(),args.cap,args.sigma,args.concentration,args.seed,
                    args.e_family,args.u_generator,methods,configs,tuple(args.deltas))

    def accept_result(rep, rep_rows, traces):
        nonlocal total_done, rows
        rows.extend(rep_rows)
        for name,tr in traces.items():
            if tr:
                pd.DataFrame(tr).to_csv(out/f'trace_rep0_{name}.csv',index=False)
        # Stable ordering makes diffs/reuse easier even though workers finish out of order.
        rows.sort(key=lambda r:(int(r['rep']),str(r['method']),float(r['delta'])))
        _atomic_write_csv(rows,trials_path)
        total_done += 1
        elapsed=time.monotonic()-t0
        rate=(total_done-len(completed))/elapsed if elapsed>0 else 0.0
        remaining=args.reps-total_done
        eta_s=remaining/rate if rate>0 else float('inf')
        if args.verbose:
            eta_txt=(f'{eta_s/60:.1f} min' if np.isfinite(eta_s) else '?')
            print(f'completed replication {total_done}/{args.reps} (rep={rep+1}) | '
                  f'elapsed={elapsed:.1f}s | {rate:.3f} rep/s | ETA={eta_txt}',flush=True)

    if args.workers<=1:
        for rep in pending:
            payload=(rep,*payload_common)
            rr,rep_rows,traces=_run_replication_task(payload)
            accept_result(rr,rep_rows,traces)
    else:
        if model=='U' and os.name=='posix':
            _prewarm_numba_for_u()
        method=args.mp_start_method
        if method=='auto':
            method='fork' if os.name=='posix' else 'spawn'
        ctx=mp.get_context(method)
        print(f'PARALLEL: {args.workers} worker processes, start_method={method}, '
              f'{len(pending)} pending replications',flush=True)
        # Replications are independent.  Only the parent writes trials.csv, so
        # checkpointing remains race-free and resumable.
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futs={ex.submit(_run_replication_task,(rep,*payload_common)):rep for rep in pending}
            try:
                for fut in as_completed(futs):
                    rep=futs[fut]
                    rr,rep_rows,traces=fut.result()
                    accept_result(rr,rep_rows,traces)
            except BaseException:
                for f in futs: f.cancel()
                print('A worker failed/interrupted. Completed replications are already checkpointed; '
                      'rerun with --resume.',flush=True)
                raise
    print(f'Wrote {out}/trials.csv, config.json, oracles.json, run_environment.json')
    return out


def make_parser():
    p=argparse.ArgumentParser(description='AISTATS one-bit E/U and optional full-feedback benchmarks')
    p.add_argument('--model',required=True,choices=['E','U'])
    p.add_argument('--preset',default=None)
    p.add_argument('--means',type=float,nargs='+',default=None)
    p.add_argument('--reps',type=int,default=10)
    p.add_argument('--deltas',type=float,nargs='+',default=[.2,.1,.05])
    p.add_argument('--beta',type=float,default=.5)
    p.add_argument('--sigma',type=float,default=.2)
    p.add_argument('--e-family',choices=['gaussian','bernoulli'],default='gaussian')
    p.add_argument('--u-generator',choices=['beta','bernoulli'],default='beta')
    p.add_argument('--u-methods',choices=['raw','marginal','both'],default='both',help='Which U stopping variant(s) to simulate. raw-only avoids duplicate raw-profile trajectories.')
    p.add_argument('--concentration',type=float,default=12.)
    p.add_argument('--qmin',type=float,default=0.)
    p.add_argument('--qmax',type=float,default=1.)
    p.add_argument('--qpoints',type=int,default=9)
    p.add_argument('--xgrid',type=int,default=17)
    p.add_argument('--qquad',type=int,default=81)
    p.add_argument('--mu-lo',type=float,default=-.2)
    p.add_argument('--mu-hi',type=float,default=1.2)
    p.add_argument('--oracle-refresh',type=int,default=80)
    p.add_argument('--refresh',type=int,default=10,help='Recompute leader/challenger each this many rounds (1 is closest to exact policy)')
    p.add_argument('--decide',type=int,default=10)
    p.add_argument('--cap',type=int,default=12000)
    p.add_argument('--seed',type=int,default=710101)
    p.add_argument('--output',default='results')
    p.add_argument('--full',action='store_true',help='Include full-feedback sampling comparator')
    p.add_argument('--skip-full-oracle',action='store_true')
    p.add_argument('--companion',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--skip-safe-bit',action='store_true')
    p.add_argument('--e-boundary',choices=['certified','practical'],default='certified')
    p.add_argument('--u-boundary',choices=['certified','practical'],default='practical')
    p.add_argument('--full-boundary',choices=['certified','practical'],default='practical')
    p.add_argument('--companion-a0',type=float,default=.12)
    p.add_argument('--trace-every',type=int,default=0,help='Save rep-0 evidence-rate trace every this many decision times; 0 disables')
    p.add_argument('--continue-after-stop',action='store_true',help='Continue trajectory to cap after all requested delta crossings; useful for lambda_hat -> Gamma diagnostics')
    p.add_argument('--verbose',action='store_true')
    p.add_argument('--resume',action='store_true',help='Resume from checkpointed complete replications in trials.csv; never overwrite them.')
    p.add_argument('--workers',type=int,default=1,help='Independent replication worker PROCESSES (CPU parallelism). 1 = serial.')
    p.add_argument('--mp-start-method',choices=['auto','fork','spawn'],default='auto',help='Linux SSH server: auto/fork recommended; spawn is portable but slower.')
    return p


def main():
    parser=make_parser();args=parser.parse_args()
    if args.preset is None:args.preset='e_sep3' if args.model=='E' else 'u_easy3'
    if args.model=='U' and args.u_boundary=='certified':
        print('WARNING: paper-certified U sieve penalty is enormous at practical sample sizes; trajectories may all censor.')
    if args.model=='E' and args.e_boundary=='certified':
        print('E certified boundary is valid under the Gaussian compact model; finite query/oracle discretization changes allocation theorem, not the boundary proof.')
    if args.model=='U' and args.u_boundary=='practical':
        print('WARNING: U practical RAW-PROFILE stopping is not δ-certified; do not interpret errors as a proof.')
    execute(args)

if __name__=='__main__':main()
