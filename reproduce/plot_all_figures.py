#!/usr/bin/env python3
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

BOOT = 1000
SEED = 314159

# Population rates recomputed with the included v39.5 oracle and the exact
# cluster5 numerical resolutions in configs/cluster5_main.json.
GAMMA_E_Q1 = 0.004080994480837258
GAMMA_U_Q1 = 0.0033946213856749087
GAMMA_E_OLD = 0.004080994480837258
GAMMA_U_OLD = 0.01344268

def bmean(x, rng):
    x = np.asarray(x, float)
    ids = rng.integers(0, len(x), size=(BOOT, len(x)))
    z = x[ids].mean(axis=1)
    return x.mean(), *np.quantile(z, [0.025,0.975])

def bnd(n,d,K=5):
    return math.log((K-1)*math.pi**2*max(n,1)**2/(6*d))

def pred(gamma,d,step,K=5,cap=500000):
    for n in range(max(K,step),cap+1,step):
        if n*gamma >= bnd(n,d,K):
            return n
    return np.nan

def axes(title,ylabel):
    fig = plt.figure(figsize=(5.0,3.55))
    ax = plt.axes()
    ax.set_title(title)
    ax.set_xlabel(r"$L_\delta=\log(1/\delta)$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.25)
    return fig,ax

def save(fig,name):
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(OUT/f"{name}.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

# 1. Main E calibration.
rng=np.random.default_rng(SEED)
df=pd.read_csv(DATA/"legacy_E_CLUSTER5_b05.csv")
rows=[]
for d,g in df[df.algorithm=="onebit_finite"].groupby("delta"):
    m,lo,hi=bmean(g.tau,rng)
    rows.append((math.log(1/d),m,lo,hi,pred(GAMMA_E_OLD,d,10)))
s=pd.DataFrame(rows,columns=["L","m","lo","hi","p"]).sort_values("L")
fig,ax=axes("(a) Gaussian · companion","Mean stopping time")
ax.plot(s.L,s.m,marker="o",label="Observed mean")
ax.fill_between(s.L,s.lo,s.hi,alpha=.18,label="95% bootstrap CI")
ax.plot(s.L,s.p,linestyle="--",label="Fixed-point predictor")
ax.legend(frameon=False); save(fig,"main_calibration_E_companion")

# 2. Main U calibration.
rng=np.random.default_rng(SEED+1)
df=pd.read_csv(DATA/"legacy_U_HARD5_b05.csv")
rows=[]
for d,g in df[df.algorithm=="onebit"].groupby("delta"):
    m,lo,hi=bmean(g.tau,rng)
    rows.append((math.log(1/d),m,lo,hi,pred(GAMMA_U_OLD,d,25)))
s=pd.DataFrame(rows,columns=["L","m","lo","hi","p"]).sort_values("L")
fig,ax=axes("(b) Bounded · current status","Mean stopping time")
ax.plot(s.L,s.m,marker="o",label="Observed mean")
ax.fill_between(s.L,s.lo,s.hi,alpha=.18,label="95% bootstrap CI")
ax.plot(s.L,s.p,linestyle="--",label="Fixed-point predictor")
ax.legend(frameon=False); save(fig,"main_calibration_U_raw")

# 3-4. Q1 first-order plots.
for i,(fn,method,gamma,title,out) in enumerate([
    ("E_Q1.csv","e_tt",GAMMA_E_Q1,"(a) Gaussian · Q1","supp_q1_E_first_order"),
    ("U_Q1.csv","u_raw",GAMMA_U_Q1,"(b) Bounded Beta · Q1","supp_q1_U_first_order"),
]):
    rng=np.random.default_rng(SEED+2+i)
    df=pd.read_csv(DATA/fn)
    rows=[]
    for d,g in df[df.method==method].groupby("delta"):
        L=math.log(1/d)
        z=g.tau.to_numpy(float)*gamma/L
        m,lo,hi=bmean(z,rng)
        rows.append((L,m,lo,hi))
    s=pd.DataFrame(rows,columns=["L","m","lo","hi"]).sort_values("L")
    fig,ax=axes(title,r"$\bar{\tau}_\delta/(T^\star L_\delta)$")
    y=s.m.to_numpy()
    ax.errorbar(s.L,y,yerr=np.vstack([y-s.lo,s.hi-y]),marker="o",capsize=3,
                label="Empirical mean (95% CI)")
    ax.axhline(1,linestyle="--",label="First-order reference = 1")
    ax.legend(frameon=False); save(fig,out)

# 5-6. Q3 full vs one-bit.
for i,(fn,methods,title,out) in enumerate([
    ("E_Q3.csv",[("e_full","Full feedback"),("e_tt","One-bit feedback")],
     "(a) Gaussian · Q3","supp_q3_E_full_vs_onebit"),
    ("U_Q3.csv",[("u_full","Full feedback"),("u_raw","One-bit feedback")],
     "(b) Bounded Beta · Q3","supp_q3_U_full_vs_onebit"),
]):
    rng=np.random.default_rng(SEED+4+i)
    df=pd.read_csv(DATA/fn)
    fig,ax=axes(title,"Mean stopping time")
    for method,label in methods:
        rows=[]
        for d,g in df[df.method==method].groupby("delta"):
            m,lo,hi=bmean(g.tau,rng)
            rows.append((math.log(1/d),m,lo,hi))
        s=pd.DataFrame(rows,columns=["L","m","lo","hi"]).sort_values("L")
        ax.plot(s.L,s.m,marker="o",label=label)
        ax.fill_between(s.L,s.lo,s.hi,alpha=.13)
    ax.legend(frameon=False); save(fig,out)

# 7. Model-E companion.
rng=np.random.default_rng(SEED+6)
df=pd.read_csv(DATA/"E_nonasym_companion.csv")
fig,ax=axes("Gaussian finite-confidence companion","Mean stopping time")
for method,label in [("e_tt","Top-Two"),
                     ("e_companion","Identification-schedule companion")]:
    rows=[]
    for d,g in df[df.method==method].groupby("delta"):
        m,lo,hi=bmean(g.tau,rng)
        rows.append((math.log(1/d),m,lo,hi))
    s=pd.DataFrame(rows,columns=["L","m","lo","hi"]).sort_values("L")
    ax.plot(s.L,s.m,marker="o",label=label)
    ax.fill_between(s.L,s.lo,s.hi,alpha=.13)
ax.legend(frameon=False); save(fig,"supp_E_companion")

print("Wrote seven PDF/PNG figure pairs to", OUT)
