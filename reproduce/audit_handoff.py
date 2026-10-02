#!/usr/bin/env python3
"""
Audit the experiment handoff data without rerunning expensive simulations.
Run from the package root:

    python reproduce/audit_handoff.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EXPECTED = {
    "E_Q1.csv": (1200, 200, 1.0, 0),
    "E_Q3.csv": (1600, 200, 1.0, 0),
    "E_correctness.csv": (2400, 600, 1.0, 0),
    "E_nonasym_companion.csv": (2400, 400, 1.0, 0),
    "U_Q1.csv": (1200, 200, 1.0, 0),
    "U_Q3.csv": (1600, 200, 1.0, 0),
    "U_correctness.csv": (4800, 600, 1.0, 0),
    "legacy_E_CLUSTER5_b05.csv": (4000, 200, 1.0, 0),
    "legacy_U_HARD5_b05.csv": (2000, 200, 1.0, 0),
}

ok = True
for name, (rows, reps, stop_rate, errors) in EXPECTED.items():
    p = DATA / name
    if not p.exists():
        print("[MISSING]", name)
        ok = False
        continue
    df = pd.read_csv(p)
    r = len(df)
    nr = df["rep"].nunique()
    sr = float(df["stopped"].mean())
    er = int(df["error"].sum())
    status = (r == rows and nr == reps and abs(sr-stop_rate) < 1e-12 and er == errors)
    print(("[OK] " if status else "[FAIL] "), name,
          f"rows={r}, reps={nr}, stop_rate={sr:.3f}, errors={er}")
    ok &= status

# Verify the known reuse relation for Model-U Q1/Q3 at common deltas.
q1 = pd.read_csv(DATA / "U_Q1.csv")
q3 = pd.read_csv(DATA / "U_Q3.csv")
q3r = q3[q3["method"] == "u_raw"]
common = sorted(set(q1.delta.unique()).intersection(q3r.delta.unique()))
a = q1[q1.delta.isin(common)][["rep","delta","tau"]].sort_values(["rep","delta"]).reset_index(drop=True)
b = q3r[q3r.delta.isin(common)][["rep","delta","tau"]].sort_values(["rep","delta"]).reset_index(drop=True)
same = a.equals(b)
print(("[OK] " if same else "[FAIL] "),
      "U_Q3 reuses U_Q1 raw trajectories at common deltas:", same)
ok &= same

# Certified stress summary.
c = pd.read_csv(DATA / "U_certified_easy5_summary.csv")
cert_ok = (len(c)==4 and (c["stops_before_cap"]==0).all() and (c["reps"]==200).all())
print(("[OK] " if cert_ok else "[FAIL] "),
      "certified-U aggregate: 0 stops for all four deltas")
ok &= cert_ok

raise SystemExit(0 if ok else 1)
