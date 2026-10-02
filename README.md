# Anonymous AISTATS 2027 reproducibility package

This archive contains the simulation source, raw synthetic Monte-Carlo trial
records used in the reported plots, experiment configurations, tests, and a
single script that regenerates the seven current PDF/PNG experimental panels.

No external dataset, GPU, cloud service, or private credential is required.

## Environment

Recommended: Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the seven current figures

```bash
python reproduce/plot_all_figures.py
```

Outputs are written to `figures/`.

The plotting script uses 1000 bootstrap percentile resamples with bootstrap
seed 314159.  It does not silently discard censored runs.

## Smoke tests

```bash
pytest -q tests
```

## Re-run the cluster5 experiment suite

The source used for the v39.5 simulations is under `onebit_bai/`.
A full cluster5 configuration is provided in `configs/cluster5_main.json`.

Example:

```bash
python scripts/run_custom_suite.py \
  --config configs/cluster5_main.json \
  --workers 8 \
  --run \
  --output-root runs/cluster5
```

Worker count can be reduced without changing the statistical configuration.

## Certified Model-U stress configuration

`configs/certified_u_easy5_final.json` records the final separated five-arm
certified-boundary stress setting.  In the reported run every trajectory was
censored at the cap; this run is a computational diagnostic and is not pooled
with the practical raw-current-status experiment.

## Data accounting

The five confidence levels evaluated on one trajectory are dependent.
Replication IDs are the Monte-Carlo resampling units for the explicitly paired
legacy benchmark.  The paired statistic is a ratio of empirical means, not the
mean of run-wise ratios.

The files `U_Q1.csv` and `U_Q3.csv` share the same raw-current-status outcomes
at their common confidence levels.  They are not independent datasets.

## Anonymity

This review package intentionally contains no author names, affiliations,
usernames, machine addresses, or identifying repository links.
