#!/usr/bin/env bash
set -euo pipefail
python reproduce/plot_all_figures.py
pytest -q tests
