# Non-asymptotic code patch — files to merge into the main repository

This bundle contains only the files added/modified for the final non-asymptotic experiment workflow.

## ADD these new files
- `configs_nonasym_final.json`
- `scripts/nonasym_pipeline.py`
- `scripts/preflight_nonasym.py`
- `scripts/analyze_nonasym_results.py`
- `scripts/estimate_runtime_from_pilot.py`
- `tests/test_nonasym_final_kit.py`

## REPLACE / MERGE these existing files
- `onebit_bai/experiment.py`
  - adds `--e-methods {tt,companion,both}` so the E finite-confidence run can simulate the companion only;
  - preserves resume/checkpoint support and existing E/U boundary switches.
- `onebit_bai/focused_report.py`
  - supports companion-only Model-E runs;
  - fixes the empty correctness-table `KeyError: 'delta'` encountered in the pilot.

## Outputs made by the non-asymptotic analysis
- `E_nonasym_correction_scale.png` / `.pdf`
- `U_practical_nonasym_shape.png` / `.pdf`
- `nonasym_summary.csv`

## Important interpretation
- E: theorem-facing companion experiment with the certified E boundary.
- U: practical raw-current-status Top-Two run, normalized by the theorem-motivated correction shape. It is NOT a direct numerical validation of the certified Model-U finite-confidence theorem.

Run syntax check after merging:

```bash
python -m py_compile onebit_bai/experiment.py onebit_bai/focused_report.py \
  scripts/nonasym_pipeline.py scripts/preflight_nonasym.py \
  scripts/analyze_nonasym_results.py scripts/estimate_runtime_from_pilot.py
python -m pytest -q tests/test_nonasym_final_kit.py
```
