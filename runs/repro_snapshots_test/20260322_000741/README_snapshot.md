# Reproducibility snapshot

Generated: 20260322_000741
Project root: C:\Users\lenovo\Desktop\因果机器学习

## Files

| File | Content |
|------|---------|
| git_summary.txt / git_info.json | Branch, commit, `git status --porcelain` |
| config_snapshot.txt / .json | Selected `config.py` attributes (imputation, Rubin, causal, cohort dirs) |
| python_version.json | `sys.executable` and `sys.version` |
| environment_pip_freeze.txt | `python -m pip freeze` |
| environment_conda_export.txt | `conda env export` (may be error text if conda unused) |

## How reviewers / you can use this

1. Checkout the recorded **commit** (if clean tree) or diff against it.
2. Restore Python env from **pip freeze** and/or **conda export**.
3. Match **config_snapshot** to the paper Methods (imputation, Rubin, temporal split, etc.).

Re-run analysis from project root: `python run_all_charls_analyses.py`
