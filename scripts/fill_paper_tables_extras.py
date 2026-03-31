# -*- coding: utf-8 -*-
"""
Fill paper-facing tables without re-running the full Rubin MI × CPM pipeline.

1) rubin_pooled_psw_exercise.csv — PSW exercise ARD per imputation (m1..mN), Rubin-pooled SE + 95% CI.
2) subgroup_age70_xlearner_exercise.csv — XLearner exercise ATE/CI on single completed dataset,
   strata: Cohort A/B/C × age <70 vs ≥70 (prespecified binary age contrast).

Run from repo root:
  python scripts/fill_paper_tables_extras.py
  python scripts/fill_paper_tables_extras.py --psw-only
  python scripts/fill_paper_tables_extras.py --age-only
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OVERLAY_RANGES = [
    ("bmi", 16, 35),
    ("systo", 90, 200),
    ("diasto", 50, 120),
    ("mwaist", 60, 150),
    ("sleep", 2, 14),
    ("family_size", 0, 12),
]


def _prepare_mi_dataframe(df_m: pd.DataFrame, df_pre: pd.DataFrame | None) -> pd.DataFrame:
    from config import AGE_MIN, COLS_TO_DROP, CESD_CUTOFF, COGNITION_CUTOFF
    from data.charls_complete_preprocessing import reapply_cohort_definition
    from scripts.run_multi_exposure_causal import prepare_exposures

    if "age" in df_m.columns:
        df_m = df_m[df_m["age"] >= AGE_MIN].copy()
    prepare_exposures(df_m)
    df_m = df_m.drop(columns=[c for c in COLS_TO_DROP if c in df_m.columns], errors="ignore")
    if df_pre is not None:
        for col, lo, hi in OVERLAY_RANGES:
            if col in df_m.columns and col in df_pre.columns:
                imp_mean = df_m[col].mean()
                pre_mean = df_pre[col].mean()
                if not (lo <= imp_mean <= hi) and (lo <= pre_mean <= hi):
                    merge_cols = ["ID", "wave", col]
                    pre_sub = df_pre[merge_cols].drop_duplicates(subset=["ID", "wave"])
                    df_m = df_m.drop(columns=[col], errors="ignore").merge(
                        pre_sub, on=["ID", "wave"], how="left"
                    )
    df_m = reapply_cohort_definition(df_m, CESD_CUTOFF, COGNITION_CUTOFF)
    return df_m


def run_psw_rubin_mi() -> pd.DataFrame | None:
    from config import (
        IMPUTED_MI_DIR,
        N_MULTIPLE_IMPUTATIONS,
        RESULTS_TABLES,
        TREATMENT_COL,
    )
    from causal.charls_causal_methods_comparison import run_causal_methods_comparison
    from utils.rubin_pooling import rubin_pool, rubin_pool_ci

    n_mi = int(N_MULTIPLE_IMPUTATIONS or 0)
    mi_paths = [os.path.join(IMPUTED_MI_DIR, f"step1_imputed_m{m}.csv") for m in range(1, n_mi + 1)]
    if n_mi < 2 or not all(os.path.exists(p) for p in mi_paths):
        logger.warning("MI files m1..m%d missing under %s — skip PSW Rubin.", n_mi, IMPUTED_MI_DIR)
        return None

    pre_path = os.path.join(_ROOT, "preprocessed_data", "CHARLS_final_preprocessed.csv")
    df_pre = pd.read_csv(pre_path, encoding="utf-8-sig") if os.path.isfile(pre_path) else None

    psw_ests = {"A": [], "B": [], "C": []}
    psw_ses = {"A": [], "B": [], "C": []}

    for m, path in enumerate(mi_paths, start=1):
        logger.info("PSW MI: imputation %s/%s", m, n_mi)
        df_m = pd.read_csv(path, encoding="utf-8-sig")
        df_m = _prepare_mi_dataframe(df_m, df_pre)
        da = df_m[df_m["baseline_group"] == 0]
        db = df_m[df_m["baseline_group"] == 1]
        dc = df_m[df_m["baseline_group"] == 2]
        for lab, dsub in [("A", da), ("B", db), ("C", dc)]:
            if len(dsub) < 50:
                psw_ests[lab].append(np.nan)
                psw_ses[lab].append(np.nan)
                continue
            tdir = tempfile.mkdtemp(prefix=f"psw_mi_{lab}_m{m}_")
            try:
                res = run_causal_methods_comparison(
                    dsub,
                    TREATMENT_COL,
                    tdir,
                    dml_ate=None,
                    dml_lb=None,
                    dml_ub=None,
                )
                if res is None or len(res) == 0:
                    psw_ests[lab].append(np.nan)
                    psw_ses[lab].append(np.nan)
                    continue
                row = res[res["Method"] == "PSW"]
                if row.empty:
                    psw_ests[lab].append(np.nan)
                    psw_ses[lab].append(np.nan)
                    continue
                ate = float(row["ATE"].iloc[0])
                lb = float(row["ATE_lb"].iloc[0])
                ub = float(row["ATE_ub"].iloc[0])
                se = (ub - lb) / 3.92 if np.isfinite(lb) and np.isfinite(ub) else np.nan
                psw_ests[lab].append(ate)
                psw_ses[lab].append(se)
            finally:
                shutil.rmtree(tdir, ignore_errors=True)

    def _pool(ests, ses):
        valid = [(e, s) for e, s in zip(ests, ses) if np.isfinite(e)]
        if len(valid) < 2:
            return {"Q_bar": np.nan, "SE": np.nan, "lb": np.nan, "ub": np.nan}
        e, s = zip(*valid)
        r = rubin_pool(e, ses=s)
        lb, ub = rubin_pool_ci(r["Q_bar"], r["SE"], r["df"])
        return {"Q_bar": r["Q_bar"], "SE": r["SE"], "lb": lb, "ub": ub}

    rows = []
    for lab in ["A", "B", "C"]:
        p = _pool(psw_ests[lab], psw_ses[lab])
        rows.append(
            {
                "Cohort": lab,
                "PSW_ARD_pooled": p["Q_bar"],
                "PSW_SE_pooled": p["SE"],
                "PSW_lb": p["lb"],
                "PSW_ub": p["ub"],
                "M_imputations": n_mi,
            }
        )

    out = pd.DataFrame(rows)
    os.makedirs(RESULTS_TABLES, exist_ok=True)
    out_path = os.path.join(RESULTS_TABLES, "rubin_pooled_psw_exercise.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote %s", out_path)
    return out


def run_age70_xlearner() -> pd.DataFrame | None:
    from config import RESULTS_TABLES, TREATMENT_COL
    from causal.charls_recalculate_causal_impact import estimate_causal_impact_xlearner
    from utils.charls_script_data_loader import load_df_for_analysis

    df = load_df_for_analysis(apply_config_drop=True)
    if df is None or "age" not in df.columns:
        logger.warning("load_df_for_analysis failed or no age column.")
        return None

    rows = []
    cohort_map = [(0, "A"), (1, "B"), (2, "C")]
    for bg, clab in cohort_map:
        base = df[df["baseline_group"] == bg].copy()
        if len(base) < 30:
            continue
        for split_name, mask, tmp_tag in [
            ("<70", base["age"] < 70, "lt70"),
            ("≥70", base["age"] >= 70, "ge70"),
        ]:
            dsub = base.loc[mask].copy()
            n = len(dsub)
            if n < 40:
                logger.warning("Cohort %s age %s: n=%s too small, skip.", clab, split_name, n)
                continue
            n_events = int(dsub["is_comorbidity_next"].sum()) if "is_comorbidity_next" in dsub.columns else -1
            tdir = tempfile.mkdtemp(prefix=f"age70_{clab}_{tmp_tag}_")
            try:
                _res, (ate, lb, ub) = estimate_causal_impact_xlearner(
                    dsub,
                    treatment_col=TREATMENT_COL,
                    output_dir=tdir,
                    run_auxiliary_steps=False,
                    bootstrap_replicates=200,
                    bootstrap_min_for_ci=20,
                )
            except Exception as ex:
                logger.warning("XLearner failed Cohort %s %s: %s", clab, split_name, ex)
                ate, lb, ub = np.nan, np.nan, np.nan
            finally:
                shutil.rmtree(tdir, ignore_errors=True)
            rows.append(
                {
                    "Cohort": clab,
                    "Age_split": split_name,
                    "N": n,
                    "N_events": n_events,
                    "XLearner_ATE": ate,
                    "ATE_lb": lb,
                    "ATE_ub": ub,
                }
            )

    if not rows:
        return None
    out = pd.DataFrame(rows)
    os.makedirs(RESULTS_TABLES, exist_ok=True)
    out_path = os.path.join(RESULTS_TABLES, "subgroup_age70_xlearner_exercise.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote %s", out_path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psw-only", action="store_true")
    ap.add_argument("--age-only", action="store_true")
    args = ap.parse_args()
    do_psw = not args.age_only
    do_age = not args.psw_only
    if args.psw_only:
        do_age = False
    if args.age_only:
        do_psw = False

    if do_psw:
        run_psw_rubin_mi()
    if do_age:
        run_age70_xlearner()


if __name__ == "__main__":
    main()
