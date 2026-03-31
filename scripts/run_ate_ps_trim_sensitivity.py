# -*- coding: utf-8 -*-
"""
PS 修剪敏感性：对三队列循环 config.PS_TRIM_SENSITIVITY_SCENARIOS，调用 XLearner（无辅助输出），
汇总 ATE、95% CI、近似 P 与修剪前后 N 至 results/tables/ate_sensitivity_trimming.csv。

输出表含：applied_subset、pct_in_band_before_subset、overlap_trimmed_pct、ate_ci_source、
bootstrap_ate_successful_draws 等，便于核对「是否真子集」与 CI / p 的来源；列 p_value_footnote 为固定脚注文案。

运行（项目根目录）:
  python scripts/run_ate_ps_trim_sensitivity.py
  python scripts/run_ate_ps_trim_sensitivity.py --treatment exercise

数据源与主因果脚本一致：utils.charls_script_data_loader.load_df_for_analysis()
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal.charls_recalculate_causal_impact import (
    _p_value_two_sided_from_ci,
    estimate_causal_impact_xlearner,
)
from config import PS_TRIM_SENSITIVITY_SCENARIOS, RESULTS_TABLES, TREATMENT_COL
from utils.charls_script_data_loader import load_df_for_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COHORTS = (
    ("Cohort_A", 0),
    ("Cohort_B", 1),
    ("Cohort_C", 2),
)

P_VALUE_FOOTNOTE = (
    "p_value_approx: two-sided normal approximation from ATE and 95% CI "
    "via SE=(UB-LB)/3.92; see causal.charls_recalculate_causal_impact._p_value_two_sided_from_ci. "
    "Not identical to a Wald p from the same bootstrap that produced percentile CIs."
)


def run_sensitivity(treatment_col: str, *, bootstrap_replicates: int) -> pd.DataFrame:
    df = load_df_for_analysis()
    if df is None or len(df) == 0:
        raise RuntimeError("load_df_for_analysis() 返回空，请检查 config 与数据路径。")
    if "baseline_group" not in df.columns:
        raise RuntimeError("数据缺少 baseline_group。")
    if treatment_col not in df.columns:
        raise RuntimeError(f"数据缺少处理列 {treatment_col!r}。")

    rows: list[dict] = []
    for cohort_label, bg in COHORTS:
        df_sub = df[df["baseline_group"] == bg].copy()
        df_sub = df_sub.dropna(subset=[treatment_col])
        if len(df_sub) < 50 or df_sub[treatment_col].nunique() < 2:
            logger.warning("跳过 %s %s：样本不足或处理无变异 (n=%s)", cohort_label, treatment_col, len(df_sub))
            continue

        for trim_lo, trim_hi, scenario_label, force_trim in PS_TRIM_SENSITIVITY_SCENARIOS:
            with tempfile.TemporaryDirectory() as td:
                res_df, (ate, ate_lb, ate_ub) = estimate_causal_impact_xlearner(
                    df_sub.copy(),
                    treatment_col=treatment_col,
                    output_dir=td,
                    ps_trim_low=float(trim_lo),
                    ps_trim_high=float(trim_hi),
                    trim_force=bool(force_trim),
                    run_auxiliary_steps=False,
                    bootstrap_replicates=bootstrap_replicates,
                )
            ti = {}
            if res_df is not None and getattr(res_df, "attrs", None):
                ti = res_df.attrs.get("ps_trim_info") or {}
            p_approx = _p_value_two_sided_from_ci(ate, ate_lb, ate_ub)
            rows.append(
                {
                    "cohort_label": cohort_label,
                    "baseline_group": bg,
                    "scenario_label": scenario_label,
                    "trim_lo": trim_lo,
                    "trim_hi": trim_hi,
                    "trim_force": force_trim,
                    "applied_subset": ti.get("applied_subset", np.nan),
                    "pct_in_band_before_subset": ti.get("pct_in_band_before_subset", np.nan),
                    "overlap_trimmed_pct": ti.get("overlap_trimmed_pct", np.nan),
                    "n_before_trim": ti.get("n_before_trim", np.nan),
                    "n_after_trim": ti.get("n_after_trim", np.nan),
                    "treatment_col": treatment_col,
                    "bootstrap_replicates_planned": ti.get("bootstrap_replicates_planned", bootstrap_replicates),
                    "bootstrap_min_for_ci": ti.get("bootstrap_min_for_ci", np.nan),
                    "ate_ci_source": ti.get("ate_ci_source", ""),
                    "bootstrap_ate_successful_draws": ti.get("bootstrap_ate_successful_draws", np.nan),
                    "ate": ate,
                    "ate_lb": ate_lb,
                    "ate_ub": ate_ub,
                    "p_value_approx": p_approx,
                    "p_value_footnote": P_VALUE_FOOTNOTE,
                }
            )
            logger.info(
                "%s | %s | trim [%.4f,%.4f] force=%s | applied_subset=%s pct_in_band=%s | N %s→%s | "
                "ATE=%.6f CI=(%.6f,%.6f) p≈%.4g | ate_ci_source=%s",
                cohort_label,
                scenario_label,
                trim_lo,
                trim_hi,
                force_trim,
                ti.get("applied_subset", "NA"),
                ti.get("pct_in_band_before_subset", "NA"),
                ti.get("n_before_trim", "NA"),
                ti.get("n_after_trim", "NA"),
                float(ate) if pd.notna(ate) else float("nan"),
                float(ate_lb) if pd.notna(ate_lb) else float("nan"),
                float(ate_ub) if pd.notna(ate_ub) else float("nan"),
                float(p_approx) if pd.notna(p_approx) else float("nan"),
                ti.get("ate_ci_source", ""),
            )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="PS trimming sensitivity → ate_sensitivity_trimming.csv")
    ap.add_argument("--treatment", default=TREATMENT_COL, help="处理变量列名（默认 config.TREATMENT_COL）")
    ap.add_argument(
        "--bootstrap-n",
        type=int,
        default=80,
        help="XLearner 聚类 bootstrap 次数（ate_interval 失败时；主分析默认 200，终稿敏感性可改为 200 再跑）",
    )
    args = ap.parse_args()

    out_df = run_sensitivity(args.treatment, bootstrap_replicates=max(1, args.bootstrap_n))
    os.makedirs(RESULTS_TABLES, exist_ok=True)
    out_path = os.path.join(RESULTS_TABLES, "ate_sensitivity_trimming.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("已写入 %s （%s 行）", out_path, len(out_df))


if __name__ == "__main__":
    main()
