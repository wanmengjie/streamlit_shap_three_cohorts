# -*- coding: utf-8 -*-
"""
仅运行 X-Learner 因果分析（快速版，跳过预测打擂、SHAP、敏感性等）。
加载数据后对三队列（A/B/C）运行 exercise 的 XLearner 因果估计，输出 ATE 与 95% CI。
运行: python archive/scripts/run_xlearner_causal_only.py（在项目根目录）
"""
import os
import sys
import logging
import pandas as pd

# 归档于 archive/scripts/，需多一层 dirname 到项目根
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import USE_IMPUTED_DATA, IMPUTED_DATA_PATH, RAW_DATA_PATH, AGE_MIN, COLS_TO_DROP
from data.charls_complete_preprocessing import preprocess_charls_data, reapply_cohort_definition
from causal.charls_recalculate_causal_impact import estimate_causal_impact_xlearner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = 'LIU_JUE_STRATEGIC_SUMMARY/xlearner_only'


def run_xlearner_only(treatment_col='exercise', output_dir=OUTPUT_DIR):
    """仅运行 X-Learner 因果分析，三队列 × 单干预"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    df_clean = None
    if USE_IMPUTED_DATA and os.path.exists(IMPUTED_DATA_PATH):
        try:
            df_clean = pd.read_csv(IMPUTED_DATA_PATH, encoding='utf-8-sig')
            df_clean = df_clean.drop(columns=[c for c in COLS_TO_DROP if c in df_clean.columns], errors='ignore')
            df_clean = df_clean[df_clean['age'] >= AGE_MIN]
            df_clean = reapply_cohort_definition(df_clean, 10, 10)
            logger.info(f"已加载插补数据，n={len(df_clean)}")
        except Exception as e:
            logger.warning(f"插补数据加载失败: {e}")
    if df_clean is None:
        df_clean = preprocess_charls_data(RAW_DATA_PATH, age_min=AGE_MIN, write_output=False)
    if df_clean is None:
        logger.error("数据加载失败")
        return pd.DataFrame()

    # 2. 三队列运行 XLearner
    results = []
    for axis_name, axis_label, df_sub in [
        ('Cohort_A', 'A', df_clean[df_clean['baseline_group'] == 0]),
        ('Cohort_B', 'B', df_clean[df_clean['baseline_group'] == 1]),
        ('Cohort_C', 'C', df_clean[df_clean['baseline_group'] == 2]),
    ]:
        df_sub = df_sub.dropna(subset=[treatment_col]).copy()
        if len(df_sub) < 50 or df_sub[treatment_col].nunique() < 2:
            logger.warning(f"跳过 {axis_name}：样本不足或干预无变异")
            continue
        out_sub = os.path.join(output_dir, axis_name)
        logger.info(f">>> {axis_name} {treatment_col} (n={len(df_sub)})")
        try:
            res_df, (ate, ate_lb, ate_ub) = estimate_causal_impact_xlearner(
                df_sub, treatment_col=treatment_col, output_dir=out_sub
            )
            results.append({
                'axis': axis_name,
                'exposure': treatment_col,
                'method': 'XLearner',
                'ate': ate,
                'ate_lb': ate_lb,
                'ate_ub': ate_ub,
                'n': len(df_sub),
                'significant': 1 if (pd.notna(ate_lb) and pd.notna(ate_ub) and (ate_lb > 0 or ate_ub < 0)) else 0,
            })
        except Exception as e:
            logger.warning(f"  {axis_name} 失败: {e}")

    if results:
        summary = pd.DataFrame(results)
        out_csv = os.path.join(output_dir, 'xlearner_ate_summary.csv')
        summary.to_csv(out_csv, index=False, encoding='utf-8-sig')
        logger.info(f"XLearner 汇总已保存: {out_csv}")
        for _, r in summary.iterrows():
            sig = "*" if r['significant'] else ""
            logger.info(f"  {r['axis']}: ATE={r['ate']:.4f} (95% CI: {r['ate_lb']:.4f}, {r['ate_ub']:.4f}) {sig}")
        return summary
    return pd.DataFrame()


if __name__ == '__main__':
    run_xlearner_only()
