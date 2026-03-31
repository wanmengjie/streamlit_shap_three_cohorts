# -*- coding: utf-8 -*-
"""
握力与步行速度的因果效应探索性分析（测试脚本）
将 lgrip、wspeed 按中位数和四分位数二值化，估计 ATE（高 vs 低）。
T=1 表示高于切点（较好），ATE<0 表示高握力/快步行降低共病风险。
"""
import os
import sys
import pandas as pd
import numpy as np
import logging

# 归档于 archive/scripts/，需多一层 dirname 到项目根
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import USE_IMPUTED_DATA, IMPUTED_DATA_PATH, RAW_DATA_PATH, AGE_MIN, COLS_TO_DROP
from data.charls_complete_preprocessing import preprocess_charls_data, reapply_cohort_definition
from scripts.run_all_interventions_analysis import prepare_interventions
from causal.charls_recalculate_causal_impact import get_estimate_causal_impact

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_binary_exposures(df, col, cutpoint_type='median'):
    """
    将连续变量二值化。T=1 表示高于切点（较好）。
    cutpoint_type: 'median' 或 'quartile'
    - median: >= 中位数为 1
    - quartile: 仅保留 Q1 与 Q4，Q4=1 vs Q1=0（高 vs 低极端对比）
    """
    if col not in df.columns:
        return None, None
    vals = df[col].dropna()
    if len(vals) < 100:
        return None, None

    if cutpoint_type == 'median':
        cut = vals.median()
        new_col = f'{col}_above_median'
        df = df.copy()
        df[new_col] = np.where(df[col].isna(), np.nan, (df[col] >= cut).astype(int))
        return new_col, df
    else:  # quartile: Q4 vs Q1
        q1, q4 = vals.quantile(0.25), vals.quantile(0.75)
        new_col = f'{col}_high_vs_low'
        df = df.copy()
        mask_low = (df[col] <= q1) & df[col].notna()
        mask_high = (df[col] >= q4) & df[col].notna()
        df[new_col] = np.nan
        df.loc[mask_low, new_col] = 0
        df.loc[mask_high, new_col] = 1
        # 仅保留 0/1 行，排除中间
        return new_col, df


def run_test():
    """加载数据，构造握力/步行速度二值变量，估计 ATE"""
    output_dir = 'test_grip_wspeed_causal'
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
        return

    df_clean = prepare_interventions(df_clean.copy())

    # 2. Define variables and cutpoints for analysis
    exposures = [
        ('lgrip', 'Grip strength', 'median'),
        ('lgrip', 'Grip strength', 'quartile'),
        ('wspeed', 'Walking speed', 'median'),
        ('wspeed', 'Walking speed', 'quartile'),
    ]

    # 3. 因果分析
    results = []
    df_a = df_clean[df_clean['baseline_group'] == 0]
    df_b = df_clean[df_clean['baseline_group'] == 1]
    df_c = df_clean[df_clean['baseline_group'] == 2]

    for col, label_cn, cut_type in exposures:
        new_col, df_with_t = create_binary_exposures(df_clean, col, cutpoint_type=cut_type)
        if new_col is None:
            logger.warning(f"跳过 {col} ({cut_type})：数据不足或列缺失")
            continue

        cut_label = 'Median' if cut_type == 'median' else 'Q4 vs Q1'
        for axis_name, df_sub in [('Cohort_A', df_with_t[df_with_t['baseline_group'] == 0]),
                                  ('Cohort_B', df_with_t[df_with_t['baseline_group'] == 1]),
                                  ('Cohort_C', df_with_t[df_with_t['baseline_group'] == 2])]:
            # 仅保留 T 非缺失的行
            df_sub = df_sub.dropna(subset=[new_col]).copy()
            if len(df_sub) < 50:
                continue
            if df_sub[new_col].nunique() < 2:
                continue

            # 排除源变量，避免泄露
            df_analyze = df_sub.drop(columns=[col], errors='ignore')

            out_sub = os.path.join(output_dir, f'{new_col}', axis_name)
            os.makedirs(out_sub, exist_ok=True)
            try:
                res_df, (ate, lb, ub) = get_estimate_causal_impact()(
                    df_analyze, treatment_col=new_col, output_dir=out_sub
                )
                results.append({
                    'exposure': new_col,
                    'label': f'{label_cn}({cut_label})',
                    'axis': axis_name,
                    'ate': ate,
                    'ate_lb': lb,
                    'ate_ub': ub,
                    'n': len(df_sub),
                    'n_t1': (df_sub[new_col] == 1).sum(),
                    'n_t0': (df_sub[new_col] == 0).sum(),
                })
                logger.info(f"  {axis_name} {new_col}: ATE={ate:.4f} (95% CI: {lb:.4f}, {ub:.4f})")
            except Exception as e:
                logger.warning(f"  {axis_name} {new_col} 失败: {e}")

    # 4. 保存结果
    if results:
        out_csv = os.path.join(output_dir, 'grip_wspeed_ate_summary.csv')
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding='utf-8-sig')
        logger.info(f"结果已保存: {out_csv}")
    else:
        logger.warning("无有效结果")
    return results


if __name__ == '__main__':
    run_test()
