# -*- coding: utf-8 -*-
"""
多方法因果估计脚本：对核心暴露（exercise、lgrip、wspeed 等）在三队列上运行 6 种因果方法，
输出汇总对比表 multi_method_causal_summary.csv。
"""
import os
import sys
import logging
import pandas as pd
import numpy as np

# 归档于 archive/scripts/，需多一层 dirname 到项目根
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import USE_IMPUTED_DATA, IMPUTED_DATA_PATH, RAW_DATA_PATH, AGE_MIN, COLS_TO_DROP
from data.charls_complete_preprocessing import preprocess_charls_data, reapply_cohort_definition
from scripts.run_all_interventions_analysis import prepare_interventions
from scripts.run_all_physio_causal import create_binary_exposure, EXPOSURES
from causal.charls_causal_multi_method import estimate_causal_multi_method

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_multi_method_causal(output_dir='multi_method_causal', max_exposures=6):
    """
    对部分核心暴露运行 6 种因果方法，输出对比汇总。
    max_exposures: 最多运行的暴露数（避免过久）
    """
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

    df_clean = prepare_interventions(df_clean.copy())

    # 2. 构造待运行暴露列表：(treat_col, label, df_use)
    tasks = []
    tasks.append(('exercise', 'Exercise', df_clean.copy()))
    for col, label_cn, binarize_type, drop_cols in EXPOSURES:
        if len(tasks) >= max_exposures:
            break
        if col == 'exercise':
            continue
        new_col, df_out = create_binary_exposure(df_clean, col, binarize_type)
        if new_col:
            df_use = df_out.drop(columns=drop_cols, errors='ignore')
            tasks.append((new_col, label_cn, df_use))

    # 3. 三队列运行
    all_results = []
    for treat_col, label, df_use in tasks:
        if treat_col not in df_use.columns:
            logger.warning(f"跳过 {treat_col}：列不存在")
            continue

        for axis_name, df_sub in [
            ('Cohort_A', df_use[df_use['baseline_group'] == 0]),
            ('Cohort_B', df_use[df_use['baseline_group'] == 1]),
            ('Cohort_C', df_use[df_use['baseline_group'] == 2]),
        ]:
            df_sub = df_sub.dropna(subset=[treat_col]).copy()
            if len(df_sub) < 50 or df_sub[treat_col].nunique() < 2:
                continue
            out_sub = os.path.join(output_dir, treat_col, axis_name)
            logger.info(f">>> {axis_name} {treat_col} (n={len(df_sub)})")
            try:
                res_df = estimate_causal_multi_method(df_sub, treatment_col=treat_col, output_dir=out_sub)
                if len(res_df) > 0:
                    for _, r in res_df.iterrows():
                        all_results.append({
                            'exposure': treat_col,
                            'label': label,
                            'axis': axis_name,
                            'method': r['method'],
                            'ate': r['ate'],
                            'ate_lb': r['ate_lb'],
                            'ate_ub': r['ate_ub'],
                            'significant': r['significant'],
                            'n': len(df_sub),
                        })
            except Exception as e:
                logger.warning(f"  {axis_name} {treat_col} 失败: {e}")

    if all_results:
        summary = pd.DataFrame(all_results)
        out_csv = os.path.join(output_dir, 'multi_method_causal_summary.csv')
        summary.to_csv(out_csv, index=False, encoding='utf-8-sig')
        logger.info(f"多方法因果汇总已保存: {out_csv}")
        return summary
    return pd.DataFrame()


if __name__ == '__main__':
    run_multi_method_causal(max_exposures=6)
