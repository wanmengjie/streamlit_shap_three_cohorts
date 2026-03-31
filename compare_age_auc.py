# -*- coding: utf-8 -*-
"""
快速对比 age≥50 vs age≥60 下三轴线的预测 AUC
"""
import logging
import pandas as pd
from charls_complete_preprocessing import preprocess_charls_data
from charls_model_comparison import compare_models

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def _parse_auc(x):
    """从 AUC 列解析数值（可能为 '0.746 (0.70-0.79)' 格式）"""
    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)
    s = str(x).split('(')[0].strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0

def run_auc_for_age(age_min):
    """对指定 age_min 跑预处理并返回三轴线最佳 AUC"""
    df = preprocess_charls_data('CHARLS.csv', age_min=age_min, write_output=False)
    if df is None:
        return None, None, None, 0
    df_a = df[df['baseline_group'] == 0]
    df_b = df[df['baseline_group'] == 1]
    df_c = df[df['baseline_group'] == 2]
    n_total = len(df)
    auc_a = auc_b = auc_c = 0.0
    if len(df_a) >= 50:
        perf_a, _ = compare_models(df_a, output_dir=f'_temp_auc/age{age_min}_A', target_col='is_comorbidity_next')
        auc_a = _parse_auc(perf_a.iloc[0]['AUC']) if len(perf_a) > 0 else 0
    if len(df_b) >= 50:
        perf_b, _ = compare_models(df_b, output_dir=f'_temp_auc/age{age_min}_B', target_col='is_comorbidity_next')
        auc_b = _parse_auc(perf_b.iloc[0]['AUC']) if len(perf_b) > 0 else 0
    if len(df_c) >= 50:
        perf_c, _ = compare_models(df_c, output_dir=f'_temp_auc/age{age_min}_C', target_col='is_comorbidity_next')
        auc_c = _parse_auc(perf_c.iloc[0]['AUC']) if len(perf_c) > 0 else 0
    return auc_a, auc_b, auc_c, n_total

if __name__ == '__main__':
    import os
    os.makedirs('_temp_auc', exist_ok=True)
    logger.info("="*60)
    logger.info("年龄标准对比：AUC (age≥50 vs age≥60)")
    logger.info("="*60)
    for age in [50, 60]:
        logger.info(f"\n>>> age >= {age} ...")
        auc_a, auc_b, auc_c, n = run_auc_for_age(age)
        if n > 0:
            logger.info(f"  样本量: {n}")
            logger.info(f"  A: {auc_a:.4f} | B: {auc_b:.4f} | C: {auc_c:.4f}")
        else:
            logger.info("  预处理失败")
    logger.info("\n" + "="*60 + "\n完成")
