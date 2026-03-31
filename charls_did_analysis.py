# -*- coding: utf-8 -*-
"""
双重差分（DID）分析（审稿意见补充）
利用 CHARLS 多波次纵向数据，评估干预暴露前后共病风险变化。
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils.charls_feature_lists import get_exclude_cols

logger = logging.getLogger(__name__)


def run_did_analysis(df, treatment_col='exercise', output_dir='did_results', target_col='is_comorbidity_next'):
    """
    双重差分：比较 (干预组_t2 - 干预组_t1) - (对照组_t2 - 对照组_t1)
    需至少 2 个 wave，且同一 ID 有前后观测。
    """
    if 'wave' not in df.columns or df['wave'].nunique() < 2:
        logger.warning("DID 需至少 2 个 wave，跳过")
        return None

    os.makedirs(output_dir, exist_ok=True)
    df = df.sort_values(['ID', 'wave'])
    waves = sorted(df['wave'].dropna().unique())
    if len(waves) < 2:
        return None

    w1, w2 = waves[0], waves[-1]
    df1 = df[df['wave'] == w1][['ID', target_col, treatment_col]].rename(
        columns={target_col: 'Y_t1', treatment_col: 'T_t1'})
    df2 = df[df['wave'] == w2][['ID', target_col, treatment_col]].rename(
        columns={target_col: 'Y_t2', treatment_col: 'T_t2'})
    merged = df1.merge(df2, on='ID', how='inner')
    merged = merged.dropna(subset=['Y_t1', 'Y_t2', 'T_t1', 'T_t2'])
    merged['T_t2'] = merged['T_t2'].fillna(0).astype(int)
    merged['T_t1'] = merged['T_t1'].fillna(0).astype(int)
    merged['delta_Y'] = merged['Y_t2'] - merged['Y_t1']
    merged['treated_t2'] = (merged['T_t2'] == 1).astype(int)
    merged['post'] = 1
    if len(merged) < 50:
        logger.warning("DID 配对样本不足 50，跳过")
        return None

    X = merged[['treated_t2', 'post']]
    X['did'] = X['treated_t2'] * X['post']
    y = merged['delta_Y']
    reg = LinearRegression().fit(X, y)
    did_coef = reg.coef_[2]
    se = np.sqrt(np.var(reg.predict(X) - y) / len(y) + 1e-9)
    lb, ub = did_coef - 1.96 * se, did_coef + 1.96 * se

    res = {'DID_estimate': did_coef, 'DID_lb': lb, 'DID_ub': ub, 'n_pairs': len(merged),
           'wave_t1': w1, 'wave_t2': w2}
    pd.DataFrame([res]).to_csv(os.path.join(output_dir, f'did_{treatment_col}.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"DID ({treatment_col}): {did_coef:.4f} (95% CI: {lb:.4f}, {ub:.4f})")
    return res
