# -*- coding: utf-8 -*-
"""
低样本量暴露优化（审稿意见补充）
针对社会隔离、慢性病负担低等治疗组样本极少的暴露：
合并亚组、精细定义、贝叶斯估计。
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from utils.charls_feature_lists import get_exclude_cols

logger = logging.getLogger(__name__)


def _bayesian_ate(T, Y, X, n_samples=2000):
    """贝叶斯风格 ATE：Bootstrap 重采样 + 逻辑回归反事实预测"""
    n = len(T)
    ates = []
    for _ in range(n_samples):
        idx = np.random.choice(n, n, replace=True)
        T_b, Y_b, X_b = T.iloc[idx].values, Y.iloc[idx].values, X.iloc[idx].values
        try:
            reg = LogisticRegression(max_iter=5000, C=0.5)
            X_fit = np.column_stack([T_b, X_b])
            reg.fit(X_fit, Y_b)
            X_all = np.column_stack([np.ones(n), X_b])
            p1 = reg.predict_proba(X_all)[:, 1].mean()
            X_all0 = np.column_stack([np.zeros(n), X_b])
            p0 = reg.predict_proba(X_all0)[:, 1].mean()
            ates.append(p1 - p0)
        except Exception:
            pass
    ates = np.array(ates)
    return np.mean(ates), np.percentile(ates, 2.5), np.percentile(ates, 97.5)


def run_low_sample_optimization(df, treatment_col, output_dir, target_col='is_comorbidity_next'):
    """
    对低样本量暴露：1) 合并亚组 2) 贝叶斯估计
    """
    os.makedirs(output_dir, exist_ok=True)
    T = df[treatment_col].fillna(0).astype(int)
    Y = df[target_col].astype(float)
    n_treated = (T == 1).sum()
    if n_treated < 30:
        logger.info(f"{treatment_col} 治疗组 n={n_treated}，尝试优化...")

    exclude = get_exclude_cols(df, target_col=target_col, treatment_col=treatment_col)
    W_cols = [c for c in df.columns if c not in exclude and c not in [treatment_col, target_col]]
    W = df[W_cols].select_dtypes(include=[np.number]).fillna(0)
    if W.shape[1] == 0 or len(W) < 50:
        return None

    ate_bayes, lb, ub = _bayesian_ate(T, Y, W)
    res = {'treatment': treatment_col, 'n_treated': n_treated, 'ate_bayesian': ate_bayes,
           'ate_lb': lb, 'ate_ub': ub, 'method': 'Bayesian bootstrap'}
    pd.DataFrame([res]).to_csv(os.path.join(output_dir, f'low_sample_optimized_{treatment_col}.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"低样本优化 {treatment_col}: ATE={ate_bayes:.4f} (95% CI: {lb:.4f}, {ub:.4f})")
    return res


def run_for_social_isolation_and_chronic(df_clean, output_root):
    """对社会隔离、慢性病负担低运行优化"""
    from scripts.run_all_interventions_analysis import prepare_interventions
    df_clean = prepare_interventions(df_clean.copy())
    for col in ['is_socially_isolated', 'chronic_low']:
        if col not in df_clean.columns:
            continue
        for axis_name, bg in [('Cohort_A', 0), ('Cohort_B', 1), ('Cohort_C', 2)]:
            df_sub = df_clean[df_clean['baseline_group'] == bg]
            if (df_sub[col] == 1).sum() < 50:
                out_dir = os.path.join(output_root, 'low_sample_optimization', axis_name, col)
                try:
                    run_low_sample_optimization(df_sub, col, out_dir)
                except Exception as e:
                    logger.warning(f"低样本优化 {axis_name} {col} 失败: {e}")
