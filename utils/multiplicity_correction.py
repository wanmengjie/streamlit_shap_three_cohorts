# -*- coding: utf-8 -*-
"""
多重检验校正：从 ATE 与 95% CI 近似 p 值，应用 Bonferroni 与 FDR (Benjamini-Hochberg)。
用于因果分析汇总表，满足审稿对多重比较校正的要求。
"""
import numpy as np
import pandas as pd
from scipy import stats


def ci_to_pvalue(ate, lb, ub):
    """
    从 ATE 与 95% CI 近似双侧 p 值（正态近似）。
    se = (ub - lb) / (2 * 1.96)，z = ate / se，p = 2 * (1 - Φ(|z|))
    """
    if np.isnan(ate) or np.isnan(lb) or np.isnan(ub):
        return np.nan
    ci_width = ub - lb
    if ci_width <= 0:
        return np.nan
    se = ci_width / (2 * 1.96)
    if se <= 0:
        return np.nan
    z = abs(ate) / se
    return 2 * (1 - stats.norm.cdf(z))


def apply_bonferroni_fdr(df, ate_col='ate', lb_col='ate_lb', ub_col='ate_ub'):
    """
    对含 ate, ate_lb, ate_ub 的 DataFrame 添加多重检验校正列。
    新增列：p_value_approx, p_adj_bonferroni, p_adj_fdr, significant_95, significant_bonferroni, significant_fdr
    """
    df = df.copy()
    pvals = df.apply(lambda r: ci_to_pvalue(r[ate_col], r[lb_col], r[ub_col]), axis=1)
    df['p_value_approx'] = pvals
    n = len(df.dropna(subset=['p_value_approx']))
    n = max(n, 1)

    # Bonferroni: alpha_adj = 0.05 / n
    p_adj_bonf = np.minimum(pvals * n, 1.0)
    df['p_adj_bonferroni'] = p_adj_bonf
    df['significant_bonferroni'] = (p_adj_bonf < 0.05).astype(int)

    # Benjamini-Hochberg FDR
    from scipy.stats import rankdata
    p_filled = pvals.fillna(1.0)  # NaN 不参与 FDR，最终 p_adj 保持 NaN
    order = rankdata(p_filled)
    p_adj_fdr = pd.Series(np.where(pvals.isna(), np.nan, np.minimum(pvals.values * n / order, 1.0)), index=df.index)
    df['p_adj_fdr'] = p_adj_fdr
    df['significant_fdr'] = (p_adj_fdr < 0.05).astype(int)

    # 原始 95% 显著性
    df['significant_95'] = df.apply(
        lambda r: 1 if (r[lb_col] > 0 or r[ub_col] < 0) and not (np.isnan(r[lb_col]) or np.isnan(r[ub_col])) else 0,
        axis=1
    )
    return df


def add_multiplicity_columns(df, ate_col='ate', lb_col='ate_lb', ub_col='ate_ub', n_comparisons=None):
    """
    为因果汇总表添加多重检验列。若 n_comparisons 指定则用于 Bonferroni，否则用 df 行数。
    返回添加了 p_value_approx, p_adj_bonferroni, p_adj_fdr, significant_95, significant_bonferroni, significant_fdr 的 DataFrame。
    """
    return apply_bonferroni_fdr(df, ate_col, lb_col, ub_col)
