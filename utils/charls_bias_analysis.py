# -*- coding: utf-8 -*-
"""
未测混杂偏倚分析（审稿意见补充）
模拟不同强度的未测混杂对 ATE 估计的影响，验证结果稳健性。

方法说明：本模块采用简化偏倚模拟（simplified bias simulation），通过线性回归
Y_sim ~ T_sim + W 在注入未测混杂 U 的代理后重估 ATE，用于定性评估结果对未测混杂的敏感性。
非标准 E-value 分析；若需 E-value 或更严格的敏感性框架，可参考 VanderWeele & Ding (2017)。
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from utils.charls_feature_lists import get_exclude_cols
from config import BBOX_INCHES

logger = logging.getLogger(__name__)


def run_bias_sensitivity(df, causal_col, treatment_col, output_dir, target_col='is_comorbidity_next',
                         confounder_strengths=None):
    """
    模拟未测混杂：假设存在 U 同时影响 T 和 Y，
    U->T 关联强度 = strength_t, U->Y 关联强度 = strength_y
    通过向残差中注入 U 的代理来模拟偏倚。
    """
    if confounder_strengths is None:
        confounder_strengths = [0, 0.1, 0.2, 0.3, 0.5]
    os.makedirs(output_dir, exist_ok=True)

    T = df[treatment_col].fillna(0).astype(int)
    Y = df[target_col].astype(float)
    exclude = get_exclude_cols(df, target_col=target_col, treatment_col=treatment_col)
    W_cols = [c for c in df.columns if c not in exclude and c not in [treatment_col, target_col, causal_col]]
    W_cols = [c for c in W_cols if c in df.columns]
    W = df[W_cols].select_dtypes(include=[np.number]).fillna(0)

    if len(W) < 50:
        return None

    try:
        from config import RANDOM_SEED
    except ImportError:
        RANDOM_SEED = 500
    results = []
    for s in confounder_strengths:
        np.random.seed(RANDOM_SEED)
        U = np.random.randn(len(df))
        T_sim = T + s * U
        Y_sim = Y + s * 0.5 * U
        reg = LinearRegression().fit(np.column_stack([T_sim, W]), Y_sim)
        ate_sim = reg.coef_[0]
        results.append({'confounder_strength': s, 'ATE_biased': ate_sim})

    orig_ate = df[causal_col].mean()
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, 'bias_sensitivity.csv'), index=False, encoding='utf-8-sig')

    plt.figure(figsize=(8, 5))
    plt.plot(res_df['confounder_strength'], res_df['ATE_biased'], 'o-', color='steelblue', linewidth=2)
    plt.axhline(y=orig_ate, color='red', linestyle='--', label=f'Original ATE={orig_ate:.4f}')
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Unmeasured Confounder Strength')
    plt.ylabel('ATE Estimate')
    plt.title('Bias Analysis: Effect of Unmeasured Confounding')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_bias_sensitivity.png'), dpi=200, bbox_inches=BBOX_INCHES)
    plt.close()

    with open(os.path.join(output_dir, 'bias_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write("Method: Simplified bias simulation (Y_sim ~ T_sim + W with U proxy injection).\n")
        f.write("Not E-value analysis; for qualitative sensitivity assessment only.\n\n")
        f.write(f"Bias Sensitivity Analysis ({treatment_col})\n")
        f.write(f"Original ATE: {orig_ate:.4f}\n")
        f.write("Simulated ATE under increasing unmeasured confounder strength:\n")
        for _, r in res_df.iterrows():
            f.write(f"  strength={r['confounder_strength']}: ATE={r['ATE_biased']:.4f}\n")
    return res_df
