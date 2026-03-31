# -*- coding: utf-8 -*-
"""
个体化干预效应列线图（审稿意见补充）
整合临床特征与 ITE，为个体化干预方案提供可视化工具。
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.charls_feature_lists import get_exclude_cols
from config import BBOX_INCHES

logger = logging.getLogger(__name__)


def run_nomogram(df, causal_col, treatment_col, output_dir, target_col='is_comorbidity_next',
                 top_features=5):
    """
    绘制简化列线图：展示关键特征 + ITE 对个体共病风险的贡献。
    """
    os.makedirs(output_dir, exist_ok=True)
    if causal_col not in df.columns:
        logger.warning("缺少因果效应列，跳过列线图")
        return False

    exclude = get_exclude_cols(df, target_col=target_col, treatment_col=treatment_col)
    feat_cols = [c for c in df.columns if c not in exclude and c not in [causal_col, target_col, treatment_col]]
    feat_cols = [c for c in feat_cols if c in ['age', 'gender', 'rural', 'bmi', 'exercise', 'sleep',
                                                'hibpe', 'diabe', 'cancre', 'lunge', 'hearte', 'stroke', 'arthre']][:top_features]
    if not feat_cols:
        feat_cols = df.select_dtypes(include=[np.number]).columns[:top_features].tolist()

    df_plot = df[feat_cols + [causal_col, target_col]].dropna()
    if len(df_plot) < 30:
        return False

    fig, ax = plt.subplots(figsize=(10, max(6, len(feat_cols) * 1.2)))
    n_show = min(5, len(df_plot))
    sample = df_plot.sample(n=n_show, random_state=500)
    y_pos = np.arange(n_show)[::-1]
    colors = plt.cm.RdYlGn_r((sample[causal_col] - sample[causal_col].min()) / (sample[causal_col].max() - sample[causal_col].min() + 1e-9))

    ax.barh(y_pos, sample[causal_col].values, color=colors, alpha=0.8)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    labels = []
    for _, r in sample.iterrows():
        lbl = ' | '.join([f'{c}={r[c]:.1f}' for c in feat_cols[:3]])
        labels.append(lbl[:40])
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Individual Treatment Effect (ITE)')
    ax.set_title('Individualized Intervention Effect Nomogram (Sample)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_nomogram_ite.png'), dpi=200, bbox_inches=BBOX_INCHES)
    plt.close()

    with open(os.path.join(output_dir, 'nomogram_guide.txt'), 'w', encoding='utf-8') as f:
        f.write("Nomogram Guide: ITE < 0 indicates benefit from intervention.\n")
        f.write(f"Features displayed: {feat_cols}\n")
        f.write("Green = higher benefit, Red = lower benefit.\n")
    return True
