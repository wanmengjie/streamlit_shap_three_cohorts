# -*- coding: utf-8 -*-
"""
额外汇总图：丰富论文/报告的可视化输出。
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from config import BBOX_INCHES, COHORT_A_DIR, COHORT_B_DIR, COHORT_C_DIR, COHORT_STEP_DIRS

def draw_baseline_summary(df, output_dir):
    """三组基线特征对比：Age, BMI, Chronic burden 分组柱状图"""
    if 'baseline_group' not in df.columns:
        return
    groups = ['Healthy', 'Depression only', 'Cognition impaired']
    cols = []
    if 'age' in df.columns: cols.append(('age', 'Age, years'))
    if 'bmi' in df.columns: cols.append(('bmi', 'BMI'))
    if 'chronic_burden' in df.columns: cols.append(('chronic_burden', 'Chronic disease count'))
    if not cols:
        return
    rows = []
    for g, label in enumerate(groups):
        sub = df[df['baseline_group'] == g]
        if len(sub) == 0:
            continue
        for col, vlabel in cols:
            if col in sub.columns:
                rows.append({'Group': label, 'Variable': vlabel, 'Mean': sub[col].mean(), 'SD': sub[col].std()})
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cols))
    width = 0.25
    for i, g in enumerate(groups):
        vals = plot_df[plot_df['Group'] == g]['Mean'].tolist()
        vals = (vals + [0.0] * len(cols))[:len(cols)]
        off = (i - 1) * width
        ax.bar(x + off, vals, width, label=g)
    ax.set_xticks(x)
    ax.set_xticklabels([v[1] for v in cols])
    ax.set_ylabel('Mean')
    ax.set_title('Baseline characteristics by group')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_baseline_summary.png'), dpi=300, bbox_inches=BBOX_INCHES)
    plt.close()

def draw_distribution_by_group(df, output_dir):
    """关键变量按组分布：Age, Exercise, Chronic burden 箱线图"""
    if 'baseline_group' not in df.columns:
        return
    df_plot = df.copy()
    df_plot['Group'] = df_plot['baseline_group'].map({0: 'Healthy', 1: 'Depression', 2: 'Cognition'})
    vars_plot = []
    if 'age' in df.columns: vars_plot.append(('age', 'Age, years'))
    if 'exercise' in df.columns: vars_plot.append(('exercise', 'Exercise (0/1)'))
    if 'chronic_burden' in df.columns: vars_plot.append(('chronic_burden', 'Chronic burden'))
    if not vars_plot:
        return
    fig, axes = plt.subplots(1, len(vars_plot), figsize=(5 * len(vars_plot), 5))
    if len(vars_plot) == 1:
        axes = [axes]
    for ax, (col, label) in zip(axes, vars_plot):
        sns.boxplot(data=df_plot, x='Group', y=col, ax=ax, hue='Group', palette='Set3', legend=False)
        ax.set_title(label)
        ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_distribution_by_group.png'), dpi=300, bbox_inches=BBOX_INCHES)
    plt.close()

def draw_auc_comparison(auc_a, auc_b, auc_c, output_dir):
    """三轴线 AUC 对比柱状图"""
    data = [
        ('Healthy (A)', auc_a),
        ('Depression→Comorbidity (B)', auc_b),
        ('Cognition→Comorbidity (C)', auc_c),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    vals = [v for _, v in data]
    labels = [l for l, _ in data]
    colors = ['#5cb85c', '#d9534f', '#5bc0de']
    ax.bar(x, vals, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('AUC')
    ax.set_title('Model performance (AUC) by axis')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.7, color='red', linestyle='--', alpha=0.4)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_auc_by_cohort.png'), dpi=300, bbox_inches=BBOX_INCHES)
    plt.close()

def draw_outcome_by_group(df, output_dir):
    """各组结局发生率（0 vs 1）堆叠柱状图"""
    if 'baseline_group' not in df.columns or 'is_comorbidity_next' not in df.columns:
        return
    groups = ['Healthy', 'Depression only', 'Cognition impaired']
    rates_0, rates_1 = [], []
    for g in range(3):
        sub = df[df['baseline_group'] == g]
        if len(sub) == 0:
            rates_0.append(0)
            rates_1.append(0)
        else:
            r1 = sub['is_comorbidity_next'].mean()
            rates_1.append(r1)
            rates_0.append(1 - r1)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    ax.bar(x, rates_0, label='No comorbidity', color='#c8e6c9', alpha=0.9)
    ax.bar(x, rates_1, bottom=rates_0, label='Incident comorbidity', color='#ff8a80', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha='right')
    ax.set_ylabel('Proportion')
    ax.set_title('Outcome distribution by baseline group')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_outcome_by_group.png'), dpi=300, bbox_inches=BBOX_INCHES)
    plt.close()

def draw_exercise_exposure_by_group(df, output_dir):
    """Exercise exposure proportion by baseline group"""
    if 'baseline_group' not in df.columns or 'exercise' not in df.columns:
        return
    groups = ['Healthy', 'Depression only', 'Cognition impaired']
    rates = []
    for g in range(3):
        sub = df[df['baseline_group'] == g]
        if len(sub) > 0:
            r = sub['exercise'].fillna(0).clip(0, 1).mean()
            rates.append(r)
        else:
            rates.append(0)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#5cb85c', '#d9534f', '#5bc0de']
    ax.bar(groups, rates, color=colors, alpha=0.8)
    ax.set_ylabel('Proportion with exercise')
    ax.set_title('Exercise exposure by baseline group')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_exercise_by_group.png'), dpi=300, bbox_inches=BBOX_INCHES)
    plt.close()

def draw_combined_subgroup_cate(output_dir):
    """汇总 A/B/C 三轴线亚组 CATE 为一张图（若存在）；路径与 config 轴线目录一致"""
    sub_dfs = []
    paths = [
        os.path.join(COHORT_A_DIR, COHORT_STEP_DIRS['subgroup'], 'subgroup_analysis_results.csv'),
        os.path.join(COHORT_B_DIR, COHORT_STEP_DIRS['subgroup'], 'subgroup_analysis_results.csv'),
        os.path.join(COHORT_C_DIR, COHORT_STEP_DIRS['subgroup'], 'subgroup_analysis_results.csv'),
    ]
    names = ['Healthy (A)', 'Depression (B)', 'Cognition (C)']
    for path, name in zip(paths, names):
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df['Axis'] = name
                sub_dfs.append(df)
            except Exception:
                pass
    if not sub_dfs:
        return
    comb = pd.concat(sub_dfs, ignore_index=True)
    comb['Label'] = comb['Subgroup'] + ': ' + comb['Value']
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ['#5cb85c', '#d9534f', '#5bc0de']  # A绿 B红 C蓝
    sns.barplot(data=comb, x='Label', y='CATE', hue='Axis', palette=palette, ax=ax)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_ylabel('Mean CATE')
    ax.set_title('Causal heterogeneity by subgroup (Axis A vs B vs C)')
    ax.legend(title='Axis')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_subgroup_cate_combined.png'), dpi=300, bbox_inches=BBOX_INCHES)
    plt.close()

def draw_all_extra_figures(df_clean, auc_a, auc_b, auc_c, output_dir):
    """一次性生成所有额外图"""
    os.makedirs(output_dir, exist_ok=True)
    draw_baseline_summary(df_clean, output_dir)
    draw_distribution_by_group(df_clean, output_dir)
    draw_outcome_by_group(df_clean, output_dir)
    draw_exercise_exposure_by_group(df_clean, output_dir)
    draw_auc_comparison(auc_a, auc_b, auc_c, output_dir)
    draw_combined_subgroup_cate(output_dir)
