# -*- coding: utf-8 -*-
"""
绘制倾向评分重叠图 (Propensity Score Overlap Plot)
对应论文 Supplementary Figure S2
"""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from data.charls_complete_preprocessing import preprocess_charls_data
from causal.charls_causal_methods_comparison import _prepare_covariates
from utils.charls_feature_lists import get_exclude_cols
from config import OUTPUT_ROOT, BBOX_INCHES, TARGET_COL

def draw_propensity_overlap():
    # 1. 加载数据
    print("Loading data...")
    df = preprocess_charls_data('CHARLS.csv', age_min=60, write_output=False)
    if df is None:
        print("Data loading failed.")
        return

    # 2. 筛选 Cohort B (Depression-only)
    # 论文重点关注 Cohort B 的运动干预效果 (PSW 显著)
    df_b = df[df['baseline_group'] == 1].copy()
    print(f"Cohort B sample size: {len(df_b)}")

    # 3. 设定干预与结局
    T = 'exercise'
    Y = TARGET_COL
    
    # 4. 准备协变量
    exclude_cols = get_exclude_cols(df_b, target_col=Y, treatment_col=T)
    X, _ = _prepare_covariates(df_b, T, Y, exclude_cols)
    
    if X is None:
        print("Covariate preparation failed.")
        return

    # 5. 计算倾向评分 (Propensity Score)
    print("Calculating propensity scores...")
    T_vals = df_b[T].fillna(0).astype(int)
    ps_model = LogisticRegression(max_iter=1000, C=1e-2, solver='lbfgs', random_state=500)
    ps_model.fit(X, T_vals)
    ps = ps_model.predict_proba(X)[:, 1]
    
    df_b['ps'] = ps

    # 6. 绘制重叠图
    print("Plotting overlap...")
    plt.figure(figsize=(10, 6))
    
    # 绘制密度图
    sns.kdeplot(data=df_b[df_b[T] == 0], x='ps', fill=True, color='#d9534f', label='Control (No Exercise)', alpha=0.3)
    sns.kdeplot(data=df_b[df_b[T] == 1], x='ps', fill=True, color='#5cb85c', label='Treated (Exercise)', alpha=0.3)
    
    # 绘制镜像直方图 (Mirror Histogram) - 可选，这里用简单的密度图即可满足 Figure S2 要求
    
    plt.title('Propensity Score Overlap (Cohort B: Exercise Intervention)', fontsize=14, fontweight='bold')
    plt.xlabel('Propensity Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    
    # 添加说明文字
    plt.text(0.5, -0.15, 'Substantial overlap indicates validity of the positivity assumption.', 
             ha='center', va='top', transform=plt.gca().transAxes, fontsize=10, style='italic')

    # 7. 保存图片
    output_dir = OUTPUT_ROOT
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig_propensity_overlap.png')
    plt.savefig(output_path, dpi=300, bbox_inches=BBOX_INCHES)
    print(f"Figure S2 saved to: {output_path}")

if __name__ == '__main__':
    draw_propensity_overlap()
