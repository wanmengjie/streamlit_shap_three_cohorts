# -*- coding: utf-8 -*-
"""
因果森林在 CHARLS 数据上的具象化展示
输出：LIU_JUE_STRATEGIC_SUMMARY/fig_causal_forest_concrete.png
展示：样本个体特征 + 个体 CATE、CATE 分布、关键协变量与 CATE 的关系
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_causal_data(axis='B'):
    """加载某轴线的因果分析结果（与 config 轴线目录一致）"""
    from config import COHORT_A_DIR, COHORT_B_DIR, COHORT_C_DIR
    paths = {
        'A': os.path.join(COHORT_A_DIR, '03_causal', 'CAUSAL_ANALYSIS_exercise.csv'),
        'B': os.path.join(COHORT_B_DIR, '03_causal', 'CAUSAL_ANALYSIS_exercise.csv'),
        'C': os.path.join(COHORT_C_DIR, '03_causal', 'CAUSAL_ANALYSIS_exercise.csv'),
    }
    path = paths.get(axis, paths['B'])
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def draw_concrete_causal_forest(output_dir='LIU_JUE_STRATEGIC_SUMMARY'):
    """生成因果森林具象化图：样本个体 + CATE 分布 + 协变量关系"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 以轴线 B 为例（仅抑郁人群）
    df = load_causal_data('B')
    if df is None:
        print("未找到因果分析结果，请先运行主流程。")
        return
    
    causal_col = 'causal_impact_exercise'
    if causal_col not in df.columns:
        print(f"未找到 {causal_col} 列")
        return
    
    cate = df[causal_col].dropna()
    if len(cate) < 10:
        print("样本过少")
        return
    
    fig = plt.figure(figsize=(14, 10))
    
    # ========== 1. 左上：5 个典型个体的“画像”+ CATE ==========
    ax1 = fig.add_subplot(2, 2, 1)
    
    # 选取 CATE 分布的代表性个体：最小、25%、50%、75%、最大
    qs = [0, 0.25, 0.5, 0.75, 1.0]
    indices = [cate.quantile(q) for q in qs]
    sample_rows = []
    for q_val in indices:
        idx = (df[causal_col] - q_val).abs().idxmin()
        sample_rows.append(df.loc[idx])
    
    labels = ['Lowest CATE\n(max exercise benefit)', '25th pctl', 'Median', '75th pctl', 'Highest CATE\n(min exercise benefit)']
    cates_show = [r[causal_col] for r in sample_rows]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 5))  # 绿=获益大，红=获益小
    
    y_pos = np.arange(5)[::-1]
    bars = ax1.barh(y_pos, cates_show, color=colors, alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Individual CATE (effect of exercise on comorbidity risk)')
    ax1.set_title('Causal effect estimates for 5 representative individuals', fontsize=12)
    ax1.set_xlim(cate.min() - 0.02, cate.max() + 0.02)
    
    # Annotate age, residence, exercise status
    for i, r in enumerate(sample_rows):
        age = r.get('age', np.nan)
        rural = 'Rural' if r.get('rural', 0) == 1 else 'Urban'
        ex = 'Exercise' if r.get('exercise', 0) == 1 else 'No exercise'
        txt = f"  Age {age:.0f} {rural} {ex}"
        ax1.text(cates_show[i] + 0.005 if cates_show[i] >= 0 else cates_show[i] - 0.03, 
                 y_pos[i], txt, va='center', fontsize=8, color='#333')
    
    # ========== 2. 右上：CATE 分布直方图 ==========
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(cate, bins=40, color='teal', alpha=0.7, edgecolor='white')
    ax2.axvline(cate.mean(), color='red', linestyle='--', linewidth=2, label=f'ATE = {cate.mean():.4f}')
    ax2.axvline(0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Individual CATE')
    ax2.set_ylabel('Count')
    ax2.set_title('Cohort B (Depression-only): distribution of individual causal effects', fontsize=12)
    ax2.legend()
    
    # ========== 3. 左下：CATE 随年龄的变化（分箱均值） ==========
    ax3 = fig.add_subplot(2, 2, 3)
    if 'age' in df.columns:
        df_plot = df[['age', causal_col]].dropna()
        df_plot['age_bin'] = pd.cut(df_plot['age'], bins=[0, 65, 70, 75, 80, 120], 
                                    labels=['<65', '65-70', '70-75', '75-80', '80+'])
        bin_means = df_plot.groupby('age_bin', observed=True)[causal_col].agg(['mean', 'count', 'std'])
        bin_means = bin_means[bin_means['count'] >= 20]  # at least 20 subjects
        x_pos = np.arange(len(bin_means))
        ax3.bar(x_pos, bin_means['mean'], color='steelblue', alpha=0.8)
        ax3.errorbar(x_pos, bin_means['mean'], yerr=1.96 * bin_means['std'] / np.sqrt(bin_means['count']), 
                     fmt='none', color='black', capsize=3)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(bin_means.index)
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Age group')
        ax3.set_ylabel('Mean CATE')
        ax3.set_title('Heterogeneity of causal effect by age', fontsize=12)
    else:
        ax3.text(0.5, 0.5, 'No age column', ha='center', va='center', transform=ax3.transAxes)
    
    # ========== 4. 右下：城乡 CATE 对比 ==========
    ax4 = fig.add_subplot(2, 2, 4)
    if 'rural' in df.columns:
        urban = df[df['rural'] == 0][causal_col].dropna()
        rural = df[df['rural'] == 1][causal_col].dropna()
        if len(urban) >= 10 and len(rural) >= 10:
            bp = ax4.boxplot([urban, rural], tick_labels=['Urban', 'Rural'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#5bc0de')
            bp['boxes'][1].set_facecolor('#5cb85c')
            ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax4.set_ylabel('Individual CATE')
            ax4.set_title('Causal effect distribution: Urban vs Rural', fontsize=12)
        else:
            ax4.text(0.5, 0.5, 'Insufficient urban/rural samples', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'No rural column', ha='center', va='center', transform=ax4.transAxes)
    
    fig.suptitle('Causal forest application: CHARLS Cohort B (Depression-only)', fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig_causal_forest_concrete.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {out_path}")

if __name__ == '__main__':
    draw_concrete_causal_forest()
