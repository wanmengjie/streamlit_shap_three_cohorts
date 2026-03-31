# -*- coding: utf-8 -*-
"""
Fig 2: 概念框架图 + 三轴线示意图
输出：LIU_JUE_STRATEGIC_SUMMARY/fig2_conceptual_framework.png
"""
import os
from config import *  # 加载字体配置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_conceptual_framework(output_path=None):
    output_dir = 'LIU_JUE_STRATEGIC_SUMMARY'
    if output_path is None:
        output_path = os.path.join(output_dir, 'fig2_conceptual_framework.png')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # 颜色
    c_healthy = '#5cb85c'
    c_dep = '#d9534f'
    c_cog = '#5bc0de'
    c_outcome = '#f0ad4e'

    # 1. 基线人群框（顶部）
    ax.add_patch(mpatches.FancyBboxPatch((1, 7.2), 12, 1.2, boxstyle="round,pad=0.05",
                                         facecolor='#f8f9fa', edgecolor='#333', linewidth=1.5))
    ax.text(7, 7.8, 'Baseline population (age ≥60, incident cohort)', ha='center', va='center', fontsize=12, fontweight='bold')

    # 2. 三轴线分支
    # 轴线 A
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 4.5), 3.5, 2.2, boxstyle="round,pad=0.05",
                                         facecolor=c_healthy, edgecolor='#2d5a2d', linewidth=1.2, alpha=0.85))
    ax.text(2.25, 6.3, 'Axis A', ha='center', fontsize=11, fontweight='bold')
    ax.text(2.25, 5.9, 'Healthy', ha='center', fontsize=10)
    ax.text(2.25, 5.4, '(No depression, no CI)', ha='center', fontsize=8)
    ax.text(2.25, 4.9, '→ Prediction', ha='center', fontsize=9)
    ax.text(2.25, 4.5, '→ Causal (exercise)', ha='center', fontsize=9)

    # 轴线 B
    ax.add_patch(mpatches.FancyBboxPatch((5.25, 4.5), 3.5, 2.2, boxstyle="round,pad=0.05",
                                         facecolor=c_dep, edgecolor='#8b3a3a', linewidth=1.2, alpha=0.85))
    ax.text(7, 6.3, 'Axis B', ha='center', fontsize=11, fontweight='bold')
    ax.text(7, 5.9, 'Depression only', ha='center', fontsize=10)
    ax.text(7, 5.4, '(CES-D-10 ≥10, CI intact)', ha='center', fontsize=8)
    ax.text(7, 4.9, '→ Prediction', ha='center', fontsize=9)
    ax.text(7, 4.5, '→ Causal + Subgroup', ha='center', fontsize=9)

    # 轴线 C
    ax.add_patch(mpatches.FancyBboxPatch((10, 4.5), 3.5, 2.2, boxstyle="round,pad=0.05",
                                         facecolor=c_cog, edgecolor='#3a7a8b', linewidth=1.2, alpha=0.85))
    ax.text(11.75, 6.3, 'Axis C', ha='center', fontsize=11, fontweight='bold')
    ax.text(11.75, 5.9, 'Cognition impaired', ha='center', fontsize=10)
    ax.text(11.75, 5.4, '(CI only, no depression)', ha='center', fontsize=8)
    ax.text(11.75, 4.9, '→ Prediction', ha='center', fontsize=9)
    ax.text(11.75, 4.5, '→ Causal + Subgroup', ha='center', fontsize=9)

    # 3. 箭头：基线 → 三轴
    for x in [2.25, 7, 11.75]:
        ax.annotate('', xy=(x, 6.5), xytext=(x, 7.0), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    # 4. 共同结局框
    ax.add_patch(mpatches.FancyBboxPatch((4, 1.5), 6, 1.5, boxstyle="round,pad=0.05",
                                         facecolor=c_outcome, edgecolor='#8b6914', linewidth=1.2, alpha=0.9))
    ax.text(7, 2.25, 'Outcome: Incident depression–cognition comorbidity', ha='center', fontsize=11, fontweight='bold')

    # 5. 箭头：三轴 → 结局
    for x in [2.25, 7, 11.75]:
        ax.annotate('', xy=(7, 2.9), xytext=(x, 4.3),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

    # 6. 干预变量标注
    ax.text(7, 0.8, 'Treatment: Exercise (binary)', ha='center', fontsize=10, style='italic')
    ax.text(7, 0.3, 'Confounders: age, sex, education, chronic disease, lifestyle, ...', ha='center', fontsize=8, color='#666')

    plt.title('Conceptual framework: Three-axis design for depression–cognition comorbidity', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Fig 2 概念框架图已保存: {output_path}")
    return output_path

if __name__ == '__main__':
    draw_conceptual_framework()
