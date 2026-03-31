# -*- coding: utf-8 -*-
"""
Fig 3: ROC 曲线 A/B/C 叠加图
读取各轴线保存的 roc_data.json，绘制三条 ROC 曲线叠加。
输出：LIU_JUE_STRATEGIC_SUMMARY/fig3_roc_combined.png
"""
import os
import json
import numpy as np
from config import *  # 加载字体配置、OUTPUT_ROOT、COHORT_*_DIR、COHORT_STEP_DIRS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def draw_roc_combined(output_path=None):
    output_dir = OUTPUT_ROOT
    if output_path is None:
        output_path = os.path.join(output_dir, 'fig3_roc_combined.png')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # 与 config 三队列目录一致，便于修改 OUTPUT_ROOT/COHORT_*_DIR 后仍能正确读取
    pred_subdir = COHORT_STEP_DIRS['prediction']
    paths = [
        (os.path.join(COHORT_A_DIR, pred_subdir, 'roc_data.json'), 'Cohort A (Healthy)', '#5cb85c'),
        (os.path.join(COHORT_B_DIR, pred_subdir, 'roc_data.json'), 'Cohort B (Depression)', '#d9534f'),
        (os.path.join(COHORT_C_DIR, pred_subdir, 'roc_data.json'), 'Cohort C (Cognition)', '#5bc0de'),
    ]
    fig, ax = plt.subplots(figsize=(8, 7))
    for path, label, color in paths:
        if not os.path.exists(path):
            print(f"未找到 {path}，跳过。请先运行主流程生成 roc_data.json。")
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            y_true = np.array(d['y_true'])
            y_prob = np.array(d['y_prob'])
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')
        except Exception as e:
            print(f"读取 {path} 失败: {e}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC curves: Model performance by axis (A/B/C)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Fig 3 ROC 叠加图已保存: {output_path}")
    return output_path

if __name__ == '__main__':
    draw_roc_combined()
