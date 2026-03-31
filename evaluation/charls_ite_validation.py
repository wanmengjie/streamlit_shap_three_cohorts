# -*- coding: utf-8 -*-
"""
个体化治疗效应（ITE）分层验证（审稿意见 P2）
按 ITE 高低分组，验证高 ITE 组接受干预后共病风险是否显著低于低 ITE 组。
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from config import BBOX_INCHES

logger = logging.getLogger(__name__)


def run_ite_stratified_validation(df, causal_col, treatment_col, output_dir, target_col='is_comorbidity_next'):
    """
    ITE 分层验证：
    1. 按 ITE 中位数分为高/低两组
    2. 在各组内比较：接受干预 vs 未接受干预 的结局发生率
    3. 高 ITE 组中，接受干预者应有更低共病率（若 ITE 有效）
    """
    os.makedirs(output_dir, exist_ok=True)
    if causal_col not in df.columns or treatment_col not in df.columns:
        logger.warning("缺少 causal_impact 或 treatment 列，跳过 ITE 验证")
        return None

    ite = df[causal_col].dropna()
    if len(ite) < 50:
        return None

    median_ite = ite.median()
    df = df.copy()
    df['_ite_high'] = df[causal_col] >= median_ite
    T = df[treatment_col].fillna(0).astype(int)
    Y = df[target_col].astype(float)

    results = []
    for ite_label, mask_ite in [('Low ITE', ~df['_ite_high']), ('High ITE', df['_ite_high'])]:
        sub = df[mask_ite]
        t_sub = T[mask_ite]
        y_sub = Y[mask_ite]
        treated = (t_sub == 1)
        control = (t_sub == 0)
        if treated.sum() < 10 or control.sum() < 10:
            continue
        rate_t = y_sub[treated].mean()
        rate_c = y_sub[control].mean()
        diff = rate_t - rate_c
        n_t, n_c = treated.sum(), control.sum()
        se = np.sqrt(y_sub[treated].var() / n_t + y_sub[control].var() / n_c + 1e-9)
        z = diff / (se + 1e-9)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        results.append({
            'ITE_Group': ite_label,
            'Treated_Rate': rate_t, 'Control_Rate': rate_c,
            'Risk_Diff': diff, 'p_value': p, 'n_treated': n_t, 'n_control': n_c
        })

    if not results:
        return None

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, 'ite_stratified_validation.csv'), index=False, encoding='utf-8-sig')

    if len(res_df) >= 2:
        high_row = res_df[res_df['ITE_Group'] == 'High ITE'].iloc[0]
        low_row = res_df[res_df['ITE_Group'] == 'Low ITE'].iloc[0]
        cate_diff = high_row['Risk_Diff'] - low_row['Risk_Diff']
        with open(os.path.join(output_dir, 'ite_validation_summary.txt'), 'w', encoding='utf-8') as f:
            f.write(f"ITE Stratified Validation\n")
            f.write(f"High ITE: Treated rate={high_row['Treated_Rate']:.4f}, Control={high_row['Control_Rate']:.4f}, RD={high_row['Risk_Diff']:.4f}, p={high_row['p_value']:.4f}\n")
            f.write(f"Low ITE: Treated rate={low_row['Treated_Rate']:.4f}, Control={low_row['Control_Rate']:.4f}, RD={low_row['Risk_Diff']:.4f}, p={low_row['p_value']:.4f}\n")
            f.write(f"CATE difference (High-Low): {cate_diff:.4f}\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(res_df))[::-1]
    rates_t = res_df['Treated_Rate'].values
    rates_c = res_df['Control_Rate'].values
    ax.barh(y_pos - 0.2, rates_t, height=0.35, label='Treated', color='steelblue', alpha=0.8)
    ax.barh(y_pos + 0.2, rates_c, height=0.35, label='Control', color='coral', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(res_df['ITE_Group'].values)
    ax.set_xlabel('Comorbidity Rate')
    ax.set_title('ITE Stratified: Treated vs Control Outcome')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_ite_stratified.png'), dpi=200, bbox_inches=BBOX_INCHES)
    plt.close()

    return res_df


def run_ite_validation_for_cohorts(output_root=None):
    """对三队列（Cohort A/B/C）、可靠干预运行 ITE 验证。output_root 默认 '.' 与主流程一致。"""
    from config import COHORT_A_DIR, COHORT_B_DIR, COHORT_C_DIR, OUTPUT_ROOT
    if output_root is None:
        output_root = '.'  # 主流程将队列输出写到 ./Cohort_*，与 OUTPUT_ROOT 分离
    cohort_paths = {
        'A': os.path.join(output_root, COHORT_A_DIR),
        'B': os.path.join(output_root, COHORT_B_DIR),
        'C': os.path.join(output_root, COHORT_C_DIR),
    }
    cohort_rel_dirs = {'A': COHORT_A_DIR, 'B': COHORT_B_DIR, 'C': COHORT_C_DIR}
    for cohort_id, base in cohort_paths.items():
        causal_dir = os.path.join(base, '03_causal')
        if not os.path.exists(causal_dir) and output_root != OUTPUT_ROOT:
            causal_dir_alt = os.path.join(OUTPUT_ROOT, cohort_rel_dirs[cohort_id], '03_causal')
            if os.path.exists(causal_dir_alt):
                causal_dir = causal_dir_alt
                base = os.path.dirname(causal_dir)  # 输出与输入同目录
        if not os.path.exists(causal_dir):
            continue
        for f in os.listdir(causal_dir):
            if f.startswith('CAUSAL_ANALYSIS_') and f.endswith('.csv'):
                treatment = f.replace('CAUSAL_ANALYSIS_', '').replace('.csv', '')
                causal_col = f'causal_impact_{treatment}'
                df = pd.read_csv(os.path.join(causal_dir, f))
                if causal_col not in df.columns:
                    continue
                out_dir = os.path.join(base, '03_causal', f'ite_validation_{treatment}')
                try:
                    run_ite_stratified_validation(df, causal_col, treatment, out_dir)
                except Exception as e:
                    logger.warning("ITE 验证 Cohort %s %s 失败: %s", cohort_id, treatment, e)


# 旧函数名兼容
run_ite_validation_for_axes = run_ite_validation_for_cohorts


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_ite_validation_for_axes()
