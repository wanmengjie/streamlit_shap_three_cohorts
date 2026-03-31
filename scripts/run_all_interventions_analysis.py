# -*- coding: utf-8 -*-
"""
扩展干预因素分析：除睡眠外，对运动、吸烟、饮酒、社会隔离、BMI、慢性病负担等
分别估计 ATE，并生成森林图与汇总表。
输出：LIU_JUE_STRATEGIC_SUMMARY/all_interventions_summary.csv
      LIU_JUE_STRATEGIC_SUMMARY/fig_all_interventions_forest.png

独立运行 __main__ 时通过 utils.charls_script_data_loader 与主流程一致的数据源（config.USE_IMPUTED_DATA）。
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import numpy as np
from config import *  # 加载字体配置，解决方框显示
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from utils.charls_script_data_loader import load_df_for_analysis
from causal.charls_recalculate_causal_impact import get_estimate_causal_impact, cleanup_temp_cat_dirs
from evaluation.charls_sensitivity_analysis import run_sensitivity_analysis

# 干预因素定义：(列名, 构造方式, 英文标签，用于图表)
# 构造方式: None=直接使用原列(需0/1); 'bmi'=从bmi构造bmi_normal; 'chronic'=从chronic_burden构造chronic_low
# sleep_adequate 由 run_xlearner_all_interventions 等处理，本脚本仅含 5 类干预
INTERVENTIONS = [
    ('exercise', None, 'Exercise'),
    ('drinkev', None, 'Drinking'),          # T=1 drinking
    ('is_socially_isolated', None, 'Social isolation'),  # T=1 isolated, ATE>0 = increased risk
    ('bmi_normal', 'bmi', 'Normal BMI (18.5-24)'),
    ('chronic_low', 'chronic', 'Low chronic disease burden (≤1)'),
]


def prepare_interventions(df):
    """构造 bmi_normal, chronic_low 等衍生干预变量（sleep_adequate 由 prepare_exposures 在主流程构造）"""
    if 'bmi' in df.columns:
        bmi_val = df['bmi'].clip(10, 50)
        df['bmi_normal'] = ((bmi_val >= 18.5) & (bmi_val <= 24)).astype(int)
        df.loc[df['bmi'].isna(), 'bmi_normal'] = np.nan
        logger.info(f"bmi_normal: {df['bmi_normal'].notna().sum()} non-missing, mean={df['bmi_normal'].mean():.2f}")
    if 'chronic_burden' in df.columns:
        df['chronic_low'] = (df['chronic_burden'].fillna(0) <= 1).astype(int)
        logger.info(f"chronic_low: {df['chronic_low'].notna().sum()} non-missing, mean={df['chronic_low'].mean():.2f}")
    return df


def run_all_interventions(df_clean, output_root='LIU_JUE_STRATEGIC_SUMMARY'):
    """对所有干预因素分别估计 ATE，并生成汇总与森林图"""
    df_clean = prepare_interventions(df_clean.copy())
    df_a = df_clean[df_clean['baseline_group'] == 0].copy()
    df_b = df_clean[df_clean['baseline_group'] == 1].copy()
    df_c = df_clean[df_clean['baseline_group'] == 2].copy()

    results = []
    causal_dir = os.path.join(output_root, 'all_interventions_causal')
    os.makedirs(causal_dir, exist_ok=True)

    for col, _, label in INTERVENTIONS:
        if col not in df_clean.columns:
            logger.warning(f"跳过 {col}：列不存在")
            continue
        for cohort_name, df_sub in [('Cohort_A', df_a), ('Cohort_B', df_b), ('Cohort_C', df_c)]:
            if len(df_sub) < 50:
                continue
            out_dir = os.path.join(causal_dir, cohort_name, col)
            os.makedirs(out_dir, exist_ok=True)
            try:
                estimate_causal = get_estimate_causal_impact()
                res_df, (ate, lb, ub) = estimate_causal(df_sub, treatment_col=col, output_dir=out_dir)
                # 因果引擎失败/跳过：返回 (None, (nan,nan,nan)) — 须以 res_df is None 判定，勿当真实 ATE
                if res_df is None:
                    ate, lb, ub = np.nan, np.nan, np.nan
                else:
                    try:
                        run_sensitivity_analysis(
                            res_df,
                            output_dir=os.path.join(out_dir, '07_sensitivity'),
                            treatment_col=col,
                        )
                    except Exception as es:
                        logger.warning(f"  {cohort_name} {col} 因果敏感性分析跳过: {es}")
                    for v in (ate, lb, ub):
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            ate, lb, ub = np.nan, np.nan, np.nan
                            break
            except Exception as e:
                logger.warning(f"{cohort_name} {col} 失败: {e}")
                ate, lb, ub = np.nan, np.nan, np.nan
            sig = 1 if (lb is not None and ub is not None and not np.isnan(lb) and not np.isnan(ub) and (lb > 0 or ub < 0)) else 0
            reliable = 1 if (ate is not None and not np.isnan(ate) and -1 <= ate <= 1) else 0
            results.append({
                'exposure': col,
                'label': label,
                'cohort': cohort_name,
                'ate': ate,
                'ate_lb': lb,
                'ate_ub': ub,
                'significant': sig,
                'reliable': reliable,
                'n': len(df_sub),
            })
            ate_str = f"{ate:.4f}" if ate is not None and not np.isnan(ate) else "NaN"
            ci_str = f"({lb:.4f}, {ub:.4f})" if lb is not None and ub is not None and not np.isnan(lb) and not np.isnan(ub) else "(N/A)"
            logger.info(f"  {cohort_name} {label}: ATE={ate_str} 95% CI={ci_str} {'*significant*' if sig else ''}")

    res_df = pd.DataFrame(results)
    # 多重检验校正（5干预×3队列=15次比较）
    try:
        from utils.multiplicity_correction import add_multiplicity_columns
        res_df = add_multiplicity_columns(res_df)
    except Exception as ex:
        logger.warning(f"多重检验校正跳过: {ex}")
    out_csv = os.path.join(output_root, 'all_interventions_summary.csv')
    # Round results to 4 decimal places
    res_df['ate'] = res_df['ate'].round(4)
    res_df['ate_lb'] = res_df['ate_lb'].round(4)
    res_df['ate_ub'] = res_df['ate_ub'].round(4)

    res_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    try:
        res_df.to_excel(os.path.join(output_root, 'all_interventions_summary.xlsx'), index=False, engine='openpyxl')
    except Exception as e:
        logger.warning(f"Excel 导出跳过 (需 openpyxl): {e}")
    logger.info(f"汇总已保存: {out_csv}")

    # 绘制森林图（仅可靠估计且 ATE 有效）
    plot_df = res_df[(res_df['reliable'] == 1) & res_df['ate'].notna()].copy()
    if len(plot_df) > 0:
        draw_forest_plot(plot_df, output_root)
    return res_df


def draw_forest_plot(res_df, output_dir):
    """绘制干预效应森林图"""
    # 按队列与干预排序
    cohort_order = {'Cohort_A': 0, 'Cohort_B': 1, 'Cohort_C': 2}
    res_df['_cohort_order'] = res_df['cohort'].map(cohort_order).fillna(99)
    res_df = res_df.sort_values(['_cohort_order', 'exposure']).drop(columns=['_cohort_order'])
    res_df['display_label'] = res_df['label'] + ' @ ' + res_df['cohort'].str.replace('Cohort_', '')

    n = len(res_df)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.35)))
    y_pos = np.arange(n)[::-1]
    colors = {'Cohort_A': '#5cb85c', 'Cohort_B': '#d9534f', 'Cohort_C': '#5bc0de'}
    bar_colors = [colors.get(r['cohort'], '#888') for _, r in res_df.iterrows()]

    ates = res_df['ate'].values
    lbs = res_df['ate_lb'].values
    ubs = res_df['ate_ub'].values
    err_lo = ates - lbs
    err_hi = ubs - ates

    ax.barh(y_pos, ates, color=bar_colors, alpha=0.8, height=0.6)
    ax.errorbar(ates, y_pos, xerr=[err_lo, err_hi], fmt='none', color='black', capsize=2)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(res_df['display_label'].values, fontsize=10)
    ax.set_xlabel('ATE (Risk Difference)', fontsize=11)
    ax.set_title('Causal Effect of Interventions on Incident Comorbidity (95% CI)', fontsize=12)
    x_min = min(-0.5, np.nanmin(lbs) - 0.05) if np.any(np.isfinite(lbs)) else -0.5
    x_max = max(0.5, np.nanmax(ubs) + 0.05) if np.any(np.isfinite(ubs)) else 0.5
    ax.set_xlim(max(-0.5, x_min), min(0.5, x_max))
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color='#5cb85c', label='Cohort A (Healthy)'),
        mpatches.Patch(color='#d9534f', label='Cohort B (Depression)'),
        mpatches.Patch(color='#5bc0de', label='Cohort C (Cognition)'),
    ], loc='lower right', fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig_all_interventions_forest.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"森林图已保存: {out_path}")


def main():
    cleanup_temp_cat_dirs()
    df = load_df_for_analysis()
    if df is None:
        logger.error("数据加载失败（检查 config.USE_IMPUTED_DATA / CHARLS.csv）")
        return
    res_df = run_all_interventions(df, output_root='LIU_JUE_STRATEGIC_SUMMARY')

    # 打印汇总
    reliable = res_df[res_df['reliable'] == 1]
    unreliable = res_df[res_df['reliable'] == 0]
    if len(unreliable) > 0:
        logger.info(f"\nUnreliable estimates: {list(unreliable['exposure'].unique())}")
    sig_df = reliable[reliable['significant'] == 1]
    if len(sig_df) > 0:
        logger.info(f"\nSignificant results ({len(sig_df)}):")
        for _, r in sig_df.iterrows():
            logger.info(f"  {r['label']} @ {r['cohort']}: ATE={r['ate']:.4f} (95% CI: {r['ate_lb']:.4f}, {r['ate_ub']:.4f})")
    else:
        logger.info("\nNo significant results (all 95% CIs include 0)")


if __name__ == '__main__':
    main()
