# -*- coding: utf-8 -*-
"""
X-Learner 全干预因果分析：运动、饮酒、社会隔离、BMI正常、慢性病负担低、睡眠充足、吸烟包年低。
对 7 类干预在三队列（A/B/C）上运行 XLearner，同步执行假设检验（重叠、SMD、E-value、PSM/PSW）。
输出：LIU_JUE_STRATEGIC_SUMMARY/xlearner_all_interventions/
运行: python scripts/run_xlearner_all_interventions.py

独立运行时数据源与主流程一致：utils.charls_script_data_loader.load_df_for_analysis()。
"""
import os
import sys
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.charls_script_data_loader import load_df_for_analysis
from causal.charls_recalculate_causal_impact import estimate_causal_impact_xlearner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = 'LIU_JUE_STRATEGIC_SUMMARY/xlearner_all_interventions'

# 干预定义：(列名, 构造方式, 英文标签)
# 构造方式: None=直接使用; 'bmi'=构造bmi_normal; 'chronic'=构造chronic_low;
# 'sleep_adequate'=从sleep构造; puff_low 已移除，改用 smokev（当前吸烟）
INTERVENTIONS = [
    ('exercise', None, 'Exercise'),
    ('drinkev', None, 'Drinking'),
    ('is_socially_isolated', None, 'Social isolation'),
    ('bmi_normal', 'bmi', 'Normal BMI (18.5-24)'),
    ('chronic_low', 'chronic', 'Low chronic disease burden (≤1)'),
    ('sleep_adequate', 'sleep_adequate', 'Adequate sleep (≥6h)'),
    ('smokev', None, 'Current smoking'),
]


def _merge_existing_results(output_dir):
    """从已有 causal_methods_comparison_*.csv 合并 X-Learner+PSM+PSW 汇总，不重跑分析。"""
    results = []
    label_map = {col: label for col, _, label in INTERVENTIONS}
    for cohort_name in ['Cohort_A', 'Cohort_B', 'Cohort_C']:
        cohort_branch = os.path.join(output_dir, cohort_name)
        if not os.path.isdir(cohort_branch):
            continue
        for col in os.listdir(cohort_branch):
            out_sub = os.path.join(cohort_branch, col)
            if not os.path.isdir(out_sub):
                continue
            comp_csv = os.path.join(out_sub, f'causal_methods_comparison_{col}.csv')
            if not os.path.exists(comp_csv):
                continue
            label = label_map.get(col, col)
            causal_csv = os.path.join(out_sub, f'CAUSAL_ANALYSIS_{col}.csv')
            n = len(pd.read_csv(causal_csv)) if os.path.exists(causal_csv) else 0
            try:
                comp_df = pd.read_csv(comp_csv, encoding='utf-8-sig')
                for _, r in comp_df.iterrows():
                    method = r.get('Method', r.get('method', ''))
                    ate_val = r.get('ATE', r.get('ate', np.nan))
                    lb_val = r.get('ATE_lb', r.get('ate_lb', np.nan))
                    ub_val = r.get('ATE_ub', r.get('ate_ub', np.nan))
                    sig = 1 if (pd.notna(lb_val) and pd.notna(ub_val) and (lb_val > 0 or ub_val < 0)) else 0
                    rel = 1 if (pd.notna(ate_val) and -1 <= ate_val <= 1) else 0
                    results.append({
                        'exposure': col, 'label': label, 'cohort': cohort_name, 'method': method,
                        'ate': round(float(ate_val), 4) if pd.notna(ate_val) else np.nan,
                        'ate_lb': round(float(lb_val), 4) if pd.notna(lb_val) else np.nan,
                        'ate_ub': round(float(ub_val), 4) if pd.notna(ub_val) else np.nan,
                        'significant': sig, 'reliable': rel, 'n': n,
                    })
            except Exception as e:
                logger.warning(f"合并 {cohort_name}/{col} 失败: {e}")
    if not results:
        logger.warning("未找到任何 causal_methods_comparison_*.csv，请先运行完整分析")
        return pd.DataFrame()
    summary = pd.DataFrame(results)
    out_csv = os.path.join(output_dir, 'xlearner_all_interventions_summary.csv')
    summary.to_csv(out_csv, index=False, encoding='utf-8-sig')
    logger.info(f"X-Learner + PSM + PSW 合并汇总已保存: {out_csv}")
    # 宽格式
    wide_rows = []
    for (exposure, cohort), grp in summary.groupby(['exposure', 'cohort']):
        row = {'exposure': exposure, 'cohort': cohort, 'n': grp['n'].iloc[0], 'label': grp['label'].iloc[0]}
        for m in ['XLearner', 'PSM', 'PSW']:
            m_df = grp[grp['method'] == m]
            if len(m_df) > 0:
                r = m_df.iloc[0]
                row[f'{m}_ATE'] = r['ate']
                lb, ub = r['ate_lb'], r['ate_ub']
                row[f'{m}_CI'] = f"({lb:.4f}, {ub:.4f})" if pd.notna(lb) and pd.notna(ub) else ''
            else:
                row[f'{m}_ATE'] = np.nan
                row[f'{m}_CI'] = ''
        wide_rows.append(row)
    wide_df = pd.DataFrame(wide_rows)
    wide_df.to_csv(os.path.join(output_dir, 'xlearner_psm_psw_wide.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"宽格式汇总已保存: xlearner_psm_psw_wide.csv")
    return summary


def prepare_interventions(df):
    """构造 bmi_normal, chronic_low, sleep_adequate；吸烟已改用 smokev，puff_low 已移除"""
    df = df.copy()
    if 'bmi' in df.columns:
        bmi_val = df['bmi'].clip(10, 50)
        df['bmi_normal'] = ((bmi_val >= 18.5) & (bmi_val <= 24)).astype(int)
        df.loc[df['bmi'].isna(), 'bmi_normal'] = np.nan
    if 'chronic_burden' in df.columns:
        df['chronic_low'] = (df['chronic_burden'].fillna(0) <= 1).astype(int)
    if 'sleep' in df.columns:
        df['sleep_adequate'] = (df['sleep'].clip(0, 24) >= 6).astype(int)
        df.loc[df['sleep'].isna(), 'sleep_adequate'] = np.nan
    return df


def run_xlearner_all_interventions(output_dir=OUTPUT_DIR, merge_only=False, df_clean=None):
    """对 7 类干预在三队列上运行 XLearner + 假设检验。
    merge_only=True 时仅从已有 causal_methods_comparison_*.csv 合并 X-Learner+PSM+PSW，不重跑分析。
    df_clean: 可选，主流程传入时复用同一数据源，保证与主流程一致。"""
    os.makedirs(output_dir, exist_ok=True)

    if merge_only:
        return _merge_existing_results(output_dir)

    # 1. 加载数据（主流程传入 df_clean 时复用；否则与 run_all_charls_analyses 同源）
    if df_clean is None:
        df_clean = load_df_for_analysis()
        if df_clean is None:
            logger.error("数据加载失败（检查 config.USE_IMPUTED_DATA / CHARLS.csv）")
            return pd.DataFrame()
    else:
        df_clean = df_clean.copy()

    df_clean = prepare_interventions(df_clean)

    # 2. 7 干预 × 3 队列
    results = []
    for col, _, label in INTERVENTIONS:
        if col not in df_clean.columns:
            logger.warning(f"跳过 {col}：列不存在")
            continue
        for cohort_name, df_sub in [
            ('Cohort_A', df_clean[df_clean['baseline_group'] == 0]),
            ('Cohort_B', df_clean[df_clean['baseline_group'] == 1]),
            ('Cohort_C', df_clean[df_clean['baseline_group'] == 2]),
        ]:
            df_sub = df_sub.dropna(subset=[col]).copy()
            if len(df_sub) < 50 or df_sub[col].nunique() < 2:
                logger.warning(f"跳过 {cohort_name} {col}：样本不足或干预无变异")
                continue
            out_sub = os.path.join(output_dir, cohort_name, col)
            logger.info(f">>> {cohort_name} {label} (n={len(df_sub)})")
            try:
                res_df, (ate, ate_lb, ate_ub) = estimate_causal_impact_xlearner(
                    df_sub, treatment_col=col, output_dir=out_sub
                )
                if res_df is None:
                    ate, ate_lb, ate_ub = np.nan, np.nan, np.nan
                else:
                    for v in (ate, ate_lb, ate_ub):
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            ate, ate_lb, ate_ub = np.nan, np.nan, np.nan
                            break
                sig = 1 if (pd.notna(ate_lb) and pd.notna(ate_ub) and (ate_lb > 0 or ate_ub < 0)) else 0
                reliable = 1 if (pd.notna(ate) and -1 <= ate <= 1) else 0
                results.append({
                    'exposure': col,
                    'label': label,
                    'cohort': cohort_name,
                    'method': 'XLearner',
                    'ate': round(ate, 4) if pd.notna(ate) else np.nan,
                    'ate_lb': round(ate_lb, 4) if pd.notna(ate_lb) else np.nan,
                    'ate_ub': round(ate_ub, 4) if pd.notna(ate_ub) else np.nan,
                    'significant': sig,
                    'reliable': reliable,
                    'n': len(df_sub),
                })
                sig_mark = "*" if sig else ""
                if pd.notna(ate):
                    logger.info(f"  {cohort_name} {label}: ATE={ate:.4f} (95% CI: {ate_lb:.4f}, {ate_ub:.4f}) {sig_mark}")
                else:
                    logger.info(f"  {cohort_name} {label}: XLearner 跳过/失败 (NaN)")

                # 合并 PSM/PSW 到汇总（与 X-Learner 同表）
                comp_csv = os.path.join(out_sub, f'causal_methods_comparison_{col}.csv')
                if os.path.exists(comp_csv):
                    try:
                        comp_df = pd.read_csv(comp_csv, encoding='utf-8-sig')
                        for _, r in comp_df.iterrows():
                            if r['Method'] in ('PSM', 'PSW'):
                                ate_val = r.get('ATE', np.nan)
                                lb_val = r.get('ATE_lb', np.nan)
                                ub_val = r.get('ATE_ub', np.nan)
                                sig_ps = 1 if (pd.notna(lb_val) and pd.notna(ub_val) and (lb_val > 0 or ub_val < 0)) else 0
                                rel_ps = 1 if (pd.notna(ate_val) and -1 <= ate_val <= 1) else 0
                                results.append({
                                    'exposure': col, 'label': label, 'cohort': cohort_name,
                                    'method': r['Method'],
                                    'ate': round(float(ate_val), 4) if pd.notna(ate_val) else np.nan,
                                    'ate_lb': round(float(lb_val), 4) if pd.notna(lb_val) else np.nan,
                                    'ate_ub': round(float(ub_val), 4) if pd.notna(ub_val) else np.nan,
                                    'significant': sig_ps, 'reliable': rel_ps, 'n': len(df_sub),
                                })
                    except Exception as ex:
                        logger.debug(f"读取 PSM/PSW 汇总失败: {ex}")
            except Exception as e:
                logger.warning(f"  {cohort_name} {col} 失败: {e}")

    if results:
        summary = pd.DataFrame(results)
        out_csv = os.path.join(output_dir, 'xlearner_all_interventions_summary.csv')
        summary.to_csv(out_csv, index=False, encoding='utf-8-sig')
        logger.info(f"X-Learner + PSM + PSW 全干预汇总已保存: {out_csv}")

        # 宽格式：每行一个干预-队列，便于论文制表
        wide_rows = []
        for (exposure, cohort), grp in summary.groupby(['exposure', 'cohort']):
            row = {'exposure': exposure, 'cohort': cohort, 'n': grp['n'].iloc[0]}
            label = grp['label'].iloc[0]
            row['label'] = label
            for m in ['XLearner', 'PSM', 'PSW']:
                m_df = grp[grp['method'] == m]
                if len(m_df) > 0:
                    r = m_df.iloc[0]
                    row[f'{m}_ATE'] = r['ate']
                    lb, ub = r['ate_lb'], r['ate_ub']
                    row[f'{m}_CI'] = f"({lb:.4f}, {ub:.4f})" if pd.notna(lb) and pd.notna(ub) else ''
                else:
                    row[f'{m}_ATE'] = np.nan
                    row[f'{m}_CI'] = ''
            wide_rows.append(row)
        wide_df = pd.DataFrame(wide_rows)
        wide_csv = os.path.join(output_dir, 'xlearner_psm_psw_wide.csv')
        wide_df.to_csv(wide_csv, index=False, encoding='utf-8-sig')
        logger.info(f"宽格式汇总（论文制表用）已保存: {wide_csv}")

        # 多重检验校正
        try:
            from utils.multiplicity_correction import add_multiplicity_columns
            summary = add_multiplicity_columns(summary, ate_col='ate', lb_col='ate_lb', ub_col='ate_ub')
            summary.to_csv(out_csv, index=False, encoding='utf-8-sig')
        except Exception as ex:
            logger.warning(f"多重检验校正跳过: {ex}")

        # 森林图
        try:
            _draw_forest_plot(summary, output_dir)
        except Exception as ex:
            logger.warning(f"森林图跳过: {ex}")

        sig_df = summary[summary['significant'] == 1]
        if len(sig_df) > 0:
            logger.info(f"\n显著结果 ({len(sig_df)}):")
            for _, r in sig_df.iterrows():
                logger.info(f"  {r['label']} @ {r['cohort']}: ATE={r['ate']:.4f} (95% CI: {r['ate_lb']:.4f}, {r['ate_ub']:.4f})")
        else:
            logger.info("\n无显著结果（所有 95% CI 均含 0）")
        return summary
    return pd.DataFrame()


def _draw_forest_plot(res_df, output_dir):
    """绘制干预效应森林图"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plot_df = res_df[(res_df['reliable'] == 1) & res_df['ate'].notna()].copy()
    if len(plot_df) == 0:
        return
    cohort_order = {'Cohort_A': 0, 'Cohort_B': 1, 'Cohort_C': 2}
    plot_df['_cohort_order'] = plot_df['cohort'].map(cohort_order).fillna(99)
    plot_df = plot_df.sort_values(['_cohort_order', 'exposure']).drop(columns=['_cohort_order'])
    # 多方法时显示方法名，便于与 PSM/PSW 对比
    plot_df['display_label'] = plot_df.apply(
        lambda r: f"{r['label']} @ {r['cohort'].replace('Cohort_', '')} ({r['method']})", axis=1
    )

    n = len(plot_df)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.35)))
    y_pos = np.arange(n)[::-1]
    colors = {'Cohort_A': '#5cb85c', 'Cohort_B': '#d9534f', 'Cohort_C': '#5bc0de'}
    bar_colors = [colors.get(r['cohort'], '#888') for _, r in plot_df.iterrows()]

    ates = plot_df['ate'].values
    lbs = plot_df['ate_lb'].values
    ubs = plot_df['ate_ub'].values
    err_lo = ates - lbs
    err_hi = ubs - ates

    ax.barh(y_pos, ates, color=bar_colors, alpha=0.8, height=0.6)
    ax.errorbar(ates, y_pos, xerr=[err_lo, err_hi], fmt='none', color='black', capsize=2)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['display_label'].values, fontsize=10)
    ax.set_xlabel('ATE (Risk Difference)', fontsize=11)
    ax.set_title('X-Learner: Causal Effect of Interventions on Incident Comorbidity (95% CI)', fontsize=12)
    x_min = min(-0.5, np.nanmin(lbs) - 0.05) if np.any(np.isfinite(lbs)) else -0.5
    x_max = max(0.5, np.nanmax(ubs) + 0.05) if np.any(np.isfinite(ubs)) else 0.5
    ax.set_xlim(max(-0.5, x_min), min(0.5, x_max))
    ax.legend(handles=[
        mpatches.Patch(color='#5cb85c', label='Cohort A (Healthy)'),
        mpatches.Patch(color='#d9534f', label='Cohort B (Depression)'),
        mpatches.Patch(color='#5bc0de', label='Cohort C (Cognition)'),
    ], loc='lower right', fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig_xlearner_all_interventions_forest.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"森林图已保存: {out_path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='X-Learner 全干预分析，可选仅合并已有 PSM/PSW 结果')
    p.add_argument('--merge-only', action='store_true', help='仅从已有 causal_methods_comparison_*.csv 合并，不重跑分析')
    args = p.parse_args()
    run_xlearner_all_interventions(merge_only=args.merge_only)
