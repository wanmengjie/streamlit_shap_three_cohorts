# -*- coding: utf-8 -*-
"""
敏感性分析：截断值变化 + 干预完整病例（论文2.5节、附录S2）。
支持多干预因素（运动、充足睡眠、吸烟、饮酒、BMI正常等）。
当传入 df_base（插补后数据）时，全流程基于插补后数据：9种诊断阈值重定义队列后重训模型、重估ATE；
完整病例为在插补后数据上剔除当前干预缺失的样本再估计。
输出：LIU_JUE_STRATEGIC_SUMMARY/sensitivity_summary.csv 与 sensitivity_ate_comparison_*.png
"""
import os
import sys
import random
import logging
import pandas as pd
import numpy as np
from config import *  # 加载字体配置、INTERVENTION_LABELS_EN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from data.charls_complete_preprocessing import preprocess_charls_data, reapply_cohort_definition
from causal.charls_recalculate_causal_impact import get_estimate_causal_impact, cleanup_temp_cat_dirs
from sklearn.model_selection import GroupShuffleSplit

# 与 run_all_interventions_analysis 保持一致（已移除 smokev）；图表使用英文
INTERVENTIONS = [
    ('exercise', INTERVENTION_LABELS_EN.get('exercise', 'Exercise')),
    ('drinkev', INTERVENTION_LABELS_EN.get('drinkev', 'Drinking')),
    ('is_socially_isolated', INTERVENTION_LABELS_EN.get('is_socially_isolated', 'Social isolation')),
    ('bmi_normal', INTERVENTION_LABELS_EN.get('bmi_normal', 'Normal BMI')),
    ('chronic_low', INTERVENTION_LABELS_EN.get('chronic_low', 'Low chronic disease burden')),
]

def _get_train_subset(df, random_state=None):
    """
    审稿修正：与 compare_models 一致的 80/20 划分，仅返回训练集（80%）。
    用于敏感性分析时隔离测试集，避免数据泄露风险。
    """
    from config import RANDOM_SEED
    if random_state is None:
        random_state = RANDOM_SEED
    if df is None or len(df) < 50 or 'ID' not in df.columns:
        return df
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, _ = next(gss.split(df, df['is_comorbidity_next'] if 'is_comorbidity_next' in df.columns else df.index, groups=df['ID']))
    return df.iloc[train_idx].copy()

def prepare_interventions(df):
    """构造 bmi_normal, chronic_low 等衍生干预变量（sleep_adequate 由 prepare_exposures 在主流程构造）"""
    if 'bmi' in df.columns and 'bmi_normal' not in df.columns:
        bmi_val = df['bmi'].clip(10, 50)
        df['bmi_normal'] = ((bmi_val >= 18.5) & (bmi_val <= 24)).astype(int)
        df.loc[df['bmi'].isna(), 'bmi_normal'] = np.nan
    if 'chronic_burden' in df.columns and 'chronic_low' not in df.columns:
        df['chronic_low'] = (df['chronic_burden'].fillna(0) <= 1).astype(int)
    return df

def run_one_scenario(df_clean, scenario_name, cohort_id, cohort_label, treatment_col='exercise', treatment_label='Exercise', restrict_complete_case=False):
    """对某一 Cohort（A/B/C）在给定 df_clean 下估计 ATE（论文2.5节）；restrict_complete_case=True 时为附录S2 完整病例敏感性。"""
    if cohort_id == 'A':
        df_sub = df_clean[df_clean['baseline_group'] == 0].copy()
    elif cohort_id == 'B':
        df_sub = df_clean[df_clean['baseline_group'] == 1].copy()
    else:
        df_sub = df_clean[df_clean['baseline_group'] == 2].copy()
    if treatment_col not in df_sub.columns:
        return {'intervention': treatment_col, 'intervention_label': treatment_label, 'scenario': scenario_name, 'cohort_id': cohort_id, 'cohort_label': cohort_label, 'n': 0,
                'incidence': np.nan, 'ate': np.nan, 'ate_lb': np.nan, 'ate_ub': np.nan}
    if restrict_complete_case:
        df_sub = df_sub.dropna(subset=[treatment_col])
    n = len(df_sub)
    if n < 30:
        return {'intervention': treatment_col, 'intervention_label': treatment_label, 'scenario': scenario_name, 'cohort_id': cohort_id, 'cohort_label': cohort_label, 'n': n,
                'incidence': np.nan, 'ate': np.nan, 'ate_lb': np.nan, 'ate_ub': np.nan}
    inc = df_sub['is_comorbidity_next'].mean()
    out_dir = os.path.join('sensitivity_temp', scenario_name.replace(' ', '_'), treatment_col, f'Cohort_{cohort_id}')
    os.makedirs(out_dir, exist_ok=True)
    res_df, (ate, ate_lb, ate_ub) = get_estimate_causal_impact()(df_sub, treatment_col=treatment_col, output_dir=out_dir)
    if res_df is None:
        ate, ate_lb, ate_ub = np.nan, np.nan, np.nan
    else:
        for v in (ate, ate_lb, ate_ub):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                ate, ate_lb, ate_ub = np.nan, np.nan, np.nan
                break
    return {'intervention': treatment_col, 'intervention_label': treatment_label, 'scenario': scenario_name, 'cohort_id': cohort_id, 'cohort_label': cohort_label, 'n': n,
            'incidence': inc, 'ate': ate, 'ate_lb': ate_lb, 'ate_ub': ate_ub}

def run_sensitivity_scenarios_analysis(final_dir='LIU_JUE_STRATEGIC_SUMMARY', data_path=None, interventions=None, df_base=None, train_only=True):
    """
    截断值敏感性（抑郁 CES-D≥8/10/12，认知 Cog≤8/10/12，共9种组合，论文2.5/附录S2）+ 完整病例敏感性。
    df_base: 若提供（如主流程的插补后数据），则全部场景基于该数据：9种阈值通过 reapply_cohort_definition 重定义队列后重训、重估ATE；完整病例为在 df_base 上 dropna(干预) 后估计。保证插补后数据贯穿敏感性全流程，无断点。
    data_path: 当 df_base 为 None 时使用，从 CSV 预处理得到各阈值数据；None 时从 config.RAW_DATA_PATH 读取。
    train_only: 审稿修正，默认 True。当 df_base 由主流程传入时，主流程已先分割并传入训练集子集（先分割再传递）；当 df_base 为 None 时，本函数内部对加载的数据做 _get_train_subset 过滤。
    """
    if data_path is None:
        try:
            from config import RAW_DATA_PATH
            data_path = RAW_DATA_PATH
        except ImportError:
            data_path = 'CHARLS.csv'
    if df_base is None and not os.path.exists(data_path):
        logger.error(f"未找到 {data_path} 且未提供 df_base，跳过敏感性场景。")
        return None

    if interventions is None:
        interventions = INTERVENTIONS
    elif interventions and isinstance(interventions[0], str):
        interventions = [(c, c) for c in interventions]
    elif interventions and not isinstance(interventions[0], tuple):
        interventions = [(c, c) for c in interventions]

    from config import RANDOM_SEED, AGE_MIN
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.makedirs(final_dir, exist_ok=True)

    # 论文2.5/附录S2：9种诊断阈值组合
    cutpoint_scenarios = [
        (10, 10, 'Main (CES-D≥10, Cog≤10)'),
        (8,  10, 'CES-D≥8'),
        (12, 10, 'CES-D≥12'),
        (10, 8,  'Cog≤8'),
        (10, 12, 'Cog≤12'),
        (8,  8,  'CES-D≥8, Cog≤8'),
        (8,  12, 'CES-D≥8, Cog≤12'),
        (12, 8,  'CES-D≥12, Cog≤8'),
        (12, 12, 'CES-D≥12, Cog≤12'),
    ]
    rows = []

    use_imputed_base = df_base is not None and len(df_base) > 0
    if use_imputed_base:
        # 毕业论文修正-对应论文2.5节：敏感性分析基于插补后数据，每种阈值重定义队列后重训、重估ATE
        df_base = prepare_interventions(df_base.copy())

    for cesd_c, cog_c, label in cutpoint_scenarios:
        logger.info(f"敏感性场景: {label}" + (" (插补后数据)" if use_imputed_base else ""))
        if use_imputed_base:
            df_clean = reapply_cohort_definition(df_base, cesd_cutoff=cesd_c, cognition_cutoff=cog_c)
        else:
            df_clean = preprocess_charls_data(data_path, cesd_cutoff=cesd_c, cognition_cutoff=cog_c, age_min=AGE_MIN, write_output=False)
            if df_clean is not None:
                df_clean = prepare_interventions(df_clean)
        if df_clean is None or len(df_clean) < 30:
            continue
        # 当 df_base 由主流程传入时，主流程已传训练集，无需再过滤；当 df_base 为 None 时内部过滤
        if train_only and not use_imputed_base:
            df_clean = _get_train_subset(df_clean)
            if df_clean is None or len(df_clean) < 30:
                continue
            logger.info(f"  使用训练集子集 (n={len(df_clean)})，测试集已隔离")
        for treatment_col, treatment_label in interventions:
            if treatment_col not in df_clean.columns:
                logger.warning(f"  跳过 {treatment_col}：列不存在")
                continue
            for cohort_id, cohort_label in [('A', 'Healthy'), ('B', 'Depression'), ('C', 'Cognition')]:
                r = run_one_scenario(df_clean, label, cohort_id, cohort_label, treatment_col=treatment_col, treatment_label=treatment_label, restrict_complete_case=False)
                rows.append(r)

    # 完整病例敏感性（论文附录S2）：在相同数据基础上仅保留当前干预无缺失的样本
    if use_imputed_base:
        df_main = reapply_cohort_definition(df_base, cesd_cutoff=10, cognition_cutoff=10)
    else:
        df_main = preprocess_charls_data(data_path, cesd_cutoff=10, cognition_cutoff=10, age_min=AGE_MIN, write_output=False)
        if df_main is not None:
            df_main = prepare_interventions(df_main)
    if df_main is not None and len(df_main) >= 30:
        # df_base 由主流程传入时已是训练集，无需再过滤
        if train_only and not use_imputed_base:
            df_main = _get_train_subset(df_main)
            if df_main is not None and len(df_main) >= 30:
                logger.info(f"  完整病例敏感性使用训练集子集 (n={len(df_main)})")
        if df_main is not None and len(df_main) >= 30:
            for treatment_col, treatment_label in interventions:
                if treatment_col not in df_main.columns:
                    continue
                logger.info(f"敏感性场景: Complete-case ({treatment_label})" + (" (插补后数据)" if use_imputed_base else ""))
                for cohort_id, cohort_label in [('A', 'Healthy'), ('B', 'Depression'), ('C', 'Cognition')]:
                    r = run_one_scenario(df_main, f'Complete-case {treatment_label}', cohort_id, cohort_label, treatment_col=treatment_col, treatment_label=treatment_label, restrict_complete_case=True)
                    rows.append(r)

    summary = pd.DataFrame(rows)
    # 四位小数 + 95% CI 格式（与全项目统一）
    for col in ['incidence', 'ate', 'ate_lb', 'ate_ub']:
        if col in summary.columns:
            summary[col] = summary[col].round(4)
    summary['95CI'] = summary.apply(
        lambda r: f"({r['ate_lb']:.4f}, {r['ate_ub']:.4f})" if pd.notna(r.get('ate_lb')) and pd.notna(r.get('ate_ub')) else '',
        axis=1
    )
    out_csv = os.path.join(final_dir, 'sensitivity_summary.csv')
    summary.to_csv(out_csv, index=False, encoding='utf-8-sig')
    # 与 config 及主流程一致的方法学说明（CPM 时间划分 vs 本脚本 ID 子集）
    try:
        from config import USE_TEMPORAL_SPLIT as _TEMP_SPLIT, RANDOM_SEED as _RS
    except ImportError:
        _TEMP_SPLIT, _RS = False, 500
    _readme_path = os.path.join(final_dir, 'sensitivity_analysis_readme.txt')
    with open(_readme_path, 'w', encoding='utf-8') as f:
        f.write("=" * 72 + "\n")
        f.write("Diagnostic threshold sensitivity — data splits vs main CPM pipeline\n")
        f.write("=" * 72 + "\n\n")
        f.write(
            "Purpose: Re-estimate XLearner (or configured causal) ATEs under alternate CES-D and cognition cutoffs "
            "(Supplementary Table S2).\n\n"
        )
        f.write("--- A) Subsample used HERE (this script) ---\n")
        if use_imputed_base:
            f.write(
                "• Source rows: imputed wide table read as in main pipeline (equivalent to post-`prepare_exposures` "
                "and `COLS_TO_DROP` on `step1_imputed_full`), **before** `reapply_cohort_definition` is applied "
                "to the full analytic cohort.\n"
                "• Subsetting: `run_all_charls_analyses.main()` passes `df_base_for_sensitivity` = "
                "`_get_train_subset(df_imputed)` from this module (same helper as below).\n"
                "• Mechanism: **sklearn.model_selection.GroupShuffleSplit**, `n_splits=1`, **test_size=0.2**, "
                f"**groups=`ID`**, **random_state=config.RANDOM_SEED** (currently {_RS}) → approximately **80%% of "
                "unique participants** retained (training-style subsample for sensitivity only).\n"
                "• Each scenario then calls `reapply_cohort_definition(df_base, cesd_cutoff, cognition_cutoff)` to "
                "rebuild incident cohorts and `baseline_group` under that scenario’s thresholds.\n\n"
            )
        else:
            f.write(
                "• Source: `preprocess_charls_data(RAW_DATA_PATH, …)` per scenario (or internal load).\n"
                "• When `train_only=True` and no imputed `df_base` is passed, an **~80%% ID subset** is taken via "
                f"**GroupShuffleSplit(test_size=0.2, groups=ID, random_state={_RS})** inside this script before "
                "running scenarios.\n\n"
            )
        f.write("--- B) Champion CPM (Table 2) — configured in `config.py` / `compare_models` ---\n")
        if _TEMP_SPLIT:
            f.write(
                "• **`config.USE_TEMPORAL_SPLIT = True`** (current default in this project): **temporal hold-out**.\n"
                "  – Training pool: all person-waves with **wave < max(wave)** in that cohort’s prediction dataframe.\n"
                "  – Held-out test: person-waves with **wave == max(wave)**.\n"
                "  – Same participant may contribute rows only to the training side or only to the test side depending "
                "on wave; splitting is **not** an 80/20 ID draw.\n"
                "• Inner hyperparameter tuning: **GroupKFold(5)** on the **training pool only**, grouped by **ID**.\n"
                "• **This is not the same split as (A).** Do not interpret threshold-sensitivity ATEs as “replication "
                "on the CPM test fold.”\n\n"
            )
        else:
            f.write(
                "• **`config.USE_TEMPORAL_SPLIT = False`**: CPM uses **GroupShuffleSplit(test_size=0.2, groups=ID, "
                f"random_state=config.RANDOM_SEED)** once for the outer train/test split (see `compare_models`).\n"
                "• Even then, the exact set of rows can **still differ** from (A) because (A) is built from "
                "**imputed `df_imputed` pre–incident-cohort filter** at main start, while CPM uses **preprocessed "
                "prediction slices `df_pa`/`df_pb`/`df_pc`** after `reapply_cohort_definition` on the preprocessing path.\n\n"
            )
        f.write("--- 中文摘要 ---\n")
        f.write(
            "截断值敏感性：使用主流程传入的插补宽表经 **按 ID 的 GroupShuffleSplit（约保留 80%% 受访者）** 后的子集；"
            "各场景内再 `reapply_cohort_definition`。**与 CPM 不同**：当 **`USE_TEMPORAL_SPLIT=True`** 时，"
            "Table 2 的测试集为 **最大波次** 时间外推，而非本说明 (A) 中的 ID 比例子集。\n"
        )
    try:
        summary.to_excel(os.path.join(final_dir, 'sensitivity_summary.xlsx'), index=False, engine='openpyxl')
    except Exception as e:
        logger.warning(f"Excel 导出跳过 (需 openpyxl): {e}")
    logger.info(f"已保存: {out_csv}")

    plot_df = summary.dropna(subset=['ate'])
    if len(plot_df) > 0:
        for treatment_col, treatment_label in interventions:
            sub = plot_df[plot_df['intervention'] == treatment_col]
            if len(sub) == 0:
                continue
            scenarios = sub['scenario'].unique().tolist()
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(scenarios))
            width = 0.25
            for i, (cohort_id, name, color) in enumerate([
                ('A', 'Healthy (A)', '#5cb85c'),
                ('B', 'Depression (B)', '#d9534f'),
                ('C', 'Cognition (C)', '#5bc0de')
            ]):
                ates, lbs, ubs = [], [], []
                for s in scenarios:
                    r = sub[(sub['cohort_id'] == cohort_id) & (sub['scenario'] == s)]
                    if len(r) > 0:
                        r = r.iloc[0]
                        ates.append(r['ate'])
                        lbs.append(r['ate_lb'])
                        ubs.append(r['ate_ub'])
                    else:
                        ates.append(np.nan)
                        lbs.append(np.nan)
                        ubs.append(np.nan)
                off = (i - 1) * width
                ax.bar(x + off, ates, width, label=name, color=color, alpha=0.8)
                err_lo = np.array(ates) - np.array(lbs)
                err_hi = np.array(ubs) - np.array(ates)
                ax.errorbar(x + off, ates, yerr=[np.nan_to_num(err_lo, nan=0), np.nan_to_num(err_hi, nan=0)], fmt='none', color='black', capsize=2)
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios, rotation=30, ha='right')
            ax.set_ylabel('ATE (risk difference)')
            ax.set_title(f'Sensitivity to Diagnostic Thresholds: {treatment_label} ATE by scenario')
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.legend()
            plt.tight_layout()
            safe_name = treatment_col.replace(' ', '_')
            plt.savefig(os.path.join(final_dir, f'sensitivity_ate_comparison_{safe_name}.png'), dpi=300)
            plt.close()
            logger.info(f"已保存: {final_dir}/sensitivity_ate_comparison_{safe_name}.png")

    return summary


def main():
    cleanup_temp_cat_dirs()
    from config import RAW_DATA_PATH
    data_path = RAW_DATA_PATH
    final_dir = 'LIU_JUE_STRATEGIC_SUMMARY'
    run_sensitivity_scenarios_analysis(final_dir=final_dir, data_path=data_path)
    logger.info("恢复主定义预处理并写盘...")
    from config import AGE_MIN
    preprocess_charls_data(data_path, cesd_cutoff=10, cognition_cutoff=10, age_min=AGE_MIN, write_output=True)
    logger.info("敏感性分析完成。")

if __name__ == '__main__':
    main()
