# -*- coding: utf-8 -*-
"""
生理与功能指标扩展因果分析：握力、步行速度、ADL/IADL、血压、自评健康等。
将连续变量按中位数/四分位数二值化，或按临床/问卷切点二值化，估计 ATE。
T=1 表示较好状态，ATE<0 表示较好状态降低共病风险。
输出：all_physio_causal/physio_ate_summary.csv
"""
import os
import sys
import pandas as pd
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.charls_script_data_loader import load_df_for_analysis
from scripts.run_all_interventions_analysis import prepare_interventions
from causal.charls_recalculate_causal_impact import get_estimate_causal_impact

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Exposure definitions: (source_col, English_label, binarize_type, cols_to_drop)
# binarize_type: 'median'|'quartile' continuous; 'adl_01' 0 vs >=1 (T=1 no difficulty); 'systo_140' <140 vs >=140 (T=1 normal);
# 'srh_good' good/very good vs fair/poor (T=1 better); 'diasto_90' <90 vs >=90 (T=1 normal); 'binary_invert' orig 1=bad, T=1 when orig=0;
# 'binary_pass' orig 1=good, T=1 when orig=1
EXPOSURES = [
    # Functional: ADL/IADL 0 vs >=1, T=1 no difficulty
    ('adlab_c', 'ADL no difficulty', 'adl_01', ['adlab_c']),
    ('iadl', 'IADL no difficulty', 'adl_01', ['iadl']),
    # Physiological: systo, diasto, lgrip, wspeed, mwaist 已恢复
    ('systo', 'Systolic BP normal', 'systo_140', ['systo']),
    ('diasto', 'Diastolic BP normal', 'diasto_90', ['diasto']),
    ('lgrip', 'Grip strength', 'median', ['lgrip']),
    ('wspeed', 'Walking speed', 'median', ['wspeed']),
    ('mwaist', 'Waist circumference', 'median', ['mwaist']),
    # Self-rated health, life satisfaction
    ('srh', 'Good self-rated health', 'srh_good', ['srh']),
    ('satlife', 'Life satisfaction', 'srh_good', ['satlife']),
    ('pulse', 'Pulse', 'median', ['pulse']),
    ('income_total', 'Income', 'median', ['income_total']),
    # Disability, fall (T=1 no disability/no fall)
    ('disability', 'No disability', 'binary_invert', ['disability']),
    ('fall_down', 'No fall', 'binary_invert', ['fall_down']),
    # Pension, insurance (T=1 has)
    ('pension', 'Has pension', 'binary_pass', ['pension']),
    ('ins', 'Has insurance', 'binary_pass', ['ins']),
]


def create_binary_exposure(df, col, binarize_type):
    """
    根据二值化方式创建暴露变量。返回 (new_col, df) 或 (None, None)。
    T=1 表示较好状态。
    """
    if col not in df.columns:
        return None, None
    vals = df[col].dropna()
    if len(vals) < 100:
        return None, None

    df = df.copy()

    if binarize_type == 'median':
        cut = vals.median()
        new_col = f'{col}_above_median'
        df[new_col] = np.where(df[col].isna(), np.nan, (df[col] >= cut).astype(int))
        return new_col, df

    if binarize_type == 'quartile':
        q1, q4 = vals.quantile(0.25), vals.quantile(0.75)
        new_col = f'{col}_high_vs_low'
        mask_low = (df[col] <= q1) & df[col].notna()
        mask_high = (df[col] >= q4) & df[col].notna()
        df[new_col] = np.nan
        df.loc[mask_low, new_col] = 0
        df.loc[mask_high, new_col] = 1
        return new_col, df

    if binarize_type == 'adl_01':
        # 0 困难 = 较好(T=1)，≥1 困难 = 较差(T=0)
        new_col = f'{col}_no_difficulty'
        df[new_col] = np.where(df[col].isna(), np.nan, (df[col] == 0).astype(int))
        return new_col, df

    if binarize_type == 'systo_140':
        # <140 正常 = T=1，≥140 高血压 = T=0
        new_col = f'{col}_normal'
        df[new_col] = np.where(df[col].isna(), np.nan, (df[col] < 140).astype(int))
        return new_col, df

    if binarize_type == 'diasto_90':
        # <90 正常 = T=1，≥90 舒张压升高 = T=0
        new_col = f'{col}_normal'
        df[new_col] = np.where(df[col].isna(), np.nan, (df[col] < 90).astype(int))
        return new_col, df

    if binarize_type == 'binary_invert':
        # 原变量 1=差，T=1 当原=0（较好）
        new_col = f'{col}_no' if col in ('disability', 'fall_down') else f'{col}_good'
        df[new_col] = np.where(df[col].isna(), np.nan, (df[col] == 0).astype(int))
        return new_col, df

    if binarize_type == 'binary_pass':
        # 原变量 1=好，直接二值化（确保 0/1）
        new_col = f'{col}_yes'
        df[new_col] = np.where(df[col].isna(), np.nan, (df[col].astype(int) >= 1).astype(int))
        return new_col, df

    if binarize_type == 'srh_good':
        # 自评健康/生活满意度：较小值=较好。CHARLS 常见 1=很好 2=好 3=一般 4=不好 5=很不好
        # LabelEncoder 后 0-4，较小=较好。取 ≤ 中位数或 ≤1 为"好"
        med = vals.median()
        # 若为 0-4 编码，1 及以下为较好；否则用中位数
        if vals.max() <= 4 and vals.min() >= 0:
            good_threshold = 1  # 0,1 = 很好/好
        else:
            good_threshold = med
        new_col = f'{col}_good'
        df[new_col] = np.where(df[col].isna(), np.nan, (df[col] <= good_threshold).astype(int))
        return new_col, df

    return None, None


def run_all_physio_causal(output_dir='all_physio_causal', df_clean=None):
    """对所有生理/功能暴露运行因果分析。df_clean 可选，主流程传入时复用同一数据源。"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据（主流程传入 df_clean 时复用，保证与主流程一致）
    if df_clean is not None:
        df_clean = df_clean.copy()
    else:
        df_clean = load_df_for_analysis()
        if df_clean is None:
            logger.error("数据加载失败")
            return []

    df_clean = prepare_interventions(df_clean)

    df_a = df_clean[df_clean['baseline_group'] == 0]
    df_b = df_clean[df_clean['baseline_group'] == 1]
    df_c = df_clean[df_clean['baseline_group'] == 2]

    results = []
    for col, label_cn, binarize_type, drop_cols in EXPOSURES:
        new_col, df_with_t = create_binary_exposure(df_clean, col, binarize_type)
        if new_col is None:
            logger.warning(f"跳过 {col} ({binarize_type})：数据不足或列缺失")
            continue

        cut_label = {
            'median': 'Median',
            'quartile': 'Q4 vs Q1',
            'adl_01': '0 vs >=1',
            'systo_140': '<140 vs >=140',
            'diasto_90': '<90 vs >=90',
            'srh_good': 'Good vs Poor',
            'binary_invert': 'No vs Yes',
            'binary_pass': 'No vs Yes',
        }.get(binarize_type, binarize_type)

        for cohort_name, df_sub in [
            ('Cohort_A', df_with_t[df_with_t['baseline_group'] == 0]),
            ('Cohort_B', df_with_t[df_with_t['baseline_group'] == 1]),
            ('Cohort_C', df_with_t[df_with_t['baseline_group'] == 2]),
        ]:
            df_sub = df_sub.dropna(subset=[new_col]).copy()
            if len(df_sub) < 50:
                continue
            if df_sub[new_col].nunique() < 2:
                continue

            df_analyze = df_sub.drop(columns=drop_cols, errors='ignore')

            out_sub = os.path.join(output_dir, new_col, cohort_name)
            os.makedirs(out_sub, exist_ok=True)
            try:
                res_df, (ate, lb, ub) = get_estimate_causal_impact()(
                    df_analyze, treatment_col=new_col, output_dir=out_sub
                )
                if res_df is None:
                    ate, lb, ub = np.nan, np.nan, np.nan
                else:
                    for v in (ate, lb, ub):
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            ate, lb, ub = np.nan, np.nan, np.nan
                            break
                results.append({
                    'exposure': new_col,
                    'label': f'{label_cn}({cut_label})',
                    'cohort': cohort_name,
                    'ate': ate,
                    'ate_lb': lb,
                    'ate_ub': ub,
                    'n': len(df_sub),
                    'n_t1': (df_sub[new_col] == 1).sum(),
                    'n_t0': (df_sub[new_col] == 0).sum(),
                })
                if not (isinstance(ate, float) and np.isnan(ate)):
                    logger.info(f"  {cohort_name} {new_col}: ATE={ate:.4f} (95% CI: {lb:.4f}, {ub:.4f})")
                else:
                    logger.info(f"  {cohort_name} {new_col}: 因果估计跳过/失败 (NaN)")
            except Exception as e:
                logger.warning(f"  {cohort_name} {new_col} 失败: {e}")

    if results:
        res_df = pd.DataFrame(results)
        # 多重检验校正（17暴露×3队列=51次比较）
        from utils.multiplicity_correction import add_multiplicity_columns
        res_df = add_multiplicity_columns(res_df)
        out_csv = os.path.join(output_dir, 'physio_ate_summary.csv')
        res_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        logger.info(f"结果已保存: {out_csv}")
    else:
        logger.warning("无有效结果")
    return results


if __name__ == '__main__':
    run_all_physio_causal()
