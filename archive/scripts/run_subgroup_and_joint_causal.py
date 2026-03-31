# -*- coding: utf-8 -*-
"""
亚组因果分析 + 联合干预因果估计（三队列统一：A/B/C）
1. 亚组分析：在年龄(<65, 65-75, 75+)、城乡、性别亚组内分别估计 exercise 的 ATE
2. 联合干预：运动+充足睡眠 (exercise_sleep_both=1) vs 其他，估计 ATE
"""
import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from data.charls_complete_preprocessing import preprocess_charls_data
from causal.charls_recalculate_causal_impact import get_estimate_causal_impact

MIN_SUBGROUP_N = 100  # 亚组最小样本量


def prepare_data(df):
    """构造 exercise_sleep_both（运动且睡眠≥6h，sleep_adequate 已移除）"""
    if 'exercise' in df.columns and 'sleep' in df.columns:
        ex = df['exercise'].fillna(0).astype(int)
        sleep_val = df['sleep'].clip(0, 24)
        df['exercise_sleep_both'] = ((ex == 1) & (sleep_val >= 6)).astype(int)
        df.loc[sleep_val.isna(), 'exercise_sleep_both'] = np.nan
    return df


def run_subgroup_causal(df_sub, treatment_col, axis_name, output_root):
    """在各亚组内分别拟合因果模型，返回 ATE"""
    results = []
    df_sub = df_sub.copy()

    # 年龄分组（文件夹名避免 < > 等非法字符）
    if 'age' in df_sub.columns:
        df_sub['age_group'] = pd.cut(df_sub['age'], bins=[0, 65, 75, 120], labels=['lt65', '65_75', '75plus'])
        ag_map = {'lt65': '<65', '65_75': '65-75', '75plus': '75+'}
        for ag, ag_label in [('lt65', '<65'), ('65_75', '65-75'), ('75plus', '75+')]:
            mask = (df_sub['age_group'] == ag)
            if mask.sum() >= MIN_SUBGROUP_N:
                sub = df_sub[mask].copy()
                out_dir = os.path.join(output_root, 'subgroup', axis_name, f'age_{ag}')
                os.makedirs(out_dir, exist_ok=True)
                _, (ate, lb, ub) = get_estimate_causal_impact()(sub, treatment_col=treatment_col, output_dir=out_dir)
                sig = 1 if (lb > 0 or ub < 0) else 0
                results.append({'subgroup': 'age', 'value': ag_map.get(ag, ag), 'axis': axis_name, 'ate': ate, 'ate_lb': lb, 'ate_ub': ub, 'significant': sig, 'n': len(sub)})

    # 城乡
    if 'rural' in df_sub.columns:
        for val, label in [(0, 'urban'), (1, 'rural')]:
            mask = (df_sub['rural'] == val)
            if mask.sum() >= MIN_SUBGROUP_N:
                sub = df_sub[mask].copy()
                out_dir = os.path.join(output_root, 'subgroup', axis_name, f'residence_{label}')
                os.makedirs(out_dir, exist_ok=True)
                _, (ate, lb, ub) = get_estimate_causal_impact()(sub, treatment_col=treatment_col, output_dir=out_dir)
                sig = 1 if (lb > 0 or ub < 0) else 0
                results.append({'subgroup': 'residence', 'value': label, 'axis': axis_name, 'ate': ate, 'ate_lb': lb, 'ate_ub': ub, 'significant': sig, 'n': len(sub)})

    # 性别
    if 'gender' in df_sub.columns:
        for val, label in [(0, 'female'), (1, 'male')]:
            mask = (df_sub['gender'] == val)
            if mask.sum() >= MIN_SUBGROUP_N:
                sub = df_sub[mask].copy()
                out_dir = os.path.join(output_root, 'subgroup', axis_name, f'sex_{label}')
                os.makedirs(out_dir, exist_ok=True)
                _, (ate, lb, ub) = get_estimate_causal_impact()(sub, treatment_col=treatment_col, output_dir=out_dir)
                sig = 1 if (lb > 0 or ub < 0) else 0
                results.append({'subgroup': 'sex', 'value': label, 'axis': axis_name, 'ate': ate, 'ate_lb': lb, 'ate_ub': ub, 'significant': sig, 'n': len(sub)})

    return results


def run_joint_intervention(df_sub, axis_name, output_root):
    """联合干预：exercise_sleep_both"""
    out_dir = os.path.join(output_root, 'joint', axis_name, 'exercise_sleep_both')
    os.makedirs(out_dir, exist_ok=True)
    _, (ate, lb, ub) = get_estimate_causal_impact()(df_sub, treatment_col='exercise_sleep_both', output_dir=out_dir)
    sig = 1 if (lb > 0 or ub < 0) else 0
    return {'exposure': 'exercise_sleep_both', 'axis': axis_name, 'ate': ate, 'ate_lb': lb, 'ate_ub': ub, 'significant': sig, 'n': len(df_sub)}


def main():
    df = preprocess_charls_data('CHARLS.csv', age_min=60)
    if df is None:
        logger.error("预处理失败")
        return

    df = prepare_data(df)
    df_a = df[df['baseline_group'] == 0].copy()
    df_b = df[df['baseline_group'] == 1].copy()
    df_c = df[df['baseline_group'] == 2].copy()

    out_root = 'subgroup_and_joint_causal'
    os.makedirs(out_root, exist_ok=True)

    # === 1. 亚组因果分析（exercise） ===
    logger.info("\n" + "="*60 + "\n>>> 1. 亚组因果分析（exercise）\n" + "="*60)
    subgroup_results = []
    for axis_name, df_sub in [('Cohort_A', df_a), ('Cohort_B', df_b), ('Cohort_C', df_c)]:
        if len(df_sub) < MIN_SUBGROUP_N:
            continue
        res = run_subgroup_causal(df_sub, 'exercise', axis_name, out_root)
        subgroup_results.extend(res)

    sub_df = pd.DataFrame(subgroup_results)
    if len(sub_df) > 0:
        sub_df.to_csv(os.path.join(out_root, 'subgroup_ate_exercise.csv'), index=False, encoding='utf-8-sig')
        logger.info(f"亚组 ATE 已保存: {out_root}/subgroup_ate_exercise.csv")
        sig_sub = sub_df[sub_df['significant'] == 1]
        if len(sig_sub) > 0:
            logger.info(f"显著亚组: {list(sig_sub[['subgroup','value','axis']].to_dict('records'))}")

    # === 2. 联合干预（运动+睡眠） ===
    logger.info("\n" + "="*60 + "\n>>> 2. 联合干预（运动+充足睡眠）\n" + "="*60)
    joint_results = []
    for axis_name, df_sub in [('Cohort_A', df_a), ('Cohort_B', df_b), ('Cohort_C', df_c)]:
        if len(df_sub) < 50:
            continue
        r = run_joint_intervention(df_sub, axis_name, out_root)
        joint_results.append(r)
        logger.info(f"  {axis_name} exercise_sleep_both: ATE={r['ate']:.4f} (95% CI: {r['ate_lb']:.4f}, {r['ate_ub']:.4f}) {'*显著*' if r['significant'] else ''}")

    joint_df = pd.DataFrame(joint_results)
    joint_df.to_csv(os.path.join(out_root, 'joint_exercise_sleep_ate.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"\n联合干预 ATE 已保存: {out_root}/joint_exercise_sleep_ate.csv")

    logger.info("\n" + "="*60 + "\n完成")
    return sub_df, joint_df


if __name__ == '__main__':
    main()
