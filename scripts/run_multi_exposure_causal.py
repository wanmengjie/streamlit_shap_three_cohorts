# -*- coding: utf-8 -*-
"""
多暴露因果分析：对运动、睡眠、吸烟、饮酒、社会隔离分别估计 ATE，
输出各暴露在各轴线的因果效应及是否显著。

独立运行 __main__ 时与主流程同源数据：utils.charls_script_data_loader.load_df_for_analysis()。
"""
import os
import sys
import logging
import pandas as pd
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from utils.charls_script_data_loader import load_df_for_analysis
from utils.charls_prepare_exposures import prepare_exposures
from causal.charls_recalculate_causal_impact import get_estimate_causal_impact

# 暴露变量定义：(列名, 二值化方式)
# None = 直接使用原列（需为 0/1）；puff_low 已移除，改用 smokev（当前吸烟）
EXPOSURES = [
    ('exercise', None),           # 规律运动 1 vs 0
    ('drinkev', None),            # 当前饮酒 1 vs 0
    ('smokev', None),             # 当前吸烟 1 vs 0（替代 puff_low）
    ('is_socially_isolated', None),  # 社会隔离 1 vs 0（风险因素）
]


def run_multi_exposure_analysis(df_clean, output_root='multi_exposure_causal'):
    """
    多暴露因果分析：对运动、睡眠、吸烟、饮酒、社会隔离分别估计 ATE。
    可被主流程调用，传入预处理后的 df_clean。
    """
    prepare_exposures(df_clean)
    df_a = df_clean[df_clean['baseline_group'] == 0].copy()
    df_b = df_clean[df_clean['baseline_group'] == 1].copy()
    df_c = df_clean[df_clean['baseline_group'] == 2].copy()

    results = []
    os.makedirs(output_root, exist_ok=True)

    for col, _ in EXPOSURES:
        treatment_col = col
        if treatment_col not in df_clean.columns:
            logger.warning(f"跳过 {treatment_col}：列不存在")
            continue

        for cohort_name, df_sub in [('Cohort_A', df_a), ('Cohort_B', df_b), ('Cohort_C', df_c)]:
            if len(df_sub) < 50:
                logger.warning(f"{cohort_name} 样本过少，跳过")
                continue
            out_dir = os.path.join(output_root, cohort_name, treatment_col)
            os.makedirs(out_dir, exist_ok=True)
            res_causal, (ate, lb, ub) = get_estimate_causal_impact()(
                df_sub, treatment_col=treatment_col, output_dir=out_dir
            )
            if res_causal is None:
                ate, lb, ub = np.nan, np.nan, np.nan
            else:
                for v in (ate, lb, ub):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        ate, lb, ub = np.nan, np.nan, np.nan
                        break
            sig = (
                1
                if (
                    lb is not None
                    and ub is not None
                    and not (isinstance(lb, float) and np.isnan(lb))
                    and not (isinstance(ub, float) and np.isnan(ub))
                    and (lb > 0 or ub < 0)
                )
                else 0
            )
            reliable = (
                1
                if (
                    ate is not None
                    and not (isinstance(ate, float) and np.isnan(ate))
                    and -1 <= ate <= 1
                )
                else 0
            )
            results.append({
                'exposure': treatment_col,
                'cohort': cohort_name,
                'ate': ate,
                'ate_lb': lb,
                'ate_ub': ub,
                'significant': sig,
                'reliable': reliable,
                'n': len(df_sub),
            })
            if not (isinstance(ate, float) and np.isnan(ate)):
                logger.info(
                    f"  {cohort_name} {treatment_col}: ATE={ate:.4f} (95% CI: {lb:.4f}, {ub:.4f}) {'*显著*' if sig else ''}"
                )
            else:
                logger.info(f"  {cohort_name} {treatment_col}: 因果估计跳过/失败 (NaN)")

    res_df = pd.DataFrame(results)
    # 多重检验校正（4暴露×3队列=12次比较）
    from utils.multiplicity_correction import add_multiplicity_columns
    res_df = add_multiplicity_columns(res_df)
    out_csv = os.path.join(output_root, 'multi_exposure_ate_summary.csv')
    res_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    logger.info(f"多暴露因果汇总已保存: {out_csv}")
    return res_df


def run_multi_exposure():
    """独立运行：对 A/B/C 轴分别跑各暴露的因果分析"""
    df = load_df_for_analysis()
    if df is None:
        logger.error("数据加载失败（检查 config / CHARLS.csv）")
        return
    res_df = run_multi_exposure_analysis(df, output_root='multi_exposure_causal')

    sig_df = res_df[(res_df['significant'] == 1) & (res_df['reliable'] == 1)]
    unreliable = res_df[res_df['reliable'] == 0]
    if len(unreliable) > 0:
        logger.info(f"\n不可靠估计（ATE 超出 [-1,1]，可能因治疗组极少）: {list(unreliable['exposure'].unique())}")
    if len(sig_df) > 0:
        logger.info(f"\n显著结果 ({len(sig_df)} 个):")
        for _, r in sig_df.iterrows():
            logger.info(f"  {r['exposure']} @ {r['cohort']}: ATE={r['ate']:.4f} (95% CI: {r['ate_lb']:.4f}, {r['ate_ub']:.4f})")
    else:
        logger.info("\n无显著结果（所有 95% CI 均包含 0）")

    return res_df


if __name__ == '__main__':
    run_multi_exposure()
