# -*- coding: utf-8 -*-
"""
AUC 对比实验：有/无 COLS_TO_DROP 下预测模型性能对比。
用于验证变量移除（rgrip、psyche、puff 等）是否导致 AUC 下降。
（sleep 保留供预测，sleep_adequate 仅用于因果）

运行: python scripts/compare_auc_with_without_drop.py
输出: LIU_JUE_STRATEGIC_SUMMARY/auc_comparison_drop_vs_full.csv
"""
import os
import sys
import pandas as pd
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import IMPUTED_DATA_PATH, COLS_TO_DROP, TARGET_COL
from modeling.charls_model_comparison import compare_models
from utils.charls_script_data_loader import load_supervised_prediction_df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    out_dir = os.path.join('LIU_JUE_STRATEGIC_SUMMARY', 'auc_comparison_temp')
    os.makedirs(out_dir, exist_ok=True)

    # 检查插补数据列
    if os.path.exists(IMPUTED_DATA_PATH):
        sample = pd.read_csv(IMPUTED_DATA_PATH, nrows=1, encoding='utf-8-sig')
        cols = set(sample.columns)
        logger.info("插补数据列检查:")
        for c in ['smokev', 'rgrip', 'grip_strength_avg', 'psyche', 'puff', 'sleep', 'sleep_adequate']:
            logger.info(f"  {c}: {'存在' if c in cols else '不存在'}")

    results = []

    # 1. 全变量（不应用 config.COLS_TO_DROP）
    logger.info("\n" + "="*60 + "\n>>> 配置 A：全变量（apply_config_drop=False）\n" + "="*60)
    df_full = load_supervised_prediction_df(apply_config_drop=False)
    if df_full is None:
        logger.error("数据加载失败")
        return

    for cohort_name, df_sub in [
        ('Cohort_A', df_full[df_full['baseline_group'] == 0]),
        ('Cohort_B', df_full[df_full['baseline_group'] == 1]),
        ('Cohort_C', df_full[df_full['baseline_group'] == 2]),
    ]:
        if len(df_sub) < 50:
            logger.warning(f"{cohort_name} 样本不足，跳过")
            continue
        out_sub = os.path.join(out_dir, 'full', cohort_name)
        perf_df, _ = compare_models(df_sub, output_dir=out_sub, target_col=TARGET_COL)
        if len(perf_df) > 0:
            best = perf_df.iloc[0]
            results.append({
                'config': 'full',
                'cohort': cohort_name,
                'model': best['Model'],
                'AUC': best['AUC_raw'],
                'n': len(df_sub),
            })
            logger.info(f"  {cohort_name}: {best['Model']} AUC={best['AUC_raw']:.4f}")

    # 2. 当前配置（与主流程一致：应用 COLS_TO_DROP）
    logger.info("\n" + "="*60 + "\n>>> 配置 B：当前 COLS_TO_DROP\n" + "="*60)
    df_drop = load_supervised_prediction_df(apply_config_drop=True)
    if df_drop is None:
        logger.error("数据加载失败（drop 配置）")
        return

    for cohort_name, df_sub in [
        ('Cohort_A', df_drop[df_drop['baseline_group'] == 0]),
        ('Cohort_B', df_drop[df_drop['baseline_group'] == 1]),
        ('Cohort_C', df_drop[df_drop['baseline_group'] == 2]),
    ]:
        if len(df_sub) < 50:
            continue
        out_sub = os.path.join(out_dir, 'dropped', cohort_name)
        perf_df, _ = compare_models(df_sub, output_dir=out_sub, target_col=TARGET_COL)
        if len(perf_df) > 0:
            best = perf_df.iloc[0]
            results.append({
                'config': 'dropped',
                'cohort': cohort_name,
                'model': best['Model'],
                'AUC': best['AUC_raw'],
                'n': len(df_sub),
            })
            logger.info(f"  {cohort_name}: {best['Model']} AUC={best['AUC_raw']:.4f}")

    # 汇总对比
    res_df = pd.DataFrame(results)
    out_csv = os.path.join('LIU_JUE_STRATEGIC_SUMMARY', 'auc_comparison_drop_vs_full.csv')
    res_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    logger.info(f"\n>>> 对比结果已保存: {out_csv}")

    # 打印对比表
    full = res_df[res_df['config'] == 'full'].set_index('cohort')['AUC']
    drop = res_df[res_df['config'] == 'dropped'].set_index('cohort')['AUC']
    diff = full - drop
    logger.info("\nAUC 对比 (全变量 - 当前drop):")
    for ax in ['Cohort_A', 'Cohort_B', 'Cohort_C']:
        if ax in full.index and ax in drop.index:
            logger.info(f"  {ax}: {full[ax]:.4f} vs {drop[ax]:.4f} (Δ={diff[ax]:+.4f})")


if __name__ == '__main__':
    main()
