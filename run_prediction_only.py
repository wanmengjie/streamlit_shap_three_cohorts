# -*- coding: utf-8 -*-
"""
仅运行预测模型打擂：加载数据、三队列分别 compare_models。
用于快速验证 sleep/sleep_adequate 修改后的预测效果。
"""
import os
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from config import TARGET_COL, OUTPUT_ROOT, COHORT_A_DIR, COHORT_B_DIR, COHORT_C_DIR
from utils.charls_script_data_loader import load_supervised_prediction_df
from modeling.charls_model_comparison import compare_models

def main():
    logger.info(">>> 仅运行预测模型（三队列打擂）")

    df_clean = load_supervised_prediction_df()
    if df_clean is None:
        logger.error("数据加载失败")
        return

    # 三队列（Cohort A/B/C）预测打擂
    cohorts = [
        ('A', COHORT_A_DIR, 'Cohort_A_Healthy_Prospective'),
        ('B', COHORT_B_DIR, 'Cohort_B_Depression_to_Comorbidity'),
        ('C', COHORT_C_DIR, 'Cohort_C_Cognition_to_Comorbidity'),
    ]
    base = os.path.join(OUTPUT_ROOT, 'prediction_only_temp')
    os.makedirs(base, exist_ok=True)
    
    for cohort_id, rel_dir, desc in cohorts:
        df_sub = df_clean[df_clean['baseline_group'] == int(cohort_id == 'B') + 2 * int(cohort_id == 'C')].copy()
        if len(df_sub) < 50:
            logger.warning(f"{desc} 样本不足 n={len(df_sub)}，跳过")
            continue
        out_dir = os.path.join(base, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"\n{'='*60}\n>>> {desc} (n={len(df_sub)})")
        perf_df, models = compare_models(df_sub, output_dir=out_dir, target_col=TARGET_COL)
        if len(perf_df) > 0:
            best = perf_df.iloc[0]
            logger.info(f"  冠军: {best['Model']} AUC={best['AUC_raw']:.4f} (95% CI: {best.get('AUC_CI', 'N/A')})")
    
    logger.info(f"\n预测完成，结果见: {base}")

if __name__ == '__main__':
    main()
