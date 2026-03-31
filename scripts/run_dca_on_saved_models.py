# -*- coding: utf-8 -*-
"""
对已保存的冠军模型直接运行临床综合评价（DCA + 校准曲线 + PR 曲线）。
无需重新训练，加载 Cohort_*/01_prediction/champion_model.joblib
或 results/models/champion_cohort*.joblib（兼容 champion_axis*.joblib）。
用法：python scripts/run_dca_on_saved_models.py [--B 仅 Cohort B]
"""

import os
import sys
import argparse
import logging
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from config import (
    COHORT_A_DIR, COHORT_B_DIR, COHORT_C_DIR, COHORT_STEP_DIRS, RESULTS_MODELS,
    TARGET_COL,
)
from utils.charls_script_data_loader import load_df_for_analysis
from evaluation.charls_clinical_evaluation import run_clinical_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_data():
    return load_df_for_analysis()


def _find_model(cohort_id):
    cohort_dirs = {'A': COHORT_A_DIR, 'B': COHORT_B_DIR, 'C': COHORT_C_DIR}
    path_dir = cohort_dirs.get(cohort_id)
    if not path_dir:
        return None
    for p in [
        os.path.join(path_dir, COHORT_STEP_DIRS['prediction'], 'champion_model.joblib'),
        os.path.join(RESULTS_MODELS, f'champion_cohort{cohort_id}.joblib'),
        os.path.join(RESULTS_MODELS, f'champion_axis{cohort_id}.joblib'),
    ]:
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', action='store_true', help='仅跑 Cohort B')
    args = parser.parse_args()

    df_clean = _load_data()
    if df_clean is None:
        logger.error("数据加载失败")
        return

    tasks = [('B', COHORT_B_DIR, 1)] if args.B else [
        ('A', COHORT_A_DIR, 0), ('B', COHORT_B_DIR, 1), ('C', COHORT_C_DIR, 2)
    ]

    for cohort_id, path_dir, baseline_val in tasks:
        df_sub = df_clean[df_clean['baseline_group'] == baseline_val]
        if len(df_sub) < 50:
            logger.warning("Cohort %s 样本过少，跳过", cohort_id)
            continue
        model_path = _find_model(cohort_id)
        if not model_path:
            logger.warning(
                "未找到 Cohort %s 的冠军模型，跳过。请先运行 run_all_charls_analyses.py", cohort_id
            )
            continue
        logger.info(">>> Cohort %s: 加载 %s 并运行 DCA/校准/PR...", cohort_id, model_path)
        model = joblib.load(model_path)
        output_dir = os.path.join(path_dir, COHORT_STEP_DIRS['eval'])
        os.makedirs(output_dir, exist_ok=True)
        run_clinical_evaluation(df_sub, model=model, output_dir=output_dir, target_col=TARGET_COL)
        logger.info("Cohort %s 完成: %s/fig3_clinical_evaluation_comprehensive.png", cohort_id, output_dir)

    logger.info(">>> DCA/校准/PR 分析完成。")


if __name__ == "__main__":
    main()
