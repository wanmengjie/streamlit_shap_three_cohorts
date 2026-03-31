# -*- coding: utf-8 -*-
"""
对已保存的冠军模型直接运行 SHAP 分析。
无需重新训练，加载 Cohort_*/01_prediction/champion_model.joblib
或 results/models/champion_cohort*.joblib（兼容旧名 champion_axis*.joblib）。
用法：
  python scripts/run_shap_on_saved_models.py           # 跑 A/B/C 三队列（若模型存在）
  python scripts/run_shap_on_saved_models.py --B      # 仅跑队列 B
"""

import os
import sys
import argparse
import logging
import joblib

# 项目根
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from config import (
    COHORT_A_DIR, COHORT_B_DIR, COHORT_C_DIR, COHORT_STEP_DIRS,
    RESULTS_MODELS, TARGET_COL,
)
from utils.charls_script_data_loader import load_df_for_analysis
from interpretability.charls_shap_analysis import run_shap_analysis_v2
from interpretability.charls_shap_stratified import run_stratified_shap, run_shap_interaction

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_data():
    """与主流程一致：utils.load_df_for_analysis()"""
    df_clean = load_df_for_analysis()
    if df_clean is None:
        logger.error("数据加载失败，退出。")
    return df_clean


def _find_model(cohort_id):
    """优先队列目录内 champion；其次 results/models（新名 cohort，旧名 axis 兼容）。"""
    cohort_dirs = {'A': COHORT_A_DIR, 'B': COHORT_B_DIR, 'C': COHORT_C_DIR}
    path_dir = cohort_dirs.get(cohort_id)
    if not path_dir:
        return None
    candidates = [
        os.path.join(path_dir, COHORT_STEP_DIRS['prediction'], 'champion_model.joblib'),
        os.path.join(RESULTS_MODELS, f'champion_cohort{cohort_id}.joblib'),
        os.path.join(RESULTS_MODELS, f'champion_axis{cohort_id}.joblib'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def run_shap_for_cohort(df_sub, cohort_id, output_dir):
    """对单队列（Cohort）加载模型并运行 SHAP"""
    model_path = _find_model(cohort_id)
    if not model_path:
        logger.warning(
            "未找到 Cohort %s 的冠军模型，跳过。请先运行 run_all_charls_analyses.py 生成。", cohort_id
        )
        return False
    logger.info(">>> Cohort %s: 加载模型 %s 并运行 SHAP...", cohort_id, model_path)
    model = joblib.load(model_path)
    run_shap_analysis_v2(df_sub, model=model, output_dir=output_dir, target_col=TARGET_COL)
    try:
        run_stratified_shap(
            df_sub, model,
            output_dir=os.path.join(os.path.dirname(output_dir), COHORT_STEP_DIRS['shap_stratified']),
            target_col=TARGET_COL,
        )
        run_shap_interaction(df_sub, model, output_dir=output_dir, target_col=TARGET_COL)
    except Exception as ex:
        logger.warning("分层/交互 SHAP 跳过: %s", ex)
    logger.info("Cohort %s SHAP 完成: %s", cohort_id, output_dir)
    return True


def main():
    parser = argparse.ArgumentParser(description='对已保存冠军模型运行 SHAP')
    parser.add_argument('--A', action='store_true', help='仅跑 Cohort A')
    parser.add_argument('--B', action='store_true', help='仅跑 Cohort B')
    parser.add_argument('--C', action='store_true', help='仅跑 Cohort C')
    args = parser.parse_args()
    only_b = getattr(args, 'B', False)
    only_a = getattr(args, 'A', False)
    only_c = getattr(args, 'C', False)

    df_clean = _load_data()
    if df_clean is None:
        return

    if only_b:
        tasks = [('B', COHORT_B_DIR, 1)]
    elif only_a:
        tasks = [('A', COHORT_A_DIR, 0)]
    elif only_c:
        tasks = [('C', COHORT_C_DIR, 2)]
    else:
        tasks = [
            ('A', COHORT_A_DIR, 0),
            ('B', COHORT_B_DIR, 1),
            ('C', COHORT_C_DIR, 2),
        ]

    for cohort_id, path_dir, baseline_val in tasks:
        df_sub = df_clean[df_clean['baseline_group'] == baseline_val]
        if len(df_sub) < 50:
            logger.warning("Cohort %s 样本过少 (n=%s)，跳过。", cohort_id, len(df_sub))
            continue
        output_dir = os.path.join(path_dir, COHORT_STEP_DIRS['shap'])
        os.makedirs(output_dir, exist_ok=True)
        run_shap_for_cohort(df_sub, cohort_id, output_dir)

    logger.info(">>> SHAP 分析完成。")


if __name__ == "__main__":
    main()
