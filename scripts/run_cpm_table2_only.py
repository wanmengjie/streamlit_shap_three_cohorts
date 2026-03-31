# -*- coding: utf-8 -*-
"""
仅运行 CPM Table 2（预测性能 + Bootstrap 95% CI），不跑 SHAP/因果/亚组等。

用法（在项目根目录）:
  # 最快：已有冠军 joblib + 旧 table2 里有 Optimal_Threshold 时，只重算 3 个队列各 1 个模型的指标（含 Precision/F1/Acc 的 CI）
  python scripts/run_cpm_table2_only.py --champion-only

  # 只要某一队列的冠军重算
  python scripts/run_cpm_table2_only.py --champion-only --cohort B

  # 完整 14 模型打擂 + CPM 表（与 run_all 里「预测步」等价，仍较慢）
  python scripts/run_cpm_table2_only.py --full

  # 完整表但单队列 + 减少调优迭代（更快，结果仅供调试）
  python scripts/run_cpm_table2_only.py --full --cohort A --quick

  # 写完后复制到 results/tables（与 consolidate 命名一致）
  python scripts/run_cpm_table2_only.py --champion-only --copy-results
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config import (
    COHORT_A_DIR,
    COHORT_B_DIR,
    COHORT_C_DIR,
    COHORT_STEP_DIRS,
    RANDOM_SEED,
    RESULTS_TABLES,
    TARGET_COL,
    USE_TEMPORAL_SPLIT,
)
from modeling.charls_cpm_evaluation import evaluate_and_report, evaluate_single_model
from modeling.charls_model_comparison import compare_models
from utils.charls_feature_lists import get_exclude_cols
from utils.charls_script_data_loader import load_df_for_analysis, load_supervised_prediction_df


def _load_df_clean():
    return load_supervised_prediction_df()


def _split_xy_test(df_sub: pd.DataFrame):
    """与 compare_models 一致的测试集划分（GroupShuffleSplit 或时间划分）；返回测试集 ID 供 cluster bootstrap。"""
    exclude = get_exclude_cols(df_sub, target_col=TARGET_COL)
    w_cols = [c for c in df_sub.columns if c not in exclude]
    x = df_sub[w_cols].select_dtypes(include=[np.number])
    y = df_sub[TARGET_COL].astype(int)
    use_temporal = USE_TEMPORAL_SPLIT and "wave" in df_sub.columns
    test_idx = None
    if use_temporal:
        max_wave = df_sub["wave"].max()
        train_mask = (df_sub["wave"] < max_wave).values
        test_mask = (df_sub["wave"] == max_wave).values
        if train_mask.sum() >= 50 and test_mask.sum() >= 20:
            test_idx = np.where(test_mask)[0]
        else:
            use_temporal = False
    if not use_temporal:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
        _, test_idx = next(gss.split(x, y, groups=df_sub["ID"]))
    x_test = x.iloc[test_idx]
    y_test = y.iloc[test_idx]
    g_test = df_sub.iloc[test_idx]["ID"].to_numpy()
    return x_test, y_test, g_test


def _cohort_dirs():
    return {"A": (COHORT_A_DIR, 0), "B": (COHORT_B_DIR, 1), "C": (COHORT_C_DIR, 2)}


def _find_champion_joblib(cohort_id: str) -> str | None:
    ad, _ = _cohort_dirs()[cohort_id]
    p = os.path.join(ad, COHORT_STEP_DIRS["prediction"], "champion_model.joblib")
    if os.path.exists(p):
        return p
    return None


def _read_threshold_from_table2(cohort_id: str) -> tuple[str | None, float | None]:
    """从已有 table2 CSV 取 AUC 最高行的 Model 与 Optimal_Threshold。"""
    ad, _ = _cohort_dirs()[cohort_id]
    t2 = os.path.join(ad, COHORT_STEP_DIRS["prediction"], f"table2_{cohort_id}_main_performance.csv")
    if not os.path.exists(t2):
        # consolidate 副本（新名优先，旧名兼容）
        t2c = os.path.join(RESULTS_TABLES, f"table2_prediction_cohort{cohort_id}.csv")
        t2a = os.path.join(RESULTS_TABLES, f"table2_prediction_axis{cohort_id}.csv")
        t2 = t2c if os.path.exists(t2c) else t2a
    if not os.path.exists(t2):
        return None, None
    try:
        df = pd.read_csv(t2, encoding="utf-8-sig")
        if df.empty or "AUC" not in df.columns:
            return None, None
        row = df.sort_values("AUC", ascending=False).iloc[0]
        name = str(row["Model"])
        th = row.get("Optimal_Threshold")
        th_f = float(th) if pd.notna(th) else None
        return name, th_f
    except Exception as ex:
        logger.debug("读取 table2 阈值失败: %s", ex)
        return None, None


def run_champion_only(cohort_list: list[str], copy_results: bool):
    import joblib

    df_clean = _load_df_clean()
    if df_clean is None:
        logger.error("数据加载失败")
        return 1

    for cohort_id in cohort_list:
        path_dir, baseline_val = _cohort_dirs()[cohort_id]
        df_sub = df_clean[df_clean["baseline_group"] == baseline_val]
        if len(df_sub) < 50:
            logger.warning("Cohort %s 样本过少，跳过", cohort_id)
            continue
        jpath = _find_champion_joblib(cohort_id)
        if not jpath:
            logger.warning("未找到 %s 的 champion_model.joblib，跳过（可先跑主流程预测步）", cohort_id)
            continue
        model = joblib.load(jpath)
        x_test, y_test, g_test = _split_xy_test(df_sub)
        name, thresh = _read_threshold_from_table2(cohort_id)
        if name is None:
            name = "Champion"
        if thresh is None:
            logger.warning(
                "Cohort %s 未读到 Optimal_Threshold，将在测试集上用 Youden 寻阈（与正式 TRIPOD 流程不一致，仅供应急）",
                cohort_id,
            )
        pred_dir = os.path.join(path_dir, COHORT_STEP_DIRS["prediction"])
        os.makedirs(pred_dir, exist_ok=True)
        res = evaluate_single_model(
            model, x_test, y_test, model_name=name, opt_threshold=thresh, groups_test=g_test
        )
        if res is None:
            logger.error("Cohort %s 评估失败", cohort_id)
            continue
        out_cols = {
            k: res[k]
            for k in res
            if k not in ("y_prob", "y_true") and not str(k).startswith("_")
        }
        df_one = pd.DataFrame([out_cols])
        out_csv = os.path.join(pred_dir, f"table2_{cohort_id}_champion_only_refresh.csv")
        df_one.to_csv(out_csv, index=False, encoding="utf-8-sig")
        logger.info("Cohort %s 已写入 %s", cohort_id, out_csv)
        if copy_results and os.path.exists(out_csv):
            os.makedirs(RESULTS_TABLES, exist_ok=True)
            shutil.copy(out_csv, os.path.join(RESULTS_TABLES, f"table2_champion_only_cohort{cohort_id}.csv"))
            shutil.copy(out_csv, os.path.join(RESULTS_TABLES, f"table2_champion_only_axis{cohort_id}.csv"))

    logger.info(">>> champion-only 完成（完整 14 模型表请用 --full）")
    return 0


def run_full(cohort_list: list[str], n_iter: int, copy_results: bool):
    df_clean = _load_df_clean()
    if df_clean is None:
        logger.error("数据加载失败")
        return 1

    for cohort_id in cohort_list:
        path_dir, baseline_val = _cohort_dirs()[cohort_id]
        df_sub = df_clean[df_clean["baseline_group"] == baseline_val]
        if len(df_sub) < 50:
            logger.warning("Cohort %s 样本过少，跳过", cohort_id)
            continue
        pred_dir = os.path.join(path_dir, COHORT_STEP_DIRS["prediction"])
        os.makedirs(pred_dir, exist_ok=True)
        logger.info(">>> Cohort %s：compare_models + evaluate_and_report（n_iter=%s）", cohort_id, n_iter)
        result = compare_models(
            df_sub,
            output_dir=pred_dir,
            target_col=TARGET_COL,
            n_iter=n_iter,
            return_search_objects=True,
            return_xy_test=True,
        )
        perf_df, models, _search_objs, x_test, y_test, _x_tr, _y_tr, _grp_tr, grp_test = result
        thresholds_dict = {}
        if "_opt_threshold" in perf_df.columns:
            thresholds_dict = {
                row["Model"]: float(row["_opt_threshold"])
                for _, row in perf_df.iterrows()
                if pd.notna(row.get("_opt_threshold"))
            }
        evaluate_and_report(
            models,
            x_test,
            y_test,
            cohort_label=cohort_id,
            output_dir=pred_dir,
            thresholds_dict=thresholds_dict or None,
            groups_test=grp_test,
        )
        src = os.path.join(pred_dir, f"table2_{cohort_id}_main_performance.csv")
        if copy_results and os.path.exists(src):
            os.makedirs(RESULTS_TABLES, exist_ok=True)
            shutil.copy(src, os.path.join(RESULTS_TABLES, f"table2_prediction_cohort{cohort_id}.csv"))
            shutil.copy(src, os.path.join(RESULTS_TABLES, f"table2_prediction_axis{cohort_id}.csv"))
            logger.info("已复制到 %s", RESULTS_TABLES)

    logger.info(">>> full CPM Table2 完成")
    return 0


def main():
    ap = argparse.ArgumentParser(description="仅跑 CPM Table2（不打全管线）")
    ap.add_argument("--full", action="store_true", help="14 模型打擂 + 全表（慢）")
    ap.add_argument(
        "--champion-only",
        action="store_true",
        help="仅对已保存 champion 重算指标（快；需 joblib，阈值优先读旧 table2）",
    )
    ap.add_argument(
        "--cohort",
        "--axis",
        choices=["A", "B", "C", "all"],
        default="all",
        dest="cohort",
        help="指定 Cohort A/B/C 或 all（--axis 为旧别名）",
    )
    ap.add_argument("--quick", action="store_true", help="--full 时调优迭代降为 12（仅调试）")
    ap.add_argument("--copy-results", action="store_true", help="同步复制到 results/tables")
    args = ap.parse_args()

    cohort_list = ["A", "B", "C"] if args.cohort == "all" else [args.cohort]

    if args.full and args.champion_only:
        logger.error("不要同时使用 --full 与 --champion-only")
        return 1
    if not args.full and not args.champion_only:
        logger.info("未指定模式，默认使用最快的 --champion-only")
        args.champion_only = True

    n_iter = 12 if args.quick else 80

    if args.full:
        return run_full(cohort_list, n_iter=n_iter, copy_results=args.copy_results)
    return run_champion_only(cohort_list, copy_results=args.copy_results)


if __name__ == "__main__":
    sys.exit(main() or 0)
