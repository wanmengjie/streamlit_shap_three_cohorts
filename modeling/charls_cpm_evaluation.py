# -*- coding: utf-8 -*-
"""
临床预测模型（CPM）评估与报告流水线
符合 TRIPOD 报告指南，支持类别不平衡数据的阈值校正与校准度分析。

适用于三队列（Cohort A/B/C）各 15 个预训练模型的统一评估。
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

try:
    from config import RANDOM_SEED, BBOX_INCHES
except ImportError:
    RANDOM_SEED = 500
    BBOX_INCHES = 'tight'

from utils.charls_ci_utils import cluster_bootstrap_indices_once

logger = logging.getLogger(__name__)


def _get_proba_1d(y_prob):
    """
    健壮处理 predict_proba 输出：支持 1 维或 2 维数组。
    返回正类概率的 1 维数组。
    """
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 2:
        if y_prob.shape[1] == 2:
            return y_prob[:, 1]
        if y_prob.shape[1] == 1:
            return y_prob[:, 0]
        raise ValueError(f"predict_proba 返回了意外的列数: {y_prob.shape[1]}")
    return y_prob


def _compute_optimal_threshold_from_validation(model, X_val, y_val, model_name="Model"):
    """
    在验证集上确定最优阈值，避免在测试集上寻优导致虚高灵敏度（审稿 P1）。
    返回 float 或 None（若无法计算）。
    """
    estimator, _, _ = _extract_search_info(model, model_name)
    try:
        y_prob = estimator.predict_proba(X_val)
    except Exception:
        return None
    y_prob = _get_proba_1d(y_prob)
    y_true = np.asarray(y_val).astype(int)
    if len(np.unique(y_true)) < 2:
        return None
    thresh, _ = _find_optimal_threshold_youden(y_true, y_prob)
    return thresh


def _find_optimal_threshold_youden(y_true, y_prob):
    """
    通过最大化约登指数 (Youden Index = Sensitivity + Specificity - 1) 找到最优阈值。
    返回 (最优阈值, 约登指数)。
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0  # 单类时无法计算 ROC，退回默认阈值
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    except (ValueError, IndexError):
        return 0.5, 0.0
    if len(thresholds) == 0 or len(fpr) == 0:
        return 0.5, 0.0
    youden = tpr - fpr  # Sensitivity + Specificity - 1 = tpr - fpr
    idx = np.argmax(youden)
    return float(thresholds[idx]), float(youden[idx])


def _format_hyperparams(params_dict):
    """
    将超参数字典格式化为易读的单行字符串。
    例如: "n_estimators=100, max_depth=5"
    """
    if params_dict is None or not isinstance(params_dict, dict):
        return ""
    parts = []
    for k, v in params_dict.items():
        key = k.replace("clf__", "") if isinstance(k, str) else k
        parts.append(f"{key}={v}")
    return ", ".join(parts)


def _extract_search_info(model, model_name):
    """
    从 RandomizedSearchCV 对象中提取 best_params_ 和 best_score_。
    若为普通 estimator，返回 (None, None)。
    """
    best_params = None
    best_score = None
    estimator = model
    if hasattr(model, "best_estimator_"):
        estimator = model.best_estimator_
        best_params = getattr(model, "best_params_", None)
        best_score = getattr(model, "best_score_", None)
    return estimator, best_params, best_score


def _bootstrap_ci_at_threshold(
    y_true, y_prob, opt_threshold, point_estimates, n_bootstraps=1000, random_state=None, groups=None
):
    """
    在固定最优阈值下计算 AUC/Recall/Specificity/Youden/Brier/Precision/F1/Accuracy 的 95% CI。

    - 提供 ``groups``（患者 ID 等）时：**cluster bootstrap**（与纵向 person-wave 设计一致）。
    - ``groups is None``：分层行级 bootstrap（遗留/非聚类数据）。
    n_bootstraps>=1000 以保证小样本队列 CI 稳定（审稿 P1）。
    """
    if random_state is None:
        random_state = RANDOM_SEED
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    boot_auc, boot_recall, boot_spec, boot_youden, boot_brier = [], [], [], [], []
    boot_precision, boot_f1, boot_acc = [], [], []
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)

    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError(f"groups length {len(groups)} != y_true length {n}")
        n_clust = len(np.unique(groups))

    for _ in range(n_bootstraps):
        if groups is not None:
            idx = cluster_bootstrap_indices_once(groups, rng, n_clusters_draw=n_clust)
        elif n_pos > 0 and n_neg > 0:
            i_pos = rng.choice(idx_pos, size=n_pos, replace=True)
            i_neg = rng.choice(idx_neg, size=n_neg, replace=True)
            idx = np.concatenate([i_pos, i_neg])
            rng.shuffle(idx)
        else:
            idx = rng.randint(0, n, n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        ypred = (yp >= opt_threshold).astype(int)
        try:
            boot_auc.append(roc_auc_score(yt, yp))
        except (ValueError, IndexError):
            continue
        boot_recall.append(recall_score(yt, ypred, zero_division=0))
        boot_precision.append(precision_score(yt, ypred, zero_division=0))
        boot_f1.append(f1_score(yt, ypred, zero_division=0))
        boot_acc.append(accuracy_score(yt, ypred))
        boot_brier.append(brier_score_loss(yt, yp))
        cm = confusion_matrix(yt, ypred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            boot_spec.append(spec)
            boot_youden.append(sens + spec - 1)
        else:
            # 与 evaluate_single_model 一致：无负类→0；全预测1→0；否则→1
            if (yt == 0).sum() == 0:
                spec = 0.0
            elif (ypred == 1).all():
                spec = 0.0
            else:
                spec = 1.0
            tp = int(((yt == 1) & (ypred == 1)).sum())
            fn = int(((yt == 1) & (ypred == 0)).sum())
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            boot_spec.append(spec)
            boot_youden.append(sens + spec - 1)

    def _ci(arr, point_est):
        if len(arr) < 10:
            return point_est, point_est
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    pe = point_estimates
    return {
        "AUC": _ci(boot_auc, pe["AUC"]) if boot_auc else (pe["AUC"], pe["AUC"]),
        "Recall": _ci(boot_recall, pe["Recall"]) if boot_recall else (pe["Recall"], pe["Recall"]),
        "Specificity": _ci(boot_spec, pe["Specificity"]) if boot_spec else (pe["Specificity"], pe["Specificity"]),
        "Youden_Index": _ci(boot_youden, pe["Youden_Index"]) if boot_youden else (pe["Youden_Index"], pe["Youden_Index"]),
        "Brier_Score": _ci(boot_brier, pe["Brier_Score"]) if boot_brier else (pe["Brier_Score"], pe["Brier_Score"]),
        "Precision": _ci(boot_precision, pe["Precision"]) if boot_precision else (pe["Precision"], pe["Precision"]),
        "F1": _ci(boot_f1, pe["F1"]) if boot_f1 else (pe["F1"], pe["F1"]),
        "Accuracy": _ci(boot_acc, pe["Accuracy"]) if boot_acc else (pe["Accuracy"], pe["Accuracy"]),
    }


def _is_imbalance_handled(model_name, best_params):
    """
    根据 best_params_ 判断该模型在调优阶段是否启用了权重补偿。
    用于向审稿人证明：在各自原生参数空间内进行了极致搜索。
    """
    if best_params is None or not isinstance(best_params, dict):
        return False
    if "clf__class_weight" in best_params and best_params["clf__class_weight"] == "balanced":
        return True
    if "clf__scale_pos_weight" in best_params:
        v = best_params["clf__scale_pos_weight"]
        if v is not None and (isinstance(v, (int, float)) and v > 1):
            return True
    if "clf__auto_class_weights" in best_params and best_params["clf__auto_class_weights"] == "Balanced":
        return True
    if "clf__is_unbalance" in best_params and best_params["clf__is_unbalance"] is True:
        return True
    return False


def evaluate_single_model(model, X_test, y_test, model_name="Model", opt_threshold=None, groups_test=None):
    """
    对单个模型执行完整评估：阈值移动、指标计算、校准度、调优信息。

    Parameters
    ----------
    model : estimator or RandomizedSearchCV
        已拟合的模型或 RandomizedSearchCV 对象
    X_test : array-like
        测试集特征
    y_test : array-like
        测试集真实标签
    model_name : str
        模型名称（用于日志）
    opt_threshold : float, optional
        最优决策阈值。若提供，则使用该阈值（应在验证集上确定）；否则在测试集上寻优（不推荐，会虚高灵敏度）。
    groups_test : array-like, optional
        与 y_test 等长的聚类标签（如参与者 ID）；用于 **cluster bootstrap** CI。CHARLS 主分析应传入。

    Returns
    -------
    dict
        包含主性能表与附录表所需的所有字段
    """
    estimator, best_params, best_score = _extract_search_info(model, model_name)
    y_true = np.asarray(y_test).astype(int)

    try:
        y_prob = estimator.predict_proba(X_test)
    except Exception as e:
        logger.warning(f"模型 {model_name} predict_proba 失败: {e}")
        return None

    y_prob = _get_proba_1d(y_prob)

    # 1. 阈值：优先使用验证集确定的阈值，避免在测试集上寻优导致虚高（审稿 P1）
    if opt_threshold is not None:
        opt_threshold = float(opt_threshold)
        y_pred_opt = (y_prob >= opt_threshold).astype(int)
        if len(np.unique(y_true)) >= 2 and len(np.unique(y_pred_opt)) >= 2:
            cm = confusion_matrix(y_true, y_pred_opt)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn + 1e-9)
                spec = tn / (tn + fp + 1e-9)
                youden_val = sens + spec - 1
            else:
                youden_val = 0.0
        else:
            youden_val = 0.0
    else:
        opt_threshold, youden_val = _find_optimal_threshold_youden(y_true, y_prob)
        y_pred_opt = (y_prob >= opt_threshold).astype(int)

    # 2. 指标计算（基于最优阈值）
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except (ValueError, IndexError):
        auc_val = 0.5  # 单类等边界情况兜底
    recall_val = recall_score(y_true, y_pred_opt, zero_division=0)
    precision_val = precision_score(y_true, y_pred_opt, zero_division=0)
    f1_val = f1_score(y_true, y_pred_opt, zero_division=0)
    acc_val = accuracy_score(y_true, y_pred_opt)

    # Specificity（边界：无负类、预测仅一类时正确计算）
    cm = confusion_matrix(y_true, y_pred_opt)
    if cm.shape == (2, 2) and np.unique(y_pred_opt).size >= 2 and np.unique(y_true).size >= 2:
        tn, fp, fn, tp = cm.ravel()
        specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # 无负类时 spec 未定义；全预测 1 且存在负类时 spec=0；全预测 0 时 spec=1
        if (y_true == 0).sum() == 0:
            specificity_val = 0.0
        elif (y_pred_opt == 1).all():
            specificity_val = 0.0
        else:
            specificity_val = 1.0

    # 3. 校准度（TRIPOD 强制输出 Brier Score）
    brier_val = brier_score_loss(y_true, y_prob)

    # Recall_raw：0.5 阈值下的 Recall，作为对比（主指标为验证集优化阈值后的 Recall）
    y_pred_05 = (y_prob >= 0.5).astype(int)
    recall_raw_05 = recall_score(y_true, y_pred_05, zero_division=0)

    # 4. 调优信息
    best_params_str = _format_hyperparams(best_params) if best_params else ""
    best_cv_auc = float(best_score) if best_score is not None else np.nan

    # 5. 是否在调优阶段启用权重补偿（Imbalance_Handled）
    imbalance_handled = _is_imbalance_handled(model_name, best_params)

    # 6. Bootstrap 95% CI（基于最优阈值；有 groups_test 时为 cluster bootstrap）
    pe = {
        "AUC": auc_val,
        "Recall": recall_val,
        "Specificity": specificity_val,
        "Youden_Index": youden_val,
        "Brier_Score": brier_val,
        "Precision": precision_val,
        "F1": f1_val,
        "Accuracy": acc_val,
    }
    ci_dict = _bootstrap_ci_at_threshold(
        y_true, y_prob, opt_threshold, pe, n_bootstraps=1000, random_state=RANDOM_SEED, groups=groups_test
    )

    def _fmt_ci(val, low, high):
        return f"{val:.4f} (95% CI: {low:.4f}, {high:.4f})"

    return {
        "Model": model_name,
        "Imbalance_Handled": imbalance_handled,
        "AUC": auc_val,
        "AUC_95CI": _fmt_ci(auc_val, ci_dict["AUC"][0], ci_dict["AUC"][1]),
        "Optimal_Threshold": opt_threshold,
        "Recall": recall_val,
        "Recall_95CI": _fmt_ci(recall_val, ci_dict["Recall"][0], ci_dict["Recall"][1]),
        "Specificity": specificity_val,
        "Specificity_95CI": _fmt_ci(specificity_val, ci_dict["Specificity"][0], ci_dict["Specificity"][1]),
        "Youden_Index": youden_val,
        "Youden_95CI": _fmt_ci(youden_val, ci_dict["Youden_Index"][0], ci_dict["Youden_Index"][1]),
        "Brier_Score": brier_val,
        "Brier_95CI": _fmt_ci(brier_val, ci_dict["Brier_Score"][0], ci_dict["Brier_Score"][1]),
        "Precision": precision_val,
        "Precision_95CI": _fmt_ci(precision_val, ci_dict["Precision"][0], ci_dict["Precision"][1]),
        "F1": f1_val,
        "F1_95CI": _fmt_ci(f1_val, ci_dict["F1"][0], ci_dict["F1"][1]),
        "Accuracy": acc_val,
        "Accuracy_95CI": _fmt_ci(acc_val, ci_dict["Accuracy"][0], ci_dict["Accuracy"][1]),
        "Best_CV_AUC": best_cv_auc,
        "Best_Params_Str": best_params_str,
        "Recall_raw_0.5": recall_raw_05,
        "y_prob": y_prob,
        "y_true": y_true,
    }


def evaluate_and_report(
    models_dict,
    X_test,
    y_test,
    cohort_label="Cohort",
    output_dir=None,
    thresholds_dict=None,
    groups_test=None,
):
    """
    对给定队列中所有模型执行评估并生成报告。

    Parameters
    ----------
    models_dict : dict
        模型名称 -> 已拟合模型（或 RandomizedSearchCV 对象）
    X_test : array-like or DataFrame
        测试集特征
    y_test : array-like
        测试集真实标签
    cohort_label : str
        队列标签（如 "A", "B", "C"）
    output_dir : str, optional
        输出目录；若为 None 则不保存文件
    groups_test : array-like, optional
        与 y_test 等长的参与者 ID；传入时 CPM 的 Bootstrap CI 使用 **cluster bootstrap**。

    Returns
    -------
    tuple
        (df_main, df_appendix, results): 主性能表、附录参数表、原始结果列表
    """
    X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
    num_cols = X_test.select_dtypes(include=[np.number])
    if num_cols.shape[1] > 0:
        X_test = X_test[num_cols.columns]
    if X_test.empty or len(X_test) == 0:
        logger.error(f"队列 {cohort_label} 测试集为空。")
        return None, None, []

    results = []
    thresholds_dict = thresholds_dict or {}
    for name, model in models_dict.items():
        opt_t = thresholds_dict.get(name)
        res = evaluate_single_model(
            model, X_test, y_test, model_name=name, opt_threshold=opt_t, groups_test=groups_test
        )
        if res is not None:
            results.append(res)

    if not results:
        logger.error(f"队列 {cohort_label} 无有效评估结果。")
        return None, None, []

    # 主性能表 (Table 2)，主指标为验证集优化阈值；Recall_raw_0.5 作对比
    # 含 Precision/F1/Accuracy 以兼容 draw_performance_radar
    df_main = pd.DataFrame(
        [
            {
                "Model": r["Model"],
                "AUC": r["AUC"],
                "AUC_95CI": r.get("AUC_95CI", ""),
                "Optimal_Threshold": r["Optimal_Threshold"],
                "Recall": r["Recall"],
                "Recall_95CI": r.get("Recall_95CI", ""),
                "Recall_raw_0.5": r.get("Recall_raw_0.5", np.nan),
                "Specificity": r["Specificity"],
                "Specificity_95CI": r.get("Specificity_95CI", ""),
                "Youden_Index": r["Youden_Index"],
                "Youden_95CI": r.get("Youden_95CI", ""),
                "Brier_Score": r["Brier_Score"],
                "Brier_95CI": r.get("Brier_95CI", ""),
                "Precision": r.get("Precision", np.nan),
                "Precision_95CI": r.get("Precision_95CI", ""),
                "F1": r.get("F1", np.nan),
                "F1_95CI": r.get("F1_95CI", ""),
                "Accuracy": r.get("Accuracy", np.nan),
                "Accuracy_95CI": r.get("Accuracy_95CI", ""),
                "Imbalance_Handled": r.get("Imbalance_Handled", False),
            }
            for r in results
        ]
    )

    # 附录参数表，含 Imbalance_Handled
    df_appendix = pd.DataFrame(
        [
            {
                "Model": r["Model"],
                "Best_CV_AUC": r["Best_CV_AUC"],
                "Best_Params_Str": r["Best_Params_Str"],
                "Imbalance_Handled": r.get("Imbalance_Handled", False),
            }
            for r in results
        ]
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        prefix = f"table2_{cohort_label}" if cohort_label else "table2"
        df_main.to_csv(os.path.join(output_dir, f"{prefix}_main_performance.csv"), index=False, encoding="utf-8-sig")
        df_appendix.to_csv(os.path.join(output_dir, f"{prefix}_appendix_params.csv"), index=False, encoding="utf-8-sig")
        logger.info(f"队列 {cohort_label} 主表与附录已保存至 {output_dir}")

    return df_main, df_appendix, results


def plot_calibration_curves_1x3(
    results_a,
    results_b,
    results_c,
    output_path=None,
):
    """
    生成 2x3 子图：上排校准曲线，下排预测概率分布直方图（低流行率下判断预测偏向）。
    """
    if not results_a or not results_b or not results_c:
        logger.warning("至少一个队列无结果，跳过校准曲线绘制。")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax_cal = axes[0]
    ax_hist = axes[1]

    def _plot_one(ax, results, title):
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        for i, r in enumerate(results):
            y_true = r["y_true"]
            y_prob = r["y_prob"]
            try:
                prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
                ax.plot(prob_pred, prob_true, marker="o", linewidth=1, label=r["Model"], color=colors[i % 20])
            except Exception as e:
                logger.warning(f"校准曲线 {r['Model']} 绘制失败: {e}")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=7, ncol=1)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    def _plot_hist(ax, results, title):
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        for i, r in enumerate(results):
            y_prob = np.asarray(r["y_prob"]).ravel()
            if len(y_prob) > 0:
                ax.hist(y_prob, bins=30, alpha=0.4, label=r["Model"], color=colors[i % 20], range=(0, 1))
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{title} (Prob. Distribution)")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlim(0, 1)

    _plot_one(ax_cal[0], results_a, "Cohort A (Healthy)")
    _plot_one(ax_cal[1], results_b, "Cohort B (Depression-only)")
    _plot_one(ax_cal[2], results_c, "Cohort C (Cognition-impaired-only)")
    _plot_hist(ax_hist[0], results_a, "Cohort A")
    _plot_hist(ax_hist[1], results_b, "Cohort B")
    _plot_hist(ax_hist[2], results_c, "Cohort C")

    plt.tight_layout()
    if output_path:
        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(output_path, bbox_inches=BBOX_INCHES, dpi=300)
        plt.close()
        logger.info(f"校准曲线已保存: {output_path}")
    else:
        plt.show()


def run_full_cpm_pipeline(
    models_A,
    models_B,
    models_C,
    X_test_A,
    y_test_A,
    X_test_B,
    y_test_B,
    X_test_C,
    y_test_C,
    output_dir="cpm_evaluation_results",
    thresholds_A=None,
    thresholds_B=None,
    thresholds_C=None,
    groups_test_A=None,
    groups_test_B=None,
    groups_test_C=None,
):
    """
    完整流水线：三队列评估 + 主表 + 附录 + 校准曲线。

    Parameters
    ----------
    models_A, models_B, models_C : dict
        各队列的模型字典 {模型名: 已拟合模型}
    X_test_*, y_test_* : array-like
        各队列的测试集特征与标签
    output_dir : str
        输出根目录

    Returns
    -------
    dict
        {
            "main_tables": {"A": df_main_A, "B": df_main_B, "C": df_main_C},
            "appendix_tables": {"A": df_app_A, "B": df_app_B, "C": df_app_C},
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    df_main_a, df_app_a, res_a = evaluate_and_report(
        models_A,
        X_test_A,
        y_test_A,
        cohort_label="A",
        output_dir=output_dir,
        thresholds_dict=thresholds_A,
        groups_test=groups_test_A,
    )
    df_main_b, df_app_b, res_b = evaluate_and_report(
        models_B,
        X_test_B,
        y_test_B,
        cohort_label="B",
        output_dir=output_dir,
        thresholds_dict=thresholds_B,
        groups_test=groups_test_B,
    )
    df_main_c, df_app_c, res_c = evaluate_and_report(
        models_C,
        X_test_C,
        y_test_C,
        cohort_label="C",
        output_dir=output_dir,
        thresholds_dict=thresholds_C,
        groups_test=groups_test_C,
    )

    plot_calibration_curves_1x3(
        res_a or [], res_b or [], res_c or [],
        output_path=os.path.join(output_dir, "calibration_curves_1x3.png"),
    )

    if df_main_a is not None and df_main_b is not None and df_main_c is not None:
        df_main_a = df_main_a.copy()
        df_main_b = df_main_b.copy()
        df_main_c = df_main_c.copy()
        df_main_a["Cohort"] = "A"
        df_main_b["Cohort"] = "B"
        df_main_c["Cohort"] = "C"
        df_combined = pd.concat([df_main_a, df_main_b, df_main_c], ignore_index=True)
        df_combined.to_csv(
            os.path.join(output_dir, "table2_main_performance_combined.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        logger.info("主表合并已保存: table2_main_performance_combined.csv")

    return {
        "main_tables": {"A": df_main_a, "B": df_main_b, "C": df_main_c},
        "appendix_tables": {"A": df_app_a, "B": df_app_b, "C": df_app_c},
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data.charls_complete_preprocessing import preprocess_charls_data
    from modeling.charls_model_comparison import compare_models
    from utils.charls_feature_lists import get_exclude_cols
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.base import clone

    logger.info(">>> 加载数据并划分训练/测试集...")
    df_clean = preprocess_charls_data("CHARLS.csv", age_min=60, cesd_cutoff=10, cognition_cutoff=10)
    if df_clean is None:
        logger.error("预处理失败，退出。")
        sys.exit(1)

    target_col = "is_comorbidity_next"
    output_dir = "cpm_evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    models_A, models_B, models_C = {}, {}, {}
    X_te_A, y_te_A = None, None
    X_te_B, y_te_B = None, None
    X_te_C, y_te_C = None, None
    grp_te_A = grp_te_B = grp_te_C = None
    thresh_A, thresh_B, thresh_C = {}, {}, {}

    for cohort_name, baseline_val, models_dict, thresh_dict in [
        ("A", 0, models_A, thresh_A), ("B", 1, models_B, thresh_B), ("C", 2, models_C, thresh_C)
    ]:
        df_sub = df_clean[df_clean["baseline_group"] == baseline_val]
        if len(df_sub) < 50:
            logger.warning(f"队列 {cohort_name} 样本过少，跳过。")
            continue
        logger.info(f">>> 队列 {cohort_name}: 训练 compare_models...")
        result = compare_models(
            df_sub,
            output_dir=os.path.join(output_dir, f"cohort_{cohort_name}_prediction"),
            target_col=target_col,
            return_search_objects=True,
        )
        perf_df, models = result[0], result[1]
        search_objs = result[2] if len(result) > 2 else {}
        # 优先使用 RandomizedSearchCV 对象以填充附录表
        models = {k: search_objs.get(k, v) for k, v in models.items()}
        # 【TRIPOD Step 1】Internal CV 阈值已在 compare_models 中计算，优先使用
        if '_opt_threshold' in perf_df.columns:
            for _, row in perf_df.iterrows():
                mn = row.get('Model')
                if mn and pd.notna(row.get('_opt_threshold')):
                    thresh_dict[mn] = float(row['_opt_threshold'])
        exclude_sub = get_exclude_cols(df_sub, target_col=target_col)
        W_sub = [c for c in df_sub.columns if c not in exclude_sub]
        X_sub = df_sub[W_sub].select_dtypes(include=[np.number])
        y_sub = df_sub[target_col].astype(int)
        gss_sub = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
        tr_idx, te_idx = next(gss_sub.split(X_sub, y_sub, groups=df_sub["ID"]))
        X_tr = X_sub.iloc[tr_idx]
        y_tr = y_sub.iloc[tr_idx]
        X_te = X_sub.iloc[te_idx]
        y_te = y_sub.iloc[te_idx]
        g_te = df_sub.iloc[te_idx]["ID"].to_numpy()
        common_cols = [c for c in X_te.columns if c in X_sub.columns]
        X_te = X_te[common_cols]

        # 对未从 compare_models 获得阈值的模型，从验证集计算（兜底）
        missing = [n for n in models.keys() if n not in thresh_dict]
        if missing:
            gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED + 1)
            tr_inner_idx, val_idx = next(gss_val.split(X_tr, y_tr, groups=df_sub.iloc[tr_idx]["ID"]))
            X_tr_inner = X_tr.iloc[tr_inner_idx]
            y_tr_inner = y_tr.iloc[tr_inner_idx]
            X_val = X_tr.iloc[val_idx]
            y_val = y_tr.iloc[val_idx]
            for name in missing:
                try:
                    model = models.get(name)
                    if model is None:
                        continue
                    est = search_objs.get(name, model)
                    if hasattr(est, "best_estimator_"):
                        m = clone(est.best_estimator_)
                    else:
                        m = clone(model)
                    m.fit(X_tr_inner, y_tr_inner)
                    t = _compute_optimal_threshold_from_validation(m, X_val, y_val, model_name=name)
                    if t is not None:
                        thresh_dict[name] = t
                except Exception as e:
                    logger.debug(f"队列 {cohort_name} 模型 {name} 阈值计算跳过: {e}")

        models_dict.update(models)

        if cohort_name == "A":
            X_te_A, y_te_A, grp_te_A = X_te, y_te, g_te
        elif cohort_name == "B":
            X_te_B, y_te_B, grp_te_B = X_te, y_te, g_te
        else:
            X_te_C, y_te_C, grp_te_C = X_te, y_te, g_te

    # 完整流水线：评估 + 主表 + 附录 + 1x3 校准曲线（使用验证集确定的阈值）
    if models_A and models_B and models_C and X_te_A is not None and X_te_B is not None and X_te_C is not None:
        run_full_cpm_pipeline(
            models_A, models_B, models_C,
            X_te_A, y_te_A, X_te_B, y_te_B, X_te_C, y_te_C,
            output_dir=output_dir,
            thresholds_A=thresh_A, thresholds_B=thresh_B, thresholds_C=thresh_C,
            groups_test_A=grp_te_A,
            groups_test_B=grp_te_B,
            groups_test_C=grp_te_C,
        )
    else:
        for cohort_name, models_dict, X_te, y_te, thresh in [
            ("A", models_A, X_te_A, y_te_A, thresh_A),
            ("B", models_B, X_te_B, y_te_B, thresh_B),
            ("C", models_C, X_te_C, y_te_C, thresh_C),
        ]:
            if models_dict and X_te is not None:
                _g = {"A": grp_te_A, "B": grp_te_B, "C": grp_te_C}[cohort_name]
                evaluate_and_report(
                    models_dict,
                    X_te,
                    y_te,
                    cohort_label=cohort_name,
                    output_dir=output_dir,
                    thresholds_dict=thresh,
                    groups_test=_g,
                )

    logger.info(">>> CPM 评估流水线完成。")
