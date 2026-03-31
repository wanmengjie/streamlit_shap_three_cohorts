import os
import re
import random
import logging
import pandas as pd
import numpy as np
import matplotlib
import warnings
import shutil
import time
from config import *  # COLS_TO_DROP, COHORT_STEP_DIRS, COHORT_*_DIR, N_MULTIPLE_IMPUTATIONS, IMPUTED_MI_DIR, USE_RUBIN_POOLING 等

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=UserWarning, module="loky")
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def safe_remove_dir(path):
    """解决 Windows 下文件占用导致的权限错误"""
    if not os.path.exists(path):
        return
    for _ in range(3):
        try:
            shutil.rmtree(path)
            break
        except OSError:
            time.sleep(1)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("LIU_JUE_FINAL_FIXED.log", encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ... 后续导入保持不变 ...
# 运行时可另开终端执行: Get-Content LIU_JUE_FINAL_FIXED.log -Wait -Tail 20  查看实时日志

from data.charls_complete_preprocessing import preprocess_charls_data, reapply_cohort_definition
from data.charls_table1_stats import generate_baseline_table
from modeling.charls_model_comparison import (
    compare_models,
    CPM_MIN_RECALL_THRESHOLD,
    parse_table2_recall_value,
    rewrite_model_performance_full_csv,
    select_champion_from_df_main,
    select_champion_from_perf_df,
)
from interpretability.charls_shap_analysis import run_shap_analysis_v2
from evaluation.charls_clinical_evaluation import run_clinical_evaluation
from evaluation.charls_clinical_decision_support import run_clinical_decision_support
from evaluation.charls_subgroup_analysis import run_subgroup_analysis, draw_performance_radar
from causal.charls_recalculate_causal_impact import get_estimate_causal_impact, cleanup_temp_cat_dirs
from evaluation.charls_imputation_audit import run_imputation_sensitivity_preprocessed
from evaluation.charls_sensitivity_analysis import run_sensitivity_analysis
from scripts.run_sensitivity_scenarios import run_sensitivity_scenarios_analysis, _get_train_subset
from scripts.run_multi_exposure_causal import prepare_exposures, run_multi_exposure_analysis
from viz.draw_attrition_flowchart import draw_flowchart
from viz.charls_extra_figures import draw_all_extra_figures
from viz.draw_conceptual_framework import draw_conceptual_framework
from viz.draw_roc_combined import draw_roc_combined
from evaluation.charls_external_validation import run_external_validation
from interpretability.charls_shap_stratified import run_stratified_shap, run_shap_interaction
from evaluation.charls_dose_response import run_dose_response
from evaluation.charls_ite_validation import run_ite_stratified_validation
from causal.charls_causal_methods_comparison import run_causal_methods_comparison
from evaluation.charls_temporal_analysis import run_temporal_analysis
from evaluation.charls_nomogram import run_nomogram
from modeling.charls_cpm_evaluation import evaluate_and_report

def _path(path_dir, step_key):
    """队列（Cohort）内步骤子路径：path_dir/step_subdir"""
    return os.path.join(path_dir, COHORT_STEP_DIRS[step_key])


def _parse_ate_ci_summary_txt(txt_path):
    """从 causal 目录下 ATE_CI_summary_<treatment>.txt 解析点估计与 95% CI。"""
    if not os.path.isfile(txt_path):
        return None
    try:
        with open(txt_path, encoding='utf-8') as f:
            text = f.read()
        m = re.search(r'ATE \(point estimate\):\s*([\-\d.eE+]+)', text)
        mci = re.search(r'95%\s*CI:\s*\(([\-\d.eE+]+)\s*,\s*([\-\d.eE+]+)\)', text)
        if m and mci:
            return float(m.group(1)), float(mci.group(1)), float(mci.group(2))
    except Exception:
        pass
    return None


def _load_skipped_cohort_metrics(path_dir, cohort_id, df_sub):
    """
    续跑时跳过某队列：从磁盘已有结果恢复 (incidence, (ate, lb, ub), champion_auc)，供汇总图与后续步骤使用。
    """
    if df_sub is None or len(df_sub) == 0:
        return 0.0, (np.nan, np.nan, np.nan), 0.0
    inc = float(df_sub[TARGET_COL].mean())
    pred_dir = os.path.join(path_dir, COHORT_STEP_DIRS['prediction'])
    t2 = os.path.join(pred_dir, f'table2_{cohort_id}_main_performance.csv')
    auc_val = 0.0
    if os.path.isfile(t2):
        try:
            tdf = pd.read_csv(t2, encoding='utf-8-sig')
            if 'AUC' in tdf.columns and len(tdf) > 0:
                row = tdf.sort_values('AUC', ascending=False).iloc[0]
                av = row['AUC']
                if isinstance(av, str):
                    av = float(str(av).split()[0])
                auc_val = float(av)
        except Exception as ex:
            logger.warning('续跑：读取 %s 的 AUC 失败: %s', t2, ex)
    causal_dir = os.path.join(path_dir, COHORT_STEP_DIRS['causal'])
    ate_txt = os.path.join(causal_dir, f'ATE_CI_summary_{TREATMENT_COL}.txt')
    parsed = _parse_ate_ci_summary_txt(ate_txt)
    if parsed:
        ate, lb, ub = parsed
    else:
        ate, lb, ub = np.nan, np.nan, np.nan
        logger.warning('续跑：未解析 %s，汇总图 ATE 记为 NaN（非真实零效应）', ate_txt)
    return inc, (ate, lb, ub), auc_val


def _read_champion_auc_from_saved_table2(cohort_id: str) -> float:
    """从已有 table2 读取冠军行 AUC（与 _load_skipped_cohort_metrics 一致：按 AUC 降序首行），供 causal_only 汇总图使用。"""
    label = {'A': COHORT_A_DIR, 'B': COHORT_B_DIR, 'C': COHORT_C_DIR}.get(cohort_id)
    if not label:
        return np.nan
    t2 = os.path.join(label, COHORT_STEP_DIRS['prediction'], f'table2_{cohort_id}_main_performance.csv')
    if not os.path.isfile(t2):
        return np.nan
    try:
        tdf = pd.read_csv(t2, encoding='utf-8-sig')
        if 'AUC' not in tdf.columns or len(tdf) == 0:
            return np.nan
        row = tdf.sort_values('AUC', ascending=False).iloc[0]
        av = row['AUC']
        if isinstance(av, str):
            av = float(str(av).split()[0])
        return float(av)
    except Exception:
        return np.nan


def _maybe_run_npj_imputation_first():
    """
    若 config.RUN_IMPUTATION_BEFORE_MAIN 且 USE_IMPUTED_DATA：先跑 archive 插补（NRMSE 选优 + full + MI），
    再生成 step1_imputed_full / m1..mN。

    Returns
    -------
    str
        'skipped' — 未执行前置插补；'ok' — 插补成功并应已刷新磁盘上的 step1_imputed_full；
        'failed' — 插补异常或返回 None，主流程可能仍读取旧的 step1_imputed_full。
    """
    run_first = getattr(__import__('config'), 'RUN_IMPUTATION_BEFORE_MAIN', False)
    if not run_first:
        return 'skipped'
    if not USE_IMPUTED_DATA:
        logger.info("RUN_IMPUTATION_BEFORE_MAIN=True 但 USE_IMPUTED_DATA=False，跳过前置插补")
        return 'skipped'
    root = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.join(root, getattr(__import__('config'), 'IMPUTATION_OUTPUT_ROOT', 'imputation_npj_results'))
    pre_csv = os.path.join(root, 'preprocessed_data', 'CHARLS_final_preprocessed.csv')
    raw_path = RAW_DATA_PATH if os.path.isabs(RAW_DATA_PATH) else os.path.join(root, RAW_DATA_PATH)
    # npj 插补要求数据含 baseline_group；原始 CHARLS.csv 无此列 → 必须用预处理表或先跑预处理
    if os.path.isfile(pre_csv):
        data_path = pre_csv
        logger.info(f">>> 前置插补输入: {pre_csv}（含 baseline_group）")
    elif os.path.isfile(raw_path):
        logger.warning(">>> 未找到预处理表，将先用 preprocess_charls_data 从 RAW 生成 CHARLS_final_preprocessed.csv …")
        _dfp = preprocess_charls_data(raw_path, age_min=AGE_MIN, cesd_cutoff=CESD_CUTOFF,
                                      cognition_cutoff=COGNITION_CUTOFF, write_output=True)
        data_path = pre_csv if (_dfp is not None and os.path.isfile(pre_csv)) else None
        if data_path is None:
            logger.error(">>> 预处理未成功写出 preprocessed_data/CHARLS_final_preprocessed.csv，前置插补可能失败")
    else:
        data_path = None
        logger.warning(f">>> 未找到 {raw_path}，插补脚本将使用其内部 load_data 默认逻辑")
    logger.info("=" * 60)
    logger.info(">>> 前置插补已开启（RUN_IMPUTATION_BEFORE_MAIN=True）")
    logger.info(">>> 将运行 NRMSE 选优 + 全量插补 + 多重插补；通常需较长时间，请耐心等待")
    logger.info(f">>> 输出目录: {out_root}")
    logger.info("=" * 60)
    try:
        import archive.charls_imputation_npj_style as _npj
        # 与主流程 config 中多重插补份数对齐
        n_mi = int(getattr(__import__('config'), 'N_MULTIPLE_IMPUTATIONS', 5))
        _npj.N_MULTIPLE_IMPUTATIONS = n_mi
        res = _npj.run_full_experiment(data_path=data_path, output_dir=out_root)
        if res is None:
            logger.error("前置插补返回 None，请检查日志；主流程将尝试读取已有 step1_imputed_full（若存在）——**可能为旧文件**")
            return 'failed'
        logger.info(f">>> 前置插补完成: 最优方法={res.get('best_method')}, NRMSE={res.get('best_nrmse')}；已写入最新 step1_imputed_full")
        return 'ok'
    except Exception as ex:
        logger.error(f"前置插补失败: {ex}", exc_info=True)
        logger.warning("将继续主流程；若 step1_imputed_full 不存在将回退预处理数据（若存在旧插补文件则仍在用旧插补）")
        return 'failed'

def _df_main_authoritative_for_champion(df_main):
    """Code freeze：仅当 Table2 主表存在且含 Recall/AUC 时，以 df_main 为唯一冠军来源。"""
    return (
        df_main is not None
        and len(df_main) > 0
        and 'Recall' in df_main.columns
        and 'AUC' in df_main.columns
        and 'Model' in df_main.columns
    )


def _select_champion(df_main, perf_df, models):
    """
    **Code freeze — Table 2 (df_main) 唯一权威**（若存在）：
    Recall（Youden 最优阈值下测试集灵敏度）>= CPM_MIN_RECALL_THRESHOLD 中取测试集 AUC 最高；
    无人达标则 Table2 内全局 AUC 最高 + 醒目 WARNING。
    同一 `models[champ_name]` 经可选校准后用于 SHAP / DCA / joblib；`rewrite_model_performance_full_csv` 同步首行。

    仅当 df_main 缺失或无法解析时，才 **唯一例外** 回退 perf_df（并 ERROR 级日志）。
    """
    keys = list(models.keys())
    if _df_main_authoritative_for_champion(df_main):
        br_main, used_fb_main = select_champion_from_df_main(df_main, keys, log=logger)
        if br_main is None or br_main['Model'] not in models:
            raise ValueError(
                'CPM Code freeze：已生成 Table2(df_main)，但无法从中解析冠军（检查 Model 是否与训练模型名一致）。'
                '禁止静默回退 perf_df。'
            )
        champ_name = br_main['Model']
        champion_auc = float(br_main['AUC'])
        r_gate = parse_table2_recall_value(br_main.get('Recall'))
        if used_fb_main:
            logger.warning(
                '*** CPM WARNING: Table2 中无一模型 Recall >= %.2f（Youden 阈值下）；'
                '退回 Table2 内全局 AUC 最高；校准/SHAP/DCA/joblib 均基于: %s ***',
                CPM_MIN_RECALL_THRESHOLD,
                champ_name,
            )
        logger.info(
            'CPM 冠军（Table2 唯一权威）: %s (AUC=%.4f, Recall=%s)',
            champ_name,
            champion_auc,
            f'{r_gate:.4f}' if pd.notna(r_gate) else 'nan',
        )
        top = df_main[df_main['Model'] == champ_name]
        rest = df_main[df_main['Model'] != champ_name].sort_values(
            'AUC', ascending=False, na_position='last'
        )
        perf_for_radar = pd.concat([top, rest], ignore_index=True)
        return models[champ_name], perf_for_radar, champion_auc, champ_name

    logger.error(
        'CPM：无权威 Table2(df_main)，回退 perf_df 选冠军（evaluate_and_report 可能失败；'
        '投稿前请确保 table2 主表可用）。'
    )
    best_row, used_recall_fallback = select_champion_from_perf_df(perf_df, model_names=keys)
    if best_row is None:
        raise ValueError('select_champion_from_perf_df 未返回冠军行，无法继续 CPM 后续步骤')
    champ_name = best_row['Model']
    if champ_name not in models:
        raise ValueError(f'冠军模型 {champ_name} 不在 models 中，无法继续')
    champion_auc = best_row.get('AUC_raw', best_row.get('AUC', 0.0))
    if isinstance(champion_auc, str):
        champion_auc = float(str(champion_auc).split(' ')[0])
    r_gate = best_row.get('Recall_at_opt_t_raw', np.nan)
    if used_recall_fallback:
        logger.warning(
            '*** CPM WARNING (_select_champion 回退 perf_df): 无模型 Recall@Youden >= %.2f；'
            '已退回全局测试集 AUC 最高: %s (Recall_at_opt_t=%s) ***',
            CPM_MIN_RECALL_THRESHOLD,
            champ_name,
            r_gate,
        )
    else:
        logger.info(
            'CPM 冠军（df_main 不可用，使用 perf_df）: %s (AUC=%.4f)',
            champ_name,
            float(champion_auc),
        )
    pf = perf_df.copy()
    pf['_champ'] = pf['Model'] == champ_name
    sort_col = 'AUC_raw' if 'AUC_raw' in pf.columns else 'AUC'
    perf_for_radar = pf.sort_values(['_champ', sort_col], ascending=[False, False]).drop(columns=['_champ'])
    return models[champ_name], perf_for_radar, champion_auc, champ_name

def _save_cpm_champion_outputs(best_model, X_test, y_test, champ_name, pred_dir):
    """保存 CPM 冠军的 ROC、joblib、model_complexity_efficiency，集中一处便于维护。"""
    import json
    import joblib
    try:
        y_prob = best_model.predict_proba(X_test)[:, 1]
        with open(os.path.join(pred_dir, 'roc_data.json'), 'w', encoding='utf-8') as f:
            json.dump({'y_true': y_test.tolist(), 'y_prob': y_prob.tolist()}, f)
        logger.info(f"ROC 数据已保存至 {pred_dir}/roc_data.json")
    except Exception as ex:
        logger.debug(f"ROC 保存跳过: {ex}")
    try:
        joblib.dump(best_model, os.path.join(pred_dir, 'champion_model.joblib'))
        n_feat = X_test.shape[1]
        pred_time_ms = np.nan
        try:
            import time as _t
            t0 = _t.time()
            _ = best_model.predict_proba(X_test.head(100))
            pred_time_ms = (_t.time() - t0) * 1000 / min(100, len(X_test))
        except Exception:
            pass
        _m = getattr(best_model, 'estimator', best_model)
        depth_str = ''
        if hasattr(_m, 'named_steps') and 'clf' in _m.named_steps:
            clf = _m.named_steps['clf']
            if hasattr(clf, 'max_depth') and clf.max_depth is not None:
                depth_str = f', max_depth={clf.max_depth}'
            if hasattr(clf, 'n_estimators'):
                depth_str += f', n_estimators={getattr(clf, "n_estimators", "N/A")}'
        if not depth_str:
            depth_str = 'see model object'
        feat_str = ', '.join(X_test.columns.tolist()) if hasattr(X_test, 'columns') else f'<{n_feat} cols>'
        with open(os.path.join(pred_dir, 'model_complexity_efficiency.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Champion Model: {champ_name}\n")
            f.write(f"n_features: {n_feat}\n")
            f.write(f"features: {feat_str}\n")
            f.write(f"prediction_time_per_100_samples_ms: {pred_time_ms:.2f}\n")
            f.write(f"hyperparams: {depth_str}\n")
        logger.info(f"CPM 冠军模型已保存: {pred_dir}/champion_model.joblib")
    except Exception as ex:
        logger.debug(f"冠军模型保存跳过: {ex}")

def run_cohort_protocol(df_sub, path_dir, cohort_id, cohort_desc, pool_only=False, df_for_causal=None, causal_only=False):
    """
    统一三队列（Cohort A/B/C）分析流程；各队列步骤一致。
    步骤：1.预测打擂+CPM评估 2.SHAP归因 3.因果分析 4.临床评价与决策支持 5.亚组分析
    pool_only=True 时仅运行预测+因果，用于多重插补 Rubin 合并（跳过 SHAP/临床评价/亚组等）
    causal_only=True 时跳过步骤 1–2 与 4 中临床/剂量等，仅跑步骤 3 及因果相关扩展（ITE/亚组等）；由 config.MAIN_COHORT_CAUSAL_ONLY 控制主入口。

    df_sub : DataFrame
        **监督学习**用表：须保留真实缺失；由 compare_models 内 Pipeline（IterativeImputer）在 CV 训练折 fit。
        causal_only 时本参数仅用于占位，不会触发 compare_models。
    df_for_causal : DataFrame, optional
        **因果/时序/剂量**用表；若主流程使用 bulk 插补，此处传入同队列的插补后切片；为 None 时与 df_sub 相同。

    cohort_id: 'A'|'B'|'C'（与 baseline_group 0/1/2 对应，勿称 axis）
    """
    df_causal_use = df_for_causal if df_for_causal is not None else df_sub
    logger.info("\n" + "#"*70 + f"\n>>> 启动【Cohort {cohort_id}】：{cohort_desc}\n" + "#"*70)
    if not pool_only and not causal_only:
        safe_remove_dir(path_dir)
    os.makedirs(path_dir, exist_ok=True)
    pred_dir = _path(path_dir, 'prediction')

    champion_auc = np.nan

    # 1. 预测打擂 + CPM 评估（验证集 Youden 阈值，AUC/Recall 更优）
    if not causal_only:
        result = compare_models(
            df_sub,
            output_dir=pred_dir,
            target_col=TARGET_COL,
            return_search_objects=True,
            return_xy_test=True,
        )
        perf_df, models, _search_objs, X_test, y_test, X_train, y_train, grp_train, grp_test = result
        # CPM 评估：使用 Internal CV 确定的 _opt_threshold，在测试集上报告 TRIPOD 合规指标
        thresholds_dict = {}
        if '_opt_threshold' in perf_df.columns:
            thresholds_dict = {
                row['Model']: float(row['_opt_threshold'])
                for _, row in perf_df.iterrows()
                if pd.notna(row.get('_opt_threshold'))
            }
        df_main, df_app, _ = evaluate_and_report(
            models,
            X_test,
            y_test,
            cohort_label=cohort_id,
            output_dir=pred_dir,
            thresholds_dict=thresholds_dict or None,
            groups_test=grp_test,
        )
        best_model, perf_for_radar, champion_auc, champ_name = _select_champion(df_main, perf_df, models)
        try:
            rewrite_model_performance_full_csv(pred_dir, TARGET_COL, perf_df, champ_name)
        except Exception as ex:
            logger.warning('model_performance_full 按冠军重排保存跳过: %s', ex)
        try:
            from config import CALIBRATE_CHAMPION_PROBA
        except ImportError:
            CALIBRATE_CHAMPION_PROBA = False
        if CALIBRATE_CHAMPION_PROBA:
            import numpy as _np
            from sklearn.base import clone
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import GroupKFold

            n_g = len(_np.unique(grp_train))
            n_splits = min(5, n_g)
            if n_splits >= 2:
                try:
                    cal = CalibratedClassifierCV(
                        estimator=clone(best_model), method='sigmoid', cv=GroupKFold(n_splits=n_splits)
                    )
                    cal.fit(X_train, y_train, groups=grp_train)
                    best_model = cal
                    logger.info(
                        'CALIBRATE_CHAMPION_PROBA=True: champion wrapped in CalibratedClassifierCV '
                        '(sigmoid, GroupKFold n_splits=%s)',
                        n_splits,
                    )
                except Exception as ex:
                    logger.warning('CALIBRATE_CHAMPION_PROBA failed, using uncalibrated champion: %s', ex)
            else:
                logger.warning(
                    'CALIBRATE_CHAMPION_PROBA skipped: need at least 2 ID groups for GroupKFold (got %s)',
                    n_g,
                )
        if not pool_only:
            draw_performance_radar(perf_for_radar, output_dir=pred_dir)
            _save_cpm_champion_outputs(best_model, X_test, y_test, champ_name, pred_dir)
    else:
        logger.info(
            'MAIN_COHORT_CAUSAL_ONLY：跳过本队列 CPM compare_models、SHAP 与冠军模型保存；'
            '仅写入因果及下游（ITE/时序/敏感性/亚组等）。'
        )

    # 2. SHAP 归因（pool_only 或 causal_only 时跳过）
    if not pool_only and not causal_only:
        run_shap_analysis_v2(df_sub, model=best_model, output_dir=_path(path_dir, 'shap'), target_col=TARGET_COL)
        try:
            run_stratified_shap(df_sub, best_model, output_dir=_path(path_dir, 'shap_stratified'), target_col=TARGET_COL)
            run_shap_interaction(df_sub, best_model, output_dir=_path(path_dir, 'shap'), target_col=TARGET_COL)
        except Exception as ex:
            logger.warning(f"分层/交互 SHAP 跳过: {ex}")

    # 3. 因果分析（根据 config.CAUSAL_METHOD 使用 XLearner/TLearner/CausalForestDML）
    causal_dir = _path(path_dir, 'causal')
    estimate_causal = get_estimate_causal_impact()
    res_df, (ate, ate_lb, ate_ub) = estimate_causal(
        df_causal_use, treatment_col=TREATMENT_COL, output_dir=causal_dir)
    if res_df is None:
        ate, ate_lb, ate_ub = np.nan, np.nan, np.nan
    else:
        # res_df 非空但点估计或 CI 任一为 NaN/Inf：与「成功因果」区分，避免下游误用 ITE 或崩溃
        def _to_float_ci(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return np.nan

        _a = _to_float_ci(ate)
        _lb = _to_float_ci(ate_lb)
        _ub = _to_float_ci(ate_ub)
        _tri_ok = np.isfinite(_a) and np.isfinite(_lb) and np.isfinite(_ub)
        if not _tri_ok:
            bad = []
            if not np.isfinite(_a):
                bad.append('ATE')
            if not np.isfinite(_lb):
                bad.append('ATE_lb')
            if not np.isfinite(_ub):
                bad.append('ATE_ub')
            _cc = f'causal_impact_{TREATMENT_COL}'
            logger.warning(
                "【Cohort %s】因果引擎返回了 res_df，但 ATE 三元组无效（%s）。"
                "已移除列 `%s` 并将汇总用 (ATE, lb, ub) 置为 NaN；"
                "敏感性分析 / ITE 验证 / 列线图 / 亚组将自动跳过或走无因果列分支。",
                cohort_id,
                ', '.join(bad),
                _cc,
            )
            if _cc in res_df.columns:
                res_df = res_df.drop(columns=[_cc], errors='ignore')
            ate, ate_lb, ate_ub = np.nan, np.nan, np.nan
        else:
            ate, ate_lb, ate_ub = _a, _lb, _ub
    if res_df is not None and not pool_only:
        run_sensitivity_analysis(res_df, output_dir=_path(path_dir, 'sensitivity'), treatment_col=TREATMENT_COL)
        causal_col = f'causal_impact_{TREATMENT_COL}'
        if causal_col in res_df.columns:
            try:
                run_ite_stratified_validation(res_df, causal_col, TREATMENT_COL,
                                              output_dir=os.path.join(causal_dir, 'ite_validation'))
                run_nomogram(res_df, causal_col, TREATMENT_COL, output_dir=os.path.join(causal_dir, 'nomogram'))
            except Exception as ex:
                logger.warning(f"ITE/列线图跳过: {ex}")
        try:
            run_temporal_analysis(
                df_causal_use, TREATMENT_COL, output_dir=os.path.join(causal_dir, 'temporal'))
        except Exception as ex:
            logger.warning(f"时序分析跳过: {ex}")

    # 4. 临床评价与决策支持（pool_only 或 causal_only 时跳过；causal_only 下无 best_model）
    if not pool_only and not causal_only:
        run_clinical_evaluation(df_sub, model=best_model, output_dir=_path(path_dir, 'eval'), target_col=TARGET_COL)
        run_clinical_decision_support(df_sub, model=best_model, output_dir=_path(path_dir, 'decision'), target_col=TARGET_COL)
        try:
            _champ_path = os.path.join(pred_dir, 'champion_model.joblib')
            run_external_validation(
                df_sub,
                champion_model_path=_champ_path if os.path.isfile(_champ_path) else None,
                model=best_model,
                output_dir=_path(path_dir, 'external_validation'),
                target_col=TARGET_COL,
                cohort_id=cohort_id,
            )
        except Exception as ex:
            logger.warning(f"外部验证跳过: {ex}")
        try:
            run_dose_response(
                df_causal_use, output_dir=_path(path_dir, 'dose_response'), target_col=TARGET_COL)
        except Exception as ex:
            logger.warning(f"剂量反应分析跳过: {ex}")

    # 5. 亚组分析（pool_only 时跳过）
    if not pool_only:
        df_for_subgroup = res_df if res_df is not None else df_causal_use
        causal_col = next((c for c in df_for_subgroup.columns if c.startswith('causal_impact_')), None)
        if causal_col is not None:
            run_subgroup_analysis(df_for_subgroup, output_dir=_path(path_dir, 'subgroup'))
        else:
            logger.warning("未找到因果效应列，跳过亚组分析。")

    auc = champion_auc
    return df_causal_use[TARGET_COL].mean(), (ate, ate_lb, ate_ub), auc


def _run_multi_imputation_rubin_analysis(df_pre, final_dir):
    """
    多重插补 + Rubin 规则合并：对 m1..mN 分别运行预测+因果，收集 AUC/ATE，按 Rubin 规则合并。
    输出至 results/tables/rubin_pooled_*.csv，供主文 Table 2/4 使用。
    """
    try:
        n_mi = getattr(__import__('config'), 'N_MULTIPLE_IMPUTATIONS', 5)
        mi_dir = getattr(__import__('config'), 'IMPUTED_MI_DIR', 'imputation_npj_results/pipeline_trace')
        use_rubin = getattr(__import__('config'), 'USE_RUBIN_POOLING', False)
    except Exception:
        n_mi, mi_dir, use_rubin = 5, 'imputation_npj_results/pipeline_trace', False
    if not use_rubin or n_mi < 2:
        return None

    mi_paths = [os.path.join(mi_dir, f'step1_imputed_m{m}.csv') for m in range(1, n_mi + 1)]
    if not all(os.path.exists(p) for p in mi_paths):
        logger.info(f"多重插补文件不完整（需 m1..m{n_mi}），跳过 Rubin 合并")
        return None

    from utils.rubin_pooling import rubin_pool, rubin_pool_ci
    OVERLAY_RANGES = [
        ('bmi', 16, 35), ('systo', 90, 200), ('diasto', 50, 120), ('mwaist', 60, 150),
        ('sleep', 2, 14), ('family_size', 0, 12),
    ]

    auc_a_list, auc_b_list, auc_c_list = [], [], []
    ate_a_list, ate_b_list, ate_c_list = [], [], []
    se_a_list, se_b_list, se_c_list = [], [], []

    pool_base = '_rubin_pooling_temp'
    os.makedirs(pool_base, exist_ok=True)

    for m in range(1, n_mi + 1):
        logger.info(f"多重插补 Rubin 合并：第 {m}/{n_mi} 份...")
        try:
            df_m = pd.read_csv(mi_paths[m - 1], encoding='utf-8-sig')
            if 'age' in df_m.columns:
                df_m = df_m[df_m['age'] >= AGE_MIN]
            prepare_exposures(df_m)
            df_m = df_m.drop(columns=[c for c in COLS_TO_DROP if c in df_m.columns], errors='ignore')
            if df_pre is not None:
                for col, lo, hi in OVERLAY_RANGES:
                    if col in df_m.columns and col in df_pre.columns:
                        imp_mean = df_m[col].mean()
                        pre_mean = df_pre[col].mean()
                        if not (lo <= imp_mean <= hi) and (lo <= pre_mean <= hi):
                            merge_cols = ['ID', 'wave', col]
                            pre_sub = df_pre[merge_cols].drop_duplicates(subset=['ID', 'wave'])
                            df_m = df_m.drop(columns=[col], errors='ignore').merge(
                                pre_sub, on=['ID', 'wave'], how='left')
            df_m = reapply_cohort_definition(df_m, CESD_CUTOFF, COGNITION_CUTOFF)
            da = df_m[df_m['baseline_group'] == 0]
            db = df_m[df_m['baseline_group'] == 1]
            dc = df_m[df_m['baseline_group'] == 2]

            path_m = os.path.join(pool_base, f'm{m}')
            cohort_tasks_mi = [
                (da, os.path.join(path_m, COHORT_A_DIR), 'A', 'Healthy'),
                (db, os.path.join(path_m, COHORT_B_DIR), 'B', 'Depression'),
                (dc, os.path.join(path_m, COHORT_C_DIR), 'C', 'Cognition'),
            ]
            results = []
            for df_sub, pdir, lab, desc in cohort_tasks_mi:
                if len(df_sub) > 0:
                    inc, (ate, ate_lb, ate_ub), auc = run_cohort_protocol(
                        df_sub, pdir, lab, desc, pool_only=True)
                    se = (ate_ub - ate_lb) / 3.92 if not (np.isnan(ate_lb) or np.isnan(ate_ub)) else np.nan
                    results.append((auc, ate, se))
                else:
                    results.append((np.nan, np.nan, np.nan))

            auc_a_list.append(results[0][0])
            auc_b_list.append(results[1][0])
            auc_c_list.append(results[2][0])
            ate_a_list.append(results[0][1])
            ate_b_list.append(results[1][1])
            ate_c_list.append(results[2][1])
            se_a_list.append(results[0][2])
            se_b_list.append(results[1][2])
            se_c_list.append(results[2][2])
        except Exception as ex:
            logger.warning(f"多重插补第{m}份失败: {ex}")
            return None

    # Rubin 合并
    def _pool(ests, ses):
        valid = [(e, s) for e, s in zip(ests, ses) if np.isfinite(e)]
        if len(valid) < 2:
            return {'Q_bar': np.nan, 'SE': np.nan, 'lb': np.nan, 'ub': np.nan}
        e, s = zip(*valid)
        r = rubin_pool(e, ses=s)
        lb, ub = rubin_pool_ci(r['Q_bar'], r['SE'], r['df'])
        return {'Q_bar': r['Q_bar'], 'SE': r['SE'], 'lb': lb, 'ub': ub}

    # AUC: Rubin 合并点估计，U_bar=0 时 T=(1+1/m)*B
    ra = rubin_pool([x for x in auc_a_list if np.isfinite(x)])
    rb = rubin_pool([x for x in auc_b_list if np.isfinite(x)])
    rc = rubin_pool([x for x in auc_c_list if np.isfinite(x)])
    pa = {'Q_bar': ra['Q_bar'], 'SE': ra['SE']}
    pb = {'Q_bar': rb['Q_bar'], 'SE': rb['SE']}
    pc = {'Q_bar': rc['Q_bar'], 'SE': rc['SE']}

    ta = _pool(ate_a_list, se_a_list)
    tb = _pool(ate_b_list, se_b_list)
    tc = _pool(ate_c_list, se_c_list)

    # 保存
    out_dir = getattr(__import__('config'), 'RESULTS_TABLES', 'results/tables')
    os.makedirs(out_dir, exist_ok=True)
    rubin_auc = pd.DataFrame([
        {'Cohort': 'A', 'AUC_pooled': pa['Q_bar'], 'AUC_SE': pa['SE']},
        {'Cohort': 'B', 'AUC_pooled': pb['Q_bar'], 'AUC_SE': pb['SE']},
        {'Cohort': 'C', 'AUC_pooled': pc['Q_bar'], 'AUC_SE': pc['SE']},
    ])
    rubin_ate = pd.DataFrame([
        {'Cohort': 'A', 'ATE_pooled': ta['Q_bar'], 'ATE_SE': ta['SE'], 'ATE_lb': ta['lb'], 'ATE_ub': ta['ub']},
        {'Cohort': 'B', 'ATE_pooled': tb['Q_bar'], 'ATE_SE': tb['SE'], 'ATE_lb': tb['lb'], 'ATE_ub': tb['ub']},
        {'Cohort': 'C', 'ATE_pooled': tc['Q_bar'], 'ATE_SE': tc['SE'], 'ATE_lb': tc['lb'], 'ATE_ub': tc['ub']},
    ])
    rubin_auc.to_csv(os.path.join(out_dir, 'rubin_pooled_auc.csv'), index=False, encoding='utf-8-sig')
    rubin_ate.to_csv(os.path.join(out_dir, 'rubin_pooled_ate.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"Rubin 合并结果已保存: {out_dir}/rubin_pooled_auc.csv, rubin_pooled_ate.csv")

    # PSW exercise only per MI draw (no CPM/XLearner): Rubin pool for Supplementary Table S16
    try:
        import importlib.util

        _fill_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'fill_paper_tables_extras.py')
        if os.path.isfile(_fill_path):
            _spec = importlib.util.spec_from_file_location('fill_paper_tables_extras', _fill_path)
            _fill_mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_fill_mod)
            _fill_mod.run_psw_rubin_mi()
        else:
            logger.warning('fill_paper_tables_extras.py 未找到，跳过 PSW×MI Rubin 导出')
    except Exception as _psw_ex:
        logger.warning('PSW×MI Rubin 导出失败（可手动运行 scripts/fill_paper_tables_extras.py --psw-only）: %s', _psw_ex)

    try:
        shutil.rmtree(pool_base)
    except Exception:
        pass
    return {'auc': (pa['Q_bar'], pb['Q_bar'], pc['Q_bar']), 'ate': (ta, tb, tc)}


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    log_path = os.path.abspath("LIU_JUE_FINAL_FIXED.log")
    logger.info("🚀 启动修复版【刘珏老师版】全量科研管线...")
    cleanup_temp_cat_dirs()  # 清理历史遗留的 CatBoost 临时目录
    logger.info(f"📂 日志文件: {log_path} （另开终端用 -Wait 可实时查看）")
    try:
        _cfg_main = __import__('config')
        _minimal_causal = bool(getattr(_cfg_main, 'MAIN_MINIMAL_CAUSAL_RERUN', False))
        if _minimal_causal:
            logger.info(
                "MAIN_MINIMAL_CAUSAL_RERUN=True：因果轻量重跑 — 跳过前置步骤与队列后扩展分析，仅用已有插补/CPM 结果；"
                "详见 config.py 注释。"
            )
            if getattr(_cfg_main, 'RUN_IMPUTATION_BEFORE_MAIN', False):
                logger.warning(
                    "MAIN_MINIMAL_CAUSAL_RERUN=True 时已跳过本轮前置 npj 插补；若需重新生成 step1_imputed_full，"
                    "请先设 MAIN_MINIMAL_CAUSAL_RERUN=False 或 RUN_IMPUTATION_BEFORE_MAIN=False。"
                )

        _imp_status = _maybe_run_npj_imputation_first() if not _minimal_causal else 'skipped'
        if _imp_status == 'failed':
            logger.error(
                "本轮前置插补未成功：若磁盘上仍有旧的 step1_imputed_full.csv，主分析可能仍在使用**旧插补**；"
                "请修日志中的报错后重跑，或暂时改用预处理数据（USE_IMPUTED_DATA=False）核对。"
            )

        _proj_root = os.path.dirname(os.path.abspath(__file__))
        _pre_csv_for_stale = os.path.join(_proj_root, 'preprocessed_data', 'CHARLS_final_preprocessed.csv')

        # 数据源：USE_IMPUTED_DATA 且插补文件存在时用插补后数据，否则用预处理数据
        df_clean = None
        df_base_for_sensitivity = None  # 敏感性分析用：插补时传完整插补数据，否则 None
        if USE_IMPUTED_DATA and os.path.exists(IMPUTED_DATA_PATH):
            try:
                from utils.imputation_data_provenance import (
                    log_imputed_csv_loaded,
                    warn_if_imputed_older_than_preprocess,
                )
                _cfg_imp = __import__('config')
                warn_if_imputed_older_than_preprocess(
                    IMPUTED_DATA_PATH,
                    _pre_csv_for_stale,
                    imputation_just_succeeded=(_imp_status == 'ok'),
                    enabled=getattr(_cfg_imp, 'WARN_IMPUTED_OLDER_THAN_PREPROCESSED', True),
                    log=logger,
                )
                df_imputed = pd.read_csv(IMPUTED_DATA_PATH, encoding='utf-8-sig')
                if 'age' in df_imputed.columns:
                    df_imputed = df_imputed[df_imputed['age'] >= AGE_MIN]
                prepare_exposures(df_imputed)  # 先构造 sleep_adequate，再 drop puff
                df_imputed = df_imputed.drop(columns=[c for c in COLS_TO_DROP if c in df_imputed.columns], errors='ignore')
                df_clean = reapply_cohort_definition(df_imputed, CESD_CUTOFF, COGNITION_CUTOFF)
                # 审稿修正：先分割再传递，确保敏感性分析仅用训练集，避免数据泄露
                df_base_for_sensitivity = _get_train_subset(df_imputed)
                log_imputed_csv_loaded(IMPUTED_DATA_PATH, log=logger)
                logger.info(f"✅ 已加载插补后数据: {IMPUTED_DATA_PATH}，主队列样本量 {len(df_clean)}，敏感性分析用训练集 n={len(df_base_for_sensitivity)}")
            except Exception as ex:
                logger.warning(f"加载插补数据失败: {ex}，回退到预处理数据")
        if df_clean is None:
            df_clean = preprocess_charls_data(RAW_DATA_PATH, age_min=AGE_MIN,
                cesd_cutoff=CESD_CUTOFF, cognition_cutoff=COGNITION_CUTOFF)
        if df_clean is None:
            return
        if df_base_for_sensitivity is None:
            prepare_exposures(df_clean)  # 预处理分支：构造暴露变量等

        # 因果时序提醒：暴露/协变量取自 Wave(t)，结局取自 Wave(t+1)；插补数据需源数据满足此结构
        logger.info("📌 因果时序：T/X 取自 Wave(t)，Y(is_comorbidity_next) 取自 Wave(t+1)")

        # 使用插补数据时：提前运行预处理，供 Table 1、BMI 覆盖、插补敏感性 复用
        df_pre = None
        if df_base_for_sensitivity is not None:
            df_pre = preprocess_charls_data(RAW_DATA_PATH, age_min=AGE_MIN, cesd_cutoff=CESD_CUTOFF,
                cognition_cutoff=COGNITION_CUTOFF, write_output=True)
            if df_pre is not None:
                prepare_exposures(df_pre)

        # 插补数据合理性校验（必须在划分 df_a/b/c 之前）：插补 pipeline 可能列错位，多变量异常时用预处理覆盖
        # 合理范围：(var, lo, hi)，均值超出则覆盖
        OVERLAY_RANGES = [
            ('bmi', 16, 35), ('systo', 90, 200), ('diasto', 50, 120), ('mwaist', 60, 150),
            ('sleep', 2, 14), ('family_size', 0, 12),
        ]
        overlay_vars = []
        if df_base_for_sensitivity is not None and df_pre is not None:
            for col, lo, hi in OVERLAY_RANGES:
                if col in df_clean.columns and col in df_pre.columns:
                    imp_mean = df_clean[col].mean()
                    pre_mean = df_pre[col].mean()
                    if not (lo <= imp_mean <= hi) and (lo <= pre_mean <= hi):
                        overlay_vars.append((col, imp_mean))
            if overlay_vars:
                merge_cols = ['ID', 'wave'] + [c for c, _ in overlay_vars]
                pre_sub = df_pre[merge_cols].drop_duplicates(subset=['ID', 'wave'])
                df_clean = df_clean.drop(columns=[c for c, _ in overlay_vars], errors='ignore').merge(
                    pre_sub, on=['ID', 'wave'], how='left')
                logger.warning(f"⚠️ 插补数据多变量异常，已用预处理覆盖: {[v[0] for v in overlay_vars]}")

        df_a = df_clean[df_clean['baseline_group'] == 0].copy()
        df_b = df_clean[df_clean['baseline_group'] == 1].copy()
        df_c = df_clean[df_clean['baseline_group'] == 2].copy()
        if len(df_a) == 0 or len(df_b) == 0 or len(df_c) == 0:
            logger.warning("某一基线亚组样本为空 (A/B/C)，请检查数据与 baseline_group。仅运行非空亚组。")

        final_dir = OUTPUT_ROOT
        os.makedirs(final_dir, exist_ok=True)

        # 人年发病密度：与 Table 1 一致，优先插补前 df_pre（STROBE/「原生」观测）；否则 df_clean
        if not _minimal_causal:
            try:
                from data.charls_incidence_density import save_incidence_density_table

                df_for_table1b = (
                    df_pre if (df_base_for_sensitivity is not None and df_pre is not None) else df_clean
                )
                save_incidence_density_table(
                    df_for_table1b,
                    output_root=final_dir,
                    results_tables=RESULTS_TABLES,
                    filename="table1b_incidence_density.csv",
                )
                if df_for_table1b is df_pre:
                    logger.info("Table 1b 发病密度使用插补前数据（与 Table 1 STROBE 一致）")
                else:
                    logger.info("Table 1b 发病密度使用当前主分析表（无 df_pre 时与 df_clean 一致）")
                try:
                    from viz.fig_incidence_cumulative_and_density import draw_incidence_combined_figure

                    fig_inc_path = os.path.join(RESULTS_FIGURES, "fig_incidence_cumulative_and_density.png")
                    draw_incidence_combined_figure(df_for_table1b, fig_inc_path)
                    logger.info("组合发病图已写: %s", fig_inc_path)
                except Exception as fig_ex:
                    logger.warning("fig_incidence_cumulative_and_density 跳过: %s", fig_ex)
            except Exception as ex:
                logger.warning("table1b 发病密度表生成跳过: %s", ex)
        else:
            logger.info("MAIN_MINIMAL_CAUSAL_RERUN：跳过 Table1b 发病密度与组合发病图（沿用此前结果）。")

        skip_prefix = bool(getattr(_cfg_main, 'MAIN_SKIP_STEPS_BEFORE_COHORTS', False)) or _minimal_causal
        if not skip_prefix:
            draw_conceptual_framework(output_path=os.path.join(final_dir, 'fig2_conceptual_framework.png'))

            # 医学规范：Table 1 基线特征使用插补前数据（STROBE 推荐，反映入组时实际观测）
            df_for_table1 = df_pre if (df_base_for_sensitivity is not None and df_pre is not None) else df_clean
            df_for_imp_sens = df_pre if (df_base_for_sensitivity is not None and df_pre is not None) else df_clean
            if df_for_table1 is df_pre:
                logger.info("Table 1 使用插补前数据（医学规范）；主分析用插补后数据（含 BMI 修正）")

            generate_baseline_table(df_for_table1, output_dir=final_dir)

            # 流失流程图数据（公共卫生/STROBE 报告）
            attrition_src = 'preprocessed_data/attrition_flow.csv'
            if os.path.exists(attrition_src):
                shutil.copy(attrition_src, os.path.join(final_dir, 'attrition_flow.csv'))
                draw_flowchart(csv_path=os.path.join(final_dir, 'attrition_flow.csv'), output_path=os.path.join(final_dir, 'attrition_flow_diagram.png'))
            # 使用插补数据时：attrition_flow 来自预处理，与当前队列可能略有差异，写入说明供方法部分引用
            if df_base_for_sensitivity is not None:  # 实际使用了插补数据（加载成功）
                with open(os.path.join(final_dir, 'attrition_flow_readme.txt'), 'w', encoding='utf-8') as f:
                    f.write("Table 1 与 attrition_flow 均基于插补前数据（医学规范/STROBE）。\n")
                    f.write(
                        "预测/CPM/SHAP：预处理宽表（保留缺失）+ compare_models 内 Pipeline IterativeImputer（仅训练折 fit）。\n"
                    )
                    f.write("因果及扩展分析：使用 step1_imputed_full 等同队列切片（bulk 插补）。\n")

            # 插补敏感性分析（基于主目标 is_comorbidity_next），每次运行更新
            imp_sens_dir = os.path.join(final_dir, '07_sensitivity_imputation')
            try:
                if df_base_for_sensitivity is not None and df_for_imp_sens is df_clean:
                    logger.warning("预处理返回 None（如 RAW_DATA_PATH 缺失），插补敏感性回退到主数据，五种方法可能无差异")
                run_imputation_sensitivity_preprocessed(df_for_imp_sens, output_dir=imp_sens_dir, target_col=TARGET_COL)
            except Exception as ex:
                logger.warning(f"插补敏感性分析跳过: {ex}")
        else:
            if _minimal_causal:
                logger.info(
                    "MAIN_MINIMAL_CAUSAL_RERUN：已跳过概念图 / Table1 / 流失图 / 队列前插补敏感性，直接进入三队列。"
                )
            else:
                logger.info(
                    "MAIN_SKIP_STEPS_BEFORE_COHORTS=True：已跳过概念图 / Table1 / 流失图 / 插补敏感性，直接进入三队列。"
                )

        # 三队列并行：Cohort A/B/C 彼此独立，可用 joblib 并行以缩短总时长
        try:
            from config import PARALLEL_COHORTS
        except ImportError:
            PARALLEL_COHORTS = False
        def _prediction_cohort_slices():
            """监督学习：预处理缺失宽表 + Pipeline 内插补；与 bulk step1_imputed_full 解耦。"""
            if df_pre is not None:
                dfp = df_pre.drop(columns=[c for c in COLS_TO_DROP if c in df_pre.columns], errors='ignore')
                pred_full = reapply_cohort_definition(dfp, CESD_CUTOFF, COGNITION_CUTOFF)
                pa = pred_full[pred_full['baseline_group'] == 0].copy()
                pb = pred_full[pred_full['baseline_group'] == 1].copy()
                pc = pred_full[pred_full['baseline_group'] == 2].copy()
                logger.info(
                    '预测/CPM/SHAP：使用预处理宽表（保留缺失）+ compare_models 内 IterativeImputer；'
                    '因果仍用插补后 df_clean 同队列切片。'
                )
                return pa, pb, pc
            logger.info('预测步：无单独 df_pre，CPM 与因果共用当前 df_clean（通常为预处理无 bulk 插补）。')
            return df_a.copy(), df_b.copy(), df_c.copy()

        df_pa, df_pb, df_pc = _prediction_cohort_slices()

        cohort_tasks = [
            (df_pa, os.path.join('.', COHORT_A_DIR), 'A', '健康人群 -> 共病风险预测', df_a),
            (df_pb, os.path.join('.', COHORT_B_DIR), 'B', '仅抑郁人群 -> 共病演化分析', df_b),
            (df_pc, os.path.join('.', COHORT_C_DIR), 'C', '仅认知受损人群 -> 共病演化分析', df_c),
        ]

        run_only = getattr(_cfg_main, 'RUN_COHORTS_ONLY', None)
        causal_only_flag = bool(getattr(_cfg_main, 'MAIN_COHORT_CAUSAL_ONLY', False)) or _minimal_causal
        if _minimal_causal:
            logger.info(
                "MAIN_MINIMAL_CAUSAL_RERUN：三队列仅跑因果（causal_only）；CPM/SHAP 等使用磁盘已有输出。"
            )
        elif causal_only_flag:
            logger.info(
                "MAIN_COHORT_CAUSAL_ONLY=True：三队列仅跑因果及因果相关扩展（ITE/时序/敏感性/亚组等），"
                "跳过 compare_models、SHAP、临床评价、决策、外部验证、剂量反应。"
            )
        only_set = None
        if run_only is not None and len(run_only) > 0:
            only_set = {str(x).strip().upper() for x in run_only}
            logger.info(
                "RUN_COHORTS_ONLY=%s：仅重跑这些队列；其余队列从磁盘恢复汇总指标（汇总图用）。",
                sorted(only_set),
            )

        def _run_cohort_task(args):
            df_pred, path_dir, cohort_id, cohort_desc, df_causal_cohort = args
            if len(df_pred) == 0:
                return (0.0, (np.nan, np.nan, np.nan), 0.0)
            return run_cohort_protocol(
                df_pred,
                path_dir,
                cohort_id,
                cohort_desc,
                df_for_causal=df_causal_cohort,
                causal_only=causal_only_flag,
            )

        tasks_to_run = [t for t in cohort_tasks if (only_set is None or t[2] in only_set)]
        if not tasks_to_run:
            logger.error("RUN_COHORTS_ONLY 过滤后无队列可跑，请检查 config")
            return

        if PARALLEL_COHORTS and len(tasks_to_run) > 1:
            from joblib import Parallel, delayed
            logger.info("三队列并行模式：同时运行 %s 个任务", len(tasks_to_run))
            cohort_results_run = Parallel(n_jobs=min(3, len(tasks_to_run)), backend='loky')(
                delayed(_run_cohort_task)(t) for t in tasks_to_run
            )
        else:
            cohort_results_run = [_run_cohort_task(t) for t in tasks_to_run]

        results_by_id = {}
        for t, res in zip(tasks_to_run, cohort_results_run):
            results_by_id[t[2]] = res
        if only_set is not None:
            for t in cohort_tasks:
                cid = t[2]
                if cid not in results_by_id:
                    df_inc = t[4] if len(t) >= 5 else t[0]
                    results_by_id[cid] = _load_skipped_cohort_metrics(t[1], cid, df_inc)

        inc_a, ate_a, auc_a = results_by_id['A']
        inc_b, ate_b, auc_b = results_by_id['B']
        inc_c, ate_c, auc_c = results_by_id['C']

        if _minimal_causal:
            _va = _read_champion_auc_from_saved_table2('A')
            _vb = _read_champion_auc_from_saved_table2('B')
            _vc = _read_champion_auc_from_saved_table2('C')
            if np.isfinite(_va):
                auc_a = _va
            if np.isfinite(_vb):
                auc_b = _vb
            if np.isfinite(_vc):
                auc_c = _vc
            if not all(np.isfinite(x) for x in (_va, _vb, _vc)):
                logger.warning(
                    "MAIN_MINIMAL_CAUSAL_RERUN：部分队列磁盘上无 table2_*_main_performance.csv，"
                    "汇总图 AUC 可能为 NaN；请先完成一次完整 CPM 或忽略 AUC 对比图。"
                )

        # 投稿补充：阴性对照结局（NCO）+ Cohort B 估计器鲁棒性（XLearner vs DML）；不改 CPM/插补
        try:
            from causal.charls_recalculate_causal_impact import (
                run_ate_method_sensitivity_cohort_b,
                run_negative_control_outcome_summary,
            )

            if getattr(_cfg_main, 'RUN_NEGATIVE_CONTROL_OUTCOME', False):
                run_negative_control_outcome_summary(df_clean)
            if getattr(_cfg_main, 'RUN_ATE_METHOD_SENSITIVITY', False):
                run_ate_method_sensitivity_cohort_b(df_clean)
        except Exception as ex:
            logger.warning('阴性对照 / ATE 方法敏感性跳过: %s', ex)

        # 多重插补 + Rubin 规则合并（m1..mN 存在且 USE_RUBIN_POOLING 时）
        rubin_result = None
        if not _minimal_causal:
            try:
                rubin_result = _run_multi_imputation_rubin_analysis(df_pre, final_dir)
            except Exception as ex:
                logger.warning(f"Rubin 合并跳过: {ex}")

            # 截断值敏感性（抑郁 CES-D≥8/10/12，认知 Cog≤8/10/12）+ 完整病例
            try:
                run_sensitivity_scenarios_analysis(
                    final_dir=final_dir, data_path=RAW_DATA_PATH, df_base=df_base_for_sensitivity
                )
            except Exception as ex:
                logger.warning(f"截断值敏感性分析跳过: {ex}")

            # 多暴露因果分析（运动、睡眠、吸烟、饮酒、社会隔离）
            try:
                run_multi_exposure_analysis(df_clean, output_root=os.path.join(final_dir, '08_multi_exposure'))
            except Exception as ex:
                logger.warning(f"多暴露因果分析跳过: {ex}")

            # 扩展干预分析 + 图4 森林图（5类干预×三轴线，使用 config.CAUSAL_METHOD）
            try:
                from scripts.run_all_interventions_analysis import run_all_interventions

                run_all_interventions(df_clean, output_root=final_dir)
            except Exception as ex:
                logger.warning(f"扩展干预分析与森林图跳过: {ex}")

            # X-Learner 全干预（7类：运动、饮酒、社会隔离、BMI、慢性病、睡眠、吸烟包年）+ PSM/PSW 合并
            try:
                from scripts.run_xlearner_all_interventions import run_xlearner_all_interventions

                run_xlearner_all_interventions(
                    output_dir=os.path.join(final_dir, 'xlearner_all_interventions'), df_clean=df_clean
                )
            except Exception as ex:
                logger.warning(f"X-Learner 全干预分析跳过: {ex}")

            # PSM/PSW 因果方法交叉验证（审稿 P1）
            try:
                from causal.charls_causal_methods_comparison import run_all_cohorts_comparison

                run_all_cohorts_comparison(df_clean, output_root=final_dir)
            except Exception as ex:
                logger.warning(f"因果方法交叉验证跳过: {ex}")

            # 低样本量暴露优化（社会隔离、慢性病负担低）
            try:
                from causal.charls_low_sample_optimization import run_for_social_isolation_and_chronic

                run_for_social_isolation_and_chronic(df_clean, output_root=final_dir)
            except Exception as ex:
                logger.warning(f"低样本优化跳过: {ex}")

            # 生理/功能指标因果分析（握力、步行速度、ADL/IADL、血压、自评健康等）
            try:
                from scripts.run_all_physio_causal import run_all_physio_causal

                run_all_physio_causal(
                    output_dir=os.path.join(final_dir, '09_physio_causal'), df_clean=df_clean
                )
            except Exception as ex:
                logger.warning(f"生理指标因果分析跳过: {ex}")
        else:
            logger.info(
                "MAIN_MINIMAL_CAUSAL_RERUN：已跳过 Rubin、截断敏感性、多暴露、扩展干预、XLearner 全干预、"
                "PSM 全队列交叉、低样本优化、生理因果（沿用此前结果）。"
            )

        # 汇总对比
        plt.figure(figsize=(10, 6))
        rates = pd.DataFrame({'Path': ['Healthy', 'Depressed', 'Cognitive'], 'Rate': [inc_a, inc_b, inc_c]})
        sns.barplot(x='Path', y='Rate', data=rates, hue='Path', palette='Set2', legend=False)
        plt.title('Incidence Rates across 3 Baselines', fontsize=14)
        plt.savefig(f'{final_dir}/incidence_comparison.png', dpi=300, bbox_inches=BBOX_INCHES)
        plt.close()

        # 汇总图：展示三条轴线的 ATE 对比（Rubin 合并可用时优先使用）
        if rubin_result is not None and 'ate' in rubin_result:
            ta, tb, tc = rubin_result['ate']
            ate_plot = [ate_a, ate_b, ate_c]
            if all('lb' in t and 'ub' in t for t in [ta, tb, tc]):
                ate_plot = [(ta['Q_bar'], ta['lb'], ta['ub']), (tb['Q_bar'], tb['lb'], tb['ub']), (tc['Q_bar'], tc['lb'], tc['ub'])]
        else:
            ate_plot = [ate_a, ate_b, ate_c]
        try:
            plt.figure(figsize=(12, 8))
            paths = ['Healthy Start (A)', 'Depressed Start (B)', 'Cognitive Start (C)']
            ate_data = pd.DataFrame({
                'Path': paths,
                'ATE': [ate_plot[0][0], ate_plot[1][0], ate_plot[2][0]],
                'LB': [ate_plot[0][1], ate_plot[1][1], ate_plot[2][1]],
                'UB': [ate_plot[0][2], ate_plot[1][2], ate_plot[2][2]],
            })
            colors = ['#5cb85c', '#d9534f', '#5bc0de']
            x = np.arange(len(paths))
            bar_h = [float(v) if np.isfinite(v) else 0.0 for v in ate_data['ATE']]
            plt.bar(x, bar_h, color=colors, alpha=0.7)
            for i in range(len(paths)):
                a, lo, hi = ate_data['ATE'].iloc[i], ate_data['LB'].iloc[i], ate_data['UB'].iloc[i]
                if np.isfinite(a) and np.isfinite(lo) and np.isfinite(hi):
                    plt.errorbar(
                        i, a,
                        yerr=[[a - lo], [hi - a]],
                        fmt='none', c='black', capsize=10,
                    )
                elif not np.isfinite(a):
                    plt.text(i, 0.0, 'NA', ha='center', va='bottom', fontsize=9, color='gray')
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.xticks(x, paths, rotation=15, ha='right')
            plt.title('Causal Effect of Exercise (ATE) across 3 Baselines', fontsize=14)
            plt.savefig(f'{final_dir}/intervention_benefit_comparison.png', dpi=300, bbox_inches=BBOX_INCHES)
            plt.close()
        except Exception as ex:
            logger.warning('intervention_benefit_comparison 图跳过: %s', ex)
            plt.close('all')

        # 额外汇总图（基线对比、分布、AUC、结局、运动暴露、亚组 CATE）
        auc_fig_a, auc_fig_b, auc_fig_c = auc_a, auc_b, auc_c
        if rubin_result is not None and 'auc' in rubin_result:
            auc_fig_a, auc_fig_b, auc_fig_c = rubin_result['auc']
        draw_all_extra_figures(df_clean, auc_fig_a, auc_fig_b, auc_fig_c, final_dir)
        # Fig 3 ROC 叠加图（需主流程已保存 roc_data.json）
        draw_roc_combined(output_path=os.path.join(final_dir, 'fig3_roc_combined.png'))
        logger.info(f"已生成额外汇总图与 Fig 3 ROC 至 {final_dir}")

        # GitHub 开源适配： consolidate 到 results/tables, results/figures, results/models
        try:
            from config import RESULTS_TABLES, RESULTS_FIGURES, RESULTS_MODELS
            for d in [RESULTS_TABLES, RESULTS_FIGURES, RESULTS_MODELS]:
                os.makedirs(d, exist_ok=True)
            # 复制关键表格
            def _copy_if_exists(src, dst):
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    return True
                return False

            def _copy_dual(src, dest_primary, dest_legacy):
                """首选 Cohort 命名；同步写旧 axis 文件名，兼容已投稿稿中的路径引用。"""
                if not os.path.exists(src):
                    return False
                shutil.copy(src, dest_primary)
                shutil.copy(src, dest_legacy)
                return True
            _copy_if_exists(os.path.join(final_dir, 'attrition_flow.csv'), os.path.join(RESULTS_TABLES, 'table1_sample_attrition.csv'))
            _copy_if_exists(os.path.join('preprocessed_data', 'attrition_flow.csv'), os.path.join(RESULTS_TABLES, 'table1_sample_attrition.csv'))
            _copy_if_exists(os.path.join(final_dir, 'table1_baseline_characteristics.csv'), os.path.join(RESULTS_TABLES, 'table1_baseline_characteristics.csv'))
            # Table 2 预测性能（CPM 主表，与 SHAP/临床评价冠军一致）
            for cid, label in [('A', COHORT_A_DIR), ('B', COHORT_B_DIR), ('C', COHORT_C_DIR)]:
                t2_src = os.path.join(label, COHORT_STEP_DIRS['prediction'], f'table2_{cid}_main_performance.csv')
                _copy_dual(
                    t2_src,
                    os.path.join(RESULTS_TABLES, f'table2_prediction_cohort{cid}.csv'),
                    os.path.join(RESULTS_TABLES, f'table2_prediction_axis{cid}.csv'),
                )
            try:
                from utils.charls_table2_combine import write_combined_table2_prediction

                _t2_comb = write_combined_table2_prediction(RESULTS_TABLES)
                if _t2_comb:
                    logger.info('合并预测 Table 2（A+B+C）: %s', _t2_comb)
            except Exception as _t2c_ex:
                logger.debug('合并预测 Table 2 跳过: %s', _t2c_ex)
            # Table 3 亚组 CATE
            for cid, label in [('A', COHORT_A_DIR), ('B', COHORT_B_DIR), ('C', COHORT_C_DIR)]:
                t3_src = os.path.join(label, COHORT_STEP_DIRS['subgroup'], 'subgroup_analysis_results.csv')
                _copy_dual(
                    t3_src,
                    os.path.join(RESULTS_TABLES, f'table3_subgroup_cohort{cid}.csv'),
                    os.path.join(RESULTS_TABLES, f'table3_subgroup_axis{cid}.csv'),
                )
            # Table 5 截断值敏感性
            _copy_if_exists(os.path.join(final_dir, 'sensitivity_summary.csv'), os.path.join(RESULTS_TABLES, 'table5_sensitivity_summary.csv'))
            if not _copy_if_exists(os.path.join(final_dir, 'all_interventions_summary.csv'), os.path.join(RESULTS_TABLES, 'table4_ate_summary.csv')):
                _copy_if_exists(os.path.join(final_dir, '08_multi_exposure', 'multi_exposure_ate_summary.csv'), os.path.join(RESULTS_TABLES, 'table4_ate_summary.csv'))
            _copy_if_exists(os.path.join(final_dir, 'xlearner_all_interventions', 'xlearner_psm_psw_wide.csv'), os.path.join(RESULTS_TABLES, 'table4_xlearner_psm_psw_wide.csv'))
            _copy_if_exists(os.path.join(final_dir, 'causal_methods_comparison_summary.csv'), os.path.join(RESULTS_TABLES, 'table7_psm_psw_dml.csv'))
            _copy_if_exists(os.path.join(final_dir, '07_sensitivity_imputation', 'imputation_sensitivity_results.csv'), os.path.join(RESULTS_TABLES, 'imputation_sensitivity_results.csv'))
            # Rubin 合并结果（多重插补 m1..mN + Rubin 规则）
            _copy_if_exists(os.path.join(RESULTS_TABLES, 'rubin_pooled_auc.csv'), os.path.join(RESULTS_TABLES, 'table2_rubin_pooled_auc.csv'))
            _copy_if_exists(os.path.join(RESULTS_TABLES, 'rubin_pooled_ate.csv'), os.path.join(RESULTS_TABLES, 'table4_rubin_pooled_ate.csv'))
            _copy_if_exists(os.path.join(final_dir, '09_physio_causal', 'physio_ate_summary.csv'), os.path.join(RESULTS_TABLES, 'table_physio_ate_summary.csv'))
            # 复制外部验证
            for cid, label in [('A', COHORT_A_DIR), ('B', COHORT_B_DIR), ('C', COHORT_C_DIR)]:
                ev_path = os.path.join(label, COHORT_STEP_DIRS['external_validation'], 'external_validation_summary.csv')
                _copy_dual(
                    ev_path,
                    os.path.join(RESULTS_TABLES, f'table6_external_validation_cohort{cid}.csv'),
                    os.path.join(RESULTS_TABLES, f'table6_external_validation_axis{cid}.csv'),
                )
            # 复制冠军模型
            for cid, label in [('A', COHORT_A_DIR), ('B', COHORT_B_DIR), ('C', COHORT_C_DIR)]:
                mp = os.path.join(label, COHORT_STEP_DIRS['prediction'], 'champion_model.joblib')
                _copy_dual(
                    mp,
                    os.path.join(RESULTS_MODELS, f'champion_cohort{cid}.joblib'),
                    os.path.join(RESULTS_MODELS, f'champion_axis{cid}.joblib'),
                )
            # 复制核心 Figure 到 results/figures（论文引用）
            fig_copies = [
                (os.path.join(final_dir, 'attrition_flow_diagram.png'), 'fig1_attrition_flow.png'),
                (os.path.join(final_dir, 'fig2_conceptual_framework.png'), 'fig2_conceptual_framework.png'),
                (os.path.join(final_dir, 'fig3_roc_combined.png'), 'fig3_roc_combined.png'),
            ]
            for src, dst in fig_copies:
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(RESULTS_FIGURES, dst))
            # 主文 Figure 4：三队列运动 ATE 对比 + 亚组汇总 + 各队列亚组森林图（consolidate 后统一从 results/figures 引用）
            fig4_copies = [
                (os.path.join(final_dir, 'intervention_benefit_comparison.png'), 'fig4_intervention_benefit.png'),
                (os.path.join(final_dir, 'fig_subgroup_cate_combined.png'), 'fig4_subgroup_cate_combined.png'),
            ]
            for src, dst in fig4_copies:
                _copy_if_exists(src, os.path.join(RESULTS_FIGURES, dst))
            for cid, label in [('A', COHORT_A_DIR), ('B', COHORT_B_DIR), ('C', COHORT_C_DIR)]:
                sub_forest = os.path.join(label, COHORT_STEP_DIRS['subgroup'], 'fig_subgroup_academic_forest.png')
                _copy_dual(
                    sub_forest,
                    os.path.join(RESULTS_FIGURES, f'fig4_subgroup_forest_cohort{cid}.png'),
                    os.path.join(RESULTS_FIGURES, f'fig4_subgroup_forest_axis{cid}.png'),
                )
            for cid, label in [('A', COHORT_A_DIR), ('B', COHORT_B_DIR), ('C', COHORT_C_DIR)]:
                shap_src = os.path.join(label, COHORT_STEP_DIRS['shap'], 'fig5a_shap_summary_is_comorbidity_next.png')
                eval_src = os.path.join(label, COHORT_STEP_DIRS['eval'], 'fig3_clinical_evaluation_comprehensive.png')
                _copy_dual(
                    shap_src,
                    os.path.join(RESULTS_FIGURES, f'fig4_shap_cohort{cid}.png'),
                    os.path.join(RESULTS_FIGURES, f'fig4_shap_axis{cid}.png'),
                )
                _copy_dual(
                    eval_src,
                    os.path.join(RESULTS_FIGURES, f'fig5_dca_cohort{cid}.png'),
                    os.path.join(RESULTS_FIGURES, f'fig5_dca_axis{cid}.png'),
                )
            logger.info("已 consolidate 至 results/tables, results/figures, results/models")
        except Exception as ex:
            logger.debug(f"results consolidate 跳过: {ex}")

        logger.info("\n" + "="*60 + "\n🎉 修复版管线运行圆满结束！\n" + "="*60)

    except Exception as e:
        logger.error(f"💥 崩溃: {e}", exc_info=True)

if __name__ == "__main__":
    main()
