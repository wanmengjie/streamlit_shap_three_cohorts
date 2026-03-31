# -*- coding: utf-8 -*-
"""
内部时间/分区域验证：使用主流程保存的 champion_model.joblib（完整 Pipeline：
IterativeImputer + 最优算法，可与 CALIBRATE_CHAMPION_PROBA 校准包装器同构），
在时间上或地域划分的验证子集上评估 AUC/AUPRC/Brier/校准曲线。
禁止在此处重训替代模型，以保证与 CPM 主分析规格一致。
"""
import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

from utils.charls_feature_lists import get_exclude_cols
from config import BBOX_INCHES, COHORT_STEP_DIRS

logger = logging.getLogger(__name__)

# CHARLS 省份编码 -> 东/中/西部（基于 CHARLS 设计文档常见映射）
PROVINCE_TO_REGION = {
    1: 'East', 2: 'East', 4: 'East', 5: 'East', 6: 'East', 16: 'East', 18: 'East', 24: 'East', 25: 'East',
    3: 'Central', 7: 'Central', 8: 'Central', 9: 'Central', 17: 'Central', 19: 'Central', 21: 'Central', 23: 'Central',
    10: 'West', 11: 'West', 12: 'West', 13: 'West', 14: 'West', 15: 'West', 22: 'West', 26: 'West', 27: 'West', 28: 'West',
}


def _add_region(df):
    """为 df 添加 region 列（East/Central/West）"""
    if 'province' not in df.columns:
        return df
    df = df.copy()
    try:
        prov = pd.to_numeric(df['province'], errors='coerce')
        df['_region'] = prov.map(PROVINCE_TO_REGION)
        df['_region'] = df['_region'].fillna('Unknown')
    except Exception:
        df['_region'] = 'Unknown'
    return df


def _champion_feature_names(champion):
    """从已 fit 的冠军估计器解析特征列顺序（支持 CalibratedClassifierCV 包装）。"""
    m = champion
    for _ in range(8):
        if m is None:
            break
        fn = getattr(m, 'feature_names_in_', None)
        if fn is not None and len(fn) > 0:
            return list(np.asarray(fn).ravel())
        cal_list = getattr(m, 'calibrated_classifiers_', None)
        if cal_list is not None and len(cal_list) > 0:
            first = cal_list[0]
            m = getattr(first, 'estimator', None) or getattr(first, 'base_estimator', None)
            continue
        m = getattr(m, 'estimator', None)
    return None


def _build_X_for_champion(df_slice, target_col, exclude_cols, champion):
    """与 compare_models 一致：排除列后仅数值列，并按冠军 Pipeline 训练时的列顺序对齐。"""
    W_cols = [c for c in df_slice.columns if c not in exclude_cols]
    X_num = df_slice[W_cols].select_dtypes(include=[np.number])
    fn = _champion_feature_names(champion)
    if fn is not None:
        missing = [c for c in fn if c not in X_num.columns]
        if missing:
            logger.error('验证集相对冠军训练特征缺列: %s', missing[:20])
            return None
        return X_num[fn]
    logger.warning('冠军模型无 feature_names_in_，按当前数值列顺序 predict（可能与训练列序不一致）')
    return X_num


def _evaluate_frozen_champion(champion, df_val, target_col, exclude_cols):
    """
    不重新 fit：用已训练冠军在验证子集上 predict_proba。
    """
    X_val = _build_X_for_champion(df_val, target_col, exclude_cols, champion)
    if X_val is None or X_val.shape[1] == 0:
        return None
    y_val = df_val[target_col].astype(int)
    if len(np.unique(y_val)) < 2:
        return None
    try:
        y_prob = champion.predict_proba(X_val)[:, 1]
    except Exception as ex:
        logger.error('冠军模型 predict_proba 失败: %s', ex)
        return None
    auc = roc_auc_score(y_val, y_prob)
    auprc = average_precision_score(y_val, y_prob)
    brier = brier_score_loss(y_val, y_prob)
    return {
        'AUC': auc,
        'AUPRC': auprc,
        'Brier': brier,
        'n_train': np.nan,
        'n_val': len(df_val),
        'y_prob': y_prob,
        'y_true': y_val.values,
    }


def run_external_validation(
    df,
    champion_model_path=None,
    model=None,
    output_dir='external_validation',
    target_col='is_comorbidity_next',
    cohort_id='',
):
    """
    时间验证与分区域验证：必须加载主流程 ``01_prediction/champion_model.joblib``（或与之一致的
    已 fit Pipeline）。不在此脚本内用 SimpleImputer+ExtraTrees 重训替代模型。

    Parameters
    ----------
    champion_model_path : str, optional
        指向 ``run_cohort_protocol`` 写入的 ``champion_model.joblib``。
    model : fitted estimator, optional
        若路径不存在时的兜底（例如保存失败）；正式复现应优先使用 joblib。
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f">>> 启动内部外部验证 (Cohort {cohort_id}, Target: {target_col})...")

    champion = None
    if champion_model_path and os.path.isfile(champion_model_path):
        champion = joblib.load(champion_model_path)
        logger.info('外部验证: 已加载主流程冠军 Pipeline — %s', champion_model_path)
    elif model is not None:
        champion = model
        logger.warning(
            '外部验证: 未找到 champion_model_path=%s，使用内存中的 model；'
            '与主流程完全同构时请确保 pred 步已成功写出 champion_model.joblib',
            champion_model_path,
        )
    else:
        raise ValueError(
            'run_external_validation 需要 champion_model_path 指向 champion_model.joblib，'
            '或在保存失败时传入已 fit 的 champion model'
        )

    exclude_cols = get_exclude_cols(df, target_col=target_col)
    df = _add_region(df)

    results = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- 1. 时间验证 (Temporal)：与主 CPM 相同 wave 划分，评估冻结冠军在末波上的泛化（可与 Table2 对照）---
    if 'wave' in df.columns:
        max_wave = df['wave'].max()
        df_val_t = df[df['wave'] == max_wave]
        if len(df_val_t) >= 50:
            res_t = _evaluate_frozen_champion(champion, df_val_t, target_col, exclude_cols)
            if res_t is not None:
                res_t['Split'] = 'Temporal'
                res_t['Split_Desc'] = f'Frozen champion on wave={max_wave} (same held-out wave as main CPM test)'
                results.append(res_t)
                prob_true, prob_pred = calibration_curve(res_t['y_true'], res_t['y_prob'], n_bins=10)
                axes[0].plot(prob_pred, prob_true, marker='o', label=f'Temporal (AUC={res_t["AUC"]:.3f})')
                logger.info(
                    "时间验证(冻结冠军): AUC=%.4f, AUPRC=%.4f, Brier=%.4f, n_val=%s",
                    res_t['AUC'],
                    res_t['AUPRC'],
                    res_t['Brier'],
                    res_t['n_val'],
                )
        else:
            logger.warning(f"时间验证样本不足: val={len(df_val_t)}")

    # --- 2. 区域验证 (Regional)：冻结冠军在东中部训练分布下，在西部子样本上的可迁移性（不重训）---
    if '_region' in df.columns:
        regions = df['_region'].dropna().unique()
        valid_regions = [r for r in regions if r != 'Unknown']
        if len(valid_regions) >= 2:
            train_regions = [r for r in valid_regions if r in ('East', 'Central')]
            val_regions = [r for r in valid_regions if r == 'West']
            if len(val_regions) == 0:
                val_regions = [valid_regions[-1]]
                train_regions = [r for r in valid_regions if r != val_regions[0]]
            if train_regions and val_regions:
                df_val_r = df[df['_region'].isin(val_regions)]
                if len(df_val_r) >= 50:
                    res_r = _evaluate_frozen_champion(champion, df_val_r, target_col, exclude_cols)
                    if res_r is not None:
                        res_r['Split'] = 'Regional'
                        res_r['Split_Desc'] = f'Frozen champion; val regions: {"+".join(val_regions)}'
                        results.append(res_r)
                        prob_true, prob_pred = calibration_curve(res_r['y_true'], res_r['y_prob'], n_bins=10)
                        axes[1].plot(prob_pred, prob_true, marker='s', label=f'Regional (AUC={res_r["AUC"]:.3f})')
                        logger.info(
                            "区域验证(冻结冠军): AUC=%.4f, AUPRC=%.4f, n_val=%s",
                            res_r['AUC'],
                            res_r['AUPRC'],
                            res_r['n_val'],
                        )
                else:
                    logger.warning(f"区域验证样本不足: val={len(df_val_r)}")
        else:
            logger.warning("区域划分不足，跳过区域验证")

    for ax in axes:
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.legend()
    axes[0].set_title('Temporal Validation Calibration (frozen champion)')
    axes[1].set_title('Regional Validation Calibration (frozen champion)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'external_validation_calibration.png'), dpi=150, bbox_inches=BBOX_INCHES)
    plt.close()

    summary = []
    for r in results:
        summary.append({
            'Split': r['Split'],
            'Split_Desc': r.get('Split_Desc', ''),
            'AUC': r['AUC'],
            'AUPRC': r['AUPRC'],
            'Brier': r['Brier'],
            'n_train': r['n_train'],
            'n_val': r['n_val'],
        })
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(output_dir, 'external_validation_summary.csv'), index=False, encoding='utf-8-sig')
        try:
            summary_df.to_excel(os.path.join(output_dir, 'external_validation_summary.xlsx'), index=False)
        except Exception:
            pass
        logger.info(f"外部验证结果已保存至 {output_dir}/external_validation_summary.csv")
        return summary_df
    return pd.DataFrame()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    from data.charls_complete_preprocessing import preprocess_charls_data
    from config import COHORT_A_DIR, COHORT_B_DIR, COHORT_C_DIR

    cohort_dirs = {'A': COHORT_A_DIR, 'B': COHORT_B_DIR, 'C': COHORT_C_DIR}
    pred_sub = COHORT_STEP_DIRS.get('prediction', '01_prediction')
    df = preprocess_charls_data('CHARLS.csv', age_min=60)
    if df is not None:
        for cohort_id, baseline_val in [('A', 0), ('B', 1), ('C', 2)]:
            df_sub = df[df['baseline_group'] == baseline_val]
            if len(df_sub) > 0:
                out = os.path.join(cohort_dirs[cohort_id], '04b_external_validation')
                champ = os.path.join(cohort_dirs[cohort_id], pred_sub, 'champion_model.joblib')
                if os.path.isfile(champ):
                    run_external_validation(
                        df_sub,
                        champion_model_path=champ,
                        output_dir=out,
                        cohort_id=cohort_id,
                    )
                else:
                    logger.error(
                        '跳过 Cohort %s：未找到 %s（请先运行 run_all_charls_analyses 生成冠军模型）',
                        cohort_id,
                        champ,
                    )
