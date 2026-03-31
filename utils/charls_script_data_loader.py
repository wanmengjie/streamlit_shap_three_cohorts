# -*- coding: utf-8 -*-
"""
独立运行 scripts 时与 run_all_charls_analyses 使用同一数据源（插补 vs 预处理）。
避免「主流程用插补、单跑脚本用原始预处理」导致结果不可比。
"""
import os
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_df_for_analysis(apply_config_drop=True):
    """
    按 config 加载分析用宽表：优先插补 CSV，失败则 preprocess_charls_data。
    返回 DataFrame 或 None。

    Parameters
    ----------
    apply_config_drop : bool
        True（默认）：删除 config.COLS_TO_DROP，与主流程一致。
        False：不删 COLS_TO_DROP，供「全变量 vs 当前 drop」对比等脚本使用。
    """
    from config import (
        USE_IMPUTED_DATA,
        IMPUTED_DATA_PATH,
        RAW_DATA_PATH,
        AGE_MIN,
        COLS_TO_DROP,
        CESD_CUTOFF,
        COGNITION_CUTOFF,
    )
    from data.charls_complete_preprocessing import preprocess_charls_data, reapply_cohort_definition
    from scripts.run_multi_exposure_causal import prepare_exposures

    df_clean = None
    if USE_IMPUTED_DATA and IMPUTED_DATA_PATH and os.path.exists(IMPUTED_DATA_PATH):
        try:
            from utils.imputation_data_provenance import (
                log_imputed_csv_loaded,
                warn_if_imputed_older_than_preprocess,
            )

            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _pre_csv = os.path.join(_root, 'preprocessed_data', 'CHARLS_final_preprocessed.csv')
            _cfg = __import__('config')
            warn_if_imputed_older_than_preprocess(
                IMPUTED_DATA_PATH,
                _pre_csv,
                imputation_just_succeeded=False,
                enabled=getattr(_cfg, 'WARN_IMPUTED_OLDER_THAN_PREPROCESSED', True),
                log=logger,
            )
            df_clean = pd.read_csv(IMPUTED_DATA_PATH, encoding='utf-8-sig')
            if 'age' in df_clean.columns:
                df_clean = df_clean[df_clean['age'] >= AGE_MIN]
            prepare_exposures(df_clean)
            if apply_config_drop:
                df_clean = df_clean.drop(
                    columns=[c for c in COLS_TO_DROP if c in df_clean.columns], errors='ignore'
                )
            df_clean = reapply_cohort_definition(df_clean, CESD_CUTOFF, COGNITION_CUTOFF)
            log_imputed_csv_loaded(IMPUTED_DATA_PATH, log=logger)
            logger.info("脚本数据: 已加载插补数据 %s, n=%s", IMPUTED_DATA_PATH, len(df_clean))
        except Exception as ex:
            logger.warning("插补数据加载失败，回退预处理: %s", ex)
            df_clean = None
    if df_clean is None:
        df_clean = preprocess_charls_data(
            RAW_DATA_PATH,
            age_min=AGE_MIN,
            cesd_cutoff=CESD_CUTOFF,
            cognition_cutoff=COGNITION_CUTOFF,
            write_output=False,
        )
        if df_clean is not None:
            prepare_exposures(df_clean)
            if apply_config_drop:
                df_clean = df_clean.drop(
                    columns=[c for c in COLS_TO_DROP if c in df_clean.columns], errors="ignore"
                )
            logger.info("脚本数据: 使用预处理 CHARLS, n=%s", len(df_clean))
    return df_clean


def load_supervised_prediction_df(apply_config_drop=True):
    """
    **CPM / compare_models** 与主流程 `run_all_charls_analyses` 预测步对齐：在 USE_IMPUTED_DATA 时仍返回
    **预处理宽表**（保留缺失）+ `reapply_cohort_definition`，由 Pipeline 内 IterativeImputer 在 CV 训练折 fit。
    因果类脚本请继续用 `load_df_for_analysis()`（插补表）。
    """
    from config import USE_IMPUTED_DATA, RAW_DATA_PATH, AGE_MIN, COLS_TO_DROP, CESD_CUTOFF, COGNITION_CUTOFF
    from data.charls_complete_preprocessing import preprocess_charls_data, reapply_cohort_definition
    from scripts.run_multi_exposure_causal import prepare_exposures

    if not USE_IMPUTED_DATA:
        return load_df_for_analysis(apply_config_drop=apply_config_drop)
    df_pre = preprocess_charls_data(
        RAW_DATA_PATH,
        age_min=AGE_MIN,
        cesd_cutoff=CESD_CUTOFF,
        cognition_cutoff=COGNITION_CUTOFF,
        write_output=False,
    )
    if df_pre is None:
        logger.warning("预处理失败，load_supervised_prediction_df 回退 load_df_for_analysis")
        return load_df_for_analysis(apply_config_drop=apply_config_drop)
    prepare_exposures(df_pre)
    if apply_config_drop:
        df_pre = df_pre.drop(columns=[c for c in COLS_TO_DROP if c in df_pre.columns], errors='ignore')
    out = reapply_cohort_definition(df_pre, CESD_CUTOFF, COGNITION_CUTOFF)
    logger.info("脚本数据(CPM): 预处理缺失宽表 n=%s（与主流程预测步一致）", len(out))
    return out
