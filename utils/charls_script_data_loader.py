# -*- coding: utf-8 -*-
"""
独立运行 scripts 时与 run_all_charls_analyses 使用同一数据源（插补 vs 预处理）。
避免「主流程用插补、单跑脚本用原始预处理」导致结果不可比。
"""
import os
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# 仓库内 sample_data.csv 未含、但 CPM 冠军 Pipeline 的 ColumnTransformer 仍要求的列（与 CHARLS 编码一致）。
_BUNDLED_DEMO_COLUMN_DEFAULTS: dict[str, float] = {
    "fall_down": 0.0,
    "pension": 0.0,
    "ins": 0.0,
    "retire": 0.0,
    "disability": 0.0,
    "adlab_c": 0.0,
    "iadl": 0.0,
    # 演示缺列时用常见步速占位，避免单列全 NaN 在部分 sklearn 版本下异常
    "wspeed": 1.0,
}


def _pad_bundled_demo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """为云端演示表补齐建模输入列，避免 Preprocessor 报 columns are missing。"""
    out = df.copy()
    for col, default in _BUNDLED_DEMO_COLUMN_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
    return out


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_repo_path(rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(_repo_root(), rel_or_abs)


def _bundled_demo_table_path() -> str | None:
    """Streamlit Cloud / 无 CHARLS 时使用仓库内演示表（与插补表后处理一致，非完整科研样本）。"""
    p = os.path.join(_repo_root(), "data", "sample_data.csv")
    return p if os.path.isfile(p) else None


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
    from utils.charls_prepare_exposures import prepare_exposures

    df_clean = None
    _imp_abs = _resolve_repo_path(IMPUTED_DATA_PATH) if IMPUTED_DATA_PATH else ""
    if USE_IMPUTED_DATA and IMPUTED_DATA_PATH and os.path.isfile(_imp_abs):
        try:
            from utils.imputation_data_provenance import (
                log_imputed_csv_loaded,
                warn_if_imputed_older_than_preprocess,
            )

            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _pre_csv = os.path.join(_root, 'preprocessed_data', 'CHARLS_final_preprocessed.csv')
            _cfg = __import__('config')
            warn_if_imputed_older_than_preprocess(
                _imp_abs,
                _pre_csv,
                imputation_just_succeeded=False,
                enabled=getattr(_cfg, 'WARN_IMPUTED_OLDER_THAN_PREPROCESSED', True),
                log=logger,
            )
            df_clean = pd.read_csv(_imp_abs, encoding='utf-8-sig')
            if 'age' in df_clean.columns:
                df_clean = df_clean[df_clean['age'] >= AGE_MIN]
            prepare_exposures(df_clean)
            if apply_config_drop:
                df_clean = df_clean.drop(
                    columns=[c for c in COLS_TO_DROP if c in df_clean.columns], errors='ignore'
                )
            df_clean = reapply_cohort_definition(df_clean, CESD_CUTOFF, COGNITION_CUTOFF)
            log_imputed_csv_loaded(_imp_abs, log=logger)
            logger.info("脚本数据: 已加载插补数据 %s, n=%s", _imp_abs, len(df_clean))
        except Exception as ex:
            logger.warning("插补数据加载失败，回退预处理: %s", ex)
            df_clean = None
    if df_clean is None:
        raw_abs = _resolve_repo_path(RAW_DATA_PATH)
        if os.path.isfile(raw_abs):
            df_clean = preprocess_charls_data(
                raw_abs,
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
        else:
            logger.info("原始 CHARLS 不存在 (%s)，跳过 preprocess_charls_data。", raw_abs)
    if df_clean is None:
        demo = _bundled_demo_table_path()
        if demo:
            try:
                logger.warning(
                    "无插补表且无原始 CHARLS：加载仓库演示数据 %s（仅用于 Streamlit / 脱机演示）。",
                    demo,
                )
                df_clean = pd.read_csv(demo, encoding="utf-8-sig")
                df_clean = _pad_bundled_demo_columns(df_clean)
                if "age" in df_clean.columns:
                    df_clean = df_clean[df_clean["age"] >= AGE_MIN]
                prepare_exposures(df_clean)
                if apply_config_drop:
                    df_clean = df_clean.drop(
                        columns=[c for c in COLS_TO_DROP if c in df_clean.columns],
                        errors="ignore",
                    )
                df_clean = reapply_cohort_definition(df_clean, CESD_CUTOFF, COGNITION_CUTOFF)
                if df_clean is not None and len(df_clean) > 0:
                    logger.info("演示数据加载成功, n=%s", len(df_clean))
                else:
                    logger.error("演示表经队列定义后为空，请检查 sample_data.csv。")
                    df_clean = None
            except Exception as ex:
                logger.warning("演示表加载失败: %s", ex)
                df_clean = None
    return df_clean


def load_supervised_prediction_df(apply_config_drop=True):
    """
    **CPM / compare_models** 与主流程 `run_all_charls_analyses` 预测步对齐：在 USE_IMPUTED_DATA 时仍返回
    **预处理宽表**（保留缺失）+ `reapply_cohort_definition`，由 Pipeline 内 IterativeImputer 在 CV 训练折 fit。
    因果类脚本请继续用 `load_df_for_analysis()`（插补表）。
    """
    from config import USE_IMPUTED_DATA, RAW_DATA_PATH, AGE_MIN, COLS_TO_DROP, CESD_CUTOFF, COGNITION_CUTOFF
    from data.charls_complete_preprocessing import preprocess_charls_data, reapply_cohort_definition
    from utils.charls_prepare_exposures import prepare_exposures

    if not USE_IMPUTED_DATA:
        return load_df_for_analysis(apply_config_drop=apply_config_drop)
    raw_abs = _resolve_repo_path(RAW_DATA_PATH)
    if not os.path.isfile(raw_abs):
        logger.info(
            "CPM 需预处理宽表但原始 CHARLS 不存在 (%s)，回退 load_df_for_analysis（含演示表）。",
            raw_abs,
        )
        return load_df_for_analysis(apply_config_drop=apply_config_drop)
    df_pre = preprocess_charls_data(
        raw_abs,
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
