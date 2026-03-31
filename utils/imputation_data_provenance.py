# -*- coding: utf-8 -*-
"""
插补结果文件溯源：记录修改时间、检测「预处理已更新但仍在用旧插补」的情况。
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def format_mtime(path: str) -> str:
    if not os.path.isfile(path):
        return "(文件不存在)"
    mtime = os.path.getmtime(path)
    return datetime.datetime.fromtimestamp(mtime).isoformat(sep=" ", timespec="seconds")


def log_imputed_csv_loaded(imputed_path: str, log: Optional[logging.Logger] = None) -> None:
    """主分析/脚本加载 step1_imputed_full 时调用，便于日志核对是否为预期版本。"""
    log = log or logger
    abs_p = os.path.abspath(imputed_path)
    if not os.path.isfile(imputed_path):
        log.warning("插补路径不存在: %s", abs_p)
        return
    log.info("📅 当前使用的插补数据: %s | 文件修改时间: %s", abs_p, format_mtime(imputed_path))


def warn_if_imputed_older_than_preprocess(
    imputed_path: str,
    preprocess_path: str,
    *,
    imputation_just_succeeded: bool = False,
    enabled: bool = True,
    log: Optional[logging.Logger] = None,
) -> None:
    """
    当未在本轮主流程内成功重跑插补时，若预处理表比插补 CSV 更新，提示可能误用旧插补。
    imputation_just_succeeded=True 时跳过（本轮已覆盖 step1_imputed_full）。
    """
    log = log or logger
    if not enabled or imputation_just_succeeded:
        return
    if not os.path.isfile(imputed_path) or not os.path.isfile(preprocess_path):
        return
    t_imp = os.path.getmtime(imputed_path)
    t_pre = os.path.getmtime(preprocess_path)
    if t_pre <= t_imp:
        return
    log.warning(
        "⚠️ 预处理表晚于当前插补文件，主分析可能仍在使用**旧插补结果**：\n"
        "   预处理: %s (%s)\n"
        "   插补:   %s (%s)\n"
        "   对策：将 config.RUN_IMPUTATION_BEFORE_MAIN=True 后重跑主流程，或手动运行 "
        "`python archive/charls_imputation_npj_style.py` 再分析。",
        os.path.abspath(preprocess_path),
        format_mtime(preprocess_path),
        os.path.abspath(imputed_path),
        format_mtime(imputed_path),
    )
