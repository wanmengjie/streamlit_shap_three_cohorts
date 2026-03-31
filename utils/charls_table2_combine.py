# -*- coding: utf-8 -*-
"""
将三队列 CPM 主表 `table2_prediction_cohort{A,B,C}.csv` 纵向合并为一张表，
便于投稿/附录一次性引用。
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_COHORT_ORDER = {"Cohort A": 0, "Cohort B": 1, "Cohort C": 2}
_DEFAULT_OUT = "table2_prediction_combined_ABC.csv"


def write_combined_table2_prediction(
    results_tables: str,
    output_filename: str = _DEFAULT_OUT,
) -> Optional[str]:
    """
    读取 results_tables 下 table2_prediction_cohortA.csv / B / C，合并后写入 output_filename。

    Returns
    -------
    写出文件的绝对路径；若三文件均不存在则返回 None。
    """
    pieces: list[pd.DataFrame] = []
    for cid in ("A", "B", "C"):
        p = os.path.join(results_tables, f"table2_prediction_cohort{cid}.csv")
        if not os.path.isfile(p):
            logger.warning("合并 Table 2：缺少 %s，该队列跳过", p)
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        label = f"Cohort {cid}"
        df.insert(0, "Cohort", label)
        pieces.append(df)

    if not pieces:
        logger.warning("合并 Table 2：未找到任何 table2_prediction_cohort*.csv")
        return None

    full = pd.concat(pieces, ignore_index=True)
    full["_co_order"] = full["Cohort"].map(_COHORT_ORDER)
    if "AUC" in full.columns:
        full = full.sort_values(["_co_order", "AUC"], ascending=[True, False])
    else:
        full = full.sort_values("_co_order")
    full = full.drop(columns=["_co_order"])

    os.makedirs(results_tables, exist_ok=True)
    out_path = os.path.join(results_tables, output_filename)
    full.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("已写入合并 Table 2: %s (%s 行)", out_path, len(full))
    return out_path
