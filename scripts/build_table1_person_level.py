# -*- coding: utf-8 -*-
"""
Person-level Table 1（与主流程 Table 1 相同的 BPS 变量结构，见 data.charls_table1_stats.BPS_SECTIONS）

数据来源：preprocessed_data/CHARLS_final_preprocessed.csv，或 config.RAW_DATA_PATH → preprocess_charls_data。

STROBE：按 Unique ID 的 entry wave（sort ID,wave 后每人保留首行）；分组 baseline_group 0/1/2。
连续变量 P 值与主表一致为 Kruskal-Wallis；分类为 Chi-square（见 tabulate_baseline_table_bps）。
"""
from __future__ import annotations

import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config import AGE_MIN, CESD_CUTOFF, COGNITION_CUTOFF, RAW_DATA_PATH, RESULTS_TABLES  # noqa: E402
from data.charls_complete_preprocessing import preprocess_charls_data  # noqa: E402
from data.charls_table1_stats import tabulate_baseline_table_bps  # noqa: E402
from scripts.run_multi_exposure_causal import prepare_exposures  # noqa: E402

GROUP_COL = "baseline_group"
COHORT_COLS = {
    0: "A: Healthy",
    1: "B: Depression-only",
    2: "C: Cognition-only",
}


def _load_incident_long_table():
    pre_csv = os.path.join(_ROOT, "preprocessed_data", "CHARLS_final_preprocessed.csv")
    if os.path.isfile(pre_csv):
        logger.info("加载预处理表: %s", pre_csv)
        import pandas as pd

        return pd.read_csv(pre_csv, encoding="utf-8-sig")
    logger.info("未找到预处理 CSV，从原始数据运行 preprocess_charls_data …")
    raw = os.path.join(_ROOT, RAW_DATA_PATH) if not os.path.isabs(RAW_DATA_PATH) else RAW_DATA_PATH
    df = preprocess_charls_data(
        raw,
        cesd_cutoff=CESD_CUTOFF,
        cognition_cutoff=COGNITION_CUTOFF,
        age_min=AGE_MIN,
        write_output=False,
    )
    if df is None:
        raise RuntimeError("无法加载或生成预处理数据。")
    return df


def main():
    os.chdir(_ROOT)
    df_long = _load_incident_long_table()
    if "ID" not in df_long.columns or "wave" not in df_long.columns:
        raise ValueError("数据需含 ID 与 wave")

    df_long = df_long.sort_values(["ID", "wave"], kind="mergesort")
    df_entry = df_long.drop_duplicates(subset=["ID"], keep="first").copy()
    logger.info("Person-waves: %s → Unique ID (entry wave): %s", len(df_long), len(df_entry))

    # sleep_adequate 与主流程 Table 1 一致（BPS Lifestyle）
    prepare_exposures(df_entry)

    # 若仅有 cesd 列而无 cesd10，供 Defining 区块使用
    if "cesd10" not in df_entry.columns:
        for c in df_entry.columns:
            if "cesd" in c.lower():
                df_entry["cesd10"] = df_entry[c]
                break

    table = tabulate_baseline_table_bps(
        df_entry,
        group_col=GROUP_COL,
        group_labels=COHORT_COLS,
        add_pvalues=True,
        p_col_name="P-value",
    )
    if table is None:
        raise RuntimeError("生成 Table 1 失败（缺少 baseline_group？）")

    out_dir = os.path.join(_ROOT, RESULTS_TABLES)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "table1_person_level.csv")
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("已写入: %s（行数 %s）", out_path, len(table))
    return table


if __name__ == "__main__":
    main()
