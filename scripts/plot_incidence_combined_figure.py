# -*- coding: utf-8 -*-
"""
生成「累积发病率 + 发病密度」组合图（与 Table 1b / charls_incidence_density 同源定义）。

默认读取 preprocessed_data/CHARLS_final_preprocessed.csv（与主文 STROBE 一致、插补前观测）。
也可指定 --csv 为插补全表（与主流程 df_clean 一致时自行选用）。

用法:
  python scripts/plot_incidence_combined_figure.py
  python scripts/plot_incidence_combined_figure.py --csv preprocessed_data/CHARLS_final_preprocessed.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.chdir(_ROOT)


def main():
    from config import AGE_MIN, CESD_CUTOFF, COGNITION_CUTOFF, RAW_DATA_PATH, RESULTS_FIGURES, RESULTS_TABLES
    from data.charls_complete_preprocessing import preprocess_charls_data
    from viz.fig_incidence_cumulative_and_density import draw_incidence_combined_figure

    p = argparse.ArgumentParser(description="Plot cumulative incidence + incidence density (1,000 PY)")
    p.add_argument(
        "--csv",
        default=os.path.join("preprocessed_data", "CHARLS_final_preprocessed.csv"),
        help="含 baseline_group, wave, is_comorbidity_next 的长表 CSV",
    )
    p.add_argument(
        "-o",
        "--output",
        default=os.path.join(RESULTS_FIGURES, "fig_incidence_cumulative_and_density.png"),
        help="输出 PNG 路径",
    )
    args = p.parse_args()

    if os.path.isfile(args.csv):
        import pandas as pd

        df = pd.read_csv(args.csv, encoding="utf-8-sig")
        logger.info("Loaded: %s (%s rows)", args.csv, len(df))
    else:
        logger.warning("未找到 %s，从 RAW 运行 preprocess_charls_data …", args.csv)
        raw = os.path.join(_ROOT, RAW_DATA_PATH) if not os.path.isabs(RAW_DATA_PATH) else RAW_DATA_PATH
        df = preprocess_charls_data(
            raw,
            cesd_cutoff=CESD_CUTOFF,
            cognition_cutoff=COGNITION_CUTOFF,
            age_min=AGE_MIN,
            write_output=False,
        )
        if df is None:
            raise SystemExit("无法加载数据")

    need = {"baseline_group", "wave", "is_comorbidity_next"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"数据缺少列: {miss}")

    os.makedirs(RESULTS_FIGURES, exist_ok=True)
    cum_df, dens_df = draw_incidence_combined_figure(df, args.output)
    side_csv = os.path.join(RESULTS_TABLES, "fig_incidence_combined_source_stats.csv")
    os.makedirs(RESULTS_TABLES, exist_ok=True)
    out_stats = cum_df.copy()
    out_stats["Incidence_Rate_per_1000_PY"] = dens_df["Incidence_Rate_per_1000_PY"].values
    out_stats["Total_Person_Years"] = dens_df["Total_Person_Years"].values
    out_stats["Baseline_Cohort_label"] = dens_df["Baseline_Cohort"].values
    out_stats.to_csv(side_csv, index=False, encoding="utf-8-sig")
    logger.info("Sidecar stats: %s", side_csv)
    print(cum_df.to_string(index=False))
    print(dens_df[["Baseline_Cohort", "Incidence_Rate_per_1000_PY", "Total_Person_Years"]].to_string(index=False))


if __name__ == "__main__":
    main()
