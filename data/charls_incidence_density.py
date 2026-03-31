# -*- coding: utf-8 -*-
"""
Discrete-time person-wave 人年（PY）与发病密度（每 1,000 人年），按导师要求的中点法 + CHARLS 波间间隔。

每行对应一次 person-wave，结局列 is_comorbidity_next 表示该区间末是否发生共病：
- 未发病 (0)：贡献完整区间长度 person_years = interval
- 发病 (1)：中点法 person_years = interval / 2

波次与至下一波间隔（年）：
  wave 1 -> 2.0；wave 2 -> 2.0；wave 3 -> 3.0；wave 4 -> 0.0（无下一波可定义结局，贡献 0 人年）
"""
from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WAVE_TO_INTERVAL_YEARS = {
    1: 2.0,
    2: 2.0,
    3: 3.0,
    4: 0.0,
}


def interval_years_for_wave(wave) -> float:
    try:
        w = int(float(wave))
    except (TypeError, ValueError):
        return 0.0
    return float(WAVE_TO_INTERVAL_YEARS.get(w, 0.0))


def person_years_per_row(is_event, interval: float) -> float:
    """Midpoint convention: event -> half interval; censored -> full interval."""
    if interval <= 0:
        return 0.0
    try:
        y = int(float(is_event))
    except (TypeError, ValueError):
        return 0.0
    if y == 1:
        return interval / 2.0
    return interval


def compute_incidence_density_by_baseline_group(
    df: pd.DataFrame,
    outcome_col: str = "is_comorbidity_next",
    wave_col: str = "wave",
    cohort_col: str = "baseline_group",
) -> pd.DataFrame:
    """
    按 baseline_group 汇总 + Total 行。

    Returns
    -------
    DataFrame columns:
        Baseline_Cohort, Person_Wave_Observations, Total_Person_Years,
        Incident_Cases, Incidence_Rate_per_1000_PY
    """
    need = [outcome_col, wave_col, cohort_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"compute_incidence_density_by_baseline_group: missing columns {missing}")

    work = df[list(dict.fromkeys(need))].copy()
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work[wave_col] = pd.to_numeric(work[wave_col], errors="coerce")
    work[cohort_col] = pd.to_numeric(work[cohort_col], errors="coerce")
    work = work.dropna(subset=[outcome_col, wave_col, cohort_col])

    intervals = work[wave_col].map(interval_years_for_wave)
    work["_interval"] = intervals
    work["_py"] = [
        person_years_per_row(ev, iv)
        for ev, iv in zip(work[outcome_col].values, work["_interval"].values)
    ]

    cohort_labels = {
        0: "Cohort A (healthy baseline)",
        1: "Cohort B (depression only)",
        2: "Cohort C (cognitive impairment only)",
    }

    rows = []
    for g in (0, 1, 2):
        sub = work[work[cohort_col] == g]
        n = len(sub)
        py = float(sub["_py"].sum())
        cases = int(sub[outcome_col].sum())
        ir = (cases / py * 1000.0) if py > 0 else np.nan
        rows.append(
            {
                "Baseline_Cohort": cohort_labels.get(g, f"Group {g}"),
                "Person_Wave_Observations": n,
                "Total_Person_Years": round(py, 4),
                "Incident_Cases": cases,
                "Incidence_Rate_per_1000_PY": round(ir, 4) if pd.notna(ir) else np.nan,
            }
        )

    py_tot = float(work["_py"].sum())
    cases_tot = int(work[outcome_col].sum())
    ir_tot = (cases_tot / py_tot * 1000.0) if py_tot > 0 else np.nan
    rows.append(
        {
            "Baseline_Cohort": "Total",
            "Person_Wave_Observations": len(work),
            "Total_Person_Years": round(py_tot, 4),
            "Incident_Cases": cases_tot,
            "Incidence_Rate_per_1000_PY": round(ir_tot, 4) if pd.notna(ir_tot) else np.nan,
        }
    )

    return pd.DataFrame(rows)


def save_incidence_density_table(
    df: pd.DataFrame,
    output_root: str,
    results_tables: Optional[str] = None,
    filename: str = "table1b_incidence_density.csv",
) -> tuple[str, Optional[str]]:
    """
    写入 OUTPUT_ROOT 与可选的 results/tables/。
    Returns (primary_path, results_path or None)
    """
    os.makedirs(output_root, exist_ok=True)
    primary = os.path.join(output_root, filename)
    out_df = compute_incidence_density_by_baseline_group(df)
    out_df.to_csv(primary, index=False, encoding="utf-8-sig")
    logger.info("Incidence density table saved: %s", primary)

    sec = None
    if results_tables:
        os.makedirs(results_tables, exist_ok=True)
        sec = os.path.join(results_tables, filename)
        out_df.to_csv(sec, index=False, encoding="utf-8-sig")
        logger.info("Incidence density table copied: %s", sec)
    return primary, sec
