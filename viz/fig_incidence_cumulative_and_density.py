# -*- coding: utf-8 -*-
"""
组合图：左栏 累积发病率（观测口径 crude %），右栏 发病密度（每 1,000 人年）。

与 `data.charls_incidence_density` 的中点法、波间人年定义一致；建议与 Table 1b 同源传入 `df`
（主流程中优先 `df_pre`，与 STROBE/插补前观测一致）。
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import BBOX_INCHES

logger = logging.getLogger(__name__)

from data.charls_incidence_density import (  # noqa: E402
    compute_incidence_density_by_baseline_group,
    interval_years_for_wave,
    person_years_per_row,
)

# 色弱友好（近似 Paul Tol muted）
COLORS = ("#4477AA", "#EE6677", "#228833")
COHORT_XLABELS = (
    "Cohort A\n(healthy baseline)",
    "Cohort B\n(depression only)",
    "Cohort C\n(cognition impaired only)",
)


def _analysis_subset(
    df: pd.DataFrame,
    outcome_col: str,
    wave_col: str,
    cohort_col: str,
) -> pd.DataFrame:
    need = [outcome_col, wave_col, cohort_col]
    work = df[list(dict.fromkeys(need))].copy()
    for c in need:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    return work.dropna(subset=need)


def compute_cumulative_incidence_percent(
    df: pd.DataFrame,
    outcome_col: str = "is_comorbidity_next",
    wave_col: str = "wave",
    cohort_col: str = "baseline_group",
) -> pd.DataFrame:
    """
    离散 person-wave 口径：每行表示一个随访区间，结局为区间末是否发病。
    Crude incidence proportion (%) = 发病行数 / 该组有效行数 × 100。
    """
    work = _analysis_subset(df, outcome_col, wave_col, cohort_col)
    rows = []
    for g in (0, 1, 2):
        sub = work[work[cohort_col] == g]
        n = len(sub)
        cases = int(sub[outcome_col].sum())
        pct = (100.0 * cases / n) if n > 0 else np.nan
        se_prop = np.sqrt((pct / 100.0) * (1 - pct / 100.0) / n) * 100.0 if n > 0 and 0 < pct < 100 else (
            np.sqrt(0.25 / n) * 100.0 if n > 0 else np.nan
        )
        ci_lo = max(0.0, pct - 1.96 * se_prop) if pd.notna(pct) else np.nan
        ci_hi = min(100.0, pct + 1.96 * se_prop) if pd.notna(pct) else np.nan
        rows.append(
            {
                "baseline_group": g,
                "n_observations": n,
                "incident_cases": cases,
                "cumulative_incidence_pct": round(pct, 2) if pd.notna(pct) else np.nan,
                "ci95_low_pct": round(ci_lo, 2) if pd.notna(ci_lo) else np.nan,
                "ci95_high_pct": round(ci_hi, 2) if pd.notna(ci_hi) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _rate_ci_per_1000_py(cases: int, person_years: float) -> tuple[float, float]:
    """Poisson 正态近似：率 = cases/PY×1000，SE ≈ sqrt(cases)/PY×1000。"""
    if person_years <= 0 or cases < 0:
        return np.nan, np.nan
    r = cases / person_years * 1000.0
    if cases == 0:
        # 简单上界：避免除零
        hi = 3.0 / person_years * 1000.0  # ~95% Poisson upper for 0 events
        return 0.0, hi
    se = np.sqrt(float(cases)) / person_years * 1000.0
    return r - 1.96 * se, r + 1.96 * se


def draw_incidence_combined_figure(
    df: pd.DataFrame,
    output_path: str,
    outcome_col: str = "is_comorbidity_next",
    wave_col: str = "wave",
    cohort_col: str = "baseline_group",
    dpi: int = 300,
    title: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    左：各 Cohort crude cumulative incidence (%)；右：incidence rate per 1,000 person-years（中点法）。
    Returns (cum_inc_df, density_df 前三行)
    """
    _dir = os.path.dirname(os.path.abspath(output_path))
    if _dir:
        os.makedirs(_dir, exist_ok=True)

    cum_df = compute_cumulative_incidence_percent(df, outcome_col, wave_col, cohort_col)
    dens_tbl = compute_incidence_density_by_baseline_group(df, outcome_col, wave_col, cohort_col)
    dens_3 = dens_tbl[dens_tbl["Baseline_Cohort"] != "Total"].copy()

    x = np.arange(3)
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.6),
        gridspec_kw={"wspace": 0.35},
        constrained_layout=True,
    )

    # --- Left: cumulative incidence % ---
    pcts = cum_df["cumulative_incidence_pct"].to_numpy(dtype=float)
    err_lo = pcts - cum_df["ci95_low_pct"].to_numpy(dtype=float)
    err_hi = cum_df["ci95_high_pct"].to_numpy(dtype=float) - pcts
    err = np.vstack([err_lo, err_hi])
    bars1 = ax1.bar(x, pcts, color=COLORS, edgecolor="0.2", linewidth=0.6, zorder=2)
    ax1.errorbar(x, pcts, yerr=err, fmt="none", ecolor="0.35", capsize=4, capthick=1.0, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(COHORT_XLABELS, fontsize=9)
    ax1.set_ylabel("Crude incidence (%)", fontsize=10)
    ax1.set_title(
        "A. Proportion with incident comorbidity\n(next-interval outcome, person-wave rows)",
        fontsize=10,
        loc="left",
    )
    ax1.set_ylim(0, max(5.0, (pcts.max() if len(pcts) else 0) * 1.25))
    for i, (b, v, n, c) in enumerate(
        zip(bars1, pcts, cum_df["n_observations"], cum_df["incident_cases"])
    ):
        if pd.isna(v):
            continue
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + max(0.3, ax1.get_ylim()[1] * 0.02),
            f"{v:.1f}%\n(n={int(c)}/{int(n)})",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.2",
        )

    # --- Right: incidence density per 1,000 PY ---
    rates = dens_3["Incidence_Rate_per_1000_PY"].to_numpy(dtype=float)
    py_vals = dens_3["Total_Person_Years"].to_numpy(dtype=float)
    case_vals = dens_3["Incident_Cases"].to_numpy(dtype=int)
    yerr_lo = np.zeros(3)
    yerr_hi = np.zeros(3)
    for i in range(3):
        lo, hi = _rate_ci_per_1000_py(int(case_vals[i]), float(py_vals[i]))
        if pd.notna(rates[i]) and pd.notna(lo):
            yerr_lo[i] = rates[i] - lo
            yerr_hi[i] = hi - rates[i]
        else:
            yerr_lo[i] = yerr_hi[i] = 0.0

    bars2 = ax2.bar(x, rates, color=COLORS, edgecolor="0.2", linewidth=0.6, zorder=2)
    ax2.errorbar(
        x,
        rates,
        yerr=np.vstack([yerr_lo, yerr_hi]),
        fmt="none",
        ecolor="0.35",
        capsize=4,
        capthick=1.0,
        zorder=3,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(COHORT_XLABELS, fontsize=9)
    ax2.set_ylabel("Incidence rate (per 1,000 person-years)", fontsize=10)
    ax2.set_title(
        "B. Incidence density (midpoint rule\nfor event time, CHARLS wave intervals)",
        fontsize=10,
        loc="left",
    )
    ymax = max(5.0, float(np.nanmax(rates)) * 1.2) if len(rates) else 10.0
    ax2.set_ylim(0, ymax)
    for b, v in zip(bars2, rates):
        if pd.isna(v):
            continue
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + ymax * 0.02,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="0.2",
        )

    if title:
        fig.suptitle(title, fontsize=11)
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.08, hspace=0, wspace=0.35)

    plt.savefig(output_path, dpi=dpi, bbox_inches=BBOX_INCHES, facecolor="white")
    plt.close()
    logger.info("Combined incidence figure saved: %s", output_path)
    return cum_df, dens_3


__all__ = [
    "compute_cumulative_incidence_percent",
    "draw_incidence_combined_figure",
    "COHORT_XLABELS",
    "COLORS",
]
