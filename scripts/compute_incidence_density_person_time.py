# -*- coding: utf-8 -*-
"""
CHARLS 多波纵向数据：抑郁–认知共病「首发」人年（Person-Years）与发病密度（Incidence Density）

流行病学约定（与本脚本一致）：
- 每个 risk interval 对应「当前波 t → 下一波 t+1」（与 is_comorbidity_next 一致），假定相邻波间隔约 2 年。
- 该区间内若发生终点（is_comorbidity_next == 1）：按中点法贡献 1 人年；若未发生（0）：贡献 2 人年（完整随访至下一调查时点）。
- 同一 ID 可有多条 person-wave；**首发共病**只计一次：一旦出现 is_comorbidity_next==1，该人不再累计后续区间人年（已不在「首发共病」风险集）。

分层变量：取每人**进入分析的第一条记录**（按 wave 最小）上的 cohort / 年龄组 / 性别 / 城乡。

用法:
  python scripts/compute_incidence_density_person_time.py
  python scripts/compute_incidence_density_person_time.py --csv imputation_npj_results/pipeline_trace/step1_imputed_full.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

# 项目根加入路径
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# 常数：与预处理一致（CHARLS 波间约 2 年；中点法发病区间计 1 年）
# ---------------------------------------------------------------------------
INTERVAL_YEARS_NO_EVENT = 2.0
INTERVAL_YEARS_INCIDENT = 1.0  # 区间内发病：中点假设

COHORT_MAP = {
    0: "Cohort A (healthy baseline)",
    1: "Cohort B (depression only)",
    2: "Cohort C (cognitive impairment only)",
}


def _age_group(age: float | int) -> str:
    """年龄组：与常见老年分层一致（入组已 ≥60 时，<65 即 60–64）。"""
    if pd.isna(age):
        return "Age missing"
    a = float(age)
    if a < 65:
        return "Age <65 years"
    if a < 75:
        return "Age 65–74 years"
    return "Age ≥75 years"


def _gender_label(v: Any) -> str:
    if pd.isna(v):
        return "Sex missing"
    try:
        x = int(float(v))
    except (TypeError, ValueError):
        return str(v)
    # 与本项目 Table 1 一致（charls_table1_stats）：Female=1, Male=0
    if x == 1:
        return "Female"
    if x == 0:
        return "Male"
    return f"Sex code {x}"


def _rural_label(v: Any) -> str:
    if pd.isna(v):
        return "Residence missing"
    try:
        x = int(float(v))
    except (TypeError, ValueError):
        return str(v)
    if x == 0:
        return "Urban"
    if x == 1:
        return "Rural"
    return f"Rural code {x}"


def compute_person_py_and_incident(
    group: pd.DataFrame,
    outcome_col: str = "is_comorbidity_next",
    wave_col: str = "wave",
) -> pd.Series:
    """
    对单个 ID 的多条 person-wave（已按风险集定义筛选过的行）：
    按 wave 升序逐段累计人年，直至首次 is_comorbidity_next==1（含该段）后停止。
    """
    g = group.sort_values(wave_col)
    total_py = 0.0
    incident = 0
    details: list[tuple[Any, float, bool]] = []

    for _, row in g.iterrows():
        y = row[outcome_col]
        if pd.isna(y):
            # 与 incident 分析一致：不应出现；若出现则终止累计以免误算
            break
        y_int = int(float(y))
        if y_int == 1:
            total_py += INTERVAL_YEARS_INCIDENT
            incident = 1
            details.append((row[wave_col], INTERVAL_YEARS_INCIDENT, True))
            break
        total_py += INTERVAL_YEARS_NO_EVENT
        details.append((row[wave_col], INTERVAL_YEARS_NO_EVENT, False))

    return pd.Series(
        {
            "total_py": total_py,
            "incident": incident,
            "n_intervals_used": len(details),
            "interval_details": details,
        }
    )


def build_person_level_table(
    df: pd.DataFrame,
    id_col: str = "ID",
    wave_col: str = "wave",
    outcome_col: str = "is_comorbidity_next",
    cohort_col: str = "baseline_group",
    age_col: str = "age",
    gender_col: str = "gender",
    rural_col: str = "rural",
    debug_print_ids: list[Any] | None = None,
) -> pd.DataFrame:
    """
    输出每人一行：总人年、是否首发共病、以及用于分层的「入组特征」（首波记录）。
    """
    need = [id_col, wave_col, outcome_col, cohort_col, age_col, gender_col, rural_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"数据缺少列: {miss}")

    dbg = set(debug_print_ids or [])

    def _one_pid(pid: pd.DataFrame) -> pd.Series:
        pid_val = pid[id_col].iloc[0]
        entry = pid.sort_values(wave_col).iloc[0]
        met = compute_person_py_and_incident(pid, outcome_col=outcome_col, wave_col=wave_col)

        if dbg and pid_val in dbg:
            print("\n" + "=" * 60)
            print(f"[DEBUG] ID = {pid_val}")
            g = pid.sort_values(wave_col)
            for _, row in g.iterrows():
                print(
                    f"  wave={row[wave_col]!r}, {outcome_col}={row[outcome_col]!r}, "
                    f"baseline_group={row[cohort_col]!r}, age={row[age_col]!r}"
                )
            print("  → 区间累计（按时间顺序）:")
            for w, py_i, ev in met["interval_details"]:
                tag = "INCIDENT (midpoint 1 PY)" if ev else "no event (2 PY)"
                print(f"     wave {w}→next: +{py_i} PY  [{tag}]")
            print(f"  → total_py={met['total_py']}, incident={met['incident']}")
            print("=" * 60)

        return pd.Series(
            {
                id_col: pid_val,
                "total_py": met["total_py"],
                "incident": met["incident"],
                "n_intervals_used": met["n_intervals_used"],
                "cohort_entry": COHORT_MAP.get(int(float(entry[cohort_col])), str(entry[cohort_col])),
                "age_group_entry": _age_group(entry[age_col]),
                "sex_entry": _gender_label(entry[gender_col]),
                "rural_entry": _rural_label(entry[rural_col]),
                "wave_entry": entry[wave_col],
            }
        )

    # 避免 pandas 2.x groupby.apply 对分组列的 FutureWarning
    rows: list[pd.Series] = []
    for _, pid_df in df.groupby(id_col, sort=False):
        rows.append(_one_pid(pid_df))
    out = pd.DataFrame(rows)
    return out


def _stratum_summary(
    pl: pd.DataFrame,
    mask: pd.Series,
    label: str,
) -> dict[str, Any]:
    sub = pl.loc[mask]
    n = sub.shape[0]
    cases = int(sub["incident"].sum())
    py = float(sub["total_py"].sum())
    dens = (cases / py * 1000.0) if py > 0 else np.nan
    return {
        "Baseline_Characteristics": label,
        "Total_N_unique_IDs": n,
        "Incident_Cases": cases,
        "Total_Person_Years": round(py, 4),
        "Incidence_Density_per_1000_PYs": round(dens, 4) if pd.notna(dens) else np.nan,
    }


def build_incidence_density_table(
    pl: pd.DataFrame,
) -> pd.DataFrame:
    """
    由每人一行汇总表生成期刊用「Person-Time / Incidence Density」长表。
    """
    rows: list[dict[str, Any]] = []

    rows.append(_stratum_summary(pl, pd.Series(True, index=pl.index), "Overall"))

    for k in sorted(COHORT_MAP.keys()):
        lab = COHORT_MAP[k]
        m = pl["cohort_entry"] == lab
        rows.append(_stratum_summary(pl, m, lab))

    for ag in ["Age <65 years", "Age 65–74 years", "Age ≥75 years", "Age missing"]:
        rows.append(_stratum_summary(pl, pl["age_group_entry"] == ag, ag))

    for sx in ["Male", "Female", "Sex missing"]:
        rows.append(_stratum_summary(pl, pl["sex_entry"] == sx, sx))

    for ru in ["Urban", "Rural", "Residence missing"]:
        rows.append(_stratum_summary(pl, pl["rural_entry"] == ru, ru))

    return pd.DataFrame(rows)


def load_incident_long_data(csv_path: str) -> pd.DataFrame:
    """读取长表；默认使用主分析插补全表（已含 incident 队列行）。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Incidence density (person-time) table for CHARLS")
    parser.add_argument(
        "--csv",
        default=None,
        help="长表 CSV（如 step1_imputed_full.csv）；默认读 config.IMPUTED_DATA_PATH",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出 CSV 路径；默认 results/tables/table_incidence_density_person_time.csv",
    )
    parser.add_argument(
        "--debug-ids",
        default="",
        help="逗号分隔的 ID，用于打印人年分解（例: 10104221002,51606214001）",
    )
    args = parser.parse_args()

    try:
        from config import IMPUTED_DATA_PATH, RESULTS_TABLES
    except ImportError:
        IMPUTED_DATA_PATH = os.path.join(_ROOT, "imputation_npj_results/pipeline_trace/step1_imputed_full.csv")
        RESULTS_TABLES = os.path.join(_ROOT, "results/tables")

    csv_path = args.csv or IMPUTED_DATA_PATH
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"找不到数据文件: {csv_path}")

    out_path = args.out or os.path.join(RESULTS_TABLES, "table_incidence_density_person_time.csv")

    df = load_incident_long_data(csv_path)
    print(f"Loaded rows: {len(df)} from {csv_path}")

    dbg_ids = [x.strip() for x in args.debug_ids.split(",") if x.strip()]
    # 尝试与 ID 列类型一致（常见为 int）
    if dbg_ids and "ID" in df.columns:
        sample_id = df["ID"].iloc[0]
        if isinstance(sample_id, (int, np.integer)):
            dbg_ids = [int(x) if x.isdigit() else x for x in dbg_ids]

    pl = build_person_level_table(df, debug_print_ids=dbg_ids or None)
    incidence_density_table = build_incidence_density_table(pl)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    incidence_density_table.to_csv(out_path, index=False, encoding="utf-8-sig")
    pl_out = out_path.replace(".csv", "_person_level.csv")
    pl.to_csv(pl_out, index=False, encoding="utf-8-sig")

    print("\nincidence_density_table:")
    print(incidence_density_table.to_string(index=False))
    print(f"\n已保存: {out_path}")
    print(f"每人一行明细: {pl_out}")


if __name__ == "__main__":
    main()
