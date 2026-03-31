# -*- coding: utf-8 -*-
"""
Love plot 风格：协变量平衡（未调整 vs IPW 加权 SMD）。

说明：assumption_balance_exercise.txt 仅含 max SMD 等摘要；**逐协变量 SMD** 在
`assumption_balance_smd_{treatment}.csv` 与 `assumption_balance_smd_{treatment}_weighted.csv`
（各队列 03_causal/），本脚本合并二者绘图。

默认：Cohort B + exercise → results/figures/love_plot_cohort_B.png

运行（项目根目录）:
  python scripts/plot_love_plot.py
  python scripts/plot_love_plot.py --cohort B --treatment exercise
  python scripts/plot_love_plot.py --smd-raw path/to/assumption_balance_smd_exercise.csv \\
      --smd-weighted path/to/assumption_balance_smd_exercise_weighted.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import (
        BBOX_INCHES,
        COHORT_A_DIR,
        COHORT_B_DIR,
        COHORT_C_DIR,
        COHORT_STEP_DIRS,
        OUTPUT_ROOT,
        RESULTS_FIGURES,
    )
except ImportError:
    BBOX_INCHES = "tight"
    COHORT_A_DIR = "Cohort_A_Healthy_Prospective"
    COHORT_B_DIR = "Cohort_B_Depression_to_Comorbidity"
    COHORT_C_DIR = "Cohort_C_Cognition_to_Comorbidity"
    COHORT_STEP_DIRS = {"causal": "03_causal"}
    OUTPUT_ROOT = "LIU_JUE_STRATEGIC_SUMMARY"
    RESULTS_FIGURES = "results/figures"


def _default_smd_paths(cohort_letter: str, treatment: str) -> tuple[str, str]:
    cohort_dir = {
        "A": COHORT_A_DIR,
        "B": COHORT_B_DIR,
        "C": COHORT_C_DIR,
    }.get(cohort_letter.upper(), COHORT_B_DIR)
    causal = os.path.join(OUTPUT_ROOT, cohort_dir, COHORT_STEP_DIRS["causal"])
    raw_p = os.path.join(causal, f"assumption_balance_smd_{treatment}.csv")
    w_p = os.path.join(causal, f"assumption_balance_smd_{treatment}_weighted.csv")
    return raw_p, w_p


def _discover_smd_paths(project_root: str, cohort_letter: str, treatment: str) -> tuple[str | None, str | None]:
    """在常见目录下查找 SMD 表；优先主流程 OUTPUT_ROOT，其次队列根下 03_causal，再全库递归。"""
    cl = cohort_letter.upper()
    cohort_dir = {"A": COHORT_A_DIR, "B": COHORT_B_DIR, "C": COHORT_C_DIR}.get(cl, COHORT_B_DIR)
    raw_name = f"assumption_balance_smd_{treatment}.csv"
    w_name = f"assumption_balance_smd_{treatment}_weighted.csv"
    tries: list[tuple[str, str]] = []
    for base in (OUTPUT_ROOT, ".", os.path.join(".", OUTPUT_ROOT)):
        causal = os.path.join(project_root, base, cohort_dir, COHORT_STEP_DIRS["causal"])
        tries.append((os.path.join(causal, raw_name), os.path.join(causal, w_name)))
    raw_p, w_p = _default_smd_paths(cl, treatment)
    tries.insert(0, (raw_p, w_p))

    def _cohort_hint(path_norm: str) -> bool:
        u = path_norm.replace("\\", "/")
        if cl == "A":
            return "Cohort_A" in u or "Healthy_Prospective" in u
        if cl == "B":
            return "Cohort_B" in u or "cohort_B" in u or "Depression" in u
        return "Cohort_C" in u or "Cognition" in u

    for rp, wp in tries:
        if os.path.isfile(rp):
            w_use = wp if os.path.isfile(wp) else None
            if w_use is None:
                alt = os.path.join(os.path.dirname(rp), w_name)
                if os.path.isfile(alt):
                    w_use = alt
            return rp, w_use

    raw_glob = glob.glob(os.path.join(project_root, "**", raw_name), recursive=True)
    raw_glob = [p for p in raw_glob if os.path.isfile(p)]
    raw_glob.sort(key=lambda p: (not _cohort_hint(p), len(p)))
    for rp in raw_glob:
        if not _cohort_hint(rp):
            continue
        wp = os.path.join(os.path.dirname(rp), w_name)
        return rp, wp if os.path.isfile(wp) else None
    for rp in raw_glob:
        wp = os.path.join(os.path.dirname(rp), w_name)
        return rp, wp if os.path.isfile(wp) else None
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Love plot: raw vs weighted SMD")
    ap.add_argument("--cohort", default="B", help="A/B/C，用于默认路径")
    ap.add_argument("--treatment", default="exercise", help="与 assumption 文件名一致")
    ap.add_argument("--smd-raw", default=None, help="未加权 SMD CSV")
    ap.add_argument("--smd-weighted", default=None, help="加权 SMD CSV")
    ap.add_argument(
        "--out",
        default=None,
        help="输出 PNG（默认 results/figures/love_plot_cohort_{letter}.png）",
    )
    args = ap.parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    raw_p = args.smd_raw
    w_p = args.smd_weighted
    if not raw_p:
        dr, dw = _discover_smd_paths(project_root, args.cohort, args.treatment)
        raw_p = dr
        if not w_p:
            w_p = dw
    elif not w_p:
        w_try = os.path.join(os.path.dirname(raw_p), f"assumption_balance_smd_{args.treatment}_weighted.csv")
        w_p = w_try if os.path.isfile(w_try) else None

    if not raw_p or not os.path.isfile(raw_p):
        raise FileNotFoundError(
            f"未找到未加权 SMD 表（请指定 --smd-raw 或先跑主流程写出 03_causal/assumption_balance_smd_*.csv）: {raw_p!r}"
        )

    df0 = pd.read_csv(raw_p, encoding="utf-8-sig")
    if "covariate" not in df0.columns or "smd" not in df0.columns:
        raise ValueError(f"{raw_p} 需含列 covariate, smd")

    weighted_ok = bool(w_p and os.path.isfile(w_p))
    if weighted_ok:
        df1 = pd.read_csv(w_p, encoding="utf-8-sig")
        if "covariate" not in df1.columns or "smd" not in df1.columns:
            raise ValueError(f"{w_p} 需含列 covariate, smd")
        m = df0[["covariate", "smd"]].merge(
            df1[["covariate", "smd"]], on="covariate", how="outer", suffixes=("_raw", "_w")
        )
        m["smd_raw"] = pd.to_numeric(m["smd_raw"], errors="coerce").abs()
        m["smd_w"] = pd.to_numeric(m["smd_w"], errors="coerce").abs()
        m["rank_key"] = m[["smd_raw", "smd_w"]].max(axis=1)
    else:
        m = df0[["covariate", "smd"]].copy()
        m["smd_raw"] = pd.to_numeric(m["smd"], errors="coerce").abs()
        m["smd_w"] = np.nan
        m["rank_key"] = m["smd_raw"]

    m = m.sort_values("rank_key", ascending=True).reset_index(drop=True)
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(7.5, max(5.0, 0.22 * len(m) + 1.5)))

    ax.scatter(m["smd_raw"], y, s=36, alpha=0.85, color="#d9534f", label="Unadjusted |SMD|", zorder=3)
    if weighted_ok:
        ax.scatter(m["smd_w"], y, s=36, alpha=0.85, color="#5cb85c", label="IPW weighted |SMD|", zorder=3)
    ax.axvline(0.1, color="gray", linestyle="--", linewidth=1.2, label="|SMD| = 0.1")
    ax.set_yticks(y)
    ax.set_yticklabels(m["covariate"], fontsize=7)
    ax.set_xlabel("Absolute standardized mean difference |SMD|")
    ax.set_ylabel("Covariate")
    sub = "" if weighted_ok else " (unadjusted only — weighted CSV not found)"
    ax.set_title(
        f"Covariate balance (Love plot): Cohort {args.cohort.upper()} — treatment={args.treatment}{sub}"
    )
    ax.legend(loc="lower right", framealpha=0.92)
    ax.set_xlim(left=0)

    out_path = args.out
    if not out_path:
        out_path = os.path.join(RESULTS_FIGURES, f"love_plot_cohort_{args.cohort.upper()}.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches=BBOX_INCHES)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
