# -*- coding: utf-8 -*-
"""
读取 XLearner 导出的个体 ITE（tau_hat），绘制 Cohort B 的 CATE 密度/直方图：
τ=0 参考线；25/50/75 分位数竖线；显著位置标注「获益比例」P(τ̂>0)。

默认输入：results/tables/ite_xlearner_{treatment}_cohort_B.csv（须先跑主流程 XLearner 且 run_auxiliary_steps 写出 ITE）。

运行（项目根目录）:
  python scripts/plot_cate_density.py
  python scripts/plot_cate_density.py --input results/tables/ite_xlearner_exercise_cohort_B.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import BBOX_INCHES, RESULTS_FIGURES, RESULTS_TABLES, TREATMENT_COL
except ImportError:
    BBOX_INCHES = "tight"
    RESULTS_FIGURES = "results/figures"
    RESULTS_TABLES = "results/tables"
    TREATMENT_COL = "exercise"


def main() -> None:
    ap = argparse.ArgumentParser(description="Cohort B CATE (tau_hat) density plot")
    ap.add_argument(
        "--input",
        default=None,
        help="ITE CSV（默认 results/tables/ite_xlearner_{treatment}_cohort_B.csv）",
    )
    ap.add_argument("--treatment", default=TREATMENT_COL, help="用于默认输入路径的处理列名")
    ap.add_argument(
        "--out",
        default=None,
        help="输出 PNG（默认 results/figures/cate_density_cohort_B_{treatment}.png）",
    )
    args = ap.parse_args()

    in_path = args.input
    if not in_path:
        in_path = os.path.join(RESULTS_TABLES, f"ite_xlearner_{args.treatment}_cohort_B.csv")
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"找不到 ITE 文件: {in_path}（请先运行主流程写出 ITE CSV）")

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    if "tau_hat" not in df.columns:
        raise ValueError(f"{in_path} 缺少 tau_hat 列")
    x = pd.to_numeric(df["tau_hat"], errors="coerce").dropna().to_numpy(dtype=float)
    if x.size < 5:
        raise ValueError(f"tau_hat 有效样本过少 (n={x.size})")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(x, bins=40, density=True, alpha=0.78, color="steelblue", edgecolor="white", label="Histogram (density)")
    try:
        from scipy.stats import gaussian_kde

        lo, hi = float(np.min(x)), float(np.max(x))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        grid = np.linspace(lo - pad, hi + pad, 256)
        kde = gaussian_kde(x)
        ax.plot(grid, kde(grid), color="darkblue", lw=2.0, label="KDE")
    except Exception:
        pass

    ax.axvline(0.0, color="red", linestyle="-", linewidth=2.0, label=r"$\tau = 0$")
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    for qv, sty, lab in [
        (q25, ":", "25th pct"),
        (q50, "--", "50th pct"),
        (q75, ":", "75th pct"),
    ]:
        ax.axvline(qv, color="darkorange", linestyle=sty, linewidth=1.4, alpha=0.9)
    benefit_pct = 100.0 * float(np.mean(x > 0))
    ax.set_xlabel(r"Individual treatment effect $\hat{\tau}$ (XLearner)")
    ax.set_ylabel("Density")
    title_cohort = df["Baseline_Cohort"].iloc[0] if "Baseline_Cohort" in df.columns else "Cohort B"
    ax.set_title(f"{title_cohort}: distribution of estimated CATEs")
    ax.legend(loc="best", framealpha=0.92)
    pct_text = (
        f"Benefit share: P($\\hat{{\\tau}}$>0) = {benefit_pct:.1f}%\n"
        f"Quartiles: q25={q25:.4f}, q50={q50:.4f}, q75={q75:.4f}"
    )
    ax.text(
        0.02,
        0.98,
        pct_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#333", alpha=0.92),
    )

    out_path = args.out
    if not out_path:
        out_path = os.path.join(RESULTS_FIGURES, f"cate_density_cohort_B_{args.treatment}.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches=BBOX_INCHES)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
