# -*- coding: utf-8 -*-
"""
根据 preprocessed_data/attrition_flow.csv 绘制纳入/排除流程图（CONSORT / STROBE 风格）。
数值与步骤与 data/charls_complete_preprocessing.py 中 attrition 列表一致。

用法:
  python -m viz.draw_attrition_flowchart
  python -m viz.draw_attrition_flowchart --csv preprocessed_data/attrition_flow.csv --out results/figures/fig1_attrition_flow_code_aligned.png
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

logger = logging.getLogger(__name__)

# 与 charls_complete_preprocessing.preprocess_charls_data 中每一步**排除**含义对应（英文，供论文图注）
DEFAULT_EXCLUSION_REASONS_EN = [
    "Did not meet age ≥ minimum or missing age",
    "Missing CES-D-10",
    "Missing cognition score",
    "Missing next-wave depression–cognition comorbidity status\n(non-consecutive wave or end of follow-up)",
    "Prevalent comorbidity or prior comorbidity\n(not incident-eligible at baseline wave)",
]


def _load_age_min() -> int:
    try:
        from config import AGE_MIN

        return int(AGE_MIN)
    except Exception:
        return 60


def _results_figures_dir() -> str:
    try:
        from config import RESULTS_FIGURES

        return RESULTS_FIGURES
    except Exception:
        return "results/figures"


def _fmt_n(n) -> str:
    try:
        return f"{int(float(n)):,}" if pd.notna(n) else "—"
    except (ValueError, TypeError):
        return "—"


def _build_consort_figure(
    steps: list,
    ns: list,
    *,
    csv_path: str,
    title: str,
    fig_w: float,
    fig_h: float,
):
    """在单个 figure 上绘制 CONSORT 布局。"""
    n_boxes = len(steps)
    age_min = _load_age_min()
    exclusion_reasons = list(DEFAULT_EXCLUSION_REASONS_EN)
    exclusion_reasons[0] = f"Did not meet age ≥ {age_min} years or missing age"

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, n_boxes * 2.2 + 2)
    ax.axis("off")

    box_w, box_h = 4.2, 0.72
    x_main = 2.35
    x_exc = 6.55
    y0 = n_boxes * 2.05 + 0.35

    for i, (step, n) in enumerate(zip(steps, ns)):
        y = y0 - i * 2.05
        rect = mpatches.FancyBboxPatch(
            (x_main - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.03",
            facecolor="#E8F4FC",
            edgecolor="#2E86AB",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            x_main,
            y,
            f"{step}\n(n = {_fmt_n(n)})",
            ha="center",
            va="center",
            fontsize=9,
            linespacing=1.15,
        )

        if i < n_boxes - 1:
            y_next = y0 - (i + 1) * 2.05
            y_arrow_top = y - box_h / 2 - 0.02
            y_arrow_bot = y_next + box_h / 2 + 0.02
            ax.annotate(
                "",
                xy=(x_main, y_arrow_bot),
                xytext=(x_main, y_arrow_top),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.35, shrinkA=2, shrinkB=2),
            )

            try:
                excluded = (
                    int(float(ns[i])) - int(float(ns[i + 1]))
                    if pd.notna(ns[i]) and pd.notna(ns[i + 1])
                    else None
                )
            except (ValueError, TypeError):
                excluded = None

            y_mid = (y_arrow_top + y_arrow_bot) / 2
            if excluded is not None and excluded > 0:
                reason = exclusion_reasons[i] if i < len(exclusion_reasons) else "Excluded"
                exc_w, exc_h = 3.25, 0.95
                left_exc = x_exc - exc_w / 2
                rect_e = mpatches.FancyBboxPatch(
                    (left_exc, y_mid - exc_h / 2),
                    exc_w,
                    exc_h,
                    boxstyle="round,pad=0.02",
                    facecolor="#FFF5F5",
                    edgecolor="#C44E52",
                    linewidth=1.0,
                )
                ax.add_patch(rect_e)
                ax.text(
                    x_exc,
                    y_mid,
                    f"Excluded\nn = {excluded:,}\n{reason}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    linespacing=1.12,
                    color="#333333",
                )
                # 自主轴水平连至排除框
                ax.plot([x_main, left_exc], [y_mid, y_mid], color="#888888", lw=0.9, linestyle="--")

    foot = (
        "Unit: person-waves (longitudinal rows in CHARLS). "
        "Covariates/outcome timing: wave t → next-wave comorbidity at consecutive t+1 "
        "(see preprocess_charls_data in data/charls_complete_preprocessing.py). "
        f"Source: {os.path.normpath(csv_path)}"
    )
    ax.text(5, -0.35, foot, ha="center", va="top", fontsize=7, color="#555555")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    return fig


def draw_flowchart(
    csv_path=None,
    output_path=None,
    *,
    also_mirror_to_results: bool = True,
    title: str | None = None,
) -> bool:
    """
    绘制与 attrition_flow.csv 一致的流程图；主列为保留样本，右侧为各步排除人数与理由。
    """
    if csv_path is None:
        for p in [
            "LIU_JUE_STRATEGIC_SUMMARY/attrition_flow.csv",
            "preprocessed_data/attrition_flow.csv",
        ]:
            if os.path.exists(p):
                csv_path = p
                break
    if csv_path is None or not os.path.exists(csv_path):
        logger.warning("未找到 attrition_flow.csv，请先运行 preprocess_charls_data 或主流程生成预处理与流失表。")
        return False

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "Step" not in df.columns or "N" not in df.columns:
        logger.error("attrition_flow.csv 需包含列 Step 与 N。")
        return False

    steps = df["Step"].tolist()
    ns = df["N"].tolist()
    if len(steps) == 0:
        return False

    if output_path is None:
        final_dir = "LIU_JUE_STRATEGIC_SUMMARY"
        if os.path.exists(final_dir):
            output_path = os.path.join(final_dir, "attrition_flow_diagram.png")
        else:
            output_path = os.path.join(os.path.dirname(csv_path) or ".", "attrition_flow_diagram.png")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n_boxes = len(steps)
    fig_w, fig_h = 9.5, max(8.0, 1.15 * n_boxes + 4.0)
    ttl = title or "Study flow aligned with preprocessing code (incident person-waves)"

    fig = _build_consort_figure(steps, ns, csv_path=csv_path, title=ttl, fig_w=fig_w, fig_h=fig_h)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("流程图已保存: %s", output_path)

    if also_mirror_to_results:
        rdir = _results_figures_dir()
        os.makedirs(rdir, exist_ok=True)
        mirror = os.path.join(rdir, "fig1_attrition_flow_code_aligned.png")
        fig2 = _build_consort_figure(steps, ns, csv_path=csv_path, title=ttl, fig_w=fig_w, fig_h=fig_h)
        fig2.savefig(mirror, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        logger.info("已同步写入论文用图: %s", mirror)

    return True


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Draw attrition flowchart from attrition_flow.csv")
    p.add_argument("--csv", default=None, help="Path to attrition_flow.csv")
    p.add_argument("--out", default=None, help="Output PNG path")
    p.add_argument("--no-mirror", action="store_true", help="Do not write results/figures/fig1_attrition_flow_code_aligned.png")
    args = p.parse_args(argv)
    ok = draw_flowchart(
        csv_path=args.csv,
        output_path=args.out,
        also_mirror_to_results=not args.no_mirror,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
