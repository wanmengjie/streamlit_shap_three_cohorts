# -*- coding: utf-8 -*-
"""
从指定 Cohort 续跑主流程，避免插补 + Cohort A 等已完成的步骤整段重来。

典型场景：Cohort B 写 model_performance_full_*.csv 时 PermissionError（文件被 Excel 占用）。
请先关闭占用 `Cohort_*/*/01_prediction/*.csv` 的程序，再运行本脚本。

用法（项目根目录）:
    python scripts/resume_charls_cohorts.py B,C
    python scripts/resume_charls_cohorts.py B C

说明:
    - 临时关闭前置插补（使用已有 step1_imputed_full）
    - 跳过概念图 / Table1 / 流失图 / 插补敏感性（可用 --full-prefix 恢复）
    - 只重跑所列队列；未列队（如 A）的汇总指标从磁盘 table2_* 与 ATE_CI_summary_* 读取

跑完后请将 config.py 中 RUN_COHORTS_ONLY 设回 None（或不再用本脚本），以免下次误只跑部分队列。
"""
from __future__ import annotations

import argparse
import os
import sys


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="CHARLS 主流程：从指定 Cohort 续跑")
    parser.add_argument(
        "cohorts",
        nargs="*",
        default=["B", "C"],
        help="要重跑的队列 ID，如 B C 或 B,C（默认 B C）",
    )
    parser.add_argument(
        "--full-prefix",
        action="store_true",
        help="仍跑 Table1 / 概念图 / 流失图 / 插补敏感性（较慢，一般不用于续跑）",
    )
    parser.add_argument(
        "--impute-first",
        action="store_true",
        help="仍执行前置插补（极慢；默认关闭以使用已有插补 CSV）",
    )
    args = parser.parse_args()

    raw = " ".join(args.cohorts) if args.cohorts else "B,C"
    parts = []
    for chunk in raw.replace(",", " ").split():
        chunk = chunk.strip().upper()
        if chunk in ("A", "B", "C"):
            parts.append(chunk)
    if not parts:
        print("请指定 A/B/C，例如: python scripts/resume_charls_cohorts.py B C", file=sys.stderr)
        sys.exit(2)

    root = _project_root()
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    import config as cfg

    cfg.RUN_COHORTS_ONLY = parts
    cfg.MAIN_SKIP_STEPS_BEFORE_COHORTS = not args.full_prefix
    cfg.RUN_IMPUTATION_BEFORE_MAIN = bool(args.impute_first)

    # 确保 run_all 从已修改的 config 执行 `from config import *`
    for mod in list(sys.modules.keys()):
        if mod == "run_all_charls_analyses" or mod.endswith(".run_all_charls_analyses"):
            del sys.modules[mod]

    print(
        f"[resume_charls_cohorts] RUN_COHORTS_ONLY={parts}, "
        f"MAIN_SKIP_STEPS_BEFORE_COHORTS={cfg.MAIN_SKIP_STEPS_BEFORE_COHORTS}, "
        f"RUN_IMPUTATION_BEFORE_MAIN={cfg.RUN_IMPUTATION_BEFORE_MAIN}"
    )

    import run_all_charls_analyses

    run_all_charls_analyses.main()


if __name__ == "__main__":
    main()
