# -*- coding: utf-8 -*-
"""
将 table2_prediction_cohortA/B/C.csv 合并为 table2_prediction_combined_ABC.csv。

用法（项目根目录）:
  python scripts/merge_table2_prediction_ABC.py
  python scripts/merge_table2_prediction_ABC.py --tables results/tables
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

os.chdir(_ROOT)


def main():
    from config import RESULTS_TABLES
    from utils.charls_table2_combine import write_combined_table2_prediction

    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=RESULTS_TABLES, help="含 table2_prediction_cohort*.csv 的目录")
    args = ap.parse_args()
    p = write_combined_table2_prediction(args.tables)
    if p is None:
        raise SystemExit(1)
    print(p)


if __name__ == "__main__":
    main()
