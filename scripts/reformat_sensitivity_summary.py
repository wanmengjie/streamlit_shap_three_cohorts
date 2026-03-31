#!/usr/bin/env python3
"""Reformat sensitivity_summary.csv: 4 decimals + 95CI column."""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "LIU_JUE_STRATEGIC_SUMMARY" / "sensitivity_summary.csv"
OUT_PATH = ROOT / "LIU_JUE_STRATEGIC_SUMMARY" / "sensitivity_summary_formatted.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    for col in ["incidence", "ate", "ate_lb", "ate_ub"]:
        if col in df.columns:
            df[col] = df[col].round(4)
    df["95CI"] = df.apply(
        lambda r: f"({r['ate_lb']:.4f}, {r['ate_ub']:.4f})"
        if pd.notna(r.get("ate_lb")) and pd.notna(r.get("ate_ub"))
        else "",
        axis=1,
    )
    df.to_csv(OUT_PATH, index=False)
    print(f"Updated: {OUT_PATH}")

if __name__ == "__main__":
    main()
