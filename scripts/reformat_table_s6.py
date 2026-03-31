#!/usr/bin/env python3
"""为 Table S6 添加 95%CI 列，ate/ate_lb/ate_ub 四位小数。"""
import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "FINAL_PAPER_TABLES")
SRC = os.path.join(ROOT, "LIU_JUE_STRATEGIC_SUMMARY", "causal_methods_comparison_summary.csv")

def main():
    df = pd.read_csv(SRC)
    for col in ["ate", "ate_lb", "ate_ub"]:
        if col in df.columns:
            df[col] = df[col].round(4)
    df["95%CI"] = df.apply(
        lambda r: f"({r['ate_lb']:.4f}, {r['ate_ub']:.4f})"
        if pd.notna(r.get("ate_lb")) and pd.notna(r.get("ate_ub"))
        else "",
        axis=1,
    )
    out = os.path.join(OUTPUT_DIR, "Table_S6_Causal_Methods_Comparison.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已更新: {out}")
    _cid = "cohort" if "cohort" in df.columns else "axis"
    print(df[[_cid, "exposure", "method", "ate", "95%CI"]].head(6).to_string())

if __name__ == "__main__":
    main()
