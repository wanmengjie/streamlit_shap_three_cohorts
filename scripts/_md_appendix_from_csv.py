"""One-off: emit markdown table fragments for manuscript appendix."""
import csv
from pathlib import Path


def md_escape(s):
    if s is None:
        return ""
    return str(s).replace("|", "\\|")


def csv_to_md(path, col_subset=None):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""
    header = rows[0]
    if col_subset:
        idx = [header.index(c) for c in col_subset if c in header]
        header = [header[i] for i in idx]
        body = [[row[i] if i < len(row) else "" for i in idx] for row in rows[1:]]
    else:
        body = rows[1:]
    lines = [
        "| " + " | ".join(md_escape(h) for h in header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for b in body:
        lines.append("| " + " | ".join(md_escape(x) for x in b) + " |")
    return "\n".join(lines)


def main():
    base = Path(__file__).resolve().parents[1] / "results" / "tables"
    out_dir = Path(__file__).resolve().parents[1]
    # S10 baseline
    (out_dir / "_tmp_s10.md").write_text(
        csv_to_md(base / "table1_baseline_characteristics.csv"), encoding="utf-8"
    )
    # S12 combined prediction
    h = [
        "Cohort",
        "Model",
        "AUC (95% CI)",
        "Recall (95% CI)",
        "Specificity (95% CI)",
        "Brier",
        "Imbalance_Handled",
    ]
    lines = [
        "| " + " | ".join(h) + " |",
        "|" + "|".join(["---"] * len(h)) + "|",
    ]
    keys = [
        "Model",
        "AUC_95CI",
        "Recall_95CI",
        "Specificity_95CI",
        "Brier_Score",
        "Imbalance_Handled",
    ]
    for co, lab in [("A", "Cohort A"), ("B", "Cohort B"), ("C", "Cohort C")]:
        p = base / f"table2_prediction_cohort{co}.csv"
        with p.open(newline="", encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            for row in r:
                lines.append(
                    "| "
                    + " | ".join(
                        md_escape(x)
                        for x in (
                            lab,
                            *[row.get(k, "") for k in keys],
                        )
                    )
                    + " |"
                )
    (out_dir / "_tmp_s12.md").write_text("\n".join(lines), encoding="utf-8")
    # S13 triangulation
    (out_dir / "_tmp_s13.md").write_text(
        csv_to_md(
            base / "table7_psm_psw_dml.csv",
            col_subset=[
                "cohort",
                "exposure",
                "method",
                "ate",
                "ate_lb",
                "ate_ub",
                "significant_95",
                "Consistency",
            ],
        ),
        encoding="utf-8",
    )
    # S16 XLearner full export (incl. chronic_low audit)
    (out_dir / "_tmp_s16_xlearner_full.md").write_text(
        csv_to_md(
            base / "table4_ate_summary.csv",
            col_subset=[
                "label",
                "cohort",
                "ate",
                "ate_lb",
                "ate_ub",
                "n",
                "p_value_approx",
                "significant_95",
            ],
        ),
        encoding="utf-8",
    )
    # S18 sensitivity
    (out_dir / "_tmp_s18.md").write_text(
        csv_to_md(
            base / "table5_sensitivity_summary.csv",
            col_subset=[
                "scenario",
                "intervention_label",
                "cohort_label",
                "n",
                "incidence",
                "ate",
                "ate_lb",
                "ate_ub",
            ],
        ),
        encoding="utf-8",
    )
    # Subgroup CATE (A/B/C) for manuscript S8
    sub_cols = ["Subgroup", "Value", "CATE", "Count", "N_events", "Sample_Size_Warning"]
    parts = ["## Subgroup CATE export (paste under Supplementary Table S8)\n"]
    for co in ("A", "B", "C"):
        p = base / f"table3_subgroup_cohort{co}.csv"
        parts.append(f"### Cohort {co}\n")
        parts.append(csv_to_md(p, col_subset=sub_cols))
        parts.append("\n")
    (out_dir / "_tmp_subgroup_S8.md").write_text("\n".join(parts), encoding="utf-8")
    print("Wrote _tmp_s10, _tmp_s12, _tmp_s13, _tmp_s16_xlearner_full, _tmp_s18, _tmp_subgroup_S8")


if __name__ == "__main__":
    main()
