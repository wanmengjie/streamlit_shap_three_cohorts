"""Inject fully inlined supplementary blocks into PAPER_Manuscript_Submission_Ready.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "PAPER_Manuscript_Submission_Ready.md"
MARKER = "# SUBMISSION CHECKLIST"

s10 = (ROOT / "_tmp_s10.md").read_text(encoding="utf-8")
s12 = (ROOT / "_tmp_s12.md").read_text(encoding="utf-8")
s13 = (ROOT / "_tmp_s13.md").read_text(encoding="utf-8")
s16 = (ROOT / "_tmp_s16_xlearner_full.md").read_text(encoding="utf-8")
s18 = (ROOT / "_tmp_s18.md").read_text(encoding="utf-8")

block = f"""
## Supplementary Table S10. Full baseline characteristics by cohort (all variables)

*Source: `results/tables/table1_baseline_characteristics.csv` (**2026-03-28** lock). **Chronic disease burden** appears only as a **baseline descriptor** here—not as a causal intervention target (§2.3).*

{s10}

---

## Supplementary Table S11. STROBE sample flow (person-waves)

*Source: `results/tables/table1_sample_attrition.csv`.*

| Step | N |
|---|---|
| Raw records (person-waves) | 96,628 |
| Age ≥ 60 years | 49,015 |
| CES-D-10 non-missing | 43,048 |
| Cognition score non-missing | 31,574 |
| Next-wave comorbidity non-missing | 16,983 |
| Incident cohort (baseline free of comorbidity) | 14,386 |

---

## Supplementary Table S12. Full predictive model comparison (held-out test), all algorithms

*Source: `results/tables/table2_prediction_cohortA/B/C.csv`. Champion selection: Table 2 footnote.*

{s12}

---

## Supplementary Table S13. Full causal triangulation (PSM, PSW, XLearner), all exported binary contrasts

*Source: `results/tables/table7_psm_psw_dml.csv`. **chronic_low** rows are **pipeline audit** only; the primary manuscript does **not** treat low chronic burden as an intervention (§2.3). **significant_95** = 1 if the exported 95% CI excludes zero.*

{s13}

---

## Supplementary Table S14. Complete XLearner ATE export (`table4_ate_summary.csv`)

*Includes **chronic_low** for reproducibility; main-text Table 5 reports **four** lifestyle interventions only.*

{s16}

---

## Supplementary Table S15. Full diagnostic-threshold and complete-case sensitivity

*Source: `results/tables/table5_sensitivity_summary.csv` (all **151** scenario × intervention × cohort rows). **Chronic burden** rows are retained for audit; primary narrative uses four lifestyle factors.*

{s18}

---
"""

text = PAPER.read_text(encoding="utf-8")
if MARKER not in text:
    raise SystemExit(f"Marker not found: {MARKER}")
if "## Supplementary Table S10." in text:
    raise SystemExit("Already injected; abort to avoid duplicate")
text = text.replace(MARKER, block + "\n" + MARKER)
PAPER.write_text(text, encoding="utf-8")
print("Injected S10–S15 before", MARKER)
