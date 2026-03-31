# Reporting checklists (observational cohort + prediction layer)

This study is an **observational secondary analysis** of **CHARLS** with an **embedded prediction module** (CPM champions, temporal/regional validation, calibration, DCA) and a **causal triangulation** layer (XLearner, PSM, PSW). Complete the items below in the journal portal or as a supplementary PDF.

## STROBE (observational cohort studies)

- **Title / abstract:** Cohort design and outcome indicated (depression–cognition comorbidity incidence; person-waves).
- **Background:** Rationale and objectives (dual actionable windows; phenotype stratification).
- **Methods:**  
  - Study design: prospective waves, incident risk set, inclusion cascade (**Figure 1**, **Table S11**).  
  - Setting: CHARLS, China.  
  - Participants: age ≥60; baseline phenotypes A/B/C; exclusion of prevalent/prior comorbidity.  
  - Variables: exposures (exercise, etc.), outcomes, confounders (**Table S3**).  
  - Bias: overlap trimming, balance (SMD), E-values, negative control, MI single-draw primary + **Table S16**.  
  - Statistical methods: grouped CV, MICE (prediction vs causal paths), bootstrap CIs by ID.
- **Results:** Participants, descriptive, outcomes, main estimates (Tables **1–7**), sensitivity (**S2, S15**).
- **Discussion:** Limitations (residual confounding, transportability, clustering), generalisability.
- **Funding / conflict:** As on title page.

*Official checklist PDF:* search “STROBE observational cohort checklist” and attach the completed form.

## TRIPOD (prediction model reporting)

- **Title:** Indicates prediction when the journal allows (prognostic CPM champions).
- **Abstract:** Discrimination (AUC), calibration context, validation (temporal / regional).
- **Methods:** Source of data (CHARLS), outcome definition, predictors, sample split (temporal), missing data (nested imputation), model type and tuning (grouped CV), measures (AUC, Brier, calibration, DCA).
- **Results:** Model performance **Table 3**, calibration **Figure S5**, DCA **Figure 5**, limitations on calibration in B/C.
- **Discussion:** Intended use (risk stratification in **Cohort A**), not conflated with universal intervention.

*Official TRIPOD statement and checklist:* Collins *et al.*, BMJ 2015 (reference **[9]** in the manuscript).

---

**Note:** Some items overlap (e.g. missing data). One consolidated **supplementary checklist PDF** plus a short **cover-letter** sentence (“STROBE and TRIPOD-style items addressed in Methods/Results and **Supplementary** materials”) is usually sufficient unless the journal mandates a specific form.
