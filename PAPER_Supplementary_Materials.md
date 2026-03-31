# Supplementary Materials

**Manuscript:** Development and validation of a prediction model for incident depression–cognition comorbidity in older adults in China

---

## Text S1. Detailed Data Preprocessing and Imputation Strategy

### Data source and waves

CHARLS waves 2011, 2013, 2015, and 2018 were used. Person-wave records were constructed such that each record represents one individual at one wave, with outcome assessed at the next wave.

### Variable definitions

- **Depression:** CES-D-10 score ≥10 (10-item Center for Epidemiologic Studies Depression scale).
- **Cognitive impairment:** Total cognitive score ≤10 (sum of immediate recall, delayed recall, and serial 7s).
- **Regular exercise:** Moderate-intensity physical activity at least once per week (yes/no).
- **Drinking:** Current drinking (yes/no).
- **Social isolation:** Living alone and having no social contact in the past month (yes/no).
- **Normal BMI:** 18.5 ≤ BMI < 24 kg/m².
- **Low chronic disease burden:** ≤1 physician-diagnosed chronic condition (hypertension, diabetes, cancer, lung disease, heart disease, stroke, arthritis).

### Missing data handling

- **Key variables (CES-D, cognition, outcome):** Complete case analysis; records with missing values were excluded.
- **Covariates:** Multiple imputation (MICE, 5 imputations) for continuous and categorical variables with <30% missing. Variables with >30% missing were excluded or handled via complete case for that variable.
- **Order of operations (must match code):** Bulk MICE that produces the analysis dataset is applied to the **entire eligible analytic cohort before** the individual-level 80:20 train–test split (i.e., **not** fold-wise imputation at the MICE stage). Sklearn `Pipeline` preprocessors during model training are fit **only on training folds** in cross-validation; CPM thresholds do not use the held-out test fold. (Aligned with Supplementary Text S1 in the full manuscript.)
- **Intervention variables:** Missing treated as 0 (absence of factor) in causal analysis; sensitivity analysis with complete case was conducted.

### Cohort exclusion criteria

- Age <60 years
- CES-D-10 missing
- Cognitive score missing
- Next-wave comorbidity status missing
- Baseline comorbidity (depression and cognitive impairment coexisting)
- Prior history of comorbidity in any preceding wave

---

## Text S2. Machine Learning Hyperparameter Optimization

### Models and search space

Fourteen algorithms were compared: Logistic Regression, Random Forest, XGBoost, GBDT, ExtraTrees, AdaBoost, Decision Tree, Bagging, KNN, MLP, Naive Bayes, SVM, HistGBM, and LightGBM. CatBoost was omitted after fitting failed (parameter compatibility). RandomizedSearchCV was used with 40–80 iterations per model. Key hyperparameters included:

- **Random Forest:** n_estimators [50, 500], max_depth [3, 15], min_samples_leaf [1, 20]
- **XGBoost/LightGBM:** learning_rate [0.01, 0.3], max_depth [3, 10], n_estimators [50, 500]
- **Logistic Regression:** C [0.01, 100], penalty [l1, l2], solver [liblinear, saga]
- **MLP:** hidden_layer_sizes, alpha, learning_rate_init
- **SVM:** C, gamma, kernel

### CPM (Comprehensive Performance Metric) selection

1. For each model, the optimal threshold maximizing Youden index was found on the internal validation set (80% of training fold).
2. AUC was computed on the held-out validation set (20% of training fold) at that threshold.
3. The model with the highest AUC was selected as champion.
4. Final evaluation was performed on the test set (never used for threshold or model selection).

### GroupKFold

Splits were performed by individual ID to ensure no patient appeared in both training and validation/test across waves, preventing leakage.

---

## Text S3. Multiple Testing, Approximate *P* Values, and FDR-Style Adjustment

**Purpose.** Many contrasts are exported when tabulating XLearner **and** triangulation methods (PSM, PSW) across exposures and cohorts. **Exploratory** Bonferroni and FDR-style columns are added in the machine-readable summary for transparency; **primary** inference remains the prespecified **XLearner** ATE, **95% CI**, and **bootstrap *P***.

**Approximate *p*.** From each row’s ATE and 95% CI (LB, UB): SE = (UB − LB) / (2 × 1.96); *z* = |ATE|/SE; two-sided *p*\_approx = 2(1 − Φ(|*z*|)) (convenience only; may differ from bootstrap *p*).

**Adjustments.** Bonferroni: *p*\_adj = min(*p*\_approx × *m*, 1) where *m* = count of non-missing *p*\_approx rows. FDR-style: rank-based adjustment on *p*\_approx via `utils/multiplicity_correction.py` (`apply_bonferroni_fdr`), applied to **all rows** in the exported file (including PSM/PSW when present).

**Output.** After a full run, see e.g. `LIU_JUE_STRATEGIC_SUMMARY/xlearner_all_interventions/xlearner_all_interventions_summary.csv` for columns `p_value_approx`, `p_adj_bonferroni`, `p_adj_fdr`, and significance flags. Full prose also in `PAPER_Manuscript_Submission_Ready.md` (Supplementary Text S3).

---

## Table S1. Full Baseline Characteristics

*See `results/tables/table1_baseline_characteristics.csv` for the complete table. Key variables are summarized in main text Table 1.*

---

## Table S2. Missing Data Mechanism Analysis

*See `results/tables/table1_missing_summary.csv` for missing proportions by variable. Missingness was highest for income (≈15%), walking speed (≈40%, excluded), and grip strength (≈35%, excluded in some analyses).*

---

## Table S3. Variable Definitions and Coding

*See `results/tables/tableS1_variable_definitions.csv` for full variable definitions, coding, and source questionnaire items.*

---

## Table S7. Hyperparameter Search Space and Optimal Parameters

*See `results/tables/tableS8_hyperparameter_search.csv` for the full hyperparameter search space and optimal parameters selected for the champion models across all cohorts.*

---

## Figure S1. Missing Data Pattern and Heatmap

*Heatmap of missing data pattern across variables and cohorts. Variables with >30% missing were excluded from primary analysis.*

---

## Figure S2. Propensity Score Overlap and Covariate Balance

*Propensity score distributions by treatment (exercise yes/no) before and after overlap trimming. Covariate balance (SMD) before and after trimming for Cohort B exercise intervention.*

---

## Figure S3. Imputation Diagnostic Plots

*Comparison of observed vs. imputed distributions for key variables. Sensitivity analysis comparing complete case vs. imputed data on prediction AUC.*

---

## TRIPOD Checklist (abbreviated)

| Item | Section |
|------|---------|
| Title/Abstract | Title identifies development/validation; abstract structured |
| Introduction | Background, objectives, intended use |
| Methods: Data | Source, eligibility, predictors, outcome |
| Methods: Analysis | Missing data, model building, validation |
| Results | Participants, model performance, limitations |
| Discussion | Interpretation, limitations, implications |
| Other | Funding, conflicts, data availability |

---

## STROBE Checklist (abbreviated)

| Item | Location |
|------|----------|
| Title/Abstract | Structured abstract with key findings |
| Introduction | Rationale, objectives |
| Methods | Design, setting, participants, variables, bias, study size, quantitative variables, statistical methods |
| Results | Participants, descriptive data, outcome data, main results, other analyses |
| Discussion | Key results, limitations, interpretation, generalizability |
| Other | Funding |

---

*Supplementary materials correspond to files in `results/tables/` and `results/figures/`.*
