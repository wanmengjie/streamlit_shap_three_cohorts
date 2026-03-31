# Supplementary Materials

**Manuscript:** Disentangling the Heterogeneous Trajectories of Depression-Cognition Comorbidity in Older Adults: A Causal Machine Learning Study

---

## Table of Contents

### Supplementary Texts
- **Text S1.** Detailed Data Preprocessing and Imputation Strategy
- **Text S2.** Machine Learning Hyperparameter Optimization Space
- **Text S3.** Multiple Testing, Approximate *P* Values, and FDR-Style Adjustment (Causal Summary Export)

### Supplementary Tables
- **Table S1.** Missing Data Mechanism and Proportions
- **Table S2.** Sensitivity Analysis Results (Varying Diagnostic Thresholds for Comorbidity)
- **Table S3.** Variable Definitions, Coding, and CHARLS Questionnaire Source Items
- **Table S4.** External Model Validation Results (Temporal and Regional Splits)
- **Table S5.** Cross-Validation of Causal Estimates (PSM, PSW, XLearner)
- **Table S6.** Subgroup CATE for Cohorts A and C (Exercise)
- **Table S7.** Hyperparameter Search Space and Optimal Parameters

### Supplementary Figures
- **Figure S1.** Missing Data Pattern Heatmap
- **Figure S2.** Propensity Score Overlap Distributions (Before and After Trimming)
- **Figure S3.** Covariate Balance (Love Plot of Standardized Mean Differences)
- **Figure S4.** Imputation Diagnostic Plots (Observed vs. Imputed Distributions)
- **Figure S5.** Calibration Curves for Champion Predictive Models

### Reporting Checklists
- **Checklist S1.** TRIPOD Checklist for Prediction Model Validation
- **Checklist S2.** STROBE Checklist for Observational Studies

---

## Detailed Description of Supplementary Figures and Tables

### 1. Supplementary Tables

**Table S1. Missing Data Mechanism and Proportions**
- *Content:* Lists the exact missingness percentage for every variable before imputation. For example, it will document that `income` had ~15% missingness, while `walking speed` (which was subsequently excluded from the primary analysis) had ~40% missingness.

**Table S2. Sensitivity Analysis Results (Varying Diagnostic Thresholds)**
- *Content:* Proves the robustness of the causal findings. It shows the ATE of exercise in Cohort B when changing the depression cutoff (e.g., CES-D ≥ 8 instead of 10) and the cognitive cutoff (e.g., Score ≤ 8 instead of 10). *Currently shows that under CES-D≥8 and Cog≤8, the ATE remains significant at -3.8%.*

**Table S3. Variable Definitions and Source Items**
- *Content:* A data dictionary mapping our variable names to the exact CHARLS questionnaire codes (e.g., `DA001` for self-rated health), ensuring perfect reproducibility for other researchers.

**Table S4. External Model Validation Results**
- *Content:* Displays the AUC, AUPRC, and Brier scores when the models are tested on out-of-distribution data:
  - *Temporal:* Trained on Waves 2011-2015, tested on Wave 2018.
  - *Regional:* Trained on Eastern/Central China, tested on Western China.

**Table S5. Cross-Validation of Causal Estimates**
- *Content:* A side-by-side comparison of the Average Treatment Effect (ATE) and 95% CI calculated by XLearner, Propensity Score Matching (PSM), and Propensity Score Weighting (PSW) across all three cohorts.

**Table S6. Subgroup CATE for Cohorts A and C (Exercise)**
- *Content:* Exploratory subgroup conditional average treatment effects (CATE) for the non-significant cohorts (A and C), provided for completeness alongside the main text Table 3 (which focuses on the significant Cohort B).

**Table S7. Hyperparameter Search Space and Optimal Parameters**
- *Content:* The full hyperparameter search space and the optimal parameters selected for the champion models across all cohorts (derived from `tableS8_hyperparameter_search.csv`).

**Text S3. Multiple testing and FDR-style columns (full prose)**
- *Content:* Journal-ready explanation that (i) approximate two-sided *p* can be derived from each exported ATE and 95% CI; (ii) Bonferroni and FDR-style adjustments are applied across **all rows** of the machine-readable causal summary CSV (including PSM/PSW when present), via `utils/multiplicity_correction.py`; (iii) **primary** inference remains bootstrap *P* and CI for prespecified contrasts; (iv) adjusted columns are for transparency. **Full text is embedded as “Supplementary Text S3” in `PAPER_Manuscript_Submission_Ready.md` and `PAPER_完整版_2026-03-20.md`.**

---

### 2. Supplementary Figures

**Figure S1. Missing Data Pattern Heatmap**
- *Content:* A visual representation (matrix) showing the co-occurrence of missing values across variables and individuals, helping to justify the Missing At Random (MAR) assumption required for MICE imputation.

**Figure S2. Propensity Score Overlap Distributions**
- *Content:* Histograms showing the distribution of propensity scores for the treated (e.g., exercised) vs. control groups. It visually demonstrates the strict trimming process (removing PS < 0.05 and > 0.95) to enforce the "Positivity" assumption in causal inference.

**Figure S3. Covariate Balance (Love Plot)**
- *Content:* A dot plot showing the Standardized Mean Difference (SMD) for all 30+ covariates before and after propensity score adjustments/trimming. It visually proves that the treatment and control groups are balanced (all dots falling within the |SMD| < 0.1 or 0.2 threshold lines).

**Figure S4. Imputation Diagnostic Plots**
- *Content:* Density plots overlaying the distribution of the original observed data with the imputed data for continuous variables (like BMI or blood pressure). This proves that the MICE imputation did not distort the underlying data distribution.

**Figure S5. Calibration Curves**
- *Content:* Plots showing the agreement between predicted probabilities and observed frequencies for the champion models. It includes the calibration slope and intercept, which are critical for assessing a model's clinical reliability alongside the AUC.
