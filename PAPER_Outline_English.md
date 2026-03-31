# Paper Outline (English Section Titles)

## Main Text

**Abstract**

**1. Introduction**

**2. Methods**
- 2.1 Study Design
  - 2.1.1 Cohort Stratification Criteria
  - 2.1.2 Definition of Modifiable Lifestyle Factors
  - 2.1.3 Data Preprocessing
  - 2.1.4 Sample Size and Statistical Power
- 2.2 Predictive Modeling
  - 2.2.1 Model Development and Validation Strategy
  - 2.2.2 Model Evaluation and Selection Criteria
  - 2.2.3 External and Temporal Validation
  - 2.2.4 Calibration and Clinical Decision Curve Analysis
- 2.3 Interpretability Analysis
- 2.4 Causal Inference Framework and Identification Assumptions
- 2.5 Causal Effect Estimation
- 2.6 Sensitivity and Robustness Analyses
- 2.7 Cross-Validation of Causal Effects with Multiple Methods

**3. Results**
- 3.1 Study Population and Cohort Stratification
- 3.2 Incidence of Depression–Cognition Comorbidity
- 3.3 Predictive Model Performance
  - 3.3.1 Internal Cross-Validation Performance
  - 3.3.2 External and Temporal Validation
- 3.4 SHAP Interpretability Analysis
- 3.5 Causal Effect of Modifiable Lifestyle Factors
  - 3.5.1 Overall Average Treatment Effect via TLearner
  - 3.5.2 Cross-Validation of Exercise ATE via PSW/PSM
- 3.6 Heterogeneous Treatment Effects of Exercise via CATE Analysis
- 3.7 Clinical Utility via Decision Curve Analysis
- 3.8 Sensitivity and Robustness Analyses

**4. Discussion**
- 4.1 Principal Findings
- 4.2 Comparison with Prior Literature
- 4.3 Mechanistic Interpretations of Cohort-Specific Exercise Effects
- 4.4 Clinical and Public Health Implications
- 4.5 Methodological Rigor and Triangulation
- 4.6 Limitations
- 4.7 Future Research Directions

**5. Conclusions**

---

## Tables (Main Text)

- **Table 1** Baseline characteristics by cohort
- **Table 2** Prediction performance by cohort
- **Table 3** Subgroup CATE estimates for regular exercise
- **Table 4** Average treatment effects of modifiable lifestyle factors
- **Table 5** Sensitivity analysis: varying diagnostic thresholds
- **Table 6** External and temporal validation results
- **Table 7** Cross-validation of causal estimates (PSM/PSW/TLearner)

---

## Figures (Main Text)

- **Figure 1** Study flow: inclusion and exclusion (STROBE)
- **Figure 2** Conceptual framework
- **Figure 3** SHAP summary plot: feature importance for prediction of incident comorbidity
- **Figure 4** Average treatment effect of regular exercise with subgroup CATE
- **Figure 5** Integrated clinical utility and calibration analysis

---

## Supplementary Materials

### Supplementary Text

**Text S1** Detailed Data Preprocessing and Imputation Strategy  

**Text S2** Machine Learning Hyperparameter Optimization  

---

### Supplementary Tables (Appendix)

| No. | Title |
|-----|-------|
| **Table S1** | Detailed Variable Definitions and Coding |
| **Table S2** | Missing Data Mechanism Analysis (Little's MCAR Test) |
| **Table S3** | Sensitivity Analysis Results: Varying Diagnostic Thresholds for Depression and Cognitive Impairment |
| **Table S4** | Exploratory Causal Analysis: Average Treatment Effects with 95% Confidence Intervals |
| **Table S5** | External Model Validation Results (Temporal and Regional Splits) |
| **Table S6** | Cross-Validation of Causal Estimates for Exercise Intervention (PSM, PSW, TLearner) |
| **Table S7** | Bias Sensitivity Analysis: Simulated Unmeasured Confounding |
| **Table S8** | Hyperparameter Search Results by Cohort |
| **Table S9** | Causal Assumption Checks: Overlap, Standardized Mean Difference, and E-value by Cohort |
| **Table S10** | Physiological and Functional Indicator Causal Effects (Grip Strength, Walking Speed, ADL/IADL, Insurance, Pension) |
| **Table S11** | Imputation Sensitivity Results: Comparison of Five Imputation Methods |

---

### Supplementary Figures (Appendix)

| No. | Title |
|-----|-------|
| **Figure S1** | Missing Data Heatmap: Pattern of Missingness Across Covariates |
| **Figure S2** | Propensity Score Overlap (Common Support) for Exercise Intervention by Cohort |
| **Figure S3** | Imputation Diagnostics: Comparison of Original vs. Imputed Data Distributions |
| **Figure S4** | Calibration Curves for External Validation (Temporal and Regional) |
| **Figure S5** | Stratified SHAP Analysis by Cohort |
| **Figure S6** | SHAP Interaction Effects (Exercise × Sleep) |
| **Figure S7** | Dose–Response Analysis for Sleep and Exercise |
| **Figure S8** | ITE Stratified Validation: Low vs. High Individual Treatment Effect |
| **Figure S9** | Nomogram for Individualized Risk Prediction |
| **Figure S10** | Sensitivity Analysis: Exercise ATE by Diagnostic Threshold Scenario |
| **Figure S11** | Low-Sample Optimization: Social Isolation and Chronic Disease Burden |
| **Figure S12** | Causal Methods Comparison Forest Plot (PSM/PSW/TLearner by Intervention) |
| **Figure S13** | Subgroup CATE Forest Plot by Residence, Age, and Sex |
| **Figure S14** | Placebo Test and E-Value Sensitivity for Unmeasured Confounding |
| **Figure S15** | Bias Sensitivity: Effect of Simulated Unmeasured Confounder Strength on ATE Estimate |  

---

## Ethics Statement

## Competing Interests

## Funding

## Authors' Contributions

## Data Availability

## References
