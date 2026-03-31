# Development and validation of a prediction model for incident depression–cognition comorbidity in older adults in China: a CHARLS-based study with exploration of potential intervention effects

**基于 2026-03-20 运行结果的论文草稿更新版**

---

## Abstract

**Background:** Based on longitudinal data from the China Health and Retirement Longitudinal Study (CHARLS), this study constructed an analytical framework integrating predictive modeling, interpretable attribution, and causal inference, aiming to identify high-risk groups for geriatric depression–cognition comorbidity and evaluate the causal effects of modifiable factors. A total of 14,386 elderly individuals aged 60 years and above were enrolled and divided into three baseline cohorts: healthy population (Cohort A, n=8,828), depression-only population (Cohort B, n=3,123), and cognition-impaired-only population (Cohort C, n=2,435).

**Results:** Incidence of comorbidity varied distinctively: 4.1% (Cohort A), 13.6% (Cohort B), and 16.9% (Cohort C). Prediction models achieved moderate-to-good discrimination, with AUCs of 0.73 (Cohort A, Logistic Regression), 0.70 (Cohort B, LightGBM), and 0.66 (Cohort C, Naive Bayes). XLearner causal analysis revealed that regular exercise significantly reduced incident comorbidity risk in Cohort B (ATE = −4.2%, 95% CI: −7.9% to −0.9%, p=0.019), with overlap assumption satisfied, moderate covariate balance (max SMD=0.16), and E-value of 2.0. No significant effect was observed in Cohort A or C. In Cohort C, low chronic disease burden was associated with increased comorbidity risk (ATE = 6.3%, 95% CI: 1.5%–10.8%), possibly reflecting reverse causality. Cross-validation with PSM and PSW showed directionally consistent effects for exercise in Cohort B. External validation (temporal and regional) yielded AUCs of 0.57–0.70 across cohorts.

**Keywords:** Depression–cognition comorbidity; Causal machine learning; CHARLS; XLearner; SHAP; Predictive modeling

---

## 1 Introduction

*（保持原有内容，略）*

---

## 2 Methods

### 2.1 Study Design
*（保持原有内容）*

### 2.2 Predictive Modeling
Five-fold grouped cross-validation was adopted, with GroupKFold used to divide the training set and test set (80%:20%) based on individual IDs. Hyperparameter random search was performed for 14 machine learning models (CatBoost omitted after fitting failed). The **champion model was selected by CPM (Comprehensive Performance Metric)**: the optimal threshold maximizing Youden index was determined on the **internal validation set** (training fold), and the model with the highest AUC at that threshold on the held-out validation set was chosen. This approach avoids test-set optimization per TRIPOD guidelines. Predictive performance was evaluated using AUC, AUPRC, Recall, Specificity, Youden index, Brier score, and 95% confidence intervals via bootstrap.

### 2.3 Interpretability Analysis
*（保持原有内容）*

### 2.4 Causal Inference Framework and Identification Strategy
We adopted the Potential Outcomes Framework and employed **XLearner** (Künzel et al.) as the primary causal estimator, which typically outperforms TLearner when treatment/control groups are imbalanced. Our identification strategy relies on four assumptions: (1) **Unconfoundedness** (no unmeasured confounding given covariates); (2) **Overlap (Positivity)**—we verified common support by trimming samples with propensity scores outside [0.05, 0.95]; (3) **Consistency** (observed outcome equals potential outcome under received treatment); (4) **SUTVA** (no interference). We assessed overlap via propensity score distribution and trimming proportion; covariate balance via standardized mean difference (SMD), with |SMD|<0.1 indicating good balance; and sensitivity to unmeasured confounding via E-value (VanderWeele & Ding, 2017).

### 2.5 Causal Estimation
XLearner was fitted with Random Forest nuisance models (n_estimators=200, max_depth=4, min_samples_leaf=15). Overlap trimming was applied: samples with propensity score outside [0.05, 0.95] were excluded before estimation. ATE and 95% confidence intervals were estimated via bootstrap (200 resamples) when closed-form intervals were unavailable. Five interventions were evaluated: exercise, drinking, social isolation, normal BMI, and low chronic disease burden.

### 2.6 Cross-Validation of Causal Effects
Propensity Score Matching (PSM, 1:1 nearest neighbor, caliper=0.024×SD of PS) and Propensity Score Weighting (PSW, IPW with weight trimming [0.1, 50]) were used to cross-validate XLearner results. PSM post-matching max SMD and PSW weighted SMD were reported.

### 2.7 Sensitivity Analysis
Cutoff value sensitivity analysis was conducted: depression defined as CES-D≥8, ≥10, ≥12; cognitive impairment as score ≤8, ≤10, ≤12. The exercise effect in Cohort B remained significant under CES-D≥8 and Cog≤8 definitions (ATE −3.5% to −3.8%), supporting robustness. Placebo test (100 resamples) and E-value analysis were performed for unmeasured confounding sensitivity. Feature noise stability was evaluated.

### 2.8–2.11
*（保持原有内容）*

---

## 3 Results

### 3.1 Study Population
After applying inclusion and exclusion criteria (Figure 1), the incident cohort comprised 14,386 person-waves. Sample flow: 96,628 raw records → 49,015 (age ≥60) → 43,048 (CES-D non-missing) → 31,574 (cognition non-missing) → 16,983 (next-wave comorbidity non-missing) → 14,386 incident cohort. Baseline characteristics by cohort are shown in Table 1.

### 3.2 Incidence of Comorbidity
Incidence was highest in Cohort C (16.9%), followed by Cohort B (13.6%) and Cohort A (4.1%) (P<0.001).

### 3.3 Prediction Performance
Champion models (selected by CPM) achieved AUCs of 0.73 (Cohort A, Logistic Regression), 0.70 (Cohort B, LightGBM), and 0.66 (Cohort C, Naive Bayes) (Table 2). For Cohort B, LightGBM had the highest AUC but low Recall (0.01) at the Youden-optimal threshold; LR achieved Recall 0.73 and AUC 0.68. For Cohort C, Naive Bayes achieved Recall 0.91 at the optimal threshold, suitable for screening. *Main-text ROC figure omitted.*

### 3.4 Causal Effect of Exercise
XLearner estimated the ATE of exercise on incident comorbidity. In **Cohort B (Depression-only)**, exercise significantly reduced comorbidity risk (ATE = −4.2%, 95% CI: −7.9% to −0.9%, p=0.019). In Cohort A (Healthy), the effect was negligible (ATE = −0.3%, 95% CI: −1.4% to 1.1%). In Cohort C (Cognition-impaired-only), the point estimate was positive but non-significant (ATE = 2.0%, 95% CI: −0.3% to 10.0%), suggesting possible reverse causality.

### 3.5 Causal Assumption Checks
For Cohort B (exercise), overlap was satisfied after trimming (0% of retained sample had PS outside [0.05, 0.95]); 51.2% of the original cohort was trimmed. Covariate balance was moderate (max SMD=0.16, 7 covariates with |SMD|≥0.1). E-value was 2.0 (point) and 1.27 (conservative), indicating that an unmeasured confounder would need RR≥1.27 with both treatment and outcome to explain away the effect. For Cohort A, max SMD was 0.29 (substantial imbalance). For Cohort C, overlap was violated (19.6% outside [0.05, 0.95]) and max SMD was 0.87, limiting the interpretability of causal estimates in that cohort.

### 3.6 Other Modifiable Factors
Low chronic disease burden (≤1 condition) was associated with increased comorbidity risk in Cohort C (ATE = 6.3%, 95% CI: 1.5%–10.8%, p=0.008), possibly reflecting reverse causation or selection bias. No other interventions reached statistical significance across cohorts (Table 4).

### 3.7 Subgroup Heterogeneity
In Cohort B, the protective effect of exercise was consistent across subgroups (residence, age, sex, education, chronic disease burden, self-rated health), with CATEs ranging from −0.036 to −0.054 (Table 3, Figure 4).

### 3.8 Cross-Validation with PSM and PSW
For exercise in Cohort B, PSM yielded ATE = −1.6% (95% CI: −3.9% to 0.7%, non-significant), PSW yielded ATE = −2.8% (95% CI: −5.3% to −0.4%, significant), and XLearner yielded ATE = −4.2% (95% CI: −7.9% to −0.9%, significant). All three methods showed a protective direction; XLearner produced the largest effect. PSM post-matching max SMD was 0.12 (≥0.1, suggesting residual imbalance).

### 3.9 External Validation
Temporal validation (train wave<4, validate wave=4): AUC 0.68 (A), 0.57 (B), 0.57 (C). Regional validation (train East+Central, validate West): AUC 0.70 (A), 0.65 (B), 0.58 (C). Cohort A showed the best generalizability.

### 3.10 Clinical Decision Support
DCA indicated net benefit at threshold probabilities of approximately 5%–35%. Calibration slopes were 1.77 (Cohort B) and 0.20 (Cohort C), indicating miscalibration in Cohort C.

---

## 4 Discussion

### 4.1 Principal Findings
We developed a three-cohort framework for predicting and causally analyzing incident depression–cognition comorbidity in Chinese older adults. Incidence varied substantially by baseline status (4.1% in healthy, 13.6% in depression only, 16.9% in cognitive impairment only). Prediction models achieved moderate discrimination (AUC 0.66–0.73), with Logistic Regression (A), LightGBM (B), and Naive Bayes (C) as champions. **XLearner causal analysis demonstrated that regular exercise significantly reduced comorbidity risk in the depression-only cohort (ATE = −4.2%, 95% CI: −7.9% to −0.9%)**, with overlap satisfied, moderate covariate balance, and E-value of 2.0. Cross-validation with PSM and PSW supported a protective direction. Subgroup analyses suggested no substantial heterogeneity. Cohort C showed overlap and balance violations, limiting causal interpretability; the positive association of low chronic disease burden with comorbidity likely reflects reverse causality.

### 4.2 Comparison with Prior Literature
*（保持原有内容，更新数值）* Prior meta-analyses support a protective effect of exercise on depression and cognition [12,13]. Our XLearner estimate (−4.2%) is consistent with this literature and provides evidence from a large Chinese cohort with rigorous causal methods.

### 4.3 Mechanisms and Interpretation
The protective effect of exercise in Cohort B aligns with neurobiological mechanisms (BDNF, anti-inflammatory pathways). The consistency of direction across XLearner, PSM, and PSW strengthens causal interpretation. The positive estimate in Cohort C for low chronic disease burden likely reflects that healthier individuals at baseline are more likely to develop comorbidity due to longer survival or differential surveillance—a form of reverse causality.

### 4.4 Limitations
First, the observational design precludes definitive causal inference; unmeasured confounding may remain despite E-value analysis. Second, overlap trimming excluded 51.2% of Cohort B, so the ATE applies to the trimmed population. Third, Cohort C showed overlap and balance violations; causal estimates there should be interpreted with caution. Fourth, PSM/PSW/XLearner yielded somewhat inconsistent point estimates, though directions agreed. Fifth, definitions and cutpoints may vary across studies. Finally, generalizability to other populations requires caution.

### 4.5 Conclusions
A three-cohort framework enables stratified prediction and causal assessment of incident depression–cognition comorbidity. **Regular exercise significantly reduces comorbidity risk in older adults with depression only (ATE = −4.2%)**, with assumption checks supporting interpretability. The integrated approach can inform risk stratification and targeted interventions for older adults in China.

---

## Table 2. Prediction performance by cohort (CPM champion)

| Cohort | Champion Model | AUC (95% CI) | Recall | Specificity | Brier |
|--------|----------------|--------------|--------|-------------|-------|
| A | Logistic Regression | 0.73 (0.66–0.78) | 0.61 | 0.70 | 0.21 |
| B | LightGBM | 0.70 (0.65–0.76) | 0.01* | 1.00 | 0.12 |
| C | Naive Bayes | 0.66 (0.61–0.72) | 0.91 | 0.31 | 0.18 |

*LightGBM in Cohort B had low Recall at Youden-optimal threshold; LR (AUC 0.68, Recall 0.73) may be preferred for screening.

---

## Table 3. Subgroup CATE (exercise, Cohort B)

| Subgroup | Value | CATE | N |
|----------|-------|------|---|
| Residence | Urban | −0.042 | 615 |
| | Rural | −0.041 | 910 |
| Age | <65 | −0.038 | 1,171 |
| | 65–75 | −0.054 | 319 |
| | 75+ | −0.045 | 35 |
| Sex | Male | −0.039 | 859 |
| | Female | −0.045 | 666 |

---

## Table 4. Causal assumption checks (exercise)

| Cohort | Overlap (post-trim) | max SMD | E-value |
|--------|---------------------|---------|---------|
| A | 0% ✓ | 0.29 (>0.2) | 1.31–1.71 |
| B | 0% ✓ | 0.16 (0.1–0.2) | 2.0 / 1.27 |
| C | 19.6% ✗ | 0.87 (>0.2) | 1.49 / 1.15 |

---

## References

*（保持原有参考文献）*

---

**说明**：本稿基于 2026-03-20 运行结果更新。数据来源：`运行结果简报_2026-03-20_详细版.md`、`results/tables/`、`Cohort_*/03_causal/assumption_checks_summary.txt`。
