# 审稿意见 3.3–3.4 方法学修订稿

根据代码实现（`charls_recalculate_causal_impact.py`、`charls_sensitivity_analysis.py`、`charls_bias_analysis.py`）整理，可直接插入或替换论文相应段落。

---

## 3.3 因果推断方法（Causal Forest DML 实施细节）

### 修订后的 2.5 Causal Effect Estimation（替换原 2.5 节）

Causal Forest DML was used to estimate the Average Treatment Effect (ATE) and Conditional Average Treatment Effect (CATE) of each of the five modifiable lifestyle factor interventions on incident comorbidity. The implementation followed the econml library (Microsoft Research) with the following specifications:

**Nuisance function estimators.** The outcome regression (E[Y|X]) and propensity score model (E[T|X]) were each fitted using **CatBoost** (gradient boosting): CatBoostRegressor for the outcome and CatBoostClassifier for the treatment, both with n_estimators=200, max_depth=4, and random_state fixed for reproducibility. CatBoost natively handles mixed continuous and categorical covariates without explicit one-hot encoding.

**Cross-fitting.** Five-fold cross-fitting was adopted (cv=5) to eliminate regularization bias. The sample was split into five folds; nuisance functions were fitted on four folds and predictions obtained on the held-out fold, with the process rotated so each observation received out-of-fold predictions. Final ATE estimation was performed on the cross-fitted residuals.

**Causal Forest hyperparameters.** The Causal Forest comprised 1000 decision trees (n_estimators=1000). The forest was grown on the cross-fitted residuals to estimate heterogeneous treatment effects. Cluster-robust inference was used to account for intra-individual correlation (groups=individual ID), as the same participant could contribute multiple person-waves.

**Covariate handling.** All 32 pre-selected covariates were numeric. Continuous covariates (age, BMI, waist circumference, blood pressure, pulse, grip strength, income, family size, ADL/IADL counts, walking speed) were standardized (zero mean, unit variance) before fitting. Binary and ordinal covariates were median-imputed for missing values but not scaled, to avoid imposing inappropriate scale assumptions. Categorical covariates were excluded or represented as numeric indicators in the preprocessing pipeline.

ATE was estimated for each intervention in all three cohorts. For the key intervention (exercise), CATE was further estimated to explore heterogeneous treatment effects across pre-defined subgroups.

---

## 3.3 重叠假设（Overlap）补充

### 新增段落（插入 2.4 或 2.5 后）

**Overlap (Positivity) and common support.** The overlap assumption was verified by visual inspection of propensity score distributions for the exercise intervention (Supplementary Figure S2), which showed substantial overlap between treated and control groups. For **Causal Forest DML**, no explicit sample trimming was applied; the estimator uses the full sample, and the cross-fitting procedure mitigates bias from potential extrapolation in regions of weak overlap. For **Propensity Score Weighting (PSW)**, propensity scores were restricted to the interval [0.01, 0.99] to avoid extreme weights, and inverse probability weights were trimmed to [0.1, 50]. For **Propensity Score Matching (PSM)**, the caliper was set to 0.2× the standard deviation of the propensity score (0.024 when SD=0.12); observations without a match within the caliper were excluded, implicitly defining the common support region. We did not apply additional propensity score trimming (e.g., 1st–99th percentile) for the DML analysis, as the non-parametric forest estimator is less sensitive to extreme propensity values than linear IPW.

---

## 3.4 敏感性分析修订

### E-Value（新增/修订 2.10 或敏感性分析节）

**E-Value calculation.** The E-Value quantifies the minimum strength of association (on the risk ratio scale) that an unmeasured confounder would need to have with both the treatment and the outcome to fully explain away the observed effect. We used the formula E = RR + √[RR×(RR−1)], where RR is the risk ratio (VanderWeele & Ding, 2017). For binary treatment (e.g., exercise: yes/no), RR was approximated from a linear probability model: we regressed the outcome on the treatment and used the coefficient and baseline risk to approximate the risk ratio. The resulting E-Values were reported in the sensitivity analysis output (e.g., Cohort B exercise: E-Value ≈ 2.01; Cohort C: ≈ 1.21). For binary exposures, E-Value ≥2 is generally considered to indicate moderate robustness to unmeasured confounding; E-Value <1.5 suggests the observed effect could be explained by a relatively weak unmeasured confounder. **For continuous exposures** (e.g., exercise frequency as a continuous variable), the standard E-Value formula applies to the risk ratio comparing a specific increment (e.g., one unit increase); our primary analysis used binary exercise (regular vs. not), so the binary E-Value formulation applies directly.

### 未测量混杂模拟（confounder_strength 定义）

**Unmeasured confounding simulation.** We conducted a simplified bias simulation to assess how ATE estimates would change under varying strengths of unmeasured confounding. We assumed a hypothetical unmeasured confounder U (standard normal) that affects both the treatment T and the outcome Y. The parameter **confounder_strength** (s) denotes the regression coefficient scaling U’s effect on T and Y in the data-generating model: T_sim = T + s×U and Y_sim = Y + 0.5×s×U. Thus, s=0 corresponds to no unmeasured confounding; s=0.1, 0.2, 0.3, 0.5 represent progressively stronger associations. For each s, we re-estimated the ATE via linear regression of Y_sim on T_sim and covariates, and recorded the biased ATE. This is a qualitative sensitivity tool (not a formal E-Value analysis) to illustrate the direction and magnitude of bias under increasing unmeasured confounding. The results are reported in Supplementary Table S3 and Figure S (bias sensitivity plot).

---

## 建议的 Table S3 图注补充

**Table S3.** Sensitivity analysis for unmeasured confounding. *confounder_strength*: coefficient scaling the effect of a hypothetical unmeasured confounder U (standard normal) on both treatment and outcome in the simulation model (T_sim = T + s×U, Y_sim = Y + 0.5×s×U). Higher values indicate stronger unmeasured confounding. ATE_biased: estimated ATE under each confounding strength. The simulation is for qualitative assessment only; formal E-Values are reported separately in the sensitivity report.

---

## 实际 E-Value 报告位置

建议在正文 3.7 或敏感性分析节补充一句：

> The E-Values for the exercise intervention were 2.01 (Cohort B) and 1.21 (Cohort C), indicating that an unmeasured confounder would need to be associated with both exercise and comorbidity at a risk ratio of at least 2.01 (Cohort B) to fully explain away the observed protective effect.

（具体数值以 `sensitivity_report.txt` 为准）
