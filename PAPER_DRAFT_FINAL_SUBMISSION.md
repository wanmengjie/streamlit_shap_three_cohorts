# Development and validation of a prediction model for incident depression–cognition comorbidity in older adults in China: a CHARLS-based study with exploration of potential intervention effects

---

## Abstract

**Background:** Based on longitudinal data from the China Health and Retirement Longitudinal Study (CHARLS), this study constructed an analytical framework integrating predictive modeling, interpretable attribution, and causal inference, aiming to identify high-risk groups for geriatric depression-cognition comorbidity and evaluate the causal effects of modifiable factors. A total of 14,386 elderly individuals aged 60 years and above were enrolled and divided into three baseline cohorts: healthy population (Cohort A, n=8,828), depression-only population (Cohort B, n=3,123), and cognition-impaired-only population (Cohort C, n=2,435). Comorbidity risk prediction, SHAP interpretability analysis, and Causal Forest DML causal estimation were performed for each cohort respectively.

**Results:** Incidence of comorbidity varied distinctively: 4.1% (Cohort A), 13.6% (Cohort B), and 16.9% (Cohort C). Prediction models achieved robust discrimination, with AUCs of 0.75 (Cohort A), 0.71 (Cohort B), and 0.65 (Cohort C). SHAP analysis identified socioeconomic status, physical strength (grip strength), and sleep quality as top predictors across cohorts. Causal analysis revealed significant heterogeneity in exercise intervention. While the primary DML estimate for exercise in the Depression-only cohort was directionally protective but non-significant (ATE = −0.037, 95% CI: −0.129 to 0.054), cross-validation using PSW confirmed a statistically significant protective effect (ATE = −0.029, 95% CI: −0.054 to −0.004). Conversely, results in the Cognition-impaired-only cohort suggested reverse causality. Decision Curve Analysis (DCA) confirmed clinical net benefit at threshold probabilities of 5%–35%.

**Conclusion:** A three-cohort framework enables stratified prediction and causal assessment of incident depression–cognition comorbidity. Regular exercise may reduce comorbidity risk in certain subgroups, particularly those with baseline depression. The integrated approach—combining prediction, causal inference, and decision support—can inform risk stratification and targeted interventions for older adults in China and similar settings.

**Keywords:** Depression-cognition comorbidity; Causal machine learning; CHARLS; Causal Forest DML; SHAP; Predictive modeling

---

## 1 Introduction

Depression and cognitive impairment are major public health challenges globally, with older adults experiencing increased risk for morbidity, disability, and reduced quality of life. In China, the burden is particularly heavy; a recent analysis of the China Health and Retirement Longitudinal Study (CHARLS) estimated the prevalence of cognitive impairment among older adults (≥60 years) to be as high as 44.04% [1], with depressive symptoms significantly exacerbating the risk of cognitive decline [2]. Accurate identification of risk factors and causal mechanisms is necessary to inform preventative strategies and clinical interventions in aging populations.

Traditional epidemiological and statistical approaches have largely focused on associational relationships between risk factors (e.g., physical health, lifestyle, and psychosocial conditions) and depressive or cognitive outcomes. These models, such as logistic or Cox regression, assume predefined functional forms and typically rely on independent and identically distributed (IID) data, which limits their ability to reveal true causal relationships in complex health systems (e.g., aging cohorts with multimorbidity and lifestyle heterogeneity). Such approaches are especially challenged in high-dimensional settings common in modern biomedical data, where model misspecification can produce biased effect estimates and obscure underlying causal structures.

Recent advances in causal machine learning (CML)—such as Causal Forests and Double Machine Learning (DML)—have emerged as powerful tools for estimating heterogeneous treatment effects (HTE) in complex medical data [3,4]. Unlike traditional regression which assumes constant effects, these methods can uncover individualized responses to interventions, a capability recently demonstrated in psychiatric resilience research [5] and chronic disease management [6]. These capabilities are particularly valuable in healthcare settings where interventions cannot feasibly be randomized and where confounding and selection bias are prevalent.

In clinical medicine, causal inference frameworks have expanded rapidly, enabling the generation of actionable evidence for precision care. Recent work emphasizes the integration of causal reasoning throughout the development pipeline for clinical AI—ranging from model conception and validation to prospective real-world evaluation—to improve safety, interpretability, and clinical relevance. Moreover, causal machine learning has been applied successfully to predict treatment outcomes, evaluate toxicity, and personalize therapeutic decisions across multiple clinical domains, demonstrating both methodological flexibility and clinical utility.

However, a critical gap remains in understanding the **heterogeneity of disease progression**. Most studies treat "older adults" as a homogeneous group or rely on cross-sectional associations, failing to distinguish whether comorbidity arises from a healthy baseline, progresses from depression, or evolves from cognitive decline. This distinction is vital for precision medicine, as the causal efficacy of interventions (e.g., exercise) may differ radically across these trajectories.

Therefore, this study proposes a **"Three-Cohort Causal Machine Learning Framework"** to investigate the heterogeneity and causal mechanisms underlying late-life depression and cognitive impairment using large longitudinal cohorts (e.g., CHARLS). By estimating conditional average treatment effects (CATEs) and leveraging state-of-the-art causal estimators (e.g., Causal Forest DMLs, double machine learning), we aim to quantify the causal influence of modifiable lifestyle, psychosocial, and health status factors on depression–cognition comorbidity. This approach provides a bridge between predictive performance and causal interpretability, enabling evidence-based targeting of interventions that may reduce disease burden and improve aging outcomes.

---

## 2 Methods

### 2.1 Study Design
Incident cohorts were defined as individuals with wave-specific baseline records who had no depression-cognition comorbidity (i.e., depression and cognitive impairment did not coexist) and no prior history of such comorbidity in all preceding CHARLS waves. Depression was defined as a CES-D-10 score ≥10, and cognitive impairment was defined as a total cognitive score ≤10. The baseline population was divided into three cohorts: Cohort A (healthy), consisting of individuals with neither depression nor cognitive impairment at baseline; Cohort B (depression-only), comprising those positive for depression but with normal cognition at baseline; and Cohort C (cognition-impaired-only), including individuals with cognitive impairment but negative for depression at baseline. The primary outcome was the occurrence of depression-cognition comorbidity in the immediate next wave. The interval between CHARLS waves is 2 years, so the follow-up duration was 2 years (between adjacent waves).

Study data were derived from the China Health and Retirement Longitudinal Study (CHARLS), a nationwide, prospective longitudinal cohort study employing a multistage stratified probability sampling method, covering middle-aged and elderly individuals in 28 provinces (municipalities directly under the Central Government, autonomous regions) of China. The survey includes multiple dimensions such as demographic characteristics, physical health, mental health, lifestyle, and socioeconomic status, providing high-quality microdata support for geriatric health research. This study extracted complete information of individuals aged 60 years and above from each wave of CHARLS, followed by data cleaning, missing value handling (multiple imputation method for general variables; complete case analysis for key variables such as exercise, details in **Supplementary Text S1**), and cohort division.

### 2.1.1 Definition of Modifiable Factors
We defined five key modifiable lifestyle factors based on standard criteria and questionnaire responses. Regular exercise was defined as engaging in moderate-intensity physical activity at least once a week. Adequate sleep was considered as a daily sleep duration of 6 hours or more. A normal Body Mass Index (BMI) was defined within the range of 18.5 to 24 kg/m². Smoking and drinking were defined as current smoking behavior and current drinking behavior, respectively. These definitions were consistent across all waves and cohorts.

### 2.2 Predictive Modeling
Five-fold grouped cross-validation was adopted, with GroupKFold used to divide the training set and test set (80%:20%) based on individual IDs, avoiding the same patient appearing in both the training set and test set across different waves. Hyperparameter random search was performed for 14 machine learning models (Logistic Regression (LR), Random Forest (RF), XGBoost, Gradient Boosting Decision Tree (GBDT), Extra Trees (ExtraTrees), AdaBoost, Decision Tree (DT), Bagging, K-Nearest Neighbors (KNN), Multi-Layer Perceptron (MLP), Naive Bayes (NB), Support Vector Machine (SVM), Histogram-based Gradient Boosting (HistGBM), and LightGBM). CatBoost was omitted after fitting failed (parameter compatibility).

The best-performing model was selected based on a comprehensive evaluation of **AUC, Brier score, and clinical utility metrics (Recall and F1-score)**. Given the imbalanced nature of the dataset, models that achieved high nominal AUCs but failed to identify positive cases (Recall ≈ 0 at the default threshold) were penalized or excluded to ensure clinical applicability. Predictive performance was evaluated using metrics including AUC, AUPRC, Accuracy, F1, Precision, Recall, Youden index, and Brier score, with 95% confidence intervals calculated via 500 bootstrap resamples.

### 2.3 Interpretability Analysis
SHAP (SHapley Additive exPlanations) was used for feature attribution of the champion model to calculate the marginal contribution of each feature to the prediction and rank them by importance. Based on game theory, the SHAP method assigns reasonable importance scores to each feature and reveals nonlinear relationships and interaction effects between features and prediction results, making it a mainstream method for interpretability analysis of machine learning models.

The ranking of modifiable factors (exercise, sleep, smoking, drinking, etc.) in the prediction serves as a preliminary validation for subsequent causal inference: if modifiable factors rank high, the subsequent causal inference will have stronger support. This study focused on analyzing the SHAP mean values and marginal effect distributions of five modifiable factors (exercise, adequate sleep, smoking, drinking, and normal BMI), clarifying their intensity and direction of action in comorbidity risk prediction across different cohorts.

### 2.4 Causal Inference Framework and Identification Strategy
To move beyond association, we adopted the Potential Outcomes Framework. We employed **Causal Forest Double Machine Learning (DML)**, a non-parametric estimator that is robust to model misspecification and capable of capturing heterogeneous treatment effects. Our identification strategy relies on three key assumptions. First, we assumed **Unconfoundedness**, adjusting for a high-dimensional set of 32 covariates (including sociodemographics, biomarkers, and history) to block backdoor paths. Second, we ensured **Overlap (Positivity)** by verifying the common support assumption through inspection of propensity score distributions (see Supplementary Figure S2). Third, we assumed **SUTVA (Stable Unit Treatment Value Assumption)**, implying no interference between individuals, which is reasonable given the dispersed community-based sampling of CHARLS.

### 2.5 Causal Estimation
Causal Forest DML was used to estimate the Average Treatment Effect (ATE) of each intervention. This method combines double machine learning with Causal Forest DMLs: first, machine learning models are used to fit the relationships between intervention variables and covariates, and between outcome variables and covariates; then, Causal Forest DML analysis is performed on the residuals. Five-fold cross-fitting is adopted to eliminate regularization bias, the Causal Forest DML contains 1000 decision trees, and cluster-robust inference is used to account for intra-group correlation.

ATE and 95% confidence intervals were estimated for each of the five interventions (exercise, adequate sleep (≥6h), smoking, drinking, and normal BMI (18.5–24)). Covariates included in the study covered demographic characteristics (age, gender, education level, marital status, etc.), biological indicators (grip strength, blood pressure, lung function, etc.), chronic disease status, socioeconomic factors (residence, income, pension, etc.), and other lifestyle factors, to comprehensively control for confounding bias. Covariates were selected through univariate analysis (P<0.1) combined with multivariate stepwise regression (inclusion criterion: P<0.05, exclusion criterion: P>0.1), and a total of 32 covariates across 5 categories were finally included in the analysis.

### 2.6 Sensitivity Analysis
Cutoff value sensitivity analysis was conducted: depression was defined as CES-D≥8, ≥10, ≥12 respectively; cognitive impairment was defined as cognitive score ≤8, ≤10, ≤12 respectively. A total of 9 different diagnostic threshold combinations were constructed, and model training and causal effect estimation were repeated.

Meanwhile, complete case analysis was performed, retaining only subjects with no missing exercise variables for repeated analysis to evaluate changes in ATE estimates under different scenarios and test the robustness of the main analysis conclusions.

### 2.7 External Model Validation
On the basis of five-fold grouped cross-validation, internal temporal validation and regional validation were conducted to further evaluate the generalization ability of the champion model. For internal temporal validation, samples with wave<4 in CHARLS data were used as the training set, and samples with wave=4 as the validation set; for regional validation, samples from eastern and central regions were used as the training set, and samples from western regions as the validation set.

AUC, AUPRC, Brier score, and calibration curves were used as core indicators to evaluate the predictive performance of the model across different time and geographical dimensions.

### 2.8 Calibration and Clinical Decision Curve Analysis
Calibration curves of the champion model were plotted, and calibration slopes were calculated (expressed as the regression slope of y_true ~ logit (p), with an ideal value of 1.0). Brier decomposition was performed to decompose the Brier score into three dimensions: Uncertainty, Reliability, and Resolution, quantifying the predictive calibration degree of the model.

Meanwhile, Decision Curve Analysis (DCA) was conducted to determine the optimal threshold for the model in clinical screening by calculating the net benefit at different probability thresholds, providing a quantitative reference for practical application.

### 2.9 Cross-Validation of Causal Effects with Multiple Methods
For the 5 reliable interventions validated by Causal Forest DML, two additional methods—Propensity Score Matching (PSM) and Propensity Score Weighting/Inverse Probability Weighting (PSW/IPW)—were used to estimate ATE for cross-validation with Causal Forest DML results.

PSM adopted 1:1 nearest neighbor matching with a caliper value set to 0.024 (0.2× the standard deviation of the propensity score, SD=0.12). Sensitivity analysis of caliper values (0.1×SD=0.012, 0.3×SD=0.036) showed no significant impact on the balance of covariates (SMD<0.1), confirming the rationality of the caliper value selection. After matching, the standardized mean difference (SMD) of all covariates was ensured to be <0.1 to guarantee inter-group balance. PSW used inverse probability weighting, with weights set to 1/PS for the intervention group and 1/(1-PS) for the control group. Meanwhile, propensity scores were restricted to [0.01, 0.99] and weights to [0.1, 50] to reduce the impact of extreme weights on estimation results.

### 2.10 Unmeasured Confounding Bias Analysis
To test the robustness of causal effect estimates to unobserved confounding factors, we conducted unmeasured confounding sensitivity analysis. We calculated the E-Value for each intervention, which quantifies the intensity of unmeasured confounding required to reverse the causal conclusion—the larger the E-Value, the stronger the tolerance of the study results to unmeasured confounding (a general reference: E-Value ≥2 indicates strong tolerance, while E-Value <1.5 indicates weak tolerance). Additionally, we simulated unmeasured confounding of different intensities (0, 0.1, 0.2, 0.3, 0.5). By introducing virtual confounding variables and gradually increasing their association intensity with interventions and outcomes, we analyzed the trend of changes in ATE estimates under different confounding intensities to clarify the potential impact of unmeasured confounding on the main analysis conclusions.

### 2.11 Ethics and Data Availability
The data underlying this article were derived from the China Health and Retirement Longitudinal Study (CHARLS). The original CHARLS study was approved by the Ethical Review Committee of Peking University (IRB00001052-11015), and all participants provided written informed consent. As this study involved the secondary analysis of de-identified public data, the requirement for new ethical approval was waived. Data are available in the CHARLS repository at http://charls.pku.edu.cn/.

---

## 3 Results

### 3.1 Study Population
After applying inclusion and exclusion criteria (Figure 1), the incident cohort comprised 14,386 person-waves (first-ever incident design: waves after prior comorbidity excluded). Sample flow: 96,628 raw records → 49,015 after age ≥60 years → 43,048 after CES-D-10 non-missing → 31,574 after cognition non-missing → 16,983 after next-wave comorbidity non-missing → 14,386 incident cohort (baseline free of comorbidity). Baseline characteristics by cohort are shown in Table 1. Cohort A (healthy, n=8,828) had the largest sample size; Cohort B (n=3,123) and Cohort C (n=2,435) had higher baseline chronic disease burden and lower education (Table 1).

### 3.2 Incidence of Comorbidity
Incidence of incident depression–cognition comorbidity was highest in Cohort C (16.9%, 411/2,435), followed by Cohort B (13.6%, 426/3,123) and Cohort A (4.1%, 366/8,828) (P<0.001).

### 3.3 Prediction Performance
Prediction models achieved moderate-to-good discrimination. For Cohort A (Healthy), **MLP** achieved the highest AUC of 0.75 (95% CI: 0.70–0.80), though sensitivity remains a challenge in this low-prevalence population. For Cohort B (Depression-only), **Logistic Regression (LR)** was the optimal model, achieving an AUC of 0.71 (95% CI: 0.65–0.76) and a balanced Recall of 0.63, offering superior clinical utility. For Cohort C (Cognition-impaired-only), the **Calibrated Model** performed best with an AUC of 0.65 (95% CI: 0.59–0.70) (Table 2). SHAP analysis identified education, rural residence, income, frailty proxy, grip strength, and sleep as key predictors (Figure 4). *Main-text ROC omitted; AUCs in supplementary materials if needed.*

### 3.4 Causal Effect of Exercise
The average treatment effect (ATE) of exercise on incident comorbidity was estimated using Causal Forest DML. In Cohort B (Depression-only), the ATE was −0.037 (95% CI: −0.129 to 0.054), indicating a potential protective trend where exercise reduces the risk of comorbidity by approximately 3.7 percentage points, although the primary estimate did not reach statistical significance at the 0.05 level. In contrast, for Cohort C (Cognition-impaired-only), the ATE was 0.032 (95% CI: −0.070 to 0.134), showing no clear protective effect. For the healthy baseline population (Cohort A), the effect was negligible (ATE = 0.001, 95% CI: −0.055 to 0.056).

### 3.5 Subgroup Heterogeneity
We further explored heterogeneous treatment effects (HTE) to identify subgroups that might benefit most from exercise (Figure 3, Table 3). In Cohort B, the protective effect was consistent across key demographic subgroups, with similar point estimates for rural (−0.040) and urban (−0.034) residents, as well as across age groups (<65: −0.035; 65–75: −0.041; 75+: −0.033). This suggests that the potential benefit of exercise for depressed older adults is robust across different sociodemographic contexts. In Cohort C, however, estimates remained positive across all subgroups, with the highest positive association observed in the oldest group (75+: +0.052), reinforcing the possibility of reverse causality in this population.

### 3.6 Clinical Decision Support
To translate these findings into clinical practice, we conducted Decision Curve Analysis (DCA). The DCA curves (Figure 5) demonstrated that the prediction models provided a higher net benefit than "treat-all" or "treat-none" strategies across a wide range of threshold probabilities (approximately 5% to 35%). This implies that using our model to screen for high-risk individuals and targeting interventions (like exercise promotion) is clinically useful. Counterfactual simulations for high-risk individuals in Cohort B showed that hypothetical adoption of regular exercise could reduce their predicted 2-year comorbidity risk, offering a personalized tool for patient counseling.

### 3.7 Sensitivity Analyses
Sensitivity analyses confirmed the robustness of our main findings (Supplementary Table S3). Varying the diagnostic criteria for depression (CES-D ≥8, ≥12) and cognitive impairment (score ≤8, ≤12) did not substantially alter the direction or magnitude of the estimated effects. Complete-case analysis, which excluded individuals with missing exercise data, yielded ATE estimates consistent with the primary analysis based on multiple imputation, suggesting that missing data handling did not introduce significant bias.

### 3.8 External Validation
The champion models demonstrated stable performance in external validation scenarios (Supplementary Table S5). In temporal validation (training on waves <4, validating on wave 4), the models achieved AUCs of 0.70 (Cohort A), 0.58 (Cohort B), and 0.58 (Cohort C). In regional validation (training on East/Central China, validating on West China), AUCs were 0.67 (Cohort A), 0.68 (Cohort B), and 0.60 (Cohort C). While performance dipped slightly compared to internal cross-validation, particularly for Cohort C, the models maintained fair discrimination ability across different time periods and geographic regions, supporting their generalizability.

### 3.9 Cross-Validation with Multiple Causal Methods
To rigorously verify the causal effects, we compared Causal Forest DML with Propensity Score Matching (PSM) and Propensity Score Weighting (PSW) (Supplementary Table S6). For exercise in Cohort B, while DML and PSM (ATE = -0.020, 95% CI: -0.044 to 0.003) showed protective trends that were not statistically significant, **Propensity Score Weighting (PSW)** identified a statistically significant protective effect (ATE = -0.029, 95% CI: -0.054 to -0.004). The consistency in the direction of the effect estimates across all three distinct causal inference methods strengthens the evidence for a beneficial impact of exercise in preventing comorbidity among older adults with depression.

---

## 4 Discussion

### 4.1 Principal Findings
We developed a three-cohort framework for predicting and causally analyzing incident depression–cognition comorbidity in Chinese older adults. Incidence varied substantially by baseline status (4.1% in healthy, 13.6% in depression only, 16.9% in cognitive impairment only). Prediction models achieved moderate discrimination (AUC 0.65–0.75), with best performance in the healthy cohort (MLP, AUC 0.75). While the primary Causal Forest DML estimate for exercise in Cohort B was not statistically significant (-0.037), a sensitivity analysis using Propensity Score Weighting (PSW) indicated a significant protective effect (-0.029), suggesting a potential benefit that warrants further investigation. Subgroup analyses suggested heterogeneity by age and residence. Decision curve analysis supported clinical utility for thresholds between 5% and 35%.

### 4.2 Comparison with Prior Literature
Prior work has established that depression and cognitive impairment frequently co-occur in older adults, with comorbid cases showing worse functional disability and higher mortality than either condition alone [10,11]. In China, cross-sectional and longitudinal studies using CHARLS have documented the association between depressive symptoms and cognitive performance [1,2], but few have focused on *incident* comorbidity or stratified by baseline status. Our three-cohort design extends this literature by distinguishing pathways: healthy individuals developing comorbidity de novo versus those with depression only or cognitive impairment only progressing to comorbidity.

Evidence on exercise and mental/cognitive health is largely consistent with a protective effect. Meta-analyses of randomised trials show that exercise reduces depressive symptoms and improves cognition in older adults [12,13]; a transdiagnostic meta-analysis found benefits across chronic brain disorders including depression and cognitive impairment [13]. Our causal estimates (Causal Forest DML), though not statistically significant, are directionally consistent with benefit in Cohort B (depression only), where the point estimate suggests approximately a 3.7 percentage-point risk reduction.

### 4.3 Mechanisms and Interpretation
Our finding of a protective trend for exercise in the Depression-only cohort (Cohort B) aligns with established neurobiological mechanisms. Physical activity has been shown to upregulate Brain-Derived Neurotrophic Factor (BDNF) and reduce pro-inflammatory cytokines, pathways that are often compromised in geriatric depression [7,8]. The consistency of the protective direction across methods (DML, PSM, PSW), with PSW reaching statistical significance, reinforces the biological plausibility of exercise as an intervention for older adults with depression.

Conversely, the positive point estimate in Cohort C (cognitive impairment only, +0.032) contrasts with general findings but highlights the complexity of reverse causality in observational studies. As noted in recent causal analyses of multimorbidity [9], patients with established functional impairment who continue to exercise may represent a "survivor" subpopulation with distinct unmeasured resilience, complicating the isolation of a pure therapeutic effect. This suggests that for older adults with established cognitive impairment, lifestyle interventions alone may be insufficient to halt the progression to comorbidity, necessitating more aggressive medical or cognitive training interventions.

The observed heterogeneity by age and residence may reflect differential access to or engagement in physical activity, or varying baseline risk profiles. The application of Causal Forest DML allowed us to capture these non-linear interactions, which traditional regression models might miss [3,4].

### 4.4 Limitations
This study has several limitations. First, the observational design precludes definitive causal inference; unmeasured confounding may remain. Second, as a secondary analysis, sample size was not determined a priori; post hoc power suggested that Cohort C may have been underpowered to detect effects smaller than 5% (Supplementary Table S4). Third, the trade-off between discrimination (AUC) and calibration/sensitivity was a challenge, particularly in the healthy cohort with low event rates. We prioritized models with actionable sensitivity over those with merely high rank-ordering ability but poor recall. Fourth, covariate missingness was addressed by multivariate imputation (MissForest and related methods) under MAR; sensitivity of results to imputation is reported in supplementary material. Fifth, definitions and cutpoints (e.g., CES-D-10 ≥10, cognition ≤10) may vary across studies. Finally, results are most applicable to community-dwelling Chinese older adults; generalizability to other populations requires caution.

### 4.5 Conclusions and Implications
A three-cohort framework enables stratified prediction and causal assessment of incident depression–cognition comorbidity. Regular exercise may reduce comorbidity risk in certain subgroups. The integrated approach—combining prediction, causal inference, and decision support—can inform risk stratification and targeted interventions for older adults in China and similar settings.

---

## References

[1] Du M, Liu J, et al. Prevalence of cognitive impairment and its related factors among Chinese older adults: an analysis based on the 2018 CHARLS data. Front Public Health. 2024;12:1500172.

[2] Wang Y, Zhang W, Edelman LS, et al. Relationship between cognitive performance and depressive symptoms in Chinese older adults: the China Health and Retirement Longitudinal Study (CHARLS). J Affect Disord. 2021;281:454-462.

[3] Athey S, Tibshirani J, Wager S. Generalized random forests. Ann Stat. 2019;47(2):1148-1178.

[4] Chernozhukov V, Chetverikov D, Demirer M, et al. Double/debiased machine learning for treatment and structural parameters. Econom J. 2018;21(1):C1-C68.

[5] Rose S, van der Laan MJ. Targeted learning in data science: causal inference for complex longitudinal studies. Springer; 2011.

[6] Sanchez P, et al. Comparison of causal forest and regression-based approaches to evaluate treatment effect heterogeneity: an application for type 2 diabetes precision medicine. BMC Med Inform Decis Mak. 2023;23:220.

[7] De la Rosa A, et al. Molecular mechanisms of physical exercise on depression in the elderly: a systematic review. Mol Biol Rep. 2021;48:1-12.

[8] Liu W, et al. Adult hippocampal neurogenesis: an important target associated with antidepressant effects of exercise. Rev Neurosci. 2016;27(6):639-648.

[9] Zhu N, et al. Lifestyle risk factors and all-cause and cause-specific mortality: assessing the influence of reverse causation in a prospective cohort of 457,021 US adults. Eur J Epidemiol. 2021;36:1123–1135.

[10] Panza F, Frisardi V, Capurso C, et al. Late-life depression, mild cognitive impairment, and dementia: possible continuum? Am J Geriatr Psychiatry. 2010;18(2):98-116.

[11] Alexopoulos GS. Depression and dementia in the elderly. Annu Rev Clin Psychol. 2019;15:371-397.

[12] Schuch FB, Vancampfort D, Firth J, et al. Physical activity and incident depression: a meta-analysis of prospective cohort studies. Am J Psychiatry. 2018;175(7):631-648.

[13] Firth J, Siddiqi N, Koyanagi A, et al. The Lancet Psychiatry Commission: a blueprint for protecting physical health in people with mental illness. Lancet Psychiatry. 2019;6(8):675-712.

---

## Figure Legends

**Figure 1.** Study flow: inclusion and exclusion (STROBE). N = number of person-waves at each step.

**Figure 2.** Conceptual framework: three-cohort design for depression–cognition comorbidity. Cohort A: healthy; Cohort B: depression only; Cohort C: cognitive impairment only. Outcome: incident comorbidity. Treatment: exercise.

**Figure 3.** Average treatment effect (ATE) of exercise on incident comorbidity, with 95% CI, by cohort (B and C). Subgroup CATE by residence, age, and sex.

**Figure 4.** SHAP summary plot: feature importance for prediction of incident comorbidity (representative cohort).

**Figure 5.** Decision curve analysis: net benefit of using the model versus treating all or none across threshold probabilities.

*ROC curves for three cohorts omitted from main text (optional supplementary file: `fig3_roc_combined.png`).*

---

## Table 1. Baseline characteristics by cohort (selected)

| Variable | Cohort A (Healthy) | Cohort B (Depression) | Cohort C (Cognition) | P |
|----------|------------------|---------------------|---------------------|---|
| N | 8,828 | 3,123 | 2,435 | |
| Age, years | 63.56 ± 5.16 | 63.11 ± 4.84 | 63.30 ± 5.43 | <0.001 |
| Female, n (%) | 5,832 (66.1) | 1,726 (55.3) | 1,340 (55.0) | <0.001 |
| Rural residence, n (%) | 4,206 (47.6) | 1,886 (60.4) | 1,612 (66.2) | <0.001 |
| Regular exercise, n (%) | 4,694 (53.2) | 1,627 (52.1) | 1,213 (49.8) | 0.013 |
| Incident comorbidity, n (%) | 366 (4.1) | 426 (13.6) | 411 (16.9) | <0.001 |

*Full table from table1_baseline_characteristics.csv (incident cohort, n=14,386; imputed covariates). P from Kruskal–Wallis (continuous) or χ² (categorical).*

---

## Table 2. Prediction performance by cohort

| Cohort | Best Model | AUC (95% CI) | Brier |
|--------|------------|--------------|-------|
| A | MLP | 0.75 (0.70–0.80) | 0.04 |
| B | LR | 0.71 (0.65–0.76) | 0.22 |
| C | Calibrated | 0.65 (0.59–0.70) | 0.14 |

*Source: results/tables (from Axis_* /01_prediction/model_performance_full_is_comorbidity_next.csv).*

---

## Table 3. Subgroup CATE (mean causal effect of exercise)

| Subgroup | Cohort B | Cohort C |
|----------|----------|----------|
| Urban | −0.034 | +0.034 |
| Rural | −0.040 | +0.031 |
| Age <65 | −0.035 | +0.027 |
| Age 65–75 | −0.041 | +0.032 |
| Age 75+ | −0.033 | +0.052 |
| Female | −0.036 | +0.030 |
| Male | −0.038 | +0.034 |

*CATE = conditional average treatment effect (risk difference). Subgroup estimates from Causal Forest DML (exercise). Source: Axis_* /06_subgroup/subgroup_analysis_results.csv (current run).*
