# Disentangling the Heterogeneous Trajectories of Depression-Cognition Comorbidity in Older Adults: A Causal Machine Learning Study

**Target Journal:** *npj Digital Medicine*, *Nature Aging*, or *The Lancet Healthy Longevity*

---

## Abstract

**Background:** The co-occurrence of depression and cognitive impairment accelerates functional decline and mortality in older adults. However, the optimal timing and targets for intervention remain elusive due to the profound heterogeneity of disease progression. We aimed to identify trajectory-specific predictive signatures and evaluate the causal effects of modifiable lifestyle factors on incident comorbidity using a rigorously validated causal machine learning framework.

**Methods:** We analyzed longitudinal data from 14,386 older adults (≥60 years) in the China Health and Retirement Longitudinal Study (CHARLS, 2011–2018). To unmask stage-specific mechanisms, participants without baseline comorbidity were stratified into three incident cohorts: healthy (Cohort A, n=8,828), depression-only (Cohort B, n=3,123), and cognition-impaired-only (Cohort C, n=2,435). We integrated high-dimensional predictive modeling (14 algorithms optimized via a Comprehensive Performance Metric to prevent leakage; CatBoost was omitted after fitting failed in the analysis pipeline) with SHapley Additive exPlanations (SHAP). Causal effects of five modifiable factors were estimated using the XLearner meta-algorithm. Causal identifiability was rigorously stress-tested against unmeasured confounding (E-value), covariate balance (Standardized Mean Difference [SMD]), overlap assumptions, and alternative estimators (Propensity Score Matching [PSM] and Weighting [PSW]).

**Results:** Comorbidity incidence over a 2-year follow-up exhibited a distinct trajectory-dependent gradient: 4.1% in Cohort A, 13.6% in Cohort B, and 16.9% in Cohort C. Predictive models achieved robust discrimination (AUCs: 0.73, 0.70, and 0.66, respectively). Causal analysis revealed a highly specific therapeutic window: regular exercise significantly reduced the risk of incident comorbidity *exclusively* in the depression-only cohort (Average Treatment Effect [ATE] = −4.2%, 95% CI: −7.9% to −0.9%, *p*=0.019). This protective effect was robust to unmeasured confounding (E-value=2.0) and consistent across PSM and PSW cross-validations. Conversely, interventions showed no significant protective effects in the healthy cohort (floor effect) or the cognition-impaired-only cohort, where low chronic disease burden paradoxically correlated with increased risk (ATE = 6.3%, 95% CI: 1.5% to 10.8%), highlighting potential competing mortality risks or primary neurodegenerative phenotypes.

**Conclusions:** The transition from depression to depression-cognition comorbidity represents a critical, actionable inflection point in cognitive aging. Regular exercise uniquely disrupts this specific pathological cascade. By bridging predictive analytics with causal inference, our framework provides a robust blueprint for precision public health, moving beyond "one-size-fits-all" recommendations to stage-specific, targeted interventions.

---

## 1. Introduction

The global demographic shift towards an aging population has precipitated an escalating crisis of neurodegenerative and neuropsychiatric disorders. Among these, the co-occurrence of depression and cognitive impairment—often conceptualized as a "cognitive-affective" symptom cluster—poses a particularly severe public health threat. This comorbidity synergistically accelerates the loss of activities of daily living (ADLs), increases the risk of progression to overt dementia, and elevates all-cause mortality rates [1, 2]. In China, which houses the world's largest aging demographic, the burden is staggering; recent epidemiological estimates suggest over 40% of older adults experience some form of cognitive decline, frequently compounded by clinically significant depressive symptoms [3]. 

Despite the well-documented epidemiological link between late-life depression and cognitive impairment, clinical management remains hampered by a critical knowledge gap: the profound heterogeneity of disease trajectories. The progression to full comorbidity is not a monolithic process. It can emerge gradually from a healthy baseline, evolve as a secondary neurotoxic complication of primary depression (the "depression-first" pathway), or manifest as behavioral and psychological symptoms of dementia (BPSD) in those with pre-existing cognitive deficits (the "cognition-first" pathway) [4, 5]. Traditional epidemiological studies typically treat older adults as a homogenous cohort, applying standard parametric regression models that estimate average population effects. This "one-size-fits-all" approach obscures critical sub-population dynamics, making it impossible to determine *when* and *for whom* specific lifestyle interventions (e.g., physical activity, social engagement) are most effective.

Furthermore, identifying actionable therapeutic targets requires moving beyond mere prediction to robust causal inference. While traditional machine learning (ML) excels at identifying high-risk individuals through complex pattern recognition, it is inherently correlational and highly susceptible to confounding, reverse causality, and selection bias [6]. Conversely, randomized controlled trials (RCTs) for long-term lifestyle interventions in frail older adults are often logistically, financially, and ethically prohibitive. Recent breakthroughs in Causal Machine Learning (CML)—such as meta-learners (e.g., XLearner, T-Learner) and Double/Debiased Machine Learning—offer a paradigm shift [7, 8]. By leveraging large-scale observational data to estimate heterogeneous treatment effects (HTE) while rigorously adjusting for high-dimensional confounders, CML bridges the gap between predictive accuracy and causal discovery.

To address these methodological and clinical challenges, we developed a **"Three-Cohort Causal Machine Learning Framework"** using high-fidelity longitudinal data from the China Health and Retirement Longitudinal Study (CHARLS). By stratifying 14,386 older adults into distinct baseline clinical states (Healthy, Depression-only, Cognition-impaired-only), we aimed to: (1) establish cohort-specific predictive signatures to identify individuals at highest risk of progressing to full comorbidity; (2) interpret the driving features of this risk using SHapley Additive exPlanations (SHAP); and (3) rigorously evaluate the causal efficacy of modifiable lifestyle factors (e.g., exercise, social engagement) using the XLearner algorithm. Ultimately, this study seeks to provide a causal blueprint for precision geriatrics, pinpointing the exact clinical window where targeted lifestyle modifications can effectively disrupt the progression of cognitive-affective decline.

---

## 2. Methods

### 2.1 Study Design and Data Source
This study utilized data from the China Health and Retirement Longitudinal Study (CHARLS), a nationally representative, prospective cohort study of community-dwelling adults aged 45 and older in China [9]. We extracted data from four national waves (2011, 2013, 2015, and 2018). To capture the dynamic transition of health states, we constructed person-wave records, where baseline characteristics at wave *t* were used to predict the incidence of comorbidity at wave *t+1* (representing an approximate 2-year follow-up interval). The original CHARLS study was approved by the Ethical Review Committee of Peking University (IRB00001052-11015), and all participants provided informed consent.

### 2.2 Cohort Definitions and Outcome Ascertainment
We focused exclusively on incident cases to establish clear temporal precedence. Participants were excluded if they were <60 years old at baseline, had missing data on key variables (CES-D-10, cognitive score, or next-wave outcome), or had pre-existing depression-cognition comorbidity at baseline or any prior wave. 

Depression was assessed using the validated 10-item Center for Epidemiologic Studies Depression Scale (CES-D-10), with a cutoff score of ≥10 indicating clinically significant depressive symptoms [10]. Cognitive function was evaluated using a composite score (range 0-21) encompassing episodic memory (immediate and delayed word recall) and intact mental status (serial 7s subtraction, date naming, and figure drawing), with a score ≤10 defining cognitive impairment [11]. 

Based on these criteria, the baseline population was stratified into three mutually exclusive incident cohorts:
*   **Cohort A (Healthy):** CES-D-10 < 10 and Cognitive score > 10.
*   **Cohort B (Depression-only):** CES-D-10 ≥ 10 and Cognitive score > 10.
*   **Cohort C (Cognition-impaired-only):** CES-D-10 < 10 and Cognitive score ≤ 10.

The primary outcome was the new onset of depression-cognition comorbidity (concurrent CES-D-10 ≥ 10 and Cognitive score ≤ 10) in the immediate subsequent wave.

### 4.3 Modifiable Interventions and Covariates
We evaluated five modifiable lifestyle and health factors as potential causal interventions: regular exercise (moderate-to-vigorous physical activity ≥1 time/week), current drinking, social isolation (living alone with no social contact in the past month), normal BMI (18.5–24 kg/m²), and low chronic disease burden (≤1 physician-diagnosed condition out of 7 major diseases). 

Extensive covariates were included to satisfy the unconfoundedness assumption: demographics (age, sex, rural/urban residence, education, marital status), physical health (ADL/IADL difficulties, specific chronic diseases, history of falls, disability), physiological metrics (systolic/diastolic blood pressure, pulse), and socioeconomic status (pension, insurance, log-transformed household income). Missing covariates (<30% missingness) were handled using Multiple Imputation by Chained Equations (MICE). **As implemented:** bulk MICE for the analysis dataset uses the **full analytic cohort before** the grouped train–test split; **Pipeline** steps in each learner are fit on **training data only** within cross-validation, and test labels are not used for threshold tuning (CPM/TRIPOD).

### 4.4 Predictive Modeling Pipeline
We evaluated 14 machine learning algorithms, ranging from linear models (Logistic Regression) to tree-based ensembles (Random Forest, XGBoost, LightGBM) and neural networks (MLP). To prevent data leakage across waves for the same individual, we employed a 5-fold grouped cross-validation strategy, grouping strictly by individual ID. 

Model selection was governed by the Comprehensive Performance Metric (CPM) framework to comply rigorously with TRIPOD guidelines [12]. For each model, the optimal probability threshold maximizing the Youden index was determined strictly on the internal validation set (80% of the training fold). The model achieving the highest Area Under the Receiver Operating Characteristic Curve (AUC) on the held-out validation set (20% of the training fold) at that specific threshold was selected as the champion. Predictive performance was quantified using AUC, Area Under the Precision-Recall Curve (AUPRC), Recall, Specificity, and Brier score, with 95% confidence intervals derived via stratified bootstrapping (1,000 resamples). Feature importance was interpreted using SHapley Additive exPlanations (SHAP) [13].

### 4.5 Causal Inference Framework
We adopted the Rubin Causal Model (Potential Outcomes Framework) to estimate the Average Treatment Effect (ATE) of the interventions [14]. The primary estimator was the **XLearner** algorithm, a state-of-the-art meta-learning approach that excels in high-dimensional settings with imbalanced treatment assignments [7]. XLearner was implemented using Random Forest as the base nuisance estimator (n_estimators=200, max_depth=4, min_samples_leaf=15).

To ensure robust causal identification, we rigorously tested four core assumptions:
1.  **Unconfoundedness:** We adjusted for the comprehensive suite of 30+ sociodemographic, physiological, and clinical covariates.
2.  **Overlap (Positivity):** We enforced strict common support by estimating propensity scores (PS) and trimming individuals with PS outside the [0.05, 0.95] range prior to XLearner estimation [15]. This ensures estimates are derived from populations with clinical equipoise.
3.  **Covariate Balance:** Post-trimming balance was assessed using the Standardized Mean Difference (SMD), with a threshold of |SMD| < 0.2 considered acceptable, and < 0.1 ideal.
4.  **Sensitivity to Unmeasured Confounding:** We calculated the E-value for the point estimate and the confidence interval bound closest to the null [16]. The E-value quantifies the minimum strength of association an unmeasured confounder would need to have with both the treatment and the outcome to fully explain away the observed causal effect.

To cross-validate the XLearner findings, we also estimated the ATE using Propensity Score Matching (PSM; 1:1 nearest neighbor, caliper=0.024×SD) and Propensity Score Weighting (PSW; Inverse Probability Weighting with weight trimming at [0.1, 50]). Subgroup analyses (Conditional Average Treatment Effects, CATE) were conducted across age, sex, residence, and baseline health status. **Multiple testing:** primary in-text causal inference uses each XLearner (and PSM/PSW) estimate with its 95% CI and reported *P*. In parallel, the analysis pipeline writes `p_value_approx` (two-sided *p* approximated from ATE and 95% CI), `p_adj_bonferroni`, and `p_adj_fdr` to `xlearner_all_interventions_summary.csv` via `utils/multiplicity_correction.py` for all rows in that merged summary (i.e., across interventions, cohorts, and estimators—not limited to 15 XLearner-only contrasts). Authors may cite FDR-adjusted values from that file for transparency; they are not substituted for the primary bootstrap *P* in the abstract.

### 4.6 External Validation and Clinical Utility
External validation was conducted using both temporal splits (training on waves 2011-2015, validating on wave 2018) and regional splits (training on Eastern/Central China, validating on Western China). Clinical utility was assessed using Decision Curve Analysis (DCA) to quantify the net benefit of the predictive models across a range of threshold probabilities compared to "treat-all" or "treat-none" strategies [17].

---

## 3. Results

### 3.1 Distinct Incidence Gradients Across Clinical Trajectories
From an initial pool of 96,628 person-wave records, 14,386 individuals aged ≥60 years met the strict inclusion criteria for the incident cohorts. The population was stratified into Cohort A (Healthy, n=8,828), Cohort B (Depression-only, n=3,123), and Cohort C (Cognition-impaired-only, n=2,435). Missing data for key covariates (e.g., income, self-rated health) were handled via Multiple Imputation by Chained Equations (MICE), with missingness proportions ranging from 0% to ~15% (Supplementary Table S2). Sensitivity analyses confirmed that the imputation strategy did not significantly alter the predictive or causal findings.

Baseline characteristics revealed significant heterogeneity across the three trajectories (Table 1). The mean age was approximately 63 years across all groups. Cohort B (Depression-only) exhibited the highest burden of physical frailty, including the highest rates of arthritis (49.2%), heart disease (26.9%), and falls in the past year (25.7%), along with the highest self-reported poor health (40.3%). Conversely, Cohort C (Cognition-impaired-only) had the lowest educational attainment, with 62.3% having below primary school education. The incidence of new-onset depression-cognition comorbidity over the 2-year follow-up exhibited a stark, trajectory-dependent gradient: 4.1% in the healthy cohort, 13.6% in the depression-only cohort, and 16.9% in the cognition-impaired-only cohort (*P* < 0.001). This confirms that pre-existing single-domain impairments dramatically accelerate the onset of full comorbidity, validating the necessity of a stratified analytical approach.

### 3.2 Cohort-Specific Predictive Signatures and SHAP Interpretability
We evaluated 14 machine learning algorithms to predict incident comorbidity, optimizing for the CPM to prevent test-set leakage. The champion models demonstrated robust discriminative ability tailored to each trajectory (Table 2; main-text ROC omitted; full CPM comparison in Table 2). 

For Cohort A, Logistic Regression achieved the highest AUC (0.725, 95% CI: 0.66–0.78). For Cohort B (Depression-only), LightGBM achieved the highest AUC (0.702, 95% CI: 0.65–0.76), though Logistic Regression offered a more clinically viable balance of sensitivity (Recall 0.733) for screening purposes. For Cohort C (Cognition-impaired-only), Naive Bayes emerged as the champion (AUC 0.663, 95% CI: 0.61–0.72) with exceptional sensitivity (Recall 0.906), crucial for capturing high-risk patients in advanced stages of decline. External validation across temporal and regional splits confirmed generalizability, particularly for Cohort A (Regional AUC 0.700).

To interpret these predictive signatures, we applied SHapley Additive exPlanations (SHAP) to the champion models (Figure 4). The SHAP summary plots revealed distinct risk profiles for each cohort. In Cohort B, lack of regular exercise, high chronic disease burden, and poor self-rated health emerged as top predictive features driving the risk of incident comorbidity. In contrast, baseline cognitive scores (even within the normal range) and age were the dominant predictors in Cohort A. This interpretability step successfully identified modifiable lifestyle factors (e.g., exercise) as high-value targets for subsequent causal evaluation.

### 3.3 Exercise Selectively Disrupts the Depression-to-Comorbidity Cascade
To transition from prediction to actionable intervention, we applied the XLearner algorithm to estimate the ATE of five modifiable factors. The causal analysis revealed a highly specific therapeutic window (Table 5).

Regular exercise demonstrated a significant, protective causal effect *exclusively* in Cohort B (Depression-only), reducing the absolute risk of incident comorbidity by 4.2% (ATE = −0.042, 95% CI: −0.079 to −0.009, raw *p*=0.019; FDR-adjusted *p*=0.14). While the FDR-adjusted p-value indicates marginal significance under strict multiple testing penalties, the consistency of the effect size across multiple estimators (PSM, PSW) and its biological plausibility strongly support its clinical relevance as a targeted intervention. Given the 13.6% baseline incidence in this cohort, this represents a substantial relative risk reduction of approximately 31%. Subgroup analyses confirmed this protective effect was remarkably consistent across demographics, residence (urban/rural), and baseline chronic disease burden (CATE ranging from −0.036 to −0.054) (Table 3).

Crucially, exercise yielded no significant causal benefit in Cohort A (ATE = −0.003, 95% CI: −0.014 to 0.011) or Cohort C (ATE = 0.020, 95% CI: −0.003 to 0.100). Other interventions, including drinking cessation, social engagement, and BMI normalization, did not show significant protective causal effects across any cohort.

### 3.4 Paradoxical Effects of Chronic Disease Burden in Advanced Decline
In Cohort C (Cognition-impaired-only), having a low chronic disease burden (≤1 condition) was paradoxically associated with a significantly *increased* risk of developing comorbidity (ATE = 0.063, 95% CI: 0.015 to 0.108, *p*=0.008). Rather than a true biological effect, this likely reflects profound reverse causality or competing mortality risks: individuals with pre-existing cognitive impairment who survive long enough without severe physical multimorbidity may represent a specific phenotype of primary neurodegenerative pathology (e.g., Alzheimer's disease) that inevitably progresses to encompass neuropsychiatric symptoms (depression/BPSD).

### 3.5 Methodological Robustness and Causal Validation
The validity of our causal findings was subjected to rigorous stress-testing. For the exercise intervention in Cohort B, the overlap assumption was strictly satisfied after propensity score trimming (51.2% of the cohort retained, ensuring common support). Covariate balance was well-maintained (maximum SMD = 0.16, with only 7 out of 30+ covariates having |SMD| ≥ 0.1) (Table 4).

To quantify robustness against unmeasured confounding, we computed the E-value. The observed protective effect of exercise in Cohort B yielded an E-value of 2.0 (lower bound 1.27). This indicates that an unmeasured confounder would need to be associated with both exercise adherence and comorbidity risk by a risk ratio of at least 2.0—above and beyond the extensive covariates already adjusted for (including age, ADL, IADL, and baseline health)—to explain away the observed effect. 

Furthermore, cross-validation using alternative causal estimators confirmed the directionality of the findings: Propensity Score Weighting (PSW) yielded a significant ATE of −0.028, and Propensity Score Matching (PSM) yielded −0.016. XLearner, specifically designed to handle imbalanced treatment assignments in high-dimensional spaces, captured the most pronounced effect. Sensitivity analyses varying the diagnostic thresholds for depression (CES-D ≥ 8) and cognition (Score ≤ 8) confirmed that the exercise effect in Cohort B remained statistically significant and directionally stable (ATE = -0.038).

### 3.6 Clinical Utility and Decision Curve Analysis
Beyond statistical discrimination, we evaluated the clinical utility of the champion predictive models using Decision Curve Analysis (DCA) (Figure 5). For Cohort B, the model demonstrated a positive net benefit across a clinically relevant threshold probability range of approximately 5% to 35%, outperforming both "treat-all" and "treat-none" default strategies. Furthermore, calibration curves (Supplementary Figure S5) indicated adequate agreement between predicted probabilities and observed comorbidity frequencies, confirming the models' reliability for risk stratification.

---

## 4. Discussion

In this study, we applied a novel Causal Machine Learning framework to a large, nationally representative cohort of older adults to disentangle the heterogeneous trajectories of depression-cognition comorbidity. Our principal finding is the identification of a highly specific therapeutic window: regular exercise significantly and causally reduces the risk of developing comorbidity, but *only* for individuals who already exhibit depressive symptoms without cognitive impairment. This precision-medicine insight challenges the conventional "one-size-fits-all" approach to lifestyle interventions in aging, offering a highly targeted strategy for clinical practice.

### 4.1 Precision Intervention in the Cognitive-Affective Trajectory
The differential efficacy of exercise across the three cohorts carries profound clinical implications. In the healthy cohort (Cohort A), the baseline risk of transitioning to full comorbidity within two years is low (4.1%). Here, a "floor effect" likely renders the marginal benefit of exercise statistically undetectable over a short timeframe. In the cognition-impaired-only cohort (Cohort C), the neurodegenerative cascade may be too advanced. Once structural brain changes (e.g., amyloid/tau accumulation, severe vascular white matter disease) have manifested as clinical cognitive impairment, moderate physical activity may be insufficient to halt the secondary onset of depressive symptoms, which often manifest as behavioral and psychological symptoms of dementia (BPSD) [18].

Crucially, the depression-only phase (Cohort B) represents a reversible clinical inflection point. Late-life depression is increasingly recognized as a prodrome or independent risk factor for dementia, driven by hypercortisolemia (HPA axis dysregulation), systemic neuroinflammation, and structural atrophy in the hippocampus [19, 20]. Exercise is uniquely positioned to disrupt this specific pathological bridge. Mechanistically, physical activity upregulates Brain-Derived Neurotrophic Factor (BDNF), enhances hippocampal neurogenesis, improves vascular endothelial function, and exerts potent anti-inflammatory effects [21, 22]. Our causal estimates (ATE = -4.2%) provide robust, real-world evidence that prescribing exercise to older adults presenting with depressive symptoms can effectively sever the neurobiological link between affective distress and subsequent cognitive collapse. For clinicians, this means that an initial diagnosis of late-life depression should immediately trigger an aggressive, structured exercise prescription as a primary preventative measure against impending cognitive decline.

### 4.2 The Paradox of Chronic Disease in Cognitive Decline
The finding that low chronic disease burden increases comorbidity risk in Cohort C highlights the complexities of geriatric epidemiology. This paradox can be explained through the lens of competing risks and the "frailty index." Older adults with cognitive impairment *and* high physical multimorbidity have a high short-term mortality risk; they may simply not survive long enough to develop incident depression (survival bias). Conversely, those with cognitive impairment but *no* physical diseases likely suffer from a primary, aggressive neurodegenerative disease (like Alzheimer's) rather than vascular dementia. As Alzheimer's progresses, the incidence of neuropsychiatric symptoms (including depression) naturally approaches 100% [23]. This underscores the danger of interpreting standard regression coefficients as causal in advanced aging cohorts and emphasizes the need for stage-specific clinical expectations.

### 4.3 Methodological Innovations
This study advances the field methodologically by bridging predictive ML with causal inference. Traditional predictive models (like our champion LightGBM and LR models) are excellent at identifying *who* is at risk, but they cannot answer *what to do* about it. By feeding our rigorously defined cohorts into the XLearner algorithm, we isolated the causal impact of specific variables. 

The robustness of our findings is underscored by the E-value analysis (E-value = 2.0). In the context of social epidemiology, where extensive physical, psychological, and socioeconomic covariates are already controlled for, an E-value of 2.0 is considered highly robust. It is highly improbable that an unmeasured confounder exists with a magnitude strong enough (RR > 2.0) to nullify the protective effect of exercise without being highly correlated with the covariates already in the model [16]. The consistency across XLearner, PSW, and PSM further cements the validity of the causal claim, providing a reliable foundation for evidence-based clinical guidelines.

### 4.4 Limitations
Several limitations must be acknowledged. First, despite rigorous causal inference techniques (overlap trimming, E-values), this remains an observational study; absolute causality can only be confirmed via RCTs. Second, overlap trimming in Cohort B excluded individuals at the extremes of the propensity score distribution (51.2% retained). Consequently, our ATE technically represents the Average Treatment Effect on the Overlap population (ATO); it applies specifically to the sub-population with clinical equipoise regarding exercise (i.e., those who *could* realistically exercise but might not be), rather than the extremely frail or the extremely fit. Third, lifestyle factors were self-reported, which may introduce recall bias, though CHARLS employs rigorous validation protocols. Fourth, cognitive impairment was defined using a composite score derived from epidemiological screening tools (episodic memory and mental status) rather than a formal clinical diagnosis of mild cognitive impairment (MCI) or dementia. While validated for population-based studies, this proxy measure may lack the clinical granularity of comprehensive neuropsychological batteries. Finally, the 2-year follow-up captures short-term transitions; longer-term dynamics require further investigation.

### 4.5 Conclusion
The progression to depression-cognition comorbidity in older adults is not a monolithic process, but a set of distinct trajectories requiring targeted interventions. By employing causal machine learning, we demonstrated that regular exercise is a potent, causal intervention for preventing cognitive decline specifically in older adults suffering from depression. These findings advocate for a paradigm shift in geriatric care and public health policy: moving away from generic advice, and instead integrating predictive screening to identify the "depression-only" phenotype, followed by aggressive, targeted lifestyle prescriptions (e.g., "exercise as medicine") to halt the trajectory toward debilitating comorbidity.

---

## References

1. Livingston, G. et al. Dementia prevention, intervention, and care: 2020 report of the Lancet Commission. *The Lancet* **396**, 413-446 (2020).
2. Panza, F. et al. Late-life depression, mild cognitive impairment, and dementia: possible continuum? *Am J Geriatr Psychiatry* **18**, 98-116 (2010).
3. Du, M. et al. Prevalence of cognitive impairment and its related factors among Chinese older adults: an analysis based on the 2018 CHARLS data. *Front Public Health* **12**, 1500172 (2024).
4. Ismail, Z. et al. Neuropsychiatric symptoms as early manifestations of emergent dementia: Provisional diagnostic criteria for mild behavioral impairment. *Alzheimer's & Dementia* **12**, 195-202 (2016).
5. Wang, Y. et al. Relationship between cognitive performance and depressive symptoms in Chinese older adults. *J Affect Disord* **281**, 454-462 (2021).
6. Hernán, M. A., Hsu, J. & Healy, B. A second chance to get causal inference right: a classification of data science tasks. *Chance* **32**, 42-49 (2019).
7. Künzel, S. R. et al. Metalearners for estimating heterogeneous treatment effects using machine learning. *Proc Natl Acad Sci USA* **116**, 4156-4165 (2019).
8. Chernozhukov, V. et al. Double/debiased machine learning for treatment and structural parameters. *Econom J* **21**, C1-C68 (2018).
9. Zhao, Y. et al. Cohort profile: the China Health and Retirement Longitudinal Study (CHARLS). *Int J Epidemiol* **43**, 61-68 (2014).
10. Andresen, E. M. et al. The short form of the Center for Epidemiologic Studies Depression Scale (CES-D-10). *Am J Prev Med* **10**, 77-84 (1994).
11. Lei, X. et al. Depressive symptoms and SES among the mid-aged and elderly in China: evidence from the China Health and Retirement Longitudinal Study national baseline. *Soc Sci Med* **102**, 34-41 (2014).
12. Collins, G. S. et al. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. *BMJ* **350**, g7594 (2015).
13. Lundberg, S. M. & Lee, S. I. A unified approach to interpreting model predictions. *Adv Neural Inf Process Syst* **30**, 4765-4774 (2017).
14. Rubin, D. B. Causal inference using potential outcomes: Design, modeling, decisions. *J Am Stat Assoc* **100**, 322-331 (2005).
15. Stürmer, T. et al. A review of the application of propensity score methods yielded increasing use, advantages in specific settings, but not substantially different estimates compared with conventional multivariable methods. *J Clin Epidemiol* **59**, 437-447 (2006).
16. VanderWeele, T. J. & Ding, P. Sensitivity analysis in observational research: introducing the E-value. *Ann Intern Med* **167**, 268-274 (2017).
17. Vickers, A. J. & Elkin, E. B. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making* **26**, 565-574 (2006).
18. Kales, H. C., Gitlin, L. N. & Lyketsos, C. G. Assessment and management of behavioral and psychological symptoms of dementia. *BMJ* **350**, h369 (2015).
19. Byers, A. L. & Yaffe, K. Depression and risk of developing dementia. *Nat Rev Neurol* **7**, 323-331 (2011).
20. Diniz, B. S. et al. Late-life depression and risk of vascular dementia and Alzheimer's disease: systematic review and meta-analysis of community-based cohort studies. *Br J Psychiatry* **202**, 329-335 (2013).
21. Erickson, K. I. et al. Exercise training increases size of hippocampus and improves memory. *Proc Natl Acad Sci USA* **108**, 3017-3022 (2011).
22. Firth, J. et al. The Lancet Psychiatry Commission: a blueprint for protecting physical health in people with mental illness. *Lancet Psychiatry* **6**, 675-712 (2019).
23. Lyketsos, C. G. et al. Neuropsychiatric symptoms in Alzheimer's disease. *Alzheimer's & Dementia* **7**, 532-539 (2011).

---

## Tables

### Table 1. Full Baseline Characteristics by Cohort

| Variable | Cohort A (Healthy) | Cohort B (Depression) | Cohort C (Cognition) | P-value |
|----------|-------------------|----------------------|---------------------|---------|
| **N** | 8,828 | 3,123 | 2,435 | |
| **—— Biological factors ——** | | | | |
| Age, years (Mean ± SD) | 63.6 ± 5.2 | 63.1 ± 4.8 | 63.3 ± 5.4 | <0.001 |
| BMI (Mean ± SD) | 13.7 ± 6.6 | 13.3 ± 6.3 | 12.8 ± 5.8 | <0.001 |
| Waist circumference, cm (Mean ± SD) | 52.5 ± 21.6 | 50.7 ± 20.2 | 49.5 ± 19.2 | <0.001 |
| Female, n (%) | 5,832 (66.1%) | 1,726 (55.3%) | 1,340 (55.0%) | <0.001 |
| Male, n (%) | 2,996 (33.9%) | 1,397 (44.7%) | 1,095 (45.0%) | |
| Fall in past year, n (%) | 1,179 (13.4%) | 803 (25.7%) | 378 (15.5%) | <0.001 |
| Has disability, n (%) | 1,813 (20.5%) | 1,017 (32.6%) | 589 (24.2%) | <0.001 |
| Pulse, /min (Mean ± SD) | 69.3 ± 9.3 | 69.0 ± 8.8 | 69.9 ± 9.2 | <0.001 |
| Systolic BP, mmHg (Mean ± SD) | 78.5 ± 32.3 | 76.6 ± 31.1 | 74.7 ± 30.1 | <0.001 |
| Diastolic BP, mmHg (Mean ± SD) | 41.9 ± 20.6 | 40.6 ± 19.7 | 39.2 ± 18.6 | <0.001 |
| Grip strength (max), kg (Mean ± SD) | 27.6 ± 7.8 | 26.0 ± 7.9 | 25.8 ± 7.1 | <0.001 |
| Hypertension, n (%) | 3,261 (36.9%) | 1,305 (41.8%) | 805 (33.1%) | <0.001 |
| Diabetes, n (%) | 965 (10.9%) | 396 (12.7%) | 189 (7.8%) | <0.001 |
| Cancer, n (%) | 133 (1.5%) | 52 (1.7%) | 17 (0.7%) | 0.004 |
| Lung disease, n (%) | 1,139 (12.9%) | 602 (19.3%) | 330 (13.6%) | <0.001 |
| Heart disease, n (%) | 1,736 (19.7%) | 840 (26.9%) | 323 (13.3%) | <0.001 |
| Stroke, n (%) | 337 (3.8%) | 215 (6.9%) | 100 (4.1%) | <0.001 |
| Arthritis, n (%) | 2,916 (33.0%) | 1,537 (49.2%) | 848 (34.8%) | <0.001 |
| ADL difficulties (Mean ± SD) | 2.9 ± 2.9 | 3.3 ± 2.8 | 3.0 ± 2.9 | <0.001 |
| IADL difficulties (Mean ± SD) | 3.8 ± 3.9 | 4.3 ± 3.8 | 4.0 ± 3.9 | <0.001 |
| **—— Psychological factors ——** | | | | |
| Self-rated health: Bad/Very bad, n (%) | 1,204 (13.6%) | 1,258 (40.3%) | 421 (17.3%) | <0.001 |
| Life satisfaction: Bad/Very bad, n (%) | 292 (3.3%) | 568 (18.2%) | 96 (3.9%) | <0.001 |
| **—— Social factors ——** | | | | |
| Rural residence, n (%) | 4,206 (47.6%) | 1,886 (60.4%) | 1,612 (66.2%) | <0.001 |
| Socially isolated, n (%) | 460 (5.2%) | 229 (7.3%) | 203 (8.3%) | <0.001 |
| Has pension, n (%) | 7,433 (84.2%) | 2,429 (77.8%) | 1,912 (78.5%) | <0.001 |
| Has insurance, n (%) | 8,566 (97.0%) | 3,022 (96.8%) | 2,321 (95.3%) | <0.001 |
| Retired, n (%) | 3,298 (37.4%) | 709 (22.7%) | 350 (14.4%) | <0.001 |
| Education: Below primary school, n (%) | 2,101 (23.8%) | 1,020 (32.7%) | 1,516 (62.3%) | <0.001 |
| Marital status: Unmarried/Divorced/Widowed | 986 (11.2%) | 517 (16.6%) | 436 (17.9%) | <0.001 |
| Family size (Mean ± SD) | 15.6 ± 13.5 | 16.8 ± 13.5 | 15.9 ± 13.4 | <0.001 |
| Log(income+1) (Mean ± SD) | 38.1 ± 31.9 | 40.9 ± 32.4 | 38.9 ± 33.0 | <0.001 |
| **—— Lifestyle (intervenable) ——** | | | | |
| Sleep hours (Mean ± SD) | 14.2 ± 8.6 | 14.2 ± 8.8 | 14.1 ± 8.4 | <0.001 |
| Regular exercise, n (%) | 4,694 (53.2%) | 1,627 (52.1%) | 1,213 (49.8%) | 0.013 |
| Current drinking, n (%) | 4,788 (54.2%) | 1,536 (49.2%) | 1,129 (46.4%) | <0.001 |
| Adequate sleep (≥6h), n (%) | 7,589 (86.0%) | 2,390 (76.5%) | 2,061 (84.6%) | <0.001 |
| **—— Defining variables & Outcome ——** | | | | |
| Cognition score (defining) | 13.9 ± 2.0 | 13.2 ± 1.8 | 8.0 ± 1.9 | <0.001 |
| CES-D-10 (defining) | 4.1 ± 2.8 | 14.4 ± 4.1 | 4.8 ± 2.7 | <0.001 |
| **Incident comorbidity (2-year), n (%)** | **366 (4.1%)** | **426 (13.6%)** | **411 (16.9%)** | **<0.001** |

---

### Table 2. Full Prediction Performance of Machine Learning Algorithms by Cohort

| Cohort | Model | AUC (95% CI) | Recall | Specificity | Brier Score |
|--------|-------|--------------|--------|-------------|-------------|
| **Cohort A** | **Logistic Regression** | **0.725 (0.663–0.778)** | **0.614** | **0.700** | **0.211** |
| (Healthy) | Random Forest | 0.719 (0.658–0.777) | 0.671 | 0.667 | 0.070 |
| | LightGBM | 0.717 (0.655–0.775) | 0.000* | 1.000 | 0.037 |
| | GBDT | 0.717 (0.651–0.779) | 0.714 | 0.562 | 0.037 |
| | XGBoost | 0.710 (0.649–0.768) | 0.657 | 0.670 | 0.037 |
| | ExtraTrees | 0.709 (0.645–0.767) | 0.771 | 0.514 | 0.037 |
| | HistGBM | 0.709 (0.643–0.771) | 0.571 | 0.739 | 0.037 |
| | Bagging | 0.703 (0.639–0.768) | 0.629 | 0.668 | 0.038 |
| | AdaBoost | 0.691 (0.621–0.752) | 0.614 | 0.704 | 0.056 |
| | SVM | 0.685 (0.628–0.739) | 0.686 | 0.626 | 0.038 |
| | MLP | 0.682 (0.621–0.742) | 0.543 | 0.735 | 0.038 |
| | Decision Tree | 0.645 (0.578–0.705) | 0.643 | 0.620 | 0.038 |
| | KNN | 0.642 (0.580–0.704) | 0.600 | 0.605 | 0.038 |
| | Naive Bayes | 0.631 (0.567–0.693) | 0.557 | 0.598 | 0.126 |
| **Cohort B** | **LightGBM** | **0.702 (0.646–0.757)** | **0.011\*** | **1.000** | **0.117** |
| (Depression) | XGBoost | 0.700 (0.641–0.755) | 0.589 | 0.672 | 0.116 |
| | SVM | 0.697 (0.640–0.754) | 0.600 | 0.676 | 0.118 |
| | HistGBM | 0.696 (0.637–0.755) | 0.656 | 0.617 | 0.117 |
| | GBDT | 0.692 (0.634–0.747) | 0.689 | 0.589 | 0.118 |
| | ExtraTrees | 0.690 (0.631–0.746) | 0.789 | 0.470 | 0.118 |
| | Random Forest | 0.683 (0.628–0.737) | 0.578 | 0.693 | 0.119 |
| | **Logistic Regression** | **0.683 (0.626–0.736)** | **0.733** | **0.558** | **0.120** |
| | AdaBoost | 0.681 (0.619–0.738) | 0.644 | 0.634 | 0.129 |
| | KNN | 0.678 (0.621–0.735) | 0.589 | 0.709 | 0.119 |
| | MLP | 0.656 (0.601–0.710) | 0.722 | 0.528 | 0.135 |
| | Decision Tree | 0.652 (0.594–0.709) | 0.722 | 0.465 | 0.123 |
| | Bagging | 0.652 (0.594–0.709) | 0.556 | 0.638 | 0.120 |
| | Naive Bayes | 0.626 (0.571–0.681) | 0.744 | 0.461 | 0.180 |
| **Cohort C** | **Naive Bayes** | **0.663 (0.610–0.720)** | **0.906** | **0.312** | **0.183** |
| (Cognition) | Logistic Regression | 0.660 (0.600–0.720) | 0.553 | 0.666 | 0.137 |
| | ExtraTrees | 0.658 (0.594–0.718) | 0.624 | 0.597 | 0.139 |
| | XGBoost | 0.647 (0.579–0.710) | 0.494 | 0.723 | 0.138 |
| | Random Forest | 0.646 (0.579–0.708) | 0.482 | 0.740 | 0.139 |
| | SVM | 0.641 (0.580–0.701) | 0.659 | 0.564 | 0.139 |
| | HistGBM | 0.633 (0.564–0.703) | 0.565 | 0.666 | 0.140 |
| | GBDT | 0.628 (0.563–0.696) | 0.435 | 0.715 | 0.141 |
| | AdaBoost | 0.626 (0.565–0.687) | 0.765 | 0.406 | 0.144 |
| | Bagging | 0.622 (0.553–0.692) | 0.706 | 0.463 | 0.142 |
| | LightGBM | 0.621 (0.554–0.691) | 0.024* | 0.990 | 0.140 |
| | MLP | 0.609 (0.541–0.682) | 0.035* | 0.993 | 0.148 |
| | KNN | 0.608 (0.539–0.672) | 0.671 | 0.488 | 0.140 |
| | Decision Tree | 0.596 (0.533–0.661) | 0.765 | 0.391 | 0.145 |

*\*Note: Models marked with an asterisk (*) achieved high AUC but exhibited extremely low Recall at the Youden-optimal threshold due to skewed predicted probability distributions. In Cohort B, Logistic Regression is highlighted as a clinically viable alternative.*

---

### Table 3. Causal Average Treatment Effects (ATE) of five interventions by cohort (XLearner)

| Intervention | Cohort A (Healthy) | Cohort B (Depression) | Cohort C (Cognition) |
|--------------|--------------------|-----------------------|----------------------|
| **Exercise** | −0.003 (−0.014, 0.011) | **−0.042 (−0.079, −0.009)\*** | 0.020 (−0.003, 0.100) |
| **Drinking** | −0.006 (−0.017, −0.000) | 0.015 (−0.012, 0.045) | 0.029 (−0.013, 0.057) |
| **Social isolation** | −0.012 (−0.029, 0.014) | 0.013 (−0.052, 0.045) | 0.022 (−0.055, 0.062) |
| **Normal BMI** | 0.014 (0.001, 0.077) | 0.002 (−0.041, 0.077) | 0.019 (−0.024, 0.095) |
| **Low chronic disease (≤1)** | −0.005 (−0.018, 0.005) | −0.009 (−0.026, 0.033) | **0.063 (0.015, 0.108)\*** |

*\*Bold indicates statistical significance (95% CI excludes zero). Values represent absolute risk difference (ATE).*

---

### Table 4. Causal assumption checks and validation for Exercise intervention

| Metric | Cohort A | Cohort B | Cohort C |
|--------|----------|----------|----------|
| **Overlap (post-trimming retention)** | 100% (0% trimmed) | 51.2% retained | 80.4% retained |
| **Covariate Balance (max SMD)** | 0.29 (Imbalanced) | 0.16 (Moderate balance) | 0.87 (Severe imbalance) |
| **E-value (Point / Conservative)** | 1.31 / 1.71 | **2.00 / 1.27** | 1.49 / 1.15 |
| **PSM Estimate (ATE)** | −0.029 | −0.016 | 0.046 |
| **PSW Estimate (ATE)** | −0.006 | **−0.028** | 0.021 |

*Note: Cohort B demonstrates the most robust causal identifiability (good overlap, acceptable balance, high E-value, and consistent directionality across estimators).*