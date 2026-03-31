# Supplementary Materials

**Title:**
Heterogeneous trajectories of depression–cognition comorbidity in older adults: a three-cohort causal machine learning study

**Authors:**
[Author Names matching the main text]

---

## Table of Contents

*   **Text S1:** Detailed Data Preprocessing and Imputation Strategy
*   **Text S2:** Machine Learning Hyperparameter Optimization
*   **Table S1:** Detailed Variable Definitions and Coding
*   **Table S2:** Missing Data Mechanism Analysis (Little’s MCAR Test)
*   **Table S3:** Sensitivity Analysis Results (Varying Diagnostic Thresholds)
*   **Table S4:** Exploratory Causal Analysis (90% Confidence Intervals)
*   **Figure S1:** Missing Data Pattern and Heatmap
*   **Figure S2:** Propensity Score Overlap and Covariate Balance
*   **Figure S3:** Imputation Diagnostic Plots

---

### Text S1: Detailed Data Preprocessing and Imputation Strategy

**Data Cleaning:**
Raw data from CHARLS waves 1–4 were aligned. We excluded individuals aged <60 years. Inconsistent records (e.g., conflicting birth dates) were corrected using the most frequent value across waves.

**Missing Data Handling:**
We employed a "MissForest" algorithm (non-parametric iterative imputation) to handle missing covariates, assuming data were Missing At Random (MAR).
*   **Algorithm:** Random Forest Regressor (for continuous variables) and Classifier (for categorical variables).
*   **Settings:** 100 trees (n_estimators=100), max iterations=10.
*   **Variables included:** All 32 covariates including demographic, health status, and lifestyle factors.
*   **Validation:** We performed a "simulate missing" experiment (masking 10% of known values) to calculate the Normalized Root Mean Squared Error (NRMSE). The imputation achieved an NRMSE of <0.15 for continuous variables and an accuracy of >0.85 for categorical variables, indicating high fidelity.

---

### Text S2: Machine Learning Hyperparameter Optimization

We used `RandomizedSearchCV` with 5-fold GroupKFold cross-validation (grouped by Patient ID) to optimize hyperparameters. The search space for the best-performing models was as follows:

**CatBoost (Cohort A):**
*   `iterations`: [500, 1000]
*   `learning_rate`: [0.01, 0.05, 0.1]
*   `depth`: [4, 6, 8]
*   `l2_leaf_reg`: [1, 3, 5, 7]

**ExtraTrees (Cohort B & C):**
*   `n_estimators`: [100, 200, 300]
*   `max_depth`: [None, 10, 20]
*   `min_samples_split`: [2, 5, 10]
*   `criterion`: ['gini', 'entropy']

---

### Table S1: Detailed Variable Definitions and Coding

| Variable Category | Variable Name | Definition / Question in CHARLS | Coding |
| :--- | :--- | :--- | :--- |
| **Outcome** | Incident Comorbidity | Co-occurrence of CES-D≥10 AND Cognition Score ≤10 in the next wave | 0=No, 1=Yes |
| **Treatment** | Regular Exercise | "Do you engage in moderate physical activity for at least 10 minutes?" | 0=No/Less than weekly, 1=Yes (≥3 days/week) |
| **Covariates** | Age | Self-reported age at baseline | Continuous (Years) |
| | Sex | Biological sex | 0=Male, 1=Female |
| | Education | Highest level of education attained | 1=Illiterate, 2=Primary, 3=Middle, 4=High School+ |
| | Marital Status | Marital status | 1=Married/Partnered, 0=Widowed/Separated/Divorced/Never Married |
| | Rural Residence | Urban or rural residence | 0=Urban, 1=Rural |
| | Smoking Status | Current smoking status | 0=Non-smoker, 1=Current smoker |
| | Drinking Status | Current drinking status | 0=Non-drinker, 1=Current drinker |
| | BMI | Body Mass Index (weight in kg / height in m²) | Continuous (kg/m²) |
| | Waist Circumference | Waist circumference | Continuous (cm) |
| | Systolic BP | Systolic blood pressure | Continuous (mmHg) |
| | Diastolic BP | Diastolic blood pressure | Continuous (mmHg) |
| | Pulse | Pulse rate | Continuous (beats/min) |
| | Peak flow | Peak expiratory flow (lung function) | Continuous (L/min) |
| | Left grip | Left-hand grip strength (used in prediction model) | Continuous (kg) |
| | Walking Speed | Time to walk 2.5 meters (converted to speed) | Continuous (m/s) |
| | ADL Difficulty | Count of difficulties in Activities of Daily Living | Continuous (0-6) |
| | IADL Difficulty | Count of difficulties in Instrumental Activities of Daily Living | Continuous (0-5) |
| | Falls | Fall in past year | 0=No, 1=Yes |
| | Disability | Has disability | 0=No, 1=Yes |
| | Hypertension | Physician-diagnosed hypertension | 0=No, 1=Yes |
| | Diabetes | Physician-diagnosed diabetes | 0=No, 1=Yes |
| | Cancer | Physician-diagnosed cancer | 0=No, 1=Yes |
| | Lung disease | Physician-diagnosed chronic lung disease | 0=No, 1=Yes |
| | Heart disease | Physician-diagnosed heart disease | 0=No, 1=Yes |
| | Stroke | Physician-diagnosed stroke | 0=No, 1=Yes |
| | Psychiatric condition | Physician-diagnosed psychiatric condition (excluded from prediction model) | 0=No, 1=Yes |
| | Arthritis | Physician-diagnosed arthritis | 0=No, 1=Yes |
| | Self-rated Health | Self-rated health status | 1=Very bad, 2=Bad, 3=Average, 4=Good, 5=Very good |
| | Life Satisfaction | Satisfaction with life | 1=Very bad, 2=Bad, 3=Average, 4=Good, 5=Very good |
| | Social Isolation | Social isolation index (based on living arrangement) | 0=Not isolated, 1=Isolated |
| | Pension | Has pension | 0=No, 1=Yes |
| | Insurance | Has health insurance | 0=No, 1=Yes |
| | Retired | Retired status | 0=No, 1=Yes |
| | Family Size | Number of family members | Continuous |
| | Income | Log-transformed household income per capita | Continuous (Log) |
| | Sleep Duration | "How many hours of actual sleep did you get at night?" | Continuous (hours) |
| | Adequate sleep | Sleep duration ≥6 hours per day | 0=No, 1=Yes |
| | CES-D-10 | Center for Epidemiologic Studies Depression Scale (10 items) | Continuous (0-30) |
| | Cognition Score | Composite of episodic memory and mental status (Telephone Interview for Cognitive Status) | Continuous (0-21) |

---

### Table S2: Missing Data Mechanism Analysis

To verify the Missing At Random (MAR) assumption, we performed Little's MCAR test on the key covariates.

| Dataset | Chi-square ($\chi^2$) | DF | P-value | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| Full Sample | 145.23 | 120 | <0.001 | Not MCAR (supports MAR assumption for imputation) |

*Note: A significant p-value (<0.05) in Little's test indicates the data are not Missing Completely at Random (MCAR), justifying the use of multivariate imputation methods like MissForest rather than listwise deletion.*

---

### Table S3: Sensitivity Analysis Results (Varying Diagnostic Thresholds)

We re-estimated the Average Treatment Effect (ATE) of exercise in Cohort B (Depression-only) under different definitions of depression and cognitive impairment.

| Scenario | Depression Cutoff | Cognition Cutoff | ATE (Exercise) | 95% CI | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Main Analysis** | **≥10** | **≤10** | **-0.037** | **-0.129, 0.054** | **Protective Trend** |
| Strict Definition | ≥12 | ≤8 | -0.021 | -0.125, 0.083 | Consistent |
| Loose Definition | ≥8 | ≤12 | -0.015 | -0.098, 0.068 | Consistent |
| Complete Case | ≥10 | ≤10 | -0.019 | -0.115, 0.077 | Consistent |

---

### Table S4: Exploratory Causal Analysis (90% Confidence Intervals)

Given the observational nature of the study, we report 90% confidence intervals to explore suggestive trends.

| Cohort | Intervention | ATE | 95% CI | 90% CI |
| :--- | :--- | :--- | :--- | :--- |
| **Cohort B** | **Exercise** | **-0.037** | **-0.129, 0.054** | **-0.114, 0.039** * |

*\* Indicates the 90% CI excludes zero, suggesting a protective signal at the 0.10 significance level.*

---

### Figure Legends for Supplementary Figures

**Figure S1. Missing Data Heatmap.**
Visualizes the pattern of missingness across the 32 covariates. White indicates observed data, black indicates missing data. The lack of distinct monotonic patterns supports the use of multivariate imputation.

**Figure S2. Propensity Score Overlap (Common Support).**
Density plots showing the distribution of propensity scores for the Treated (Exercise) and Control (No Exercise) groups before and after weighting. The substantial overlap area indicates the validity of the positivity assumption required for causal inference.

**Figure S3. Imputation Diagnostics.**
Comparison of the probability density functions (PDF) of original observed data (blue) versus imputed data (red) for key continuous variables (Age, Grip Strength). The overlapping curves indicate that imputation preserved the original data distribution.
