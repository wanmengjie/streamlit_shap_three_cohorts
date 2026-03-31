# GEMINI 用完整数据简报：CHARLS 预测 + 因果推断（XLearner / PS / PSM / PSW）

> **用途**：整份或分节上传给 Gemini，用于扩写/润色 Methods、Results、Discussion。  
> **数据快照**：与仓库 **`results/tables/`** 及投稿稿 **`PAPER_Manuscript_Submission_Ready.md`** 对齐；主文数字锁 **`2026-03-28`**。  
> **重要**：主分析 **XLearner 的 ATE/95% CI 以 `table4_ate_summary.csv` 为准**。`table7_psm_psw_dml.csv` 中 **XLearner 行的 CI 可能与 table4 不同**（导出逻辑/区间来源不一致），写作时 **勿混用**。

---

## 0. 目录

1. [分析单元与样本量](#1-分析单元与样本量)  
2. [结局与三队列定义](#2-结局与三队列定义)  
3. [描述性：发病率（Table 1b）](#3-描述性发病率table-1b)  
4. [预测模型（CPM）冠军与性能](#4-预测模型cpm冠军与性能)  
5. [因果分析数据入口与协变量逻辑](#5-因果分析数据入口与协变量逻辑)  
6. [方法参数一览（PS / 修剪 / XLearner / PSM / PSW）](#6-方法参数一览ps--修剪--xlearner--psm--psw)  
7. [主表：XLearner 全暴露×队列（table4）](#7-主表xlearner-全暴露队列table4)  
8. [方法比较：PSM / PSW / XLearner（table7）](#8-方法比较psm--psw--xlearnertable7)  
9. [阴性对照结局（negative_control_results）](#9-阴性对照结局negative_control_results)  
10. [Cohort B：XLearner vs CausalForestDML](#10-cohort-bxlearner-vs-causalforestdml)  
11. [诊断阈值与完整病例敏感性（table5 摘要）](#11-诊断阈值与完整病例敏感性table5-摘要)  
12. [外部验证（table6）](#12-外部验证table6)  
13. [亚组 CATE：Cohort B 运动（table3）](#13-亚组-catecohort-b-运动table3)  
14. [假设检验叙事：重叠 / SMD / E-value（运动，稿中数字）](#14-假设检验叙事重叠--smd--e-value运动稿中数字)  
15. [权威文件路径清单](#15-权威文件路径清单)  
16. [给 Gemini 的写作任务](#16-给-gemini-的写作任务)

---

## 1. 分析单元与样本量

| 项目 | 数值 / 说明 |
|------|-------------|
| **分析行** | **Person-wave**：每名参与者在每一调查波一行；预测因子在波 *t*，结局在 *t*+1 |
| **总分析 person-waves** | **14,386** |
| **约 unique participants** | **~7,027**（稿中摘要） |
| **Cohort A person-waves** | **8,828** |
| **Cohort B person-waves** | **3,123** |
| **Cohort C person-waves** | **2,435** |
| **纳排链（稿 §3.1）** | 96,628 → 49,015（age≥60）→ 43,048（CES-D 非缺失）→ 31,574（认知非缺失）→ 16,983（下一波共病可定义）→ **14,386** |

---

## 2. 结局与三队列定义

| 项目 | 定义 |
|------|------|
| **主结局 Y** | `is_comorbidity_next`：下一波是否发生 **抑郁与认知障碍同时存在**（抑郁–认知共病） |
| **抑郁（基线/划分）** | CES-D-10 **≥10** |
| **认知受损** | 认知总分 **≤10** |
| **Cohort A** | 基线 **无**抑郁且 **无**认知受损 |
| **Cohort B** | 基线 **有**抑郁、认知 **正常** |
| **Cohort C** | 基线 **有**认知受损、**无**抑郁 |

---

## 3. 描述性：发病率（Table 1b）

**文件**：`results/tables/table1b_incidence_density.csv`（与稿中 Table 1b 一致；人年为中点法，见稿 §2.2）

| Baseline cohort | Person-wave obs | Person-years | Incident cases | Rate / 1000 PY |
|-----------------|-----------------|-------------|------------------|----------------|
| Cohort A | 8828 | 15020.0 | 366 | 24.37 |
| Cohort B | 3123 | 5046.0 | 426 | 84.42 |
| Cohort C | 2435 | 3837.0 | 411 | 107.11 |
| **Total** | **14386** | **23903.0** | **1203** | **50.33** |

**粗发病率（稿 §3.2）**：A **4.1%**，B **13.6%**，C **16.9%**（person-wave 上 χ²，P<0.001）。

---

## 4. 预测模型（CPM）冠军与性能

**规则**：在 **Youden 最优阈值** 下 **Recall ≥ 0.05** 的模型中，取 **测试集 AUC 最高** 者为冠军；阈值仅在 **训练池** 上确定（时间外推：wave < max 训练，max wave 测试）。

**文件**：`results/tables/table2_prediction_combined_ABC.csv` 及各队列 `Cohort_*/01_prediction/table2_*_main_performance.csv`

| Cohort | Champion | Test AUC | AUC 95% CI（约） | Recall | Specificity | Brier |
|--------|----------|----------|------------------|--------|-------------|-------|
| A | **XGB** | 0.7244 | 0.6699–0.7799 | 0.571 | 0.693 | 0.0239 |
| B | **LR** | 0.6425 | 0.5788–0.7050 | 0.547 | 0.670 | 0.0873 |
| C | **HistGBM** | 0.6450 | 0.5854–0.7044 | 0.554 | 0.641 | 0.1203 |

**说明**：预测用 **Pipeline 内 IterativeImputer**；**与因果用的 `step1_imputed_full` 不是同一条数据链**。

---

## 5. 因果分析数据入口与协变量逻辑

| 项目 | 内容 |
|------|------|
| **因果主数据** | 单次完成集 **`step1_imputed_full`**（外置 MICE 流程） |
| **主推断** | **不**采用 Rubin 合并 `table*_rubin_pooled_*` 作为主文数字 |
| **Y** | `is_comorbidity_next` |
| **T（每次跑一个）** | exercise, drinkev, is_socially_isolated, bmi_normal, chronic_low |
| **协变量** | `get_exclude_cols(df, target_col, treatment_col)`：排除 ID、wave、结局、队列定义相关列、当前 treatment、衍生列等；**数值列**进入模型；预处理与假设检验模块用 **`fit_transform_numeric_train_only`**（与训练子集一致哲学） |
| **运动特殊项** | 当 **T = exercise** 且存在 **`adlab_c`** 时，构造 **`exercise_x_adl`**（运动×基线 ADL）以缓解选择偏倚（代码：`charls_recalculate_causal_impact.py` 中 XLearner / TLearner / DML 敏感性路径） |
| **二分类暴露缺失** | `fillna(0)` 叙事（与稿 §2.1 一致） |

---

## 6. 方法参数一览（PS / 修剪 / XLearner / PSM / PSW）

| 组件 | 设定（稿 + 代码） |
|------|-------------------|
| **PS 重叠修剪** | 倾向得分 **∉ [0.05, 0.95]** 的样本剔除后再估计（`config.PS_TRIM_LOW/HIGH` 可引用） |
| **XLearner** | `econml` XLearner；nuisance：**RandomForestRegressor / RandomForestClassifier**，**n_estimators=200, max_depth=4, min_samples_leaf=15** |
| **ATE / CI** | 优先 `ate_interval`；否则 **按 ID 的 cluster bootstrap**，**200** 次 |
| **PSM** | **1:1 最近邻**；**caliper = 0.024 × SD(PS)** |
| **PSW** | **IPW**；权重限制 **[0.1, 50]** |
| **假设诊断** | 重叠图、**SMD**（|SMD|<0.1 为常用参考）、**E-value**；Love plot → **Figure S3** |

---

## 7. 主表：XLearner 全暴露×队列（table4）

**权威文件**：`results/tables/table4_ate_summary.csv`  
**尺度**：`ate`, `ate_lb`, `ate_ub` 为 **风险差（比例差）**；×100 为百分点。  
**列说明**：`significant_95=1` 表示 **95% CI 不含 0**；`n` 为分析样本量；`p_value_approx` 为由 CI 反推的近似 *p*。

| exposure | label | cohort | ate | ate_lb | ate_ub | significant_95 | n | p_value_approx |
|----------|-------|--------|-----|--------|--------|----------------|---|----------------|
| exercise | Exercise | Cohort_A | -0.0014 | -0.0099 | 0.0070 | 0 | 8828 | 0.750 |
| exercise | Exercise | Cohort_B | -0.0212 | -0.0416 | 0.0034 | 0 | 3123 | 0.064 |
| exercise | Exercise | Cohort_C | 0.0347 | 0.0024 | 0.0586 | **1** | 2435 | 0.016 |
| drinkev | Drinking | Cohort_A | -0.0056 | -0.0145 | 0.0017 | 0 | 8828 | 0.177 |
| drinkev | Drinking | Cohort_B | 0.0205 | -0.0117 | 0.0474 | 0 | 3123 | 0.174 |
| drinkev | Drinking | Cohort_C | 0.0273 | -0.0070 | 0.0622 | 0 | 2435 | 0.122 |
| is_socially_isolated | Social isolation | Cohort_A | 0.0035 | -0.0175 | 0.0321 | 0 | 8828 | 0.782 |
| is_socially_isolated | Social isolation | Cohort_B | 0.0015 | -0.0459 | 0.0476 | 0 | 3123 | 0.950 |
| is_socially_isolated | Social isolation | Cohort_C | 0.0081 | -0.0492 | 0.0680 | 0 | 2435 | 0.787 |
| bmi_normal | Normal BMI (18.5-24) | Cohort_A | 0.0019 | -0.0065 | 0.0167 | 0 | 8828 | 0.748 |
| bmi_normal | Normal BMI (18.5-24) | Cohort_B | 0.0164 | -0.0090 | 0.0485 | 0 | 3123 | 0.264 |
| bmi_normal | Normal BMI (18.5-24) | Cohort_C | 0.0137 | -0.0206 | 0.0600 | 0 | 2435 | 0.506 |
| chronic_low | Low chronic disease burden (≤1) | Cohort_A | -0.0062 | -0.0193 | 0.0021 | 0 | 8828 | 0.260 |
| chronic_low | Low chronic disease burden (≤1) | Cohort_B | -0.0123 | -0.0298 | 0.0159 | 0 | 3123 | 0.293 |
| chronic_low | Low chronic disease burden (≤1) | Cohort_C | 0.0442 | 0.0034 | 0.0816 | **1** | 2435 | 0.027 |

**写作要点**：

- **仅 C 的 exercise 与 chronic_low** 在 table4 上 **significant_95=1**。  
- **B 的 exercise**：**边缘**（*p*≈0.064），**CI 含 0**。  
- **C 的 exercise**：统计显著但 **禁止写成「运动有害」的因果推荐**（混杂/反向因果/平衡差）。  
- **chronic_low in C**：显著但 **竞争风险/选择** 解释，**非**临床「应多得病」。

---

## 8. 方法比较：PSM / PSW / XLearner（table7）

**文件**：`results/tables/table7_psm_psw_dml.csv`  
**Consistency 列**：多方法方向/区间一致性标签（**Inconsistent** 表示与主 XLearner 或彼此在「显著性/方向」上不完全对齐）。

**再次强调**：此表中 **XLearner 的 95% CI 与 table4 可能不一致**（例如 B 的 exercise：table7 为约 (−0.08, 0.058)，table4 为 (−0.0416, 0.0034)）。**论文主文以 table4 的 XLearner 为准**；table7 用于 **PSM/PSW 与 triangulation**。

### 8.1 Exercise

| cohort | method | ate | ate_lb | ate_ub | significant_95 | Consistency |
|--------|--------|-----|--------|--------|----------------|-------------|
| A | PSM | -0.0124 | -0.0205 | -0.0043 | 1 | Inconsistent |
| A | PSW | -0.0069 | -0.0156 | 0.0017 | 0 | Inconsistent |
| A | XLearner | -0.0014 | -0.0397 | 0.0271 | 0 | Inconsistent |
| B | PSM | -0.0194 | -0.0423 | 0.0036 | 0 | Inconsistent |
| B | PSW | -0.0314 | -0.0560 | -0.0069 | **1** | Inconsistent |
| B | XLearner | -0.0212 | -0.0800 | 0.0582 | 0 | Inconsistent |
| C | PSM | 0.0368 | 0.0073 | 0.0662 | **1** | Inconsistent |
| C | PSW | 0.0217 | -0.0085 | 0.0520 | 0 | Inconsistent |
| C | XLearner | 0.0347 | -0.0396 | 0.1120 | 0 | Inconsistent |

**叙述用数字（与稿 §3.8 对齐，运动 B）**：PSM **−1.9%**（−4.2%~+0.4%），PSW **−3.1%**（−5.6%~−0.7%），XLearner（**table4**）**−2.1%**（−4.2%~+0.3%）。

### 8.2 Drinking（drinkev）

| cohort | method | ate | ate_lb | ate_ub | significant_95 | Consistency |
|--------|--------|-----|--------|--------|----------------|-------------|
| A | PSM | 0.0036 | -0.0037 | 0.0108 | 0 | Consistent |
| A | PSW | -0.0022 | -0.0110 | 0.0065 | 0 | Consistent |
| A | XLearner | -0.0056 | -0.0304 | 0.0440 | 0 | Consistent |
| B | PSM | 0.0216 | -0.0014 | 0.0446 | 0 | Consistent |
| B | PSW | 0.0175 | -0.0078 | 0.0427 | 0 | Consistent |
| B | XLearner | 0.0205 | -0.0401 | 0.0882 | 0 | Consistent |
| C | PSM | 0.0215 | -0.0089 | 0.0520 | 0 | Consistent |
| C | PSW | 0.0260 | -0.0050 | 0.0571 | 0 | Consistent |
| C | XLearner | 0.0273 | -0.0558 | 0.1514 | 0 | Consistent |

### 8.3 Social isolation

| cohort | method | ate | ate_lb | ate_ub | significant_95 | Consistency |
|--------|--------|-----|--------|--------|----------------|-------------|
| A | PSM | -0.0556 | -0.1100 | -0.0012 | **1** | Inconsistent |
| A | PSW | 0.0035 | -0.0012 | 0.0082 | 0 | Inconsistent |
| A | XLearner | 0.0035 | -0.0440 | 0.0609 | 0 | Inconsistent |
| B | PSM | -0.1250 | -0.2517 | 0.0017 | 0 | Consistent |
| B | PSW | 0.0078 | -0.0079 | 0.0235 | 0 | Consistent |
| B | XLearner | 0.0015 | -0.1008 | 0.0951 | 0 | Consistent |
| C | PSM | 0.0652 | -0.0445 | 0.1749 | 0 | Inconsistent |
| C | PSW | 0.0222 | 0.0014 | 0.0431 | **1** | Inconsistent |
| C | XLearner | 0.0081 | -0.1448 | 0.1519 | 0 | Inconsistent |

### 8.4 Normal BMI

| cohort | method | ate | ate_lb | ate_ub | significant_95 | Consistency |
|--------|--------|-----|--------|--------|----------------|-------------|
| A | PSM | 0.0018 | -0.0054 | 0.0090 | 0 | Consistent |
| A | PSW | -0.0202 | -0.0410 | 0.0006 | 0 | Consistent |
| A | XLearner | 0.0019 | -0.0416 | 0.0444 | 0 | Consistent |
| B | PSM | -0.0015 | -0.0222 | 0.0193 | 0 | Consistent |
| B | PSW | -0.0211 | -0.0580 | 0.0159 | 0 | Consistent |
| B | XLearner | 0.0164 | -0.1023 | 0.0935 | 0 | Consistent |
| C | PSM | -0.0327 | -0.0594 | -0.0060 | **1** | Inconsistent |
| C | PSW | -0.0087 | -0.0530 | 0.0357 | 0 | Inconsistent |
| C | XLearner | 0.0137 | -0.0720 | 0.0733 | 0 | Inconsistent |

### 8.5 Chronic low burden

| cohort | method | ate | ate_lb | ate_ub | significant_95 | Consistency |
|--------|--------|-----|--------|--------|----------------|-------------|
| A | PSM | 0.0374 | 0.0287 | 0.0461 | **1** | Inconsistent |
| A | PSW | -0.0099 | -0.0155 | -0.0044 | **1** | Inconsistent |
| A | XLearner | -0.0062 | -0.0567 | 0.0250 | 0 | Inconsistent |
| B | PSM | 0.0940 | 0.0615 | 0.1266 | **1** | Inconsistent |
| B | PSW | -0.0151 | -0.0324 | 0.0022 | 0 | Inconsistent |
| B | XLearner | -0.0123 | -0.0615 | 0.0770 | 0 | Inconsistent |
| C | PSM | 0.0128 | -0.0309 | 0.0564 | 0 | Inconsistent |
| C | PSW | 0.0265 | 0.0045 | 0.0484 | **1** | Inconsistent |
| C | XLearner | 0.0442 | -0.0962 | 0.1181 | 0 | Inconsistent |

**解读提示**：chronic_low 上 **PSM/PSW 与 XLearner 在 A/B/C 常标记 Inconsistent**——适合写 **「估计量与样本重新定义（匹配/加权）敏感」**，避免单一叙事。

---

## 9. 阴性对照结局（negative_control_results）

**文件**：`results/tables/negative_control_results.csv`  
**设定**：暴露仍为 **exercise**；结局 **`is_fall_next`**（下一波跌倒）；**同一 XLearner + 重叠框架**（与主分析一致）。

| cohort | n_after_trim | n_before_trim | ate | ate_lb | ate_ub | p_two_sided_approx | ate_ci_source |
|--------|--------------|---------------|-----|--------|--------|--------------------|---------------|
| A | 6532 | 6532 | 0.000794 | -0.02006 | 0.01390 | 0.927 | cluster_bootstrap_percentile |
| B | 2352 | 2352 | -0.02024 | -0.04177 | 0.01471 | 0.160 | cluster_bootstrap_percentile |
| C | 1856 | 1856 | 0.01757 | -0.01743 | 0.04635 | 0.280 | cluster_bootstrap_percentile |

**结论**：**无一** 95% CI 排除 0；支持「并非所有结局都被同一偏倚同等放大」，**属支持性而非确证性**证据。

---

## 10. Cohort B：XLearner vs CausalForestDML

**文件**：`results/tables/ate_method_sensitivity.csv`  
**样本**：`n_after_trim=3123`；结局 `is_comorbidity_next`；暴露 `exercise`；**与 XLearner 同源 PS 修剪**。

| method | ate | ate_lb | ate_ub | p_two_sided_approx | ate_ci_source |
|--------|-----|--------|--------|--------------------|---------------|
| XLearner | -0.021085 | -0.043471 | 0.002878 | 0.0745 | cluster_bootstrap_percentile |
| CausalForestDML | -0.031696 | -0.130031 | 0.066639 | 0.5275 | econml_ate_interval |

**叙述**：**方向一致**（均保护），DML **区间更宽、跨 0** → **估计量依赖精度**，非方向反转。

---

## 11. 诊断阈值与完整病例敏感性（table5 摘要）

**文件**：`results/tables/table5_sensitivity_summary.csv`（**全表约 150+ 行**；此处仅 **Main 场景** 五行暴露×三队列，供 Gemini 抓主模式）

**场景**：`Main (CES-D≥10, Cog≤10)`；注意 **n 与主分析 14,386 不同**（稿：约 **80% ID 子样本** 等，见 `sensitivity_analysis_readme.txt`）

| intervention | cohort | n | incidence | ate | ate_lb | ate_ub |
|----------------|--------|---|-----------|-----|--------|--------|
| exercise | A | 7093 | 0.0407 | -0.0004 | -0.0101 | 0.0082 |
| exercise | B | 2486 | 0.1388 | -0.0165 | -0.0404 | 0.0181 |
| exercise | C | 1942 | 0.1632 | 0.0323 | 0.0067 | 0.0626 |
| drinkev | A | 7093 | 0.0407 | -0.0030 | -0.0135 | 0.0046 |
| drinkev | B | 2486 | 0.1388 | 0.0082 | -0.0221 | 0.0394 |
| drinkev | C | 1942 | 0.1632 | 0.0495 | 0.0058 | 0.0837 |
| is_socially_isolated | A | 7093 | 0.0407 | -0.0005 | -0.0194 | 0.0241 |
| is_socially_isolated | B | 2486 | 0.1388 | -0.0081 | -0.0604 | 0.0386 |
| is_socially_isolated | C | 1942 | 0.1632 | 0.0074 | -0.0418 | 0.0705 |
| bmi_normal | A | 7093 | 0.0407 | 0.0012 | -0.0079 | 0.0174 |
| bmi_normal | B | 2486 | 0.1388 | 0.0120 | -0.0204 | 0.0428 |
| bmi_normal | C | 1942 | 0.1632 | 0.0147 | -0.0293 | 0.0611 |
| chronic_low | A | 7093 | 0.0407 | -0.0065 | -0.0182 | 0.0035 |
| chronic_low | B | 2486 | 0.1388 | -0.0082 | -0.0368 | 0.0198 |
| chronic_low | C | 1942 | 0.1632 | 0.0348 | -0.0005 | 0.0884 |

**其余场景**（同一 CSV）：CES-D≥8 / ≥12；Cog≤8 / ≤12；组合切分；**Complete-case** 按暴露分列。完整数值请直接解析该 CSV。

---

## 12. 外部验证（table6）

**文件**：`results/tables/table6_external_validation_cohortA.csv`（B、C 同理）  
**方法**：**冻结**各队列 CPM 冠军 `champion_model.joblib`，**时间**（wave=4）与 **区域**（西部位验证集）

### Cohort A

| Split | AUC | AUPRC | Brier |
|-------|-----|-------|-------|
| Temporal | 0.7244 | 0.0630 | 0.0239 |
| Regional (West) | 0.8216 | 0.2962 | 0.0373 |

### Cohort B

| Split | AUC | AUPRC | Brier |
|-------|-----|-------|-------|
| Temporal | 0.6425 | 0.1761 | 0.0873 |
| Regional (West) | 0.7116 | 0.2774 | 0.0981 |

### Cohort C

| Split | AUC | AUPRC | Brier |
|-------|-----|-------|-------|
| Temporal | 0.6450 | 0.2223 | 0.1203 |
| Regional (West) | 0.9479 | 0.8508 | 0.0860 |

**稿中提醒**：C **区域 AUC 极高** 但验证子集 **n 小**（表内 n_train/n_val 常为空，需自算）；**谨慎解读可迁移性**。

---

## 13. 亚组 CATE：Cohort B 运动（table3）

**文件**：`results/tables/table3_subgroup_cohortB.csv`  
**尺度**：CATE 为 **风险差**（与主因果一致方向理解）

| Subgroup | Value | CATE | Count | N_events | Sample_Size_Warning |
|----------|-------|------|-------|----------|---------------------|
| Residence | Urban | -0.01479 | 1237 | 134 | OK |
| Residence | Rural | -0.02540 | 1886 | 292 | OK |
| Age_Group | <65 | -0.02056 | 1677 | 192 | OK |
| Age_Group | 65-75 | -0.02416 | 1275 | 203 | OK |
| Age_Group | 75+ | -0.00531 | 171 | 31 | OK |
| Gender | Male | -0.01758 | 1726 | 198 | OK |
| Gender | Female | -0.02566 | 1397 | 228 | OK |
| Education | Edu_1 | -0.03270 | 1020 | 214 | OK |
| Education | Edu_2 | -0.02064 | 1064 | 137 | OK |
| Education | Edu_3 | -0.01403 | 695 | 52 | OK |
| Education | Edu_4 | -0.00326 | 344 | 23 | Caution: Underpowered |
| Chronic | 0 | -0.02080 | 663 | 77 | OK |
| Chronic | 1-2 | -0.02094 | 1772 | 235 | OK |
| Chronic | 3+ | -0.02223 | 688 | 114 | OK |
| SRH | SRH_1 | -0.01925 | 257 | 52 | OK |
| SRH | SRH_2 | -0.02077 | 1001 | 155 | OK |
| SRH | SRH_3 | -0.02183 | 1576 | 194 | OK |
| SRH | SRH_4 | -0.01905 | 194 | 17 | Caution: Underpowered |
| SRH | SRH_5 | -0.02476 | 95 | 8 | Caution: Underpowered |

**写作**：**探索性**；**未**做交互多重校正；低事件亚组标 **Caution**。

---

## 14. 假设检验叙事：重叠 / SMD / E-value（运动，稿中数字）

**来源**：稿 §3.5、Table 4；队列输出 `Cohort_*/03_causal/`  

| Cohort | Overlap（稿述） | max SMD | \|SMD\|≥0.1 协变量数 | E-value（point / conservative） |
|--------|----------------|---------|------------------------|----------------------------------|
| A | 修剪后 0% 越界 | ~0.20 | 6 | ~1.24 / ~1.43 |
| B | ~1.6% 预分析修剪；保留样本 0% 越界 | ~0.21 | 8 | ~1.59 / ~1.19 |
| C | ~0.1% 越界（稿述） | ~0.28 | 13 | ~1.7 / ~1.29 |

---

## 15. 权威文件路径清单

| 内容 | 路径 |
|------|------|
| XLearner 主汇总 | `results/tables/table4_ate_summary.csv` |
| PSM/PSW/宽表 | `results/tables/table7_psm_psw_dml.csv` |
| 阴性对照 | `results/tables/negative_control_results.csv` |
| B 队列 DML 敏感 | `results/tables/ate_method_sensitivity.csv` |
| 阈值敏感性 | `results/tables/table5_sensitivity_summary.csv` |
| 外部验证 | `results/tables/table6_external_validation_cohortA/B/C.csv` |
| 亚组 B | `results/tables/table3_subgroup_cohortB.csv` |
| 发病率 | `results/tables/table1b_incidence_density.csv` |
| CPM 合并 | `results/tables/table2_prediction_combined_ABC.csv` |
| 假设/重叠 | `Cohort_*_*/03_causal/assumption_*.txt`, `fig_propensity_overlap_*.png` |
| 投稿主文 | `PAPER_Manuscript_Submission_Ready.md` |

---

## 16. 给 Gemini 的写作任务

1. 生成 **英文 Methods**：因果部分按 **Estimands → Identification → Data (`step1_imputed_full`) → Covariates & exercise×ADL → PS overlap trimming → XLearner → Bootstrap → PSM → PSW → Diagnostics → Sensitivities** 顺序。  
2. 明确 **table4 为 XLearner 主表**；**table7 的 XLearner 行不得与 table4 数字混用**；PSM/PSW 用 table7。  
3. 写 **Results**：五暴露×三队列 + **运动 B 的 triangulation** + **C 运动/慢病的谨慎解释** + **阴性对照与 DML** 各一段。  
4. **Discussion**：估计量依赖、短 horizon、观察性局限、试验启示；**禁止** C 运动「少动」类因果推荐。  
5. 风格：**BMC Medicine / JAD** 级别正式英文。

---

**维护**：重跑 `run_all_charls_analyses` 后请用新 CSV 更新本文件中的表格单元格；并同步 `PAPER_Manuscript_Submission_Ready.md` 的 Data lock-in 日期。
