# 方法学表述草稿（中英）

可直接或稍作修改放入论文 Methods 部分，与代码和 ANALYSIS_DEFINITIONS.md 一致。

---

## 一、中文

### 1. 研究人群与数据来源

数据来源于中国健康与养老追踪调查（CHARLS）。研究对象为基线年龄 ≥ 60 岁的社区老年人，分析单位为人–波次（person-wave）。纳入流程与每步样本量见补充材料中的流失流程图（attrition flow diagram）；关键排除步骤包括：无 CES-D-10 或认知得分、无下一波共病结局、基线已罹患抑郁–认知共病（仅保留基线无共病者构成入射队列）。

### 2. 结局、暴露与定义

**抑郁**：采用 10 题版流调中心抑郁量表（CES-D-10），总分 ≥ 10 定义为抑郁症状阳性（参考文献：Andresen 等；CHARLS 用户手册）。**认知受损**：采用 CHARLS 认知综合得分（低分表示功能较差），得分 ≤ 10 定义为认知受损（依据 CHARLS 问卷与手册）。**抑郁–认知共病**：同时满足上述抑郁阳性与认知受损。**随访结局**：下一轮调查时是否新发抑郁–认知共病（is_comorbidity_next），仅使用紧邻下一波次且间隔为一期的观测。

**暴露/干预**：因果分析中以“运动”（exercise）为处理变量，定义为 CHARLS 中与过去一段时间规律运动/锻炼相关的二值变量（具体题项见问卷）。主分析中，运动缺失的观测在分析中记为未运动（0）；敏感性分析中，我们限制在运动变量无缺失的完整病例中重新估计平均处理效应（ATE）。

### 3. 混杂控制与因果假定

我们采用基于双机器学习（DML）的因果森林（Causal Forest）估计运动对随访共病发生率的平均处理效应。控制的协变量包括年龄、性别、教育、城乡、体质指数、慢性病负担、生活方式（如睡眠、吸烟、饮酒）、躯体功能相关指标等，且不包含与结局定义直接相关的量表得分（如 CES-D-10、认知总分），以避免信息泄露。我们假定：在给定上述协变量下，治疗组与对照组可交换（无未测混杂）；且各协变量组合下均存在运动与非运动者（正性）。效应尺度为平均处理效应，对应二值结局时可解释为风险差（RD）的估计。

### 4. 缺失数据

基线抑郁、认知及下一波共病结局缺失者自分析中排除；其余协变量缺失在建模与因果分析中采用中位数插补。干预（运动）缺失在主分析中按“未运动”处理，敏感性分析中改为仅保留运动无缺失的完整病例。

### 5. 敏感性分析

我们进行了以下敏感性分析：（1）采用不同截断值定义抑郁（CES-D-10 ≥ 8 或 ≥ 12）与认知受损（得分 ≤ 8 或 ≤ 12），重新构建队列并估计各亚组 ATE；（2）将分析限制在运动变量无缺失的完整病例中重新估计 ATE。结果以表格与图示汇总（见补充材料）。

### 6. 伦理与泛化性

CHARLS 已获相应伦理审批及知情同意。本研究结果主要适用于中国 60 岁及以上社区老年人，外推至其他人群或地区需谨慎。

---

## 二、English

### 1. Study population and data source

Data were from the China Health and Retirement Longitudinal Study (CHARLS). The study population comprised community-dwelling adults aged ≥60 years at baseline, with person-wave as the unit of analysis. The inclusion flow and sample size at each step are reported in the attrition flow diagram (see Supplementary Material). Key exclusions included missing CES-D-10 or cognition score, missing next-wave comorbidity outcome, and baseline depression–cognitive impairment comorbidity (only those free of comorbidity at baseline were included in the incident cohort).

### 2. Outcomes, exposure, and definitions

**Depression** was defined using the 10-item Center for Epidemiologic Studies Depression Scale (CES-D-10), with a score ≥10 indicating depressive symptoms (Andresen et al.; CHARLS user manual). **Cognitive impairment** was defined using the CHARLS composite cognition score (lower score indicating worse function), with a score ≤10 indicating impairment (per CHARLS questionnaire and manual). **Depression–cognition comorbidity** required both depression and cognitive impairment as defined above. The **follow-up outcome** was incident depression–cognitive impairment comorbidity at the next survey wave; only observations with a valid next wave (one wave ahead) were used.

**Exposure/treatment**: In causal analyses, “exercise” was the treatment variable, defined as the binary indicator of regular physical activity in CHARLS (see questionnaire for exact item). In the main analysis, missing exercise was coded as no exercise (0); in sensitivity analyses, we restricted to complete cases with non-missing exercise and re-estimated the average treatment effect (ATE).

### 3. Confounding and causal assumptions

We estimated the average treatment effect of exercise on incident comorbidity using a causal forest based on double machine learning (DML). Confounders included age, sex, education, urban/rural residence, body mass index, chronic disease burden, lifestyle (e.g., sleep, smoking, alcohol), and physical function–related measures; we excluded variables that directly define the outcome (e.g., CES-D-10, total cognition score) to avoid information leakage. We assumed exchangeability given these covariates (no unmeasured confounding) and positivity (both treated and untreated individuals within covariate strata). The effect scale is the ATE, interpretable as a risk difference (RD) for the binary outcome.

### 4. Missing data

Observations with missing baseline depression, cognition, or next-wave comorbidity outcome were excluded. Remaining covariate missingness was handled by median imputation in modeling and causal estimation. Treatment (exercise) missingness was coded as unexposed in the main analysis; sensitivity analyses used complete cases with non-missing exercise.

### 5. Sensitivity analyses

We performed sensitivity analyses by (1) varying cutpoints for depression (CES-D-10 ≥8 or ≥12) and cognitive impairment (score ≤8 or ≤12) to redefine the cohort and re-estimate ATE by subgroup, and (2) restricting to complete cases on exercise and re-estimating the ATE. Results are summarized in tables and figures (Supplementary Material).

### 6. Ethics and generalizability

CHARLS was approved by the relevant ethics committees with informed consent. Findings are most applicable to community-dwelling adults aged ≥60 years in China; extrapolation to other populations or settings should be made with caution.
