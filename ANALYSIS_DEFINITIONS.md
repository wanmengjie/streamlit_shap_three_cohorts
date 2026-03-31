# 分析定义与变量说明（与代码一致）

用于论文方法部分与审稿对照，保证正文定义与代码实现一致。

---

## 1. 数据来源与人群

- **数据**：中国健康与养老追踪调查（CHARLS）。
- **人群**：基线年龄 ≥ 60 岁的社区老年人（person-wave 为分析单位）。
- **伦理**：CHARLS 已获相应伦理审批及知情同意；本研究使用公开/授权数据，符合伦理要求。

---

## 2. 结局与暴露定义

### 2.1 抑郁（Depression）

| 项目 | 说明 |
|------|------|
| **变量** | `cesd10`（CHARLS 中 10 题版抑郁量表 CES-D-10） |
| **定义** | CES-D-10 总分 ≥ **10** 为抑郁症状阳性 |
| **代码** | `is_depression = (cesd10 >= 10)` |
| **参考** | Andresen EM et al. (1994). *J Aging Health*; CHARLS 用户手册。常用截断值 10（或 8/12 用于敏感性分析）。 |

### 2.2 认知受损（Cognitive impairment）

| 项目 | 说明 |
|------|------|
| **变量** | `total_cognition` 或 `total_cog`（CHARLS 认知综合得分，低分=差） |
| **定义** | 认知得分 ≤ **10** 为认知受损 |
| **代码** | `is_cognitive_impairment = (total_cognition <= 10)` |
| **参考** | CHARLS 问卷与认知模块（如 TICS、即时/延迟回忆、画图等）汇总；截断值依据手册或文献。 |

### 2.3 抑郁–认知共病（Comorbidity）

| 项目 | 说明 |
|------|------|
| **基线共病** | 同时满足抑郁阳性且认知受损：`is_comorbidity = (is_depression==1) & (is_cognitive_impairment==1)` |
| **随访结局** | 下一波（紧邻下一轮调查）是否发生共病：`is_comorbidity_next`（仅保留下一波间隔为 1 的观测）。 |
| **表述** | 文中统一用“抑郁–认知共病”或“incident depression–cognitive impairment comorbidity”。 |

### 2.4 干预/暴露：运动（Exercise）

| 项目 | 说明 |
|------|------|
| **变量** | `exercise`（CHARLS 中与规律运动/锻炼相关的二值或有序变量，具体以问卷题项为准） |
| **因果分析** | 作为二值处理变量 T：编码为 0/1；**缺失在主分析中记为 0（未运动）**。 |
| **说明** | 主分析采用“缺失=0”；敏感性分析采用“仅干预变量无缺失的完整病例”重新估计 ATE。 |

---

## 3. 截断值（主分析与敏感性分析）

| 变量 | 主分析 | 敏感性分析（可选） |
|------|--------|-------------------|
| CES-D-10（抑郁） | ≥ 10 | ≥ 8、≥ 12 |
| 认知得分（认知受损） | ≤ 10 | ≤ 8、≤ 12 |

代码中通过 `preprocess_charls_data(..., cesd_cutoff=10, cognition_cutoff=10)` 传入；敏感性分析脚本可调用不同截断值并汇总 ATE/发病率。

---

## 4. 因果推断中的假定（与 METHODS_DRAFT 对应）

- **可交换性**：在给定协变量 X 下，治疗组与对照组可交换；X 包含年龄、性别、教育、城乡、基线慢性病、基线功能等（见特征列表与排除列表）。
- **正性**：各协变量组合下均存在 T=0 与 T=1 的个体（若违反则需在文中说明并限制人群或合并层）。
- **因果效应尺度**：ATE 为平均处理效应，对二值结局 Y 可解释为**风险差（RD）**的估计（EconML Causal Forest DML 输出）。

---

## 5. 缺失数据处理

- **listwise deletion**：CES-D-10、认知得分、下一波共病结局缺失的观测不进入分析；基线已共病者不进入入射队列。
- **协变量缺失**：建模与因果分析中对数值型协变量采用**中位数插补**；分类变量在预处理中标签编码，缺失在插补阶段一并处理。
- **干预缺失**：主分析将运动缺失视为 0；敏感性分析中采用“仅运动无缺失”的完整病例重新估计 ATE。

---

## 6. 代码与输出对应

- **预处理**：`charls_complete_preprocessing.py`，入口 `preprocess_charls_data(input_path, cesd_cutoff, cognition_cutoff)`。
- **流失表**：`preprocessed_data/attrition_flow.csv` → 主流程复制至 `LIU_JUE_STRATEGIC_SUMMARY/attrition_flow.csv`。
- **因果估计**：`charls_recalculate_causal_impact.estimate_causal_impact(..., treatment_col='exercise')`。
- **敏感性分析**：`run_sensitivity_scenarios.py` 输出 `LIU_JUE_STRATEGIC_SUMMARY/sensitivity_summary.csv` 与 `sensitivity_ate_comparison.png`。
