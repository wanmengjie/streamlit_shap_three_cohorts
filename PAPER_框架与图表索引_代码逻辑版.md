# 论文框架与图表索引（基于当前代码逻辑）

**依据**：`run_all_charls_analyses.py` 主流程及依赖模块  
**更新日期**：2025-03-16（主文 ROC 已移除，图号与 `PAPER_Manuscript_Submission_Ready.md` 对齐）

---

## 一、论文结构框架

| 章节 | 内容概要 | 对应代码/输出 |
|------|----------|---------------|
| **Abstract** | 摘要、方法、主要结果 | 全文结果汇总 |
| **1 Introduction** | 背景、研究 gap、目标 | — |
| **2 Methods** | — | — |
| 2.1 研究设计与数据 | 三队列、流失、基线 | `preprocess_charls_data`、`generate_baseline_table`、`draw_flowchart` |
| 2.2 预测建模 | 15 模型打擂、冠军选择、阈值 | `compare_models`、`charls_model_comparison` |
| 2.3 可解释性 | SHAP 归因、分层、交互 | `run_shap_analysis_v2`、`run_stratified_shap`、`run_shap_interaction` |
| 2.4 因果推断 | XLearner/TLearner、PSM/PSW、假设检验 | `get_estimate_causal_impact`、`run_sensitivity_analysis`、`run_all_axes_comparison` |
| 2.5 临床评价 | 校准、DCA、列线图 | `run_clinical_evaluation`、`run_clinical_decision_support`、`run_nomogram` |
| 2.6 敏感性分析 | 截断值、插补、偏倚、Placebo、E-value | `run_sensitivity_scenarios_analysis`、`run_imputation_sensitivity_preprocessed`、`run_sensitivity_analysis` |
| 21 | **3 Results** | — | — |
| 22 | 3.1 样本与基线 | 流失、基线特征 | Table 1、Figure 1 |
| 23 | 3.2 预测性能 | 三队列冠军、判别指标 | **Table 2**（含 CPM 全模型比较，按主文呈现）；主文不设 ROC（可选 `fig3_roc_combined.png`） |
| 24 | 3.3 可解释性 | SHAP 归因、可干预性 | **Figure 3**、Figure S5/S6 |
| 25 | 3.4 因果效应 | 主干预 ATE、亚组 CATE、多干预、PSM/PSW 验证 | Table 4/4b、Table 7、**Figure 4** |
| 26 | 3.5 亚组与异质性 | 亚组 CATE | Table 3、Table S6、Figure 4B、Figure S15 |
| 27 | 3.6 外部验证 | 时间/区域 | Table 6、Figure S4 |
| 28 | 3.7 敏感性 | 截断值、插补、偏倚、E-value | Table 5、Table S8、Figure S10、Figure S16/S17 |
| 29 | **4 Discussion** | 解释、局限、临床意义 | — |
| 30 | **5 Conclusions** | 结论 | — |
| 31 | **Supplementary** | 附录 | Table S1–S12、Figure S1–S17 |

---

## 二、正文表格（Table）

| 序号 | 名称 | 数据来源 | 路径/文件 |
|------|------|----------|-----------|
| **Table 1** | 样本流失 | `draw_flowchart` + `attrition_flow.csv` | `results/tables/table1_sample_attrition.csv` |
| **Table 1** | 基线特征 | `generate_baseline_table` | `results/tables/table1_baseline_characteristics.csv` |
| **Table 2** | 预测性能（15 模型，CPM 主表） | `evaluate_and_report`（CPM） | **首选** `results/tables/table2_prediction_cohortA/B/C.csv`；旧名 `table2_prediction_axis*` 双写；或 `Cohort_*/01_prediction/table2_*_main_performance.csv` |
| **Table 3** | 亚组 CATE | `run_subgroup_analysis` | `Cohort_A/B/C/06_subgroup/subgroup_analysis_results.csv` |
| **Table 4** | 多干预 ATE | `run_all_interventions` | `results/tables/table4_ate_summary.csv` |
| **Table 4b** | X-Learner 全干预 + PSM/PSW 宽表 | `run_xlearner_all_interventions` | `results/tables/table4_xlearner_psm_psw_wide.csv` |
| **Table 5** | 截断值敏感性 | `run_sensitivity_scenarios_analysis` | `LIU_JUE_STRATEGIC_SUMMARY/sensitivity_summary.csv` |
| **Table 6** | 外部验证 | `run_external_validation` | **首选** `table6_external_validation_cohortA/B/C.csv`；旧名 `table6_external_validation_axis*` |
| **Table 7** | PSM/PSW/XLearner 因果方法对比 | `run_all_axes_comparison` | `results/tables/table7_psm_psw_dml.csv` |

---

## 三、正文图形（Figure）— 建议重组版

**叙事逻辑**：按「减法」拆分原 Figure 过载内容，使每图主题单一、逻辑清晰。

| 序号 | 名称 | 数据来源 | 路径/文件 |
|------|------|----------|-----------|
| **Figure 1** | 样本流失流程图 | `draw_flowchart` | `LIU_JUE_STRATEGIC_SUMMARY/attrition_flow_diagram.png` |
| **Figure 2** | 概念框架图 | `draw_conceptual_framework` | `LIU_JUE_STRATEGIC_SUMMARY/fig2_conceptual_framework.png` |
| **Figure 3** | SHAP 归因（主文） | `run_shap_analysis_v2` | `Cohort_*/02_shap/fig5a_shap_summary_*.png`、`fig5b_shap_intervenability_*.png` |
| **Figure 4** | 因果效应森林图（主文） | `run_xlearner_all_interventions` 等 | `LIU_JUE_STRATEGIC_SUMMARY/fig_all_interventions_forest.png`、`fig_subgroup_cate_combined.png` |
| **Figure 5** | 校准、DCA、PR 等临床评价（主文） | `run_clinical_evaluation` | `Cohort_*/04_eval/fig3_clinical_evaluation_comprehensive.png`（consolidate 后可能为 `fig5_dca_*` 等） |
| **（可选补充）** | 三队列合并 ROC | `draw_roc_combined` | `fig3_roc_combined.png` — **不作主文 Figure** |
| **附录** | 截断值敏感性 | `run_sensitivity_scenarios_analysis` | `sensitivity_ate_comparison_exercise.png` → **Figure S10** |
| **附录/正文表** | 多干预森林等 | `run_all_interventions` | `fig_all_interventions_forest.png` 等，按期刊要求放主文或附录 |

---

## 四、附录表格（Table S）

| 序号 | 名称 | 数据来源 | 路径/文件 |
|------|------|----------|-----------|
| **Table S1** | 变量定义 | 手动/脚本 | `results/tables/tableS1_variable_definitions.csv` |
| **Table S2** | 缺失数据汇总 | `generate_baseline_table` | `results/tables/table1_missing_summary.csv` |
| **Table S3** | 截断值敏感性 | `run_sensitivity_scenarios_analysis` | `LIU_JUE_STRATEGIC_SUMMARY/sensitivity_summary.csv` |
| **Table S4** | 探索性因果 ATE | `run_all_interventions` | `results/tables/table4_ate_summary.csv` |
| **Table S4b** | X-Learner+PSM+PSW 宽表 | `run_xlearner_all_interventions` | `results/tables/table4_xlearner_psm_psw_wide.csv` |
| **Table S5** | 外部验证 | `run_external_validation` | **首选** `results/tables/table6_external_validation_cohortA/B/C.csv`；旧名 `table6_external_validation_axis*` |
| **Table S6** | 因果方法交叉验证 | `run_all_cohorts_comparison`（旧名 `run_all_axes_comparison`） | `results/tables/table7_psm_psw_dml.csv`（汇总列名 `cohort`，旧 CSV 可能仍为 `axis`） |
| **Table S7** | A/C 亚组 CATE | `run_subgroup_analysis` | **首选** `table3_subgroup_cohortA.csv`、`table3_subgroup_cohortC.csv`；旧名 `table3_subgroup_axis*` |
| **Table S8** | 偏倚敏感性 | `run_sensitivity_analysis` | `Cohort_A/B/C/07_sensitivity/sensitivity_exercise/bias_sensitivity.csv` |
| **Table S9** | 超参搜索 | `compare_models` | `results/tables/tableS8_hyperparameter_search.csv` |
| **Table S10** | 因果假设检验 | `get_estimate_causal_impact` | `Cohort_A/B/C/03_causal/assumption_checks_summary.txt` |
| **Table S11** | 生理指标 ATE | `run_all_physio_causal` | `results/tables/table_physio_ate_summary.csv` |
| **Table S12** | 插补敏感性 | `run_imputation_sensitivity_preprocessed` | `results/tables/imputation_sensitivity_results.csv` |

---

## 五、附录图形（Figure S）

| 序号 | 名称 | 数据来源 | 路径/文件 |
|------|------|----------|-----------|
| **Figure S1** | 缺失数据热图 | 待生成 | — |
| **Figure S2** | 倾向得分重叠 | `get_estimate_causal_impact` | `Cohort_A/B/C/03_causal/fig_propensity_overlap_exercise.png` |
| **Figure S3** | 插补敏感性 | `run_imputation_sensitivity_preprocessed` | `LIU_JUE_STRATEGIC_SUMMARY/07_sensitivity_imputation/figS3_imputation_sensitivity.png` |
| **Figure S4** | 外部验证校准 | `run_external_validation` | `Cohort_A/B/C/04b_external_validation/external_validation_calibration.png` |
| **Figure S5** | SHAP 分层 | `run_stratified_shap` | `Cohort_A/B/C/02_shap_stratified/fig_shap_stratified.png` |
| **Figure S6** | SHAP 交互 | `run_shap_interaction` | `Cohort_A/B/C/02_shap/fig_shap_interaction.png` |
| **Figure S7** | 剂量反应 RCS | `run_dose_response` | `Cohort_A/B/C/04c_dose_response/fig_dose_response_rcs.png` |
| **Figure S8** | ITE 分层验证 | `run_ite_stratified_validation` | `Cohort_A/B/C/03_causal/ite_validation/fig_ite_stratified.png` |
| **Figure S9** | 列线图 | `run_nomogram` | `Cohort_A/B/C/03_causal/nomogram/fig_nomogram_ite.png` |
| **Figure S10** | 截断值敏感性 | `run_sensitivity_scenarios_analysis` | `LIU_JUE_STRATEGIC_SUMMARY/sensitivity_ate_comparison_exercise.png` |
| **Figure S11** | 因果方法森林图（运动） | `run_all_axes_comparison` | `Cohort_A/B/C/03_causal/fig_causal_methods_forest_exercise.png` |
| **Figure S12** | 运动 ATE 与亚组 CATE（原主文图） | `run_subgroup_analysis`、汇总图 | `LIU_JUE_STRATEGIC_SUMMARY/fig4_intervention_benefit*.png` |
| **Figure S13** | 多干预因果森林图 | `run_all_interventions` + `run_all_axes_comparison` | `all_interventions_causal/Cohort_A/B/C/{exercise,drinkev,bmi_normal,chronic_low,is_socially_isolated}/fig_causal_methods_forest_*.png` |
| **Figure S14** | X-Learner 全干预森林图 | `run_xlearner_all_interventions` | `xlearner_all_interventions/fig_xlearner_all_interventions_forest.png` |
| **Figure S15** | 亚组森林图 | `run_subgroup_analysis` | `Cohort_A/B/C/06_subgroup/fig_subgroup_academic_forest.png` |
| **Figure S16** | Placebo + E-value | `run_sensitivity_analysis` | `Cohort_A/B/C/07_sensitivity/sensitivity_exercise/fig_s14_placebo_e_value.png` |
| **Figure S17** | 偏倚敏感性 | `run_sensitivity_analysis` | `Cohort_A/B/C/07_sensitivity/sensitivity_exercise/fig_bias_sensitivity.png` |

---

## 六、额外汇总图（可选）

| 图名 | 数据来源 | 路径/文件 |
|------|----------|-----------|
| 发病率对比 | 主流程 | `LIU_JUE_STRATEGIC_SUMMARY/incidence_comparison.png` |
| 运动 ATE 三轴线 | 主流程 | `LIU_JUE_STRATEGIC_SUMMARY/intervention_benefit_comparison.png` |
| 基线汇总 | `draw_all_extra_figures` | `LIU_JUE_STRATEGIC_SUMMARY/fig_baseline_summary.png` |
| 亚组 CATE 合并 | `draw_all_extra_figures` | `LIU_JUE_STRATEGIC_SUMMARY/fig_subgroup_cate_combined.png` |
| 模型比较雷达图 | `draw_performance_radar` | `Cohort_A/B/C/01_prediction/fig4a_performance_radar.png` |
| 15 模型比较柱状图 | `compare_models` | `Cohort_A/B/C/01_prediction/fig2c_comparison_*.png` |

---

## 七、代码流程与图表对应关系

```
run_all_charls_analyses.main()
├── 数据加载 → Table 1 基线、Table 1 流失
├── draw_conceptual_framework → Figure 2
├── generate_baseline_table → Table 1、Table S2
├── draw_flowchart → Figure 1
├── run_imputation_sensitivity_preprocessed → Table S11、Figure S3
├── run_cohort_protocol（Cohort A/B/C 各一次）
│   ├── compare_models → Table 2（及 `table2_prediction_cohort*.csv`，旧名 `table2_prediction_axis*`）、Table S8
│   ├── draw_performance_radar → 雷达图
│   ├── run_shap_analysis_v2 → SHAP 图
│   ├── run_stratified_shap → Figure S5
│   ├── run_shap_interaction → Figure S6
│   ├── get_estimate_causal_impact → Figure S2、Table S9、Figure S11
│   ├── run_sensitivity_analysis → Table S7、Figure S13、Figure S14
│   ├── run_ite_stratified_validation → Figure S8
│   ├── run_nomogram → Figure S9
│   ├── run_clinical_evaluation → Figure 5（校准+DCA 等）
│   ├── run_external_validation → Table 6、Figure S4
│   ├── run_dose_response → Figure S7
│   └── run_subgroup_analysis → Table 3、Figure S12
├── run_sensitivity_scenarios_analysis → Table 5、Table S3、Figure S10
├── run_multi_exposure_analysis → Table 4 备选
├── run_all_interventions → Table 4、附录森林图等
├── run_xlearner_all_interventions → Table 4b、Figure S11 X-Learner
├── run_all_axes_comparison → Table 7、Figure S11 因果森林
├── run_all_physio_causal → Table S10
├── draw_all_extra_figures → 额外汇总图
└── draw_roc_combined → 可选补充图 `fig3_roc_combined.png`（非主文 Figure 3）
```

---

## 八、三队列说明

所有 `Cohort_*` 路径均对应三个队列：

- **Cohort_A**：`Cohort_A_Healthy_Prospective`（健康人群）
- **Cohort_B**：`Cohort_B_Depression_to_Comorbidity`（抑郁人群）
- **Cohort_C**：`Cohort_C_Cognition_to_Comorbidity`（认知受损人群）

上述表格与图形中，凡标注为 `Cohort_A/B/C` 的，均表示三队列均有对应输出。

---

## 九、论文写作建议与叙事逻辑

### 1. 因果推断叙事（Methods 2.4 & Results 3.4）

| 要点 | 建议表述 |
|------|----------|
| **主分析** | X-Learner 为主分析（Primary Analysis），因其能估计异质性治疗效应（CATE），在治疗/对照组不平衡时通常优于传统 PSM。 |
| **验证分析** | PSM、PSW 作为验证分析（Validation），证明结论在不同因果识别策略下的一致性。 |
| **Table 7 增补** | 已实现：`charls_causal_methods_comparison` 自动生成 `Consistency` 列（Consistent / Direction_only / Inconsistent）。 |

### 2. 讨论（Discussion）深度挖掘

| 主题 | 建议内容 |
|------|----------|
| **Cost-Benefit 视角** | 基于 DCA 图，讨论在不同决策阈值下实施干预（如运动）的净获益与性价比；指出「最优阈值区间」及临床适用场景。 |
| **干预优先级** | 利用 SHAP 可干预性结果，区分：① 重要但不可改变（如年龄）；② 高回报且可改变（如运动、社交孤立）；③ 可改变但回报中等（如 BMI、慢性病负担）。为临床医生提供「优先干预清单」。 |

### 3. 附录 Figure S2：倾向得分重叠

| 要点 | 说明 |
|------|------|
| **重要性** | 倾向得分重叠是因果推断的「门面」；重叠度不足时 ATE 估计不可靠。 |
| **写作建议** | 在附录中确保 Figure S2（三队列）清晰展示；若某队列重叠差（如 overlap < 20%），在 Limitations 中明确说明，并解释 trimming 或敏感性分析如何应对。 |
| **路径** | `Cohort_A/B/C/03_causal/fig_propensity_overlap_exercise.png` |
