# 主文 Figure 1–5 与输出文件对照（与 `run_all_charls_analyses.py` consolidate 一致）

**项目根目录**：`因果机器学习`（以下相对路径均相对根目录）

**说明：** 主文**不再收录**三队列合并 ROC 图（原 Figure 3）；判别与全模型比较以主文 **Table 2** 为准（不再单独设补充表 S6）。流水线仍可能生成 `fig3_roc_combined.png`，可作**可选补充材料**；原始宽表见 **`results/tables/table2_prediction_cohort*.csv`**（首选）；旧名 `table2_prediction_axis*.csv` 为 consolidate 双写同内容。

| 主文 Figure | 图注（稿件版） | 插入 Word 时优先使用的文件 | 生成 / 来源 |
|------------|----------------|---------------------------|-------------|
| **1** | STROBE 纳入排除流程 | `results/figures/fig1_attrition_flow.png` | `LIU_JUE_STRATEGIC_SUMMARY/attrition_flow_diagram.png` → consolidate 复制 |
| **2** | 三队列概念框架 | `results/figures/fig2_conceptual_framework.png` | `LIU_JUE_STRATEGIC_SUMMARY/fig2_conceptual_framework.png` |
| **3** | 运动 ATE / 亚组 CATE | `results/figures/fig4_intervention_benefit.png`；亚组：`fig4_subgroup_cate_combined.png`；森林：**首选** `fig4_subgroup_forest_cohortA/B/C.png`（旧名 `fig4_subgroup_forest_axis*` 同图） | `LIU_JUE_STRATEGIC_SUMMARY/intervention_benefit_comparison.png` 等 → consolidate |
| **4** | SHAP 摘要（代表队列） | **首选** `results/figures/fig4_shap_cohortA/B/C.png`（旧名 `fig4_shap_axis*`） | 自 `Cohort_*/02_shap/fig5a_shap_summary_is_comorbidity_next.png` 复制；**fig4_shap_cohort* 对应主文 Figure 4** |
| **5** | DCA（及同文件中的校准/PR 子图） | **首选** `results/figures/fig5_dca_cohortA/B/C.png`（旧名 `fig5_dca_axis*`） | 自 `Cohort_*/04_eval/fig3_clinical_evaluation_comprehensive.png` 复制；**fig5_dca_cohort* 对应主文 Figure 5** |

## （可选）补充材料用 ROC

| 用途 | 文件 |
|------|------|
| 三队列 ROC 合并图（非主文） | `results/figures/fig3_roc_combined.png` 或 `LIU_JUE_STRATEGIC_SUMMARY/fig3_roc_combined.png` |

## 注意

1. **fig4_shap_cohort*** / **fig5_dca_cohort***（及兼容旧名 **fig4_shap_axis*** / **fig5_dca_axis***）是 consolidate 时的内部命名（来自流水线 `fig5a_shap_*`、`fig3_clinical_evaluation_*`），与主文 **Figure 4（SHAP）、Figure 5（DCA）** 对应；勿与 **Figure 3（因果/亚组）** 的 `fig4_intervention_benefit*` 混淆。
2. **Figure 3**（因果 / CATE）若缺失，回 `OUTPUT_ROOT` 或 `Cohort_*/06_subgroup/` 取源图。
3. 若 `results/figures` 为空，请先完整跑通 `run_all_charls_analyses.py` 以触发 consolidate。
