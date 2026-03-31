# 图表文件核对清单（对齐 `PAPER_Manuscript_Submission_Ready.md`）

**用途**：跑完 `run_all_charls_analyses.py`（或完整主流程）后，按下列路径逐项检查文件是否存在；在 `[ ]` 打勾表示已找到。

**项目根目录**：`因果机器学习`（下列均为相对路径）。

**当前磁盘快照（识别用，2026-03 核对）**：

- **`LIU_JUE_STRATEGIC_SUMMARY/`** 仅有：`table1_baseline_characteristics.csv`、`table1_academic_final.csv`、`table1_missing_summary.csv`、`attrition_flow.csv` 及说明 txt；**无** `fig1_attrition_flow.png` / `fig3_roc_combined.png` 等（需另跑绘图脚本或等主流程汇总）。
- **`Cohort_A/B/C_*`** 各约 **33** 个文件：以 **`01_prediction/table2_*_main_performance.csv`**、**`06_subgroup/subgroup_analysis_results.csv`**、**`03_causal/*`**、**`04_eval/calibration_brier_report.txt`**、**`04b_external_validation/external_validation_summary.csv`** 为主；**当前无** `02_shap/*.png`（若需主文 SHAP 图，需从运行日志确认是否另存或补跑 SHAP 绘图）。
- **`results/tables/`** 中 `table2_prediction_axis*.csv` 的 **AUC 排序可与 `Cohort_*` 的 `table2_*_main_performance.csv` 不一致** → **改稿以 `Cohort_*` CPM 表为准**，或重新跑完整主流程至 consolidate。

**说明**：

- **consolidate** 成功后，主文图优先看 `results/figures/`；若为空，再到 `LIU_JUE_STRATEGIC_SUMMARY/` 与各 `Cohort_*` 子目录找源文件。
- 部分**补充图**在当前主流程中**不保证**自动生成（见「可能缺失」栏），需单独脚本或投稿前补图。
- **路径命名（Cohort，与代码一致）**：`run_all_charls_analyses` consolidate 后**优先**生成 `table2_prediction_cohortA.csv`、`fig4_shap_cohortA.png`、`fig5_dca_cohortA.png` 等；并**双写**旧名 `*_axis*`（内容相同）。下列核对**任一名称存在即可**打勾；新稿写作请优先写 `*_cohort*`。

---

## 一、主文表格（Table 1–5）— 数据文件

| 核对 | 稿件 | 建议对照文件（`results/tables/` 为主） |
|------|------|----------------------------------------|
| [ ] | **Table 1** 基线 / 样本流 | **`LIU_JUE_STRATEGIC_SUMMARY/table1_baseline_characteristics.csv`**；样本流 **`attrition_flow.csv`**；`results/tables/table1_sample_attrition.csv`（若 consolidate 有） |
| [ ] | **Table 2** 预测（CPM） | **首选** `Cohort_*/01_prediction/table2_A/B/C_main_performance.csv`；次选 `results/tables/table2_prediction_cohort*` 或 `table2_prediction_axis*`（**须与 Cohort 时间戳一致**） |
| [ ] | **Table 3** 亚组 CATE | **首选** `Cohort_*/06_subgroup/subgroup_analysis_results.csv`；次选 `results/tables/table3_subgroup_cohort*` 或 `table3_subgroup_axis*` |
| [ ] | **Table 4** 因果假设（重叠、SMD、E-value） | 各队列 `Cohort_A/B/C_*/03_causal/assumption_checks_summary.txt`（及同目录 `assumption_*.txt` / `assumption_balance_smd_*.csv`） |
| [ ] | **Table 5** 五干预 ATE（XLearner） | `table4_ate_summary.csv`；三角验证另见 `table7_psm_psw_dml.csv`、`table4_xlearner_psm_psw_wide.csv` |

---

## 二、主文图形（Figure 1–5）— 图片文件

| 核对 | 稿件 | `results/figures/`（consolidate 目标名） | 常见源路径（若 consolidate 未复制） |
|------|------|------------------------------------------|-------------------------------------|
| [ ] | **Figure 1** STROBE 流程 | `fig1_attrition_flow.png`；**与预处理代码一致版**：`fig1_attrition_flow_code_aligned.png` | 运行 `python -m viz.draw_attrition_flowchart`（读 `preprocessed_data/attrition_flow.csv`）→ 同步写入 `results/figures/fig1_attrition_flow_code_aligned.png`；汇总副本见 `LIU_JUE_STRATEGIC_SUMMARY/attrition_flow_diagram.png` |
| [ ] | **Figure 2** 概念框架 | `fig2_conceptual_framework.png` | `LIU_JUE_STRATEGIC_SUMMARY/fig2_conceptual_framework.png` |
| [ ] | **Figure 3** SHAP | `fig4_shap_cohortA/B/C.png`（或旧名 `fig4_shap_axisA/B/C.png`） | `Cohort_*/02_shap/fig5a_shap_summary_is_comorbidity_next.png` |
| [ ] | **Figure 4** 运动 ATE / 亚组森林图 | `fig4_all_interventions_forest.png` (或 `fig_all_interventions_forest.png`)、`fig4_subgroup_cate_combined.png` | `LIU_JUE_STRATEGIC_SUMMARY/fig_all_interventions_forest.png`；亚组森林：`LIU_JUE_STRATEGIC_SUMMARY/fig_subgroup_cate_combined.png` |
| [ ] | **Figure 5** DCA + 校准 + PR | `fig5_dca_cohortA/B/C.png`（或旧名 `fig5_dca_axisA/B/C.png`） | `Cohort_*/04_eval/fig3_clinical_evaluation_comprehensive.png` |

### 可选（非主文）

| 核对 | 用途 | 路径 |
|------|------|------|
| [ ] | 三队列合并 ROC（补充材料可选） | `results/figures/fig3_roc_combined.png` 或 `LIU_JUE_STRATEGIC_SUMMARY/fig3_roc_combined.png` |

---

## 三、补充表格（Table S1–S7）— 数据文件

| 核对 | 稿件 | 建议对照文件 |
|------|------|----------------|
| [ ] | **Table S1** 缺失机制 | `table1_missing_summary.csv` |
| [ ] | **Table S2** 诊断截断敏感性 | `table5_sensitivity_summary.csv` |
| [ ] | **Table S3** 变量定义与编码 | `tableS1_variable_definitions.csv` |
| [ ] | **Table S4** 外部验证 | **首选** `Cohort_*/04b_external_validation/external_validation_summary.csv`；次选 `results/tables/table6_external_validation_cohort*` 或 `table6_external_validation_axis*` |
| [ ] | **Table S5** PSM / PSW / XLearner 对比 | `table7_psm_psw_dml.csv`（及宽表 `table4_xlearner_psm_psw_wide.csv` 如需） |
| [ ] | **Table S6** A/C 亚组 CATE | `table3_subgroup_cohortA.csv`、`table3_subgroup_cohortC.csv`（或旧名 `table3_subgroup_axisA/C.csv`） |
| [ ] | **Table S7** 超参数搜索 | `tableS8_hyperparameter_search.csv` |

**Text S1–S3**：正文写在 `PAPER_Manuscript_Submission_Ready.md` / `PAPER_Supplementary_Materials.md`，无单独二进制文件。

---

## 四、补充图形（Figure S1–S5）— 图片文件

| 核对 | 稿件 | 建议查找路径 | 可能缺失 / 备注 |
|------|------|----------------|-----------------|
| [ ] | **Figure S1** 缺失热图 | 主流程**不保证**产出；历史脚本见 `archive/charls_imputation_npj_style.py` → `fig1_missing_heatmap.png` | 若无可基于 `table1_missing_summary.csv` 自行出图 |
| [ ] | **Figure S2** PS 重叠（修剪前后） | `Cohort_A/B/C_*/03_causal/fig_propensity_overlap_exercise.png`（及带 `_pre_trim` 等后缀变体，以实际为准） | 运动主暴露 |
| [ ] | **Figure S3** Love plot（SMD） | 主流程产出多为 `assumption_balance_smd_exercise.csv` + txt，**未必**有独立 Love plot PNG | 若期刊要经典 Love plot，需另写绘图脚本 |
| [ ] | **Figure S4** 插补诊断 | `LIU_JUE_STRATEGIC_SUMMARY/07_sensitivity_imputation/figS3_imputation_sensitivity.png` | 代码里文件名仍为 `figS3_*`，与稿件 **Figure S4** 对应关系以投稿说明为准 |
| [ ] | **Figure S5** 校准曲线（冠军模型） | 与主文 Figure 5 中校准面板同源：`Cohort_*/04_eval/fig3_clinical_evaluation_comprehensive.png`；或见 `Cohort_*/01_prediction/calibration_curves_1x3.png`（若 CPM 评价步已生成） | 以你最终采用的「单独校准图」版本为准 |

---

## 五、跑通主流程后快速自检命令（PowerShell）

在项目根目录执行：

```powershell
# 主文 consolidate 图
Get-ChildItem -Path "results\figures" -ErrorAction SilentlyContinue

# 主表
Get-ChildItem -Path "results\tables" -Filter "table*.csv" -ErrorAction SilentlyContinue
```

若 `results\figures` 仍为空，请检查是否执行到 `run_all_charls_analyses.py` 末尾的 **consolidate** 段，或查看日志是否出现 `results consolidate 跳过`。

---

## 六、与 `scripts/check_paths.py` 的关系

仓库内 `scripts/check_paths.py` 使用**另一套**附录图编号（如将外部验证校准标为 Figure S4）。以 **`PAPER_Manuscript_Submission_Ready.md` 目录**与**本清单**为准投稿；若与 `check_paths.py` 不一致，以投稿稿为准更新脚本或清单其一。

---

*最后更新：与当前主文 Figure 1–5、Table S1–S7 一致；`results/*` 文件名以 `*_cohort*` 为首选，`*_axis*` 为 consolidate 兼容副本。*
