# 分析代码逻辑与衔接核查报告（2025-03）

## 结论概览

**未发现新的逻辑错误或上下衔接断点。** 主流程、敏感性、因果与亚组的数据流与路径一致；90% CI 为在 95% CI 基础上的增加项，未替换任何既有逻辑。

---

## 1. 主流程 `run_all_charls_analyses.py`

| 环节 | 检查项 | 结果 |
|------|--------|------|
| 数据源 | 优先插补数据 `IMPUTED_DATA_PATH`，失败回退预处理 | ✓ 一致 |
| 队列划分 | `baseline_group` 0/1/2 → A/B/C，与 config 一致 | ✓ |
| 轴线协议 | `run_axis_protocol(df_*, path_dir=AXIS_*_DIR)`，路径来自 config | ✓ |
| 因果→亚组 | `estimate_causal_impact` 在原 `df_sub` 上新增 `causal_impact_{T}` 并返回同一对象，后续 `df_sub` 带因果列供亚组/临床评价使用 | ✓ 衔接正确 |
| 汇总图 | ATE 对比图使用 95% CI（ate_lb, ate_ub）；90% CI 仅写入各轴线 `03_causal/ATE_CI_summary_*.txt` | ✓ |

---

## 2. 因果模块 `charls_recalculate_causal_impact.py`

- 95% CI：`dml.ate_interval(X_scaled)`（默认 alpha=0.05），继续作为主推断并返回给调用方。
- 90% CI：`dml.ate_interval(X_scaled, alpha=0.10)`，仅用于日志与写入 `ATE_CI_summary_{T}.txt`，不改变返回值。
- 所有调用方（主流程、敏感性、多暴露、扩展干预）仍解包 `(ate, ate_lb, ate_ub)`，无需修改。

---

## 3. 敏感性分析衔接

| 项目 | 说明 | 结果 |
|------|------|------|
| 插补敏感性 | 主分析用插补数据时，用「带缺失」的预处理数据做五种插补比较 | ✓ 与设计一致 |
| 截断值敏感性 | `USE_IMPUTED_DATA` 且存在插补文件时传入 `df_base=df_clean`；9 种阈值通过 `reapply_cohort_definition(df_base, cesd_c, cog_c)` 重定义队列 | ✓ |
| 完整病例 | 在 `df_base` 上 `reapply_cohort_definition(..., 10, 10)` 得主阈值队列，再 `dropna(subset=[treatment_col])` | ✓ |

`reapply_cohort_definition` 要求 df 含 `cesd*`、`total_cog*`/`total_cognition`、`is_comorbidity_next`；插补输出保留这些列名，与 `charls_imputation_npj_style.py` 中 DEFINING_COLS 一致。

---

## 4. 路径与配置一致性

- **Fig 3 ROC**：`draw_roc_combined` 从 `config` 读 `OUTPUT_ROOT`、`AXIS_A_DIR`、`AXIS_B_DIR`、`AXIS_C_DIR`，读取各轴线 `01_prediction/roc_data.json`。主流程用 `path_dir=os.path.join('.', AXIS_*_DIR)` 保存，路径一致。
- **亚组 CATE 图**：`charls_extra_figures.draw_combined_subgroup_cate` 使用 `AXIS_*_DIR/06_subgroup/subgroup_analysis_results.csv`，与 `run_subgroup_analysis` 输出路径一致。
- **Consolidate**：复制到 `results/tables`、`results/figures`、`results/models` 时使用同一 config 轴线目录与 `OUTPUT_ROOT`。

---

## 5. 临床评价与亚组

- **临床评价**：优先用模型预测概率；失败时用 `causal_impact_*` 或 `causal_impact` 列做兜底，列名动态识别，无硬编码。
- **亚组分析**：通过 `next((c for c in df.columns if c.startswith('causal_impact_')), ...)` 识别因果列；接收的 `df_sub` 已在因果步骤中就地写入因果列，衔接正确。

---

## 6. 可选改进（非必须）

- **可读性**：在 `run_axis_protocol` 中，当 `res_df is not None` 时，可用 `run_subgroup_analysis(res_df, ...)` 替代 `run_subgroup_analysis(df_sub, ...)`，明确“亚组使用的是带因果列的 dataframe”。当前因 `res_df` 与 `df_sub` 为同一引用，行为已正确。
- **敏感性汇总表**：若希望敏感性场景的汇总表也包含 90% CI，需在 `run_sensitivity_scenarios.run_one_scenario` 中读取 `ATE_CI_summary_{T}.txt` 或让 `estimate_causal_impact` 额外返回 90% CI 再写入 summary；当前敏感性表格仍仅含 95% CI，与主结果一致。

---

## 7. 总结

- 所有分析代码逻辑与论文/附录设计一致，未发现新的逻辑错误。
- 上下衔接：数据源 → 预处理/插补 → 三轴线 → 因果 → 亚组/临床/敏感性 → 汇总图与 consolidate 路径一致。
- 90% CI 为增量输出，不替代 95% CI，不破坏现有调用与表格。

如后续新增分析模块，建议继续从 `config` 读取目录与关键列名，并保持「因果模块只写文件/日志、返回值仍为 (ate, ate_lb, ate_ub)」的约定，以保持衔接一致。
