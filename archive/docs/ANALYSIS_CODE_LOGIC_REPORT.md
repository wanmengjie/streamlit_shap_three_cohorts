# 全部分析代码逻辑检查报告

## 一、检查范围与数据流总览

- **主入口**：`run_all_charls_analyses.py` 的 `main()` 与 `run_axis_protocol()`。
- **数据流**：数据源选择（插补/预处理）→ `df_clean` → 分轴线 `df_a/df_b/df_c` → 各分析模块；插补敏感性单独使用「带缺失」的预处理数据。
- **配置来源**：`config.py`（TARGET_COL、TREATMENT_COL、RANDOM_SEED、OUTPUT_ROOT、AXIS_*_DIR、IMPUTED_DATA_PATH 等）。

---

## 二、模块级逻辑状态

| 模块 | 入口函数 | 输入依赖 | 输出/副作用 | 逻辑状态 |
|------|----------|----------|-------------|----------|
| charls_complete_preprocessing | preprocess_charls_data | CSV 路径、截断值、age_min、write_output | df；可选写盘 attrition/preprocessed | ✅ 已检查：缺列用 next() 查找，返回 None 有保障 |
| charls_table1_stats | generate_baseline_table | df 含 baseline_group | table1_*.csv | ✅ 正常 |
| charls_model_comparison | compare_models | df、target_col、save_roc_path | perf_df、models、roc_data.json、champion_model.joblib | ✅ 使用 get_exclude_cols、TARGET_COL 一致 |
| charls_shap_analysis | run_shap_analysis_v2 | df、model、target_col | SHAP 图/CSV | ✅ 正常 |
| charls_recalculate_causal_impact | estimate_causal_impact | df、treatment_col | res_df 含 causal_impact_{T}、(ate,lb,ub) | ⚠️ 结局名硬编码为 is_comorbidity_next，与 TARGET_COL 当前一致 |
| charls_sensitivity_analysis | run_sensitivity_analysis | res_df（含 causal_impact_*）、treatment_col | Placebo/E-Value/噪声/偏倚 图与报告 | ✅ 正常；内部用 is_comorbidity_next |
| charls_imputation_audit | run_imputation_sensitivity_preprocessed | 带缺失的 df、target_col | imputation_sensitivity_results.csv、figS3 | ✅ 已修正：主流程传入带缺失数据；空 bootstrap/NaN 已防护 |
| run_sensitivity_scenarios | run_sensitivity_scenarios_analysis | data_path、final_dir | sensitivity_summary.csv、ATE 对比图 | ✅ 有 df_main is not None 判断 |
| run_multi_exposure_causal | prepare_exposures, run_multi_exposure_analysis | df_clean 含 baseline_group、sleep 等 | multi_exposure_ate_summary.csv | ✅ 正常；is_socially_isolated 依赖预处理 marry/family_size |
| charls_clinical_evaluation | run_clinical_evaluation | df、model、target_col | 校准/DCA/PR 图、calibration_brier_report.txt | ✅ 主流程传 model；fallback 用 causal_impact 单列（与 causal_impact_{T} 不同，仅异常时） |
| charls_clinical_decision_support | run_clinical_decision_support | df、model、target_col | 决策支持输出 | ✅ 主流程始终传 model，路径一致 |
| charls_subgroup_analysis | run_subgroup_analysis, draw_performance_radar | df 含 causal_impact_* | subgroup_analysis_results.csv、雷达图 | ✅ 主流程在 causal_col 非空时调用 |
| charls_external_validation | run_external_validation | df、model、target_col、axis_label | external_validation_summary.csv、校准图 | ✅ 正常 |
| charls_shap_stratified | run_stratified_shap, run_shap_interaction | df、model、target_col | 分层/交互 SHAP | ✅ 正常 |
| charls_dose_response | run_dose_response | df、target_col | dose_response 图/CSV | ✅ 正常 |
| charls_ite_validation | run_ite_stratified_validation | res_df、causal_col、treatment_col | ite_stratified_validation.csv | ✅ 正常 |
| charls_nomogram | run_nomogram | res_df、causal_col、treatment_col | 列线图、说明 | ✅ 正常 |
| charls_temporal_analysis | run_temporal_analysis | df、treatment_col | temporal_*.csv/图 | ✅ 正常 |
| charls_causal_methods_comparison | run_all_axes_comparison | df_clean | causal_methods_comparison_summary.csv | ✅ 依赖轴线/08_multi_exposure 输出，有回退路径 |
| run_all_interventions_analysis | run_all_interventions | df_clean | all_interventions_summary、森林图 | ✅ 正常 |
| charls_low_sample_optimization | run_for_social_isolation_and_chronic | df_clean（需 prepare_exposures） | low_sample_optimized_*.csv | ✅ 主流程先 prepare_exposures |
| draw_attrition_flowchart | draw_flowchart | attrition CSV（Step, N） | 流失图 PNG | ✅ 主流程先复制 attrition 到 final_dir |
| draw_conceptual_framework | draw_conceptual_framework | output_path | fig2_conceptual_framework.png | ✅ 主流程传 final_dir 路径 |
| draw_roc_combined | draw_roc_combined | output_path | fig3_roc_combined.png | ✅ 已修正：使用 config OUTPUT_ROOT、AXIS_*_DIR |
| charls_extra_figures | draw_all_extra_figures, draw_combined_subgroup_cate | df_clean、auc_a/b/c、output_dir | 基线/运动/亚组 CATE 等图 | ✅ 已修正：亚组路径使用 config AXIS_*_DIR |

---

## 三、已修复的逻辑/一致性问题

1. **轴线路径与 config 统一**
   - **run_all_charls_analyses.py**：复制外部验证时改为使用 `AXIS_A_DIR`、`AXIS_B_DIR`、`AXIS_C_DIR`，与复制冠军模型一致。
   - **draw_roc_combined.py**：默认输出目录改为 `OUTPUT_ROOT`；roc_data.json 路径改为 `os.path.join(AXIS_*_DIR, '01_prediction', 'roc_data.json')`。
   - **charls_extra_figures.py**：`draw_combined_subgroup_cate` 中亚组 CSV 路径改为使用 `AXIS_A_DIR`、`AXIS_B_DIR`、`AXIS_C_DIR`。

2. **插补敏感性分析（此前已修）**
   - 主流程在使用插补数据时，插补敏感性改为使用「带缺失」的预处理数据；`charls_imputation_audit` 内目标缺失剔除、空 bootstrap、NaN、绘图边界已防护。

---

## 四、已知设计约定（无需改代码）

| 项目 | 说明 |
|------|------|
| 因果结局名 | `charls_recalculate_causal_impact` 内 Y 固定为 `is_comorbidity_next`；与当前 config.TARGET_COL 一致，若日后改 TARGET_COL 须同步改该模块。 |
| 临床评价 fallback | 仅当模型预测失败时使用 `df['causal_impact']`；实际列为 `causal_impact_{T}`，主流程有 model 故一般不触发。 |
| 决策支持 model=None | 会读 `evaluation_results/best_predictive_model.joblib`；主流程始终传各轴线 `01_prediction/champion_model.joblib`，无冲突。 |
| 泄露关键字 | `charls_feature_lists.LEAKAGE_KEYWORDS` 已含 `memory`（无 memeory 拼写错误）。 |
| run_sensitivity_scenarios | 使用 `CHARLS.csv` 与预处理，complete-case 分支已有 `if df_main is not None`。 |

---

## 五、与 self.md 记录的对齐情况

- **group_ids**：已改为 `df['ID']`。
- **Phase 1 任务与 compare_models**：compare_models 预测 is_comorbidity_next，与轴线描述一致（文档层面已说明）。
- **分类变量/StandardScaler**：见 self.md；因果/预处理中按现有实现使用。
- **memeory 拼写**：已在 feature_lists 中为 memory。
- **临床决策支持反事实**：见 self.md；未在本次改逻辑。
- **confusion_matrix ravel / Bootstrap CI / Table1 分类**：self.md 记录已防护。
- **插补 LossySetitemError / 插补敏感性数据源**：已按 self.md 与前述报告修正。

---

## 六、结论与建议

- **全部分析代码逻辑**：主流程调用关系、参数传递（TARGET_COL、TREATMENT_COL、轴线目录）、数据源选择与插补敏感性数据源、结果汇总与复制路径均已检查；轴线相关路径已统一为 config，无发现新的逻辑错误。
- **建议**：若将来修改 `config.py` 中 TARGET_COL 或轴线目录名，需同步检查 `charls_recalculate_causal_impact`（Y 名）及所有直接引用轴线目录的脚本（当前已统一为 config，仅需改 config 即可生效）。
