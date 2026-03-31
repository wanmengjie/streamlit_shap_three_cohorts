# 归档脚本 (Archive)

本目录存放非主流程的辅助脚本，供参考或按需运行。

## 运行方式

从项目根目录执行，例如：
```powershell
cd "c:\Users\lenovo\Desktop\因果机器学习"
python -m archive.regenerate_table1
python archive/regenerate_table1.py
```

若遇导入错误，确保在项目根目录运行（脚本会从 data/modeling/causal 等包导入）。

## 目录说明

- **docs/**：归档文档（审稿回复、代码审计报告、草稿模板等）
- **实验性**：imputation_experiment, imbalance_comparison_experiment, imputation_npj_style, robust_imputation_audit, real_data_imputation_audit
- **审计/检查**：missing_feature_audit, autoencoder_audit, mcar_test, methodology_audit, validation_suite, correlation_analysis, variable_dictionary, check_*
- **表格/图表整理**：regenerate_table1, collect_final_tables*, consolidate_figures_for_word, update_*, generate_table_s*, generate_fig_*, replot_*, finalize_tables_figures, draw_study_flow, draw_framework_improved, draw_propensity_overlap, generate_full_table2

主流程 `run_all_charls_analyses.py` 已内置表格 consolidate 至 `results/tables`，通常无需单独运行上述脚本。
