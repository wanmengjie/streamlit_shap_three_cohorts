# 项目结构说明

## 目录结构

```
因果机器学习/
├── run_all_charls_analyses.py   # 主入口
├── config.py                    # 全局配置
├── data/                        # 数据预处理
│   ├── charls_complete_preprocessing.py
│   └── charls_table1_stats.py
├── modeling/                    # 预测建模
│   ├── charls_model_comparison.py
│   └── charls_cpm_evaluation.py
├── causal/                      # 因果推断
│   ├── charls_recalculate_causal_impact.py
│   ├── charls_causal_methods_comparison.py
│   ├── charls_causal_refinement.py
│   ├── charls_low_sample_optimization.py
│   └── charls_cognitive_subdomain_analysis.py
├── evaluation/                  # 评估
│   ├── charls_clinical_evaluation.py
│   ├── charls_clinical_decision_support.py
│   ├── charls_subgroup_analysis.py
│   ├── charls_sensitivity_analysis.py
│   ├── charls_external_validation.py
│   ├── charls_dose_response.py
│   ├── charls_ite_validation.py
│   ├── charls_temporal_analysis.py
│   ├── charls_nomogram.py
│   └── charls_imputation_audit.py
├── interpretability/            # 可解释性
│   ├── charls_shap_analysis.py
│   └── charls_shap_stratified.py
├── viz/                         # 可视化
│   ├── draw_attrition_flowchart.py
│   ├── draw_conceptual_framework.py
│   ├── draw_roc_combined.py
│   └── charls_extra_figures.py
├── scripts/                     # 独立运行脚本
│   ├── run_sensitivity_scenarios.py
│   ├── run_multi_exposure_causal.py
│   ├── run_all_interventions_analysis.py
│   ├── run_xlearner_all_interventions.py
│   ├── run_dca_on_saved_models.py
│   ├── run_shap_on_saved_models.py
│   ├── run_baseline_table_only.py
│   ├── run_verification_checklist.py
│   └── check_paths.py
├── external/                    # 外部验证 (CLHLS)
│   ├── clhls_load_utils.py
│   ├── clhls_external_validation_plan_a.py
│   └── clhls_full_pipeline.py
├── utils/                       # 共享工具
│   ├── charls_feature_lists.py
│   ├── charls_ci_utils.py
│   └── charls_bias_analysis.py
└── archive/                     # 归档脚本
```

## 运行方式

主流程（在项目根目录执行）：
```powershell
cd "c:\Users\lenovo\Desktop\因果机器学习"
$env:PYTHONIOENCODING='utf-8'; python run_all_charls_analyses.py
```

独立脚本示例：
```powershell
python -m scripts.run_baseline_table_only
python -m scripts.run_dca_on_saved_models
```
