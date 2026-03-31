# 代码逻辑一致性审计报告

## 1. 年龄标准 (age_min) 不一致

| 脚本 | 当前调用 | 主流程标准 | 建议 |
|------|----------|------------|------|
| `run_all_charls_analyses.py` | age_min=60 | ✓ | 已统一 |
| `run_multi_exposure_causal.py` | age_min=60 | ✓ | 已统一 |
| `run_subgroup_and_joint_causal.py` | age_min=60 | ✓ | 已统一 |
| `compare_age_auc.py` | age_min 由参数传入 | ✓ | 已统一 |
| `patch_missing_plots.py` | age_min=60 | ✓ | 已统一 |
| `run_sensitivity_scenarios.py` | age_min=60 | ✓ | 已统一 |

---

## 2. patch_missing_plots.py 因果逻辑错误

**问题**：`estimate_causal_impact(df)` 对**全量数据** (A+B+C 混合) 拟合因果模型。

**主流程设计**：因果分析应对 **轴线 B** 和 **轴线 C** 分别拟合，因不同基线人群的干预效应可能异质。

**影响**：补全的 CATE、亚组、验证图基于混合人群的因果估计，与主流程结果不一致。

**建议**：改为对 df_b、df_c 分别调用 `estimate_causal_impact`，或明确标注该脚本为「探索性全人群分析」。

---

## 3. 特征排除列表分散定义

| 模块 | 排除逻辑 | 与 charls_feature_lists 一致？ |
|------|----------|-------------------------------|
| `charls_model_comparison.py` | 使用 `get_exclude_cols()` | ✓ |
| `charls_recalculate_causal_impact.py` | 使用 `get_exclude_cols()` | ✓ |
| `charls_shap_analysis.py` | 硬编码 exclude + leakage_keywords | 基本一致，但未用 get_exclude_cols |
| `charls_clinical_evaluation.py` | 硬编码 exclude（更细） | 部分一致 |
| `charls_clinical_decision_support.py` | 硬编码 exclude | 部分一致 |
| `charls_cate_visualization.py` | 硬编码 exclude，treatment=is_depression | 不同分析目标 |
| `charls_validation_suite.py` | 硬编码 exclude | 部分一致 |

**风险**：若 charls_feature_lists 新增泄露列，SHAP/临床模块可能未同步排除。

**建议**：SHAP、临床评价、决策支持逐步改为调用 `get_exclude_cols()`，或至少与 EXCLUDE_COLS_BASE + LEAKAGE_KEYWORDS 对齐。

---

## 4. 干预变量 (treatment) 不一致

| 模块 | 默认 treatment | 主流程 |
|------|----------------|--------|
| 主流程 / 因果 | exercise | ✓ |
| `charls_cate_visualization.py` | is_depression | 不同分析 |
| `charls_validation_suite.py` | is_depression | 不同分析 |

**说明**：CATE 与验证套件以 is_depression / is_cognitive_impairment 为「治疗」，属于对称轴分析；主因果以 exercise 为干预。两者分析目标不同，可保留，但应在注释中说明。

---

## 5. 读取预处理数据的方式

| 脚本 | 数据来源 | 与主流程一致？ |
|------|----------|----------------|
| `replot_main_figures.py` | CHARLS_final_preprocessed.csv | 依赖主流程先运行并写盘 |
| `charls_cate_visualization.py` | 传入的 df | 取决于调用方 |
| `emergency_replot.py` | CHARLS_final_preprocessed.csv | 同上 |
| `run_final_completion.py` | CHARLS_final_preprocessed.csv | 同上 |

**说明**：这些脚本读取的是主流程写入的预处理文件；若主流程使用 age_min=50，则数据一致。仅当单独运行 patch_missing_plots 等脚本且未传 age_min 时，会与主流程不一致。

---

## 6. 修复优先级

1. **高**：`patch_missing_plots.py` — 年龄标准 + 因果分析应按 B/C 分轴
2. **高**：`run_sensitivity_scenarios.py` — 补充 age_min=50
3. **中**：SHAP/临床模块 — 统一使用 get_exclude_cols 或明确对齐规则
4. **低**：CATE/验证套件的 treatment 差异 — 文档说明即可
