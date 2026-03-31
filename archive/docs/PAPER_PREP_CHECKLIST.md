# 论文写作前检查清单

## ✅ 已就绪

| 项目 | 状态 |
|------|------|
| 年龄标准 | 已统一为 age≥60 |
| 主流程、多暴露、亚组、敏感性、补图脚本 | 均使用 age_min=60 |
| 泄露排除 | charls_feature_lists 统一管理，memory 拼写正确 |
| Bootstrap/混淆矩阵 | charls_ci_utils 已防护单类、少次迭代 |
| Table1 分类变量 | 已有 try/except 防护 astype(int) |
| 临床决策支持反事实 | 已与 exercise 干预对齐 |
| ANALYSIS_DEFINITIONS.md | 与代码一致，可直接用于 Methods |

---

## ⚠️ 写论文前必做

### 1. 重新运行主流程（重要）

当前 `preprocessed_data/attrition_flow.csv` 可能仍是 age≥50 的旧结果。请执行：

```bash
python run_all_charls_analyses.py
```

完成后将得到：
- 正确的流失流程图（age≥60）
- 更新的 Table 1、发病率、ATE、AUC
- 所有汇总图与 Fig 2/3

### 2. 核对论文数字与输出

运行后请对照以下文件核对：

| 论文位置 | 核对来源 |
|----------|----------|
| 样本量 14,386；A/B/C 人数 | `LIU_JUE_STRATEGIC_SUMMARY/table1_baseline_characteristics.csv` 首行 N |
| 流失步骤各步 N | `preprocessed_data/attrition_flow.csv` |
| 发病率 | `incidence_comparison.png` 或各轴线 `is_comorbidity_next.mean()` |
| AUC | `Axis_*/0*_prediction/model_performance_full_*.csv` 首行（最佳模型） |
| ATE 及 95% CI | `Axis_B/01_causal/`、`Axis_C/01_causal/` 或日志 |
| 最佳模型名称 | 可能与论文草稿不同（如 A 轴 ExtraTrees 0.75 vs CatBoost 0.72） |

### 3. 论文草稿中可能需更新的数字

- **最佳模型**：当前实现可能选出 ExtraTrees（A）、SVM（B）、LR/AdaBoost（C），与草稿中的 CatBoost/SVM/LR 可能不同，需按实际输出修改。
- **AUC**：以最新 `model_performance_*.csv` 为准。
- **流失步骤**：若预处理逻辑有微调，各步 N 可能略有差异，以 `attrition_flow.csv` 为准。

---

## 低优先级（可选）

| 项目 | 说明 |
|------|------|
| patch_missing_plots 因果逻辑 | 对全量做因果估计，与主流程 B/C 分轴设计不同；若需补图，可后续按 B/C 分轴调用 |
| SHAP/临床模块 exclude | 硬编码，与 get_exclude_cols 基本一致；若 charls_feature_lists 新增泄露列，需同步更新 |
| CATE 可视化、验证套件 | 默认 treatment=is_depression，与主因果 exercise 不同；属不同分析，可保留 |

---

## 建议的 Methods 引用

- **人群**：基线年龄 ≥60 岁；见 `ANALYSIS_DEFINITIONS.md` §1
- **结局/暴露定义**：见 `ANALYSIS_DEFINITIONS.md` §2
- **截断值**：CES-D-10 ≥10，认知 ≤10；见 §3
- **因果假定**：见 §4
- **缺失处理**：见 §5

---

## 输出文件与论文对应

| 论文 Figure | 文件路径 |
|-------------|----------|
| Fig 1 流失图 | `LIU_JUE_STRATEGIC_SUMMARY/attrition_flow_diagram.png` |
| Fig 2 概念框架 | `LIU_JUE_STRATEGIC_SUMMARY/fig2_conceptual_framework.png` |
| Fig 3 ROC | `LIU_JUE_STRATEGIC_SUMMARY/fig3_roc_combined.png` |
| Fig 4 ATE | `LIU_JUE_STRATEGIC_SUMMARY/intervention_benefit_comparison.png` |
| Table 1 | `LIU_JUE_STRATEGIC_SUMMARY/table1_baseline_characteristics.csv` |
| 发病率图 | `LIU_JUE_STRATEGIC_SUMMARY/incidence_comparison.png` |
