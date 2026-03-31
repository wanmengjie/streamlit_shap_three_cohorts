# 论文写作视角：代码优化建议

**目的**：若由我来写这篇论文，会做哪些优化以消除冗余、合并逻辑、修复潜在矛盾与疏漏。

---

## 一、潜在矛盾（需优先修复）

### 1. Table 2 数据源与冠军逻辑不一致

| 问题 | 现状 | 建议 |
|------|------|------|
| **论文引用** | 所有 PAPER_*.md 均指向 `model_performance_full_*.csv` | 主流程已用 CPM 冠军，应引用 `table2_*_main_performance.csv` |
| **冠军定义** | compare_models 用 Recall≥0.05 + AUC；主流程用 CPM（AUC 排序） | 两套逻辑并存，`model_performance_full` 首行可能与实际冠军不符 |
| **consolidate** | 未将 CPM 的 table2 复制到 results/tables | 已实现：`table2_prediction_cohortA/B/C.csv` 并双写旧名 `table2_prediction_axis*` |

**影响**：写论文时若按文档引用 model_performance_full，会与 SHAP/临床评价实际使用的冠军不一致。

### 2. fig2c_comparison 与冠军不一致

- `fig2c_comparison_*.png` 由 compare_models 生成，按 compare_models 冠军排序
- 实际冠军为 CPM 冠军，两者可能不同
- **建议**：fig2c 改为用 CPM df_main 排序后绘制，或加注「展示 15 模型 AUC，冠军以 CPM 为准」

---

## 二、冗余可删/可并

### 1. 双份预测性能表

| 文件 | 来源 | 用途 |
|------|------|------|
| `model_performance_full_*.csv` | compare_models | 0.5 阈值、旧冠军逻辑 |
| `table2_*_main_performance.csv` | CPM | Youden 阈值、Bootstrap CI、CPM 冠军 |

**建议**：
- **主表**：以 CPM 的 table2 为准，论文 Table 2 引用此表
- **model_performance_full**：保留作审计/对比，但 consolidate 时以 CPM 表为主；或在 compare_models 中加参数，CPM 模式下不再写入此表，减少混淆

### 2. compare_models 内冠军选择逻辑

- 主流程已不用 compare_models 的冠军，仅用其 `_opt_threshold` 和 `perf_df`
- 但 `run_prediction_only`、`compare_auc_with_without_drop` 仍依赖该逻辑
- **建议**：保持现状，在 compare_models 注释中写明「主流程由 CPM 选冠军，此处逻辑供独立脚本使用」

### 3. 多处 try/except 静默跳过

- 各步骤失败时多为 `logger.warning` 或 `logger.debug`，无汇总
- **建议**：在 main 末尾收集 `skipped_steps` 列表并输出，便于检查漏跑步骤

---

## 三、consolidate 疏漏（论文直接引用）

### 当前未 consolidate 的论文用表

| 论文表 | 当前路径 | 建议 |
|--------|----------|------|
| **Table 2** | Cohort_*/01_prediction/table2_*_main_performance.csv | 复制到 `table2_prediction_cohortA/B/C.csv`（并双写 `table2_prediction_axis*`） |
| **Table 3** | Cohort_*/06_subgroup/subgroup_analysis_results.csv | 复制到 `table3_subgroup_cohortA/B/C.csv`（并双写 `table3_subgroup_axis*`） |
| **Table 5** | LIU_JUE_STRATEGIC_SUMMARY/sensitivity_summary.csv | 复制到 results/tables/table5_sensitivity_summary.csv |

### results/figures 为空

- 创建了 `results/figures` 目录，但未复制任何图形
- **建议**：增加核心图 consolidate，例如：
  - Figure 1: attrition_flow_diagram.png
  - Figure 2: fig2_conceptual_framework.png
  - Figure 3: fig3_roc_combined.png
  - Figure 4: Cohort_*/02_shap/fig5a_shap_summary_*.png
  - Figure 5: Cohort_*/04_eval/fig3_clinical_evaluation_comprehensive.png

---

## 四、文档与代码同步

### 需更新的文档

| 文档 | 需改内容 | 状态 |
|------|----------|------|
| PAPER_写作前最终检查清单 | Table 2 改为 table2_*_main_performance；冠军选择改为 CPM 逻辑 | ✅ 已实施 |
| PAPER_框架与图表索引_代码逻辑版 | Table 2 数据来源改为 CPM | ✅ 已实施 |
| PAPER_绝对路径索引 | 增加 table2_*_main_performance、Table 3/5 的 results 路径 | ✅ 已实施 |
| PAPER_DRAFT_* | Source 改为 CPM 表 | 待实施 |

### 方法学表述统一

- 论文 2.2 节应写：「冠军模型按 CPM 评估选取（验证集 Youden 最优阈值下 AUC 最高）」
- 不再写「Recall≥0.05 + 最高 AUC」

---

## 五、建议实施优先级

| 优先级 | 项目 | 工作量 | 状态 |
|--------|------|--------|------|
| **P0** | consolidate 增加 Table 2/3/5 的 CPM 与 subgroup、sensitivity 复制 | 小 | ✅ 已实施 |
| **P0** | 更新 PAPER_写作前最终检查清单 中 Table 2 与冠军逻辑 | 小 | ✅ 已实施 |
| **P1** | consolidate 增加核心 Figure 复制到 results/figures | 中 | ✅ 已实施 |
| **P1** | 在 main 末尾输出 skipped_steps 汇总 | 小 |
| **P2** | fig2c 改为基于 CPM 排序或加说明 | 小 |
| **P2** | 同步更新其余 PAPER_*.md | 中 |

---

## 六、总结

当前代码**可以**产出论文所需结果，但存在：

1. **数据源不一致**：文档指向 model_performance_full，实际应以 CPM table2 为准
2. **consolidate 不完整**：Table 2/3/5 和 Figures 未集中到 results/
3. **文档滞后**：PAPER_* 与当前实现不同步

按上述 P0 项修改后，论文写作与复现会更顺畅。
