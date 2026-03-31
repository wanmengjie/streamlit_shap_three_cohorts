# 多重检验 / FDR 与代码对齐说明

## 代码在做什么

1. **`scripts/run_xlearner_all_interventions.py`**（约 242–248 行）在写出 `xlearner_all_interventions_summary.csv` 后调用：
   - `from utils.multiplicity_correction import add_multiplicity_columns`
   - 对当前 `summary` 表的每一行（含 **XLearner / PSM / PSW** 等不同 `method`、不同 **`cohort`（队列，旧表可能为列名 `axis`）**、不同 `exposure`）在列 `ate`, `ate_lb`, `ate_ub` 上计算：
     - `p_value_approx`：由 ATE 与 95% CI 用正态近似反推的双侧 *p*（见 `ci_to_pvalue`）。
     - `p_adj_bonferroni`：`min(p × n, 1)`，其中 *n* 为表中非缺失 `p_value_approx` 的行数。
     - `p_adj_fdr`：基于 `p_value_approx` 秩的 **BH 型**调整（实现见 `apply_bonferroni_fdr`；若需与 `statsmodels` 标准 `multipletests(method='fdr_bh')` 逐行完全一致，可再核对）。
     - `significant_95` / `significant_bonferroni` / `significant_fdr`：与 0.05 比较的示性列。

2. **与「5 干预 × 3 队列 = 15 个主假设」的关系**  
   - 汇总表里通常 **远多于 15 行**（同一干预-队列会拆成 XLearner、PSM、PSW 多行），因此 **FDR 的名义检验数 = 该 CSV 行数（有近似 *p* 的）**，不是严格的「仅 15 个 XLearner」。

## 论文应如何表述

- **主文 *P*（如 *P* = 0.019）**：来自主分析（如 XLearner + bootstrap）的原始报告，**不是** `p_adj_fdr`。
- **若要报告多重校正**：建议写成「我们在补充材料 / 输出 `xlearner_all_interventions_summary.csv` 中报告了基于 CI 近似 *p* 的 Bonferroni 与 FDR 调整列，供审查；**主要推断仍以 95% CI 与预设的主分析 *P* 为准**。」

## 相关文件

- `utils/multiplicity_correction.py`
- 运行输出（若已跑）：`LIU_JUE_STRATEGIC_SUMMARY/xlearner_all_interventions/xlearner_all_interventions_summary.csv`

## 英文补充材料定稿位置（期刊投稿用）

- **`PAPER_Manuscript_Submission_Ready.md`**：**Supplementary Text S3**（目录与正文内嵌）。
- **`PAPER_完整版_2026-03-20.md`**：同上 **Text S3**，与 Brief 版对齐。
- **`PAPER_Supplementary_Materials_Detailed.md`**：目录增加 **Text S3** 条目与摘要说明。
