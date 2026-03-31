# 代码检查报告

**检查时间**：2026-03-03  
**范围**：主流程、预处理、因果分析、模型比较、临床决策支持、图表生成

---

## 1. 已修复问题

### 1.1 临床决策支持 `is_socially_isolated` 缺失

**问题**：部分病例生成决策支持图时报错 `columns are missing: {'is_socially_isolated'}`。  
**原因**：模型训练时使用了 `is_socially_isolated`，但构建预测输入时若 df 缺少该列会漏传。  
**修复**：在 `charls_clinical_decision_support.py` 中：
- 从模型 preprocessor 提取期望列，识别 `missing_for_model`
- 对 df 中缺失的列用 0 填充
- 使用 `pd.api.types.is_numeric_dtype` 替代固定 dtype 判断

---

## 2. 当前逻辑确认

### 2.1 预处理（首次发病队列）

- 年龄 ≥60、CES-D/认知非缺失、下一波结局非缺失
- **首次发病**：任一波次发生共病后，该个体后续波次全部排除
- 流失表：`preprocessed_data/attrition_flow.csv`（最后一步：14,386）

### 2.2 主流程 `run_all_charls_analyses.py`

- 三轴线 A/B/C 流程正确
- 因果分析使用 `groups=cluster_ids`（communityID 或 ID）
- ROC 数据保存路径：A 用 01_prediction，B/C 用 02_prediction

### 2.3 因果分析 `charls_recalculate_causal_impact.py`

- 使用 `get_exclude_cols` 排除泄露列
- 干预变量缺失按 0 处理
- 聚类 ID 使用正确

### 2.4 特征排除 `charls_feature_lists.py`

- `exercise`、`sleep_adequate`、`exercise_sleep_both` 时排除 `lifestyle_active_sleep`
- `sleep_adequate` 时排除 `sleep`
- `exercise_sleep_both` 时排除 `exercise`、`sleep_adequate`、`sleep`

---

## 3. 注意事项（非错误）

### 3.1 样本量更新

采用首次发病设计后，样本量由 15,024 变为 14,386。论文草稿 `PAPER_DRAFT_完整版.md` 中 Table 1、Results、Abstract 等处的样本量需在重新运行全流程后更新。

### 3.2 流失流程图

新步骤名称 `"Baseline free of comorbidity, first-ever incident only"` 较长，流程图会自动换行，可正常显示。

### 3.3 临床决策支持

反事实场景为「增加运动」「改善睡眠」「综合」，与主分析干预变量一致。

---

## 4. 建议后续操作

1. **重新运行全流程**：`python run_all_charls_analyses.py`，以得到基于 14,386 样本的完整结果。
2. **更新论文**：用新生成的 Table 1、发病率、ATE 等更新 `PAPER_DRAFT_完整版.md`。
3. **样本量/效能**：运行 `python sample_size_power_analysis.py`，用新样本量更新 Supplementary Table S4。
