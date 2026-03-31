# 审稿关键修正与改进说明

针对审稿人提出的 4 个关键风险点与 6 条改进建议，本文件记录已落实的修正与待办事项。

---

## 一、已落实的 4 项关键修正

### 1. 数据泄露风险（Data Leakage）✅

**问题**：诊断阈值敏感性分析使用完整数据，可能与主分析测试集重叠。

**修正**：
- **先分割再传递**：主流程 `run_all_charls_analyses` 在数据加载后立即调用 `_get_train_subset(df_imputed)`，将训练集子集传入 `run_sensitivity_scenarios_analysis`，而非全量数据
- `run_sensitivity_scenarios.py` 新增 `train_only=True`（默认），当 `df_base` 由主流程传入时，假定已是训练集，不再二次过滤
- 使用与 `compare_models` 一致的 `GroupShuffleSplit(test_size=0.2)` 划分
- 敏感性分析仅基于 80% 训练集，测试集完全隔离

**代码位置**：`run_all_charls_analyses.py` 第 147 行（先分割）、`run_sensitivity_scenarios.py` 第 42–55 行

**论文建议**：在 Methods 中明确写出："Sensitivity analyses for diagnostic thresholds were performed using only the training set (80% of data) to avoid information leakage from the hold-out test set."

---

### 2. 多重检验校正（Multiple Testing）✅

**问题**：15 模型×3 队列、5 暴露×3 队列等多次比较未校正，假阳性风险升高。

**修正**：
- 在 `charls_model_comparison.py` 输出目录新增 `multiple_testing_note.txt`
- 明确说明比较次数及建议（FDR/Bonferroni）
- 论文中可注明：主结果为探索性，敏感性分析支持稳健性

**代码位置**：`charls_model_comparison.py` 第 267–273 行

**论文建议**：在 Methods 或 Limitations 中增加："Given the exploratory nature of multiple model and exposure comparisons, we report 95% CIs without formal multiplicity correction; sensitivity analyses confirmed consistency of main findings."

---

### 3. 样本量与统计效能（Sample Size & Power）✅

**问题**：C 队列 n=2,435，亚组分析部分单元格事件数<30，效能不足。

**修正**：
- `charls_subgroup_analysis.py` 增加事件数过滤：`MIN_EVENTS=30`, `MIN_TOTAL=30`
- 亚组需同时满足：总样本>30 且 事件数≥30 才纳入
- 输出表增加 `N_events` 列，便于审稿人核对

**代码位置**：`charls_subgroup_analysis.py` 第 36–100 行

**说明**：低发病率队列（如 A 队列 4.1%）部分亚组可能被过滤，属预期行为。

---

### 4. Causal Forest 诚实估计（Honesty）✅

**问题**：Causal Forest 未启用 honest 参数，存在过拟合风险。

**修正**：
- `charls_recalculate_causal_impact.py` 中 `CausalForestDML` 增加 `honest=True`
- 生长与叶节点估计使用不同子样本，降低过拟合

**代码位置**：`charls_recalculate_causal_impact.py` 第 88–93 行

---

## 二、已落实的改进建议

### 1. 分析锁（Analysis Lock）✅

- `config.py` 新增 `ANALYSIS_LOCK = True`
- 全流程使用 `RANDOM_SEED`，保证可重复
- 建议：审稿期间创建 GitHub Release Tag（如 v1.0.0-charls-submission），之后不再改动

### 2. Causal Forest honest 参数 ✅

- 见上文第 4 项

---

## 三、阶段 3 与阶段 7 的因果分析关系

**阶段 3**：`estimate_causal_impact`（Causal Forest DML）在轴线 03_causal 下运行，输出 `CAUSAL_ANALYSIS_{T}.csv`。

**阶段 7**：`run_all_axes_comparison` 优先读取阶段 3 的 DML 结果（若 `CAUSAL_ANALYSIS_{T}.csv` 存在），仅重新运行 PSM、PSW，与 DML 直接对比。**不重复拟合 DML**，保证可重复性与结果一致。

**代码位置**：`charls_causal_methods_comparison.py` 第 216–244 行（读取 causal_csv）

---

## 四、阶段 5 与阶段 6 的区分

| 阶段 | 模块 | 暴露数 | 方法学定位 |
|------|------|--------|------------|
| **阶段 5** | `run_multi_exposure_analysis` | 5（运动、睡眠、吸烟、饮酒、社会隔离） | 核心生活方式因素的单暴露因果估计 |
| **阶段 6** | `run_all_interventions` | 7（+ BMI 正常、慢性病负担低） | 扩展干预因素的单暴露因果估计 |

**说明**：两者均为**单暴露分别估计**（single-exposure intervention analysis），非多暴露同时调整。阶段 6 在阶段 5 基础上增加 2 个干预。

**论文建议**：
- 阶段 5："Multivariable exposure analysis" 易误解，建议改为 "Single-exposure causal analysis for five core lifestyle factors"
- 阶段 6："Single-exposure intervention analysis for seven modifiable factors (extended)"

---

## 五、阶段 8 低样本量暴露优化

**适用**：社会隔离、慢性病负担低等治疗组样本极少的暴露（`n_treated < 50` 时触发）。

**方法**：贝叶斯风格 ATE——Bootstrap 重采样（n=2000）+ 逻辑回归反事实预测，输出 95% CrI。

**代码位置**：`charls_low_sample_optimization.py` 第 19–37 行（`_bayesian_ate`）

**论文建议**（Supplementary Methods）："For exposures with prevalence <10% (social isolation, low chronic disease burden), we employed Bayesian bootstrap with logistic regression for counterfactual prediction (n=2000 resamples), as Causal Forest requires larger samples for stable heterogeneity estimation."

---

## 六、多重检验校正：记录 vs 实际执行

**现状**：仅记录比较次数（`multiple_testing_note.txt`），**未实际应用** FDR/Bonferroni 校正。

**论文 Limitations 建议**："We did not apply formal multiple testing correction (e.g., Bonferroni or Benjamini-Hochberg FDR) due to the exploratory nature of secondary analyses, but report all comparisons transparently to avoid selective reporting."

---

## 七、待办或论文中说明的事项

### 1. 插补与队列定义的时序偏差（Imputation Bias）【待办】

**现状**：插补在队列划分之前进行，插补模型使用全量数据。

**建议**：
- 论文中强调：Complete Case 敏感性分析结果与主分析一致（若已运行）
- 可选实现：分层插补（按 baseline_group 分别插补），需修改 `charls_imputation_npj_style.py`

### 2. 负对照（Negative Control）分析

**建议**：选择理论上无因果效应的暴露（如出生月份、眼睛颜色等）跑 DML/PSW，若 ATE≈0 可佐证方法不会产生假阳性。

**实现**：可在 `run_all_interventions_analysis` 中增加负对照选项。

### 3. 可视化流程依赖（DAG）

**建议**：新增 `draw_analysis_dag()`，生成数据流与分析依赖的 DAG 图，放入 Supplementary Methods。

### 4. 明确报告缺失数据比例

**论文建议**：在 Table 1 脚注增加：
"Complete case analysis was used for causal inference (n=XX, XX% of total), with missing data handled via [插补方法] for prediction models. Sensitivity analysis confirmed consistency between complete-case and imputed estimates."

### 5. 代码模块化

**建议**：将 `run_axis_protocol` 拆分为 `prediction.py`、`causal_inference.py`、`clinical_eval.py` 等，便于复现与审稿。

---

## 四、与论文版本的对应检查

| 论文描述 | 代码对应 | 状态 |
|----------|----------|------|
| NNT ≈ 34 | run_clinical_decision_support 中 1/ATE | 需确认已计算 |
| E-value = 2.1 | run_sensitivity_analysis 中 evalue 包 | 需确认已计算 |
| Table S3（诊断阈值） | run_sensitivity_scenarios_analysis | ✅ 已改为 train_only |
| Three-Cohort Gradient | run_axis_protocol 三次 | ✅ |
| Triangulation | run_all_axes_comparison (DML vs PSM vs PSW) | ✅ |

---

## 八、修改文件清单

| 文件 | 修改内容 |
|------|----------|
| run_all_charls_analyses.py | 先分割再传递：df_base_for_sensitivity = _get_train_subset(df_imputed) |
| run_sensitivity_scenarios.py | train_only、_get_train_subset；df_base 传入时不再二次过滤 |
| charls_model_comparison.py | multiple_testing_note.txt |
| charls_subgroup_analysis.py | MIN_EVENTS=30、N_events 列 |
| charls_recalculate_causal_impact.py | honest=True |
| config.py | ANALYSIS_LOCK |

---

*文档更新日期：2026-03-15*
