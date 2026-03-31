# CHARLS 老年抑郁-认知共病研究 系统性代码审计报告

**审计日期**：2025年3月  
**审计范围**：数据处理→预测建模→因果推断→敏感性分析→代码规范性  
**核心定义**：抑郁=CES-D-10≥10，认知受损=认知总分≤10，共病=抑郁+认知受损，入射队列 n=14,386（A=8,828, B=3,123, C=2,435）

---

## 一、数据处理层

### 1.1 队列筛选逻辑 ✅ 已正确实现

| 检查项 | 代码实现 | 论文/要求 | 结论 |
|--------|----------|-----------|------|
| 基线无共病 | `df['is_comorbidity']==0` | ✓ | 一致 |
| 既往无共病 | `had_comorbidity_before==0`（cummax 追溯） | ✓ | 一致 |
| 下一波共病非缺失 | `dropna(subset=['is_comorbidity_next'])` | ✓ | 一致 |
| 紧邻下一波 | `mask_valid = (next_wave_val == wave + 1)` | ✓ | 一致 |

**流失流程与表1一致**：`attrition_flow.csv` 输出 Raw 96,628 → Age≥60 49,015 → CES-D 43,048 → Cognition 31,574 → Next-wave 16,983 → Incident 14,386。

### 1.2 变量定义 ✅ 正确

| 变量 | 代码 | 要求 | 结论 |
|------|------|------|------|
| 抑郁 | `cesd10 >= cesd_cutoff`（默认10） | CES-D≥10 | ✓ |
| 认知受损 | `total_cognition <= cognition_cutoff`（默认10） | 认知≤10 | ✓ |
| 充足睡眠 | `sleep >= 6`（run_multi_exposure/prepare_interventions） | ≥6h | ✓ |
| BMI正常 | `18.5 <= bmi <= 24` | 18.5–24 | ✓ |
| 慢性病负担低 | `chronic_burden <= 1` | ≤1 | ✓ |

### 1.3 缺失值处理

| 问题 | 等级 | 表现 | 原因 | 修正建议 |
|------|------|------|------|----------|
| 运动缺失纳入主分析 | **低危** | 运动变量缺失率 43.42%，主分析用 `SimpleImputer(median)` 插补 | 主分析未排除运动缺失样本 | 敏感性分析已有 `restrict_complete_case=True` 的完整病例场景；建议在方法部分明确说明主分析采用中位数插补 |
| 缺失处理方式未集中标注 | **低危** | 各模块分散使用 median 插补 | 无统一文档 | 在 `charls_complete_preprocessing.py` 或 README 中集中说明：主分析=中位数插补，完整病例=敏感性分析 |

### 1.4 分组合理性 ✅ 正确

- 三轴线互斥：`baseline_group` 0/1/2 由 `is_depression` 与 `is_cognitive_impairment` 组合定义，无重叠。
- 样本量与表2一致：A=8,828, B=3,123, C=2,435（由 `generate_baseline_table` 输出验证）。

---

## 二、预测建模层

### 2.1 交叉验证 ✅ 基本正确

| 检查项 | 代码 | 要求 | 结论 |
|--------|------|------|------|
| 分组划分 | `GroupShuffleSplit(..., groups=df['ID'])` | 按 ID 分组 | ✓ |
| 训练/测试比例 | `test_size=0.2` | 8:2 | ✓ |
| 超参搜索 CV | `GroupKFold(n_splits=5)` + `groups=df.iloc[train_idx]['ID']` | 5折分组 | ✓ |

### 2.2 概率校准存在潜在泄露 ⚠️

| 问题 | 等级 | 表现 | 原因 | 修正建议 |
|------|------|------|------|----------|
| CalibratedClassifierCV 未用 GroupKFold | **高危** | `CalibratedClassifierCV(best_pipe, cv=3, method='isotonic')` 默认 KFold | 同一患者不同波次可能分入不同折，存在组内泄露 | 改为 `cv=GroupKFold(n_splits=3)` 并传入 `groups=df.iloc[train_idx]['ID']`，或在校准阶段显式按 ID 分组 |

### 2.3 指标计算 ✅ 正确

- AUC/AUPRC/Accuracy/F1：`charls_ci_utils.get_metrics_with_ci` 正确计算。
- Bootstrap 500 次、95% CI：`n_bootstraps=500`，`np.percentile(arr, [2.5, 97.5])`。
- 类别不平衡：`class_weight='balanced'`、`scale_pos_weight` 已使用；论文已说明“全负预测”时以 AUC 为主。

### 2.4 模型选择 ✅ 正确

- 按 AUC 排序：`perf_df.sort_values('AUC', ascending=False)`。
- 冠军模型选取：`perf_df.iloc[0]`，排除高 Accuracy 但 F1=0 的误导。

### 2.5 超参数与附录 S8

- 15 种模型均有扩展搜索空间，与论文描述一致；若附录 S8 有具体范围，需逐项核对。

---

## 三、因果推断层

### 3.1 Causal Forest DML ✅ 正确

| 检查项 | 代码 | 要求 | 结论 |
|--------|------|------|------|
| 5 折交叉拟合 | `cv=5` | 5 折 | ✓ |
| 决策树数量 | `n_estimators=1000` | 1000 | ✓ |
| 聚类稳健 | `groups=cluster_ids`（communityID 或 ID） | 考虑组内相关 | ✓ |
| ATE 95% CI | `dml.ate_interval(X_scaled)` | 正确 | ✓ |
| 可靠性判定 | 论文中 ATE∈[-1,1] | 在 run_all_interventions 等模块中实现 | ✓ |

### 3.2 PSM ✅ 正确

| 检查项 | 代码 | 要求 | 结论 |
|--------|------|------|------|
| 1:1 最近邻 | `NearestNeighbors(n_neighbors=1)` | 1:1 | ✓ |
| 卡尺 | `caliper=0.2`，`caliper_val = 0.2 * sd_ps` | 0.2×PS 标准差 | ✓ |
| SMD 验证 | 未实现 | 匹配后 SMD<0.1 | **建议补充**：匹配后计算 SMD，若不满足可报告 |

### 3.3 PSW ⚠️ 与任务描述略有差异

| 检查项 | 代码 | 任务要求 | 论文/附录 |
|--------|------|----------|-----------|
| PS 截断 | `np.clip(ps, 0.01, 0.99)` | — | 一致 |
| 权重截断 | `np.clip(w, 0.1, 50)` | “trim=1%” | 附录 S7 写为“权重限制在 [0.1, 50]” |

**结论**：代码与论文一致，采用固定 clip(0.1, 50)，未做 1% 分位数 trim；若审稿要求 1% trim，需单独实现并对比。

### 3.4 结果数值一致性

| 轴线 | 干预 | 方法 | table7 值 | 论文表4/7 | 备注 |
|------|------|------|------------|-----------|------|
| B | 运动 | PSM | -0.028 | -0.036 | 存在差异，可能因随机种子或数据版本 |
| B | 运动 | PSW | -0.034 | -0.034 | 一致 |
| B | 运动 | DML | -0.037 | -0.037 | 一致 |

**建议**：固定 `RANDOM_SEED` 后重跑全流程，确认 PSM 数值可复现；若仍与论文不符，需核对数据版本与预处理步骤。

---

## 四、敏感性分析层

### 4.1 截断值敏感性 ⚠️ 未遍历全部组合

| 问题 | 等级 | 表现 | 原因 | 修正建议 |
|------|------|------|------|----------|
| 未遍历 9 种组合 | **中危** | 仅 (10,10),(8,10),(12,10),(10,8),(10,12) 共 5 种 | `cutpoint_scenarios` 未包含 (8,8),(8,12),(12,8),(12,12) | 补充 4 种组合，或明确说明仅做单因素敏感性 |

```python
# 建议补充
(8, 8, 'CES-D≥8, Cog≤8'),
(8, 12, 'CES-D≥8, Cog≤12'),
(12, 8, 'CES-D≥12, Cog≤8'),
(12, 12, 'CES-D≥12, Cog≤12'),
```

### 4.2 完整病例分析 ✅ 正确

- `restrict_complete_case=True` 时执行 `df_sub.dropna(subset=[treatment_col])`，运动缺失被排除。

### 4.3 偏倚分析 ⚠️ 模型简化

| 问题 | 等级 | 表现 | 原因 | 修正建议 |
|------|------|------|------|----------|
| 使用线性回归近似 | **低危** | `LinearRegression().fit(..., Y_sim)` | 未测混杂通过线性模型近似 | 论文已说明为简化模拟；若需更严谨，可考虑 DML 或工具变量框架下的偏倚分析 |
| “强度=0.3 时 ATE 逆转” | — | `confounder_strengths=[0,0.1,0.2,0.3,0.5]` | 需根据实际输出判断 | 运行后检查 `bias_sensitivity.csv`，确认 0.3 时是否发生方向逆转 |

### 4.4 随机种子未统一

| 问题 | 等级 | 表现 | 修正建议 |
|------|------|------|----------|
| run_sensitivity_scenarios 硬编码 | **低危** | `random.seed(500)`, `np.random.seed(500)` | 改为 `from config import RANDOM_SEED` 并统一使用 |
| charls_bias_analysis 硬编码 | **低危** | `np.random.seed(500)` | 同上 |

---

## 五、代码规范性

### 5.1 数据泄露防护 ✅ 已落实

| 检查项 | 实现 | 结论 |
|--------|------|------|
| 认知/抑郁定义变量排除 | `LEAKAGE_KEYWORDS` 含 cesd/cognition/memory/executive | ✓ |
| 结局变量排除 | `is_comorbidity_next` 在 `EXCLUDE_COLS_BASE` | ✓ |
| 认知细分指标 | `memory`、`executive` 在 LEAKAGE_KEYWORDS 中 | ✓ |

### 5.2 可复现性 ✅ 基本满足

- `RANDOM_SEED=500` 已在 config 中定义，多数模块已引用。
- 需统一：`run_sensitivity_scenarios`、`charls_bias_analysis` 仍硬编码 500。

### 5.3 异常处理 ✅ 基本完善

- 空样本：`len(X)<30`、`df['ID'].nunique()<5` 等有检查。
- 单类别：`charls_ci_utils` 对 `confusion_matrix` 形状有检查。
- 日志：关键步骤有 `logger.info` 输出。

---

## 六、核心数值验证

| 指标 | 代码输出 | 论文表 | 一致性 |
|------|----------|--------|--------|
| 入射队列 n | 14,386 | 14,386 | ✓ |
| 轴线 A n | 8,828 | 8,828 | ✓ |
| 轴线 B n | 3,123 | 3,123 | ✓ |
| 轴线 C n | 2,435 | 2,435 | ✓ |
| 轴线 A 共病率 | 4.1% | 4.1% | ✓ |
| 轴线 B 共病率 | 13.6% | 13.6% | ✓ |
| 轴线 C 共病率 | 16.9% | 16.9% | ✓ |
| 轴线 A 冠军 MLP AUC | 0.7420 | 0.7420 | ✓ |
| 轴线 B 冠军 SVM AUC | 0.6834 | 0.6834 | ✓ |
| 轴线 C 冠军 ExtraTrees AUC | 0.6386 | 0.6386 | ✓ |
| 轴线 B 运动 DML ATE | -0.037 | -0.037 | ✓ |
| 轴线 B 运动 PSM ATE | -0.028 | -0.036 | ⚠️ 有差异 |

---

## 七、问题汇总（按等级）

### 致命（P0）
- 无。

### 高危（P1）— 已修复
1. ~~**CalibratedClassifierCV 未用 GroupKFold**~~：已改为使用 `GroupKFold(n_splits=3)` 预生成分组折，传入 `CalibratedClassifierCV(cv=cv_splits)`。

### 中危（P2）— 已修复
1. ~~**截断值敏感性未覆盖 9 种组合**~~：已补全 (8,8),(8,12),(12,8),(12,12) 共 9 种组合。
2. ~~**PSM 匹配后未验证 SMD**~~：已增加 `_compute_smd`，匹配后计算 max_SMD 并写入结果 CSV，日志输出平衡提示。
3. **PSM 数值与论文不一致**：已为 PSM/PSW 的 LogisticRegression 增加 `random_state=RANDOM_SEED`，重跑后可验证复现性。

### 低危（P3）— 已修复
1. 运动缺失主分析采用插补，需在方法中明确说明。
2. ~~**敏感性分析、偏倚分析中随机种子未统一**~~：已改为使用 `config.RANDOM_SEED`。
3. PSW 采用 clip(0.1, 50) 而非 1% trim，与论文一致，若审稿要求 1% trim 需单独实现。

---

## 八、代码整体可用性评分

| 维度 | 得分 | 说明 |
|------|------|------|
| 数据处理 | 9/10 | 队列筛选、变量定义正确，缺失处理有说明空间 |
| 预测建模 | 8/10 | GroupKFold 正确，校准阶段存在分组泄露风险 |
| 因果推断 | 8.5/10 | DML/PSM/PSW 逻辑正确，缺 SMD 验证，PSM 数值有差异 |
| 敏感性分析 | 7.5/10 | 截断值组合不全，偏倚分析为简化模型 |
| 规范性 | 8.5/10 | 泄露防护到位，随机种子部分未统一 |

**综合评分：8.3/10**

---

## 九、优化建议（优先级排序）

1. **高**：将 `CalibratedClassifierCV` 改为使用 `GroupKFold`，或在校准说明中明确其局限性。
2. **高**：固定随机种子后重跑全流程，核对 PSM 与论文的差异来源。
3. **中**：补全截断值敏感性 9 种组合，或在方法中说明仅做单因素敏感性。
4. **中**：在 PSM 流程中增加匹配后 SMD 计算与报告。
5. **低**：统一所有模块的 `RANDOM_SEED` 引用。
6. **低**：在方法部分集中说明缺失值处理策略（主分析 vs 完整病例）。
