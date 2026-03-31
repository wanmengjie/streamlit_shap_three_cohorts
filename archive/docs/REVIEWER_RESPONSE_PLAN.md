# 审稿意见落地实施计划

根据审稿人反馈，制定分阶段补充实验与修改计划。**已实现**项标注 ✓，**待实现**项标注 ○。

---

## 一、预测建模部分

### 1.1 外部验证 ○

| 建议 | 现状 | 实施 |
|------|------|------|
| 内部时间/分区域验证 | ✓ 已实现 | `charls_external_validation.py`：按 wave 划分时间验证、按东中西部划分区域验证 |
| 跨数据集验证（CLHLS） | 无 | 需 CLHLS 数据，可列为展望 |
| 校准曲线 | ✓ 已有 | `charls_clinical_evaluation.py` 已绘制 |
| 校准斜率、Brier 分解 | ✓ 已实现 | `charls_clinical_evaluation.py` 输出 calibration_brier_report.txt |

### 1.2 DCA ✓

- 已实现：`charls_clinical_evaluation.py` 含 DCA、校准曲线、PR 曲线
- 建议：确保三轴线均有输出，并补充最优阈值说明

### 1.3 模型复杂度与效率 ○

- 新增：记录冠军模型超参数、特征数、训练/预测耗时

---

## 二、因果推断部分

### 2.1 多种方法交叉验证 ○

| 方法 | 可行性 | 说明 |
|------|--------|------|
| PSM（倾向得分匹配） | 高 | 可用 `statsmodels` 或 `causalml` |
| PSW（倾向得分加权） | 高 | IPW 加权回归 |
| DID（双重差分） | 中 | 需多波次面板，CHARLS 可尝试 |
| Causal Forest DML | ✓ 已有 | 主方法 |

- 新增 `charls_causal_methods_comparison.py`：对运动、睡眠等 6 类可靠干预，用 PSM/PSW 重估 ATE，与 DML 对比

### 2.2 未测混杂敏感性 ✓ / ○

| 项目 | 现状 | 实施 |
|------|------|------|
| E-Value | ✓ 已有 | `charls_sensitivity_analysis.py` 已计算并保存 |
| 偏倚分析模型 | ○ 无 | 可补充 E-Value 曲线、模拟不同未测混杂强度 |

### 2.3 剂量反应与时序分析 ○

| 建议 | 实施 |
|------|------|
| 运动频率：0/1-3/≥4 次/周 | 需原始变量，若有则用 RCS |
| 睡眠：<5h/5-6h/≥6h | 将 sleep 分箱，RCS 或分段回归 |
| 时序分析 | 利用 wave 信息，分析暴露持续时间与结局 |

- 新增 `charls_dose_response.py`：RCS 剂量反应分析

### 2.4 低样本量暴露优化 ○

| 暴露 | 建议 |
|------|------|
| 社会隔离 | 合并亚组、精细定义、贝叶斯估计 |
| 慢性病负担低 | 同上 |

---

## 三、可解释性与亚组分析

### 3.1 分层 SHAP ○

- 按年龄（<65/65-75/≥75）、性别、城乡分层
- 各亚组绘制 SHAP 重要性图、依赖图
- 新增 `charls_shap_stratified.py`

### 3.2 SHAP 交互分析 ○

- 使用 `shap.TreeExplainer(...).shap_interaction_values()`
- 分析运动×睡眠、BMI×慢性病等交互
- 在 `charls_shap_analysis.py` 中扩展

### 3.3 亚组拓展 ○

- 增加教育、慢性病数量、自评健康等分层
- 亚组效应异质性检验（交互项）
- 在 `charls_subgroup_analysis.py` 中扩展

### 3.4 ITE 刻画与验证 ✓ / ○

| 项目 | 现状 | 实施 |
|------|------|------|
| ITE 分布 | ✓ 已有 | `visualize_causal_forest_concrete.py` |
| 分层验证（高/低 ITE） | ○ 无 | 按 ITE 分组，比较干预后风险 |
| 个体化列线图 | ○ 无 | 可补充 |

---

## 四、结果与讨论

### 4.1 轴线 C 预测性能分析 ○

- 对比 A/B/C 的 SHAP 特征排名与分布
- 结合机制讨论认知受损人群预测难度高的原因

### 4.2 运动在轴线 C 的 ATE 为正 ○

- 反向因果排查
- 按运动强度分层
- 补充认知相关协变量重估

### 4.3 临床转化方案 ○

- 基于 DCA 最优阈值制定筛查流程
- 分层干预建议（仅抑郁→运动；健康→睡眠/BMI）

---

## 五、实施优先级

| 优先级 | 任务 | 工作量 | 产出 |
|--------|------|--------|------|
| P0 | 内部时间/区域验证 | 中 | 验证集 AUC、校准 |
| P0 | 校准斜率、Brier 分解 | 低 | 方法补充 |
| P1 | PSM/PSW 因果交叉验证 | 中 | 多方法 ATE 对比表 |
| P1 | 分层 SHAP | 中 | 各亚组 SHAP 图 |
| P1 | E-Value 完善（ATE 尺度） | 低 | 更规范 E-Value 报告 |
| P2 | 剂量反应 RCS | 中 | 运动/睡眠剂量曲线 |
| P2 | 亚组拓展+异质性检验 | 中 | 扩展亚组表 |
| P2 | ITE 分层验证 | 中 | 高/低 ITE 组对比 |
| P3 | SHAP 交互、列线图、社会隔离优化 | 高 | 补充分析 |

---

## 六、已实现清单（可直接引用）

- ✓ DCA、校准曲线、PR 曲线（`charls_clinical_evaluation.py`）
- ✓ E-Value、Placebo Test（`charls_sensitivity_analysis.py`）
- ✓ ITE 分布、CATE 与年龄/城乡关系（`visualize_causal_forest_concrete.py`）
- ✓ 居住地、年龄、性别亚组 CATE（`charls_subgroup_analysis.py`）
- ✓ 截断值敏感性（`run_sensitivity_scenarios.py`）
- ✓ 7 类干预 ATE（`run_all_interventions_analysis.py`）

---

## 七、建议执行顺序

1. **第一阶段**：P0 任务（验证+校准），快速增强方法学完整性
2. **第二阶段**：P1 任务（因果交叉验证+分层 SHAP），提升论证强度
3. **第三阶段**：P2 任务（剂量反应+亚组+ITE 验证），深化分析
4. **第四阶段**：P3 任务及讨论部分修改，完善论文叙事
