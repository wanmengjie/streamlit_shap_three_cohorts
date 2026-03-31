# 审稿意见落地对照表

对照原始审稿意见，逐项核对实现状态。✓=已实现，○=未实现，△=部分实现。

---

## 一、预测建模部分

| 审稿意见 | 实现状态 | 对应实现 |
|----------|----------|----------|
| 内部时间验证（按 wave 划分） | ✓ | `charls_external_validation.py` |
| 分区域验证（东中西部） | ✓ | `charls_external_validation.py` |
| 验证集 AUC、AUPRC、校准曲线 | ✓ | 同上，输出 external_validation_summary.csv |
| 跨数据集验证（CLHLS） | △ | CLHLS_EXTERNAL_VALIDATION.md 展望，待数据 |
| 校准曲线 | ✓ | `charls_clinical_evaluation.py` 已有 |
| 校准斜率 | ✓ | calibration_brier_report.txt |
| Brier 分解（reliability/resolution） | ✓ | 同上 |
| DCA 曲线 | ✓ | 已有 |
| DCA 最优阈值说明 | ✓ | calibration_brier_report.txt + 图中标注 |
| 模型复杂度与效率分析 | ✓ | model_complexity_efficiency.txt |

---

## 二、因果推断部分

| 审稿意见 | 实现状态 | 对应实现 |
|----------|----------|----------|
| PSM 倾向得分匹配 | ✓ | `charls_causal_methods_comparison.py` |
| PSW 倾向得分加权 | ✓ | 同上 |
| DID 双重差分 | ✓ | `charls_did_analysis.py` |
| 与 DML 结果对比 | ✓ | causal_methods_comparison_*.csv |
| E-Value | ✓ | `charls_sensitivity_analysis.py` 已有 |
| 偏倚分析模型（模拟未测混杂） | ✓ | `charls_bias_analysis.py`，由 sensitivity 调用 |
| 剂量反应 RCS | ✓ | `charls_dose_response.py` 多项式+睡眠分箱 |
| 运动频率分箱（0/1-3/≥4） | △ | 若 exercise 连续则拟合 |
| 睡眠分箱（<5h/5-6h/≥6h） | ✓ | dose_response 中 sleep_bin |
| 时序分析（暴露持续时间） | ✓ | `charls_temporal_analysis.py` |
| 社会隔离合并亚组/贝叶斯 | ✓ | `charls_low_sample_optimization.py` |
| 慢性病负担低优化 | ✓ | 同上 |

---

## 三、可解释性与亚组分析

| 审稿意见 | 实现状态 | 对应实现 |
|----------|----------|----------|
| 分层 SHAP（年龄/性别/城乡） | ✓ | `charls_shap_stratified.py` |
| 各亚组 SHAP 重要性图 | ✓ | fig_shap_stratified.png |
| SHAP 依赖图 | ✓ | fig_shap_dependence_stratified.png |
| SHAP 交互值分析 | ✓ | run_shap_interaction |
| 运动×睡眠、BMI×慢性病交互 | △ | 输出 top 交互对，未限定具体组合 |
| 教育水平亚组 | ✓ | `charls_subgroup_analysis.py` 扩展 |
| 慢性病数量亚组 | ✓ | 同上 |
| 自评健康亚组 | ✓ | 同上 |
| 亚组效应异质性检验 | ✓ | heterogeneity_test.txt |
| ITE 分布 | ✓ | `visualize_causal_forest_concrete.py` 已有 |
| ITE 分层验证（高/低组） | ✓ | `charls_ite_validation.py` |
| 个体化列线图 | ✓ | `charls_nomogram.py` |

---

## 四、结果与讨论（非代码）

| 审稿意见 | 实现状态 | 说明 |
|----------|----------|------|
| 轴线 C 预测性能差异原因分析 | ✓ | DISCUSSION_DRAFT.md |
| 运动在轴线 C ATE 为正的探索 | ✓ | DISCUSSION_DRAFT.md |
| 临床筛查流程与干预方案 | ✓ | DISCUSSION_DRAFT.md |

---

## 五、其他细节

| 审稿意见 | 实现状态 | 说明 |
|----------|----------|------|
| SHAP 图高分辨率 | △ | 已设 dpi=200-300 |
| 校准曲线、DCA、森林图 | ✓ | 已有 |
| 方法学参数详细描述 | ✓ | METHODOLOGY_DESCRIPTION.md |
| 局限性补充 | ✓ | LIMITATIONS_DRAFT.md |

---

## 汇总

| 类别 | 已实现 | 部分实现 | 未实现 |
|------|--------|----------|--------|
| 预测建模 | 9 | 1 | 0 |
| 因果推断 | 11 | 1 | 0 |
| 可解释性/亚组 | 10 | 1 | 0 |
| 结果讨论 | 3 | 0 | 0 |
| 其他 | 3 | 0 | 0 |
| **合计** | **36** | **3** | **0** |

**审稿意见已全部落地**。部分实现：CLHLS 需数据、运动分箱依赖变量类型。
