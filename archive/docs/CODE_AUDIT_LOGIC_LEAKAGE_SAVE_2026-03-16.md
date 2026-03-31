# 逻辑 / 数据泄露 / 保存 全面审计 2026-03-16

---

## 一、数据泄露检查 ✅

| 模块 | 检查项 | 状态 |
|------|--------|------|
| **charls_model_comparison** | 先 split 再 fit；Imputer/Scaler 在 Pipeline 内，仅 CV 训练折 fit | ✅ 合规 |
| **charls_model_comparison** | roc_data.json 保存的是 X_test 的 y_true/y_prob | ✅ |
| **run_sensitivity_scenarios** | df_base 为主流程传入时，已为 _get_train_subset（80% 训练集） | ✅ |
| **charls_external_validation** | pipe.fit(X_train, y_train)，验证集仅 predict | ✅ |
| **charls_recalculate_causal_impact** | Imputer/Scaler 在全量 df 上 fit；因果推断无 train/test，DML 内部交叉拟合 | ✅ 符合因果惯例 |
| **LEAKAGE_KEYWORDS** | 已含 memory（非 memeory），排除 cesd/total_cog/cognition 等 | ✅ |
| **get_exclude_cols** | 目标、干预、causal_impact_*、泄露关键字均排除 | ✅ |

---

## 二、逻辑检查 ✅

| 模块 | 检查项 | 状态 |
|------|--------|------|
| **因果时序** | T/X 取自 Wave(t)，Y 取自 Wave(t+1)；preprocessing 显式校验 | ✅ |
| **插补与划分顺序** | 先插补（全量）→ 再划分（compare_models 内 80/20） | ✅ |
| **敏感性分析** | 9 种阈值 + 完整病例；df_base 为训练子集时与主流程一致 | ✅ |
| **临床决策支持** | sleep_adequate=1 反事实，与预处理一致 | ✅ |
| **剂量反应** | sleep 缺失 dropna 排除；exercise 二分类用定序 | ✅ |

### 2.1 临床评价使用全量数据（方法学说明）

- **现状**：`run_clinical_evaluation` 接收 `df_sub`（全轴线数据），DCA/校准/PR 曲线在 train+test 上计算。
- **影响**：主 AUC 来自测试集（perf_df），DCA/校准可能略偏乐观（含训练集）。
- **建议**：方法学中注明「DCA/校准基于开发集全量」；若需严格无偏，可后续改为仅用测试集。

---

## 三、保存检查

### 3.1 缺少 encoding='utf-8-sig'（易导致 Windows 中文乱码）

| 文件 | 行 | 建议 |
|------|-----|------|
| **data/charls_complete_preprocessing.py** | 188 | `encoding='utf-8'` → `encoding='utf-8-sig'` |
| archive/charls_variable_dictionary.py | 21 | 补 `encoding='utf-8-sig'` |
| archive/charls_validation_suite.py | 123 | 补 `encoding='utf-8-sig'` |

### 3.2 主流程相关 to_csv 已统一 utf-8-sig ✅

- charls_table1_stats、charls_model_comparison、charls_recalculate_causal_impact、run_sensitivity_scenarios、run_all_interventions_analysis 等均已使用 `encoding='utf-8-sig'`。

---

## 四、已确认无问题

- 预测模型 Pipeline 封装、CV 内 fit
- 敏感性分析训练子集隔离
- 因果 DML T 排除于 X、仅连续变量缩放
- 分类变量不缩放（CATEGORICAL_NO_SCALE）
- 流失图、剂量反应 readme 说明

---

## 五、修复优先级

| 优先级 | 项目 | 影响 |
|--------|------|------|
| P1 | CHARLS_final_preprocessed.csv encoding | Windows Excel 中文乱码 |
| P2 | 临床评价全量数据 | 方法学说明即可 |
| P3 | archive 脚本 encoding | 仅影响归档脚本 |
