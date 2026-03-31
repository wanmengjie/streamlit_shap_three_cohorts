# CHARLS 因果机器学习项目 - 代码审查报告

---

## 1. 问题汇总

### 高严重程度

| # | 问题 | 位置 | 说明 |
|---|------|------|------|
| 1 | **预处理缺少必要列时未提前返回** | `charls_complete_preprocessing.py` 第 64 行 | 若数据缺少 `cesd10` 或认知列，`is_depression`/`is_cognitive_impairment` 未创建，第 64 行 `df['is_comorbidity']` 会触发 `KeyError` |
| 2 | **estimate_causal_impact 失败时调用方未检查返回值** | `run_all_charls_analyses.py` 第 60 行 | 因果分析失败返回 `(None, (0,0,0))` 时，`res_df` 为 None，但后续 `run_subgroup_analysis(df_sub)` 仍使用原 `df_sub`，而 `df_sub` 未包含 `causal_impact_*` 列，亚组分析会报错或跳过。当前实现通过 `causal_col is not None` 检查可跳过，但 `df_sub` 在失败时未被更新，逻辑上不够清晰 |
| 3 | **硬编码文件路径** | 多处 | `CHARLS.csv`、`preprocessed_data/`、`LIU_JUE_STRATEGIC_SUMMARY` 等路径硬编码，不利于部署与测试 |

### 中严重程度

| # | 问题 | 位置 | 说明 |
|---|------|------|------|
| 4 | **pandas clip 用法** | `charls_complete_preprocessing.py` 第 50 行 | `df['income_total'].clip(lower=0)` 用法正确；若 `income_total` 含 NaN，`np.log1p` 会保留 NaN，建议在 clip 前 `fillna(0)` 或 `dropna` 以明确处理 |
| 5 | **多暴露循环中冗余逻辑** | `run_multi_exposure_causal.py` 第 59-64 行 | `if binarize_from: treatment_col = col else: treatment_col = col` 恒成立，可简化 |
| 6 | **日志文件未显式关闭** | `run_all_charls_analyses.py` 第 19 行 | `logging.FileHandler` 会随进程结束释放，但长时间运行或多次调用时，建议显式管理日志句柄 |
| 7 | **JSON 文件写入缺少编码** | `charls_model_comparison.py` 第 171 行 | `open(save_roc_path, 'w')` 未指定 `encoding='utf-8'`，若路径含中文或内容含非 ASCII 可能出问题 |
| 8 | **draw_attrition_flowchart 数值转换** | `draw_attrition_flowchart.py` 第 61 行 | `int(ns[i])` 在 `ns[i]` 为 `np.nan` 或非数字字符串时会抛错，应加 try/except 或 `pd.notna` 检查 |

### 低严重程度

| # | 问题 | 位置 | 说明 |
|---|------|------|------|
| 9 | **未使用的 import** | `run_multi_exposure_causal.py` 第 7 行 | `import sys` 未使用 |
| 10 | **magic number** | 多处 | 如 `random_state=500`、`n_bootstraps=500`、`MIN_SUBGROUP_N=100` 等，建议抽取为常量或配置 |
| 11 | **compare_age_auc 返回值不一致** | `compare_age_auc.py` 第 27 行 | 失败时返回 `(None, None, None, 0)`，成功时返回 `(float, float, float, int)`，类型不一致，调用方需处理 None |
| 12 | **异常捕获过于宽泛** | 多处 | `except Exception as e` 会捕获所有异常，可能掩盖严重错误，建议区分可恢复异常与应传播的异常 |

---

## 2. 修正建议

### 问题 1：预处理缺少列时提前返回

```python
# charls_complete_preprocessing.py，在第 64 行之前添加：
if 'is_depression' not in df.columns or 'is_cognitive_impairment' not in df.columns:
    logger.error("预处理失败：缺少 is_depression 或 is_cognitive_impairment，请检查数据是否含 cesd10 及认知列。")
    return None
```

### 问题 2：因果估计失败时的处理

当前 `run_axis_BC_protocol` 中 `df_sub` 在因果失败时不会更新，`run_subgroup_analysis` 会因找不到 `causal_impact_*` 而跳过。建议在因果失败时显式记录并跳过依赖因果的步骤：

```python
# run_all_charls_analyses.py run_axis_BC_protocol 中：
res_df, (ate, ate_lb, ate_ub) = estimate_causal_impact(...)
if res_df is None:
    logger.warning(f"轴线 {axis_label} 因果分析失败，跳过亚组分析。")
    causal_col = None
else:
    df_sub = res_df  # 使用更新后的 df
    causal_col = next((c for c in df_sub.columns if c.startswith('causal_impact_')), None)
```

### 问题 2 补充：estimate_causal_impact 失败时不应修改 df_sub

当前 `estimate_causal_impact` 失败时返回 `(None, (0,0,0))`，不修改传入的 `df_sub`，因此 `df_sub` 没有 `causal_impact_*`。`run_subgroup_analysis` 通过 `causal_col is not None` 检查会跳过，逻辑正确。问题 2 可降级为“建议在失败时显式打日志”，无需大改。

### 问题 4：income_total 含 NaN 时的处理（可选）

```python
# charls_complete_preprocessing.py 第 50-51 行，若需明确处理 NaN：
if 'income_total' in df.columns:
    df['income_total'] = np.log1p(df['income_total'].clip(lower=0).fillna(0))
```

### 问题 5：冗余逻辑简化

```python
# run_multi_exposure_causal.py 第 59-64 行，简化为：
for col, binarize_from in EXPOSURES:
    treatment_col = col
    if treatment_col not in df.columns:
        ...
```

### 问题 7：JSON 写入指定编码

```python
# charls_model_comparison.py 第 171 行：
with open(save_roc_path, 'w', encoding='utf-8') as f:
    json.dump({'y_true': y_test.tolist(), 'y_prob': y_prob.tolist()}, f)
```

### 问题 8：attrition 数值安全转换

```python
# draw_attrition_flowchart.py 第 61 行：
try:
    excluded = int(float(ns[i])) - int(float(ns[i+1])) if i+1 < len(ns) and pd.notna(ns[i]) and pd.notna(ns[i+1]) else None
except (ValueError, TypeError):
    excluded = None
```

### 问题 9：移除未使用 import

```python
# run_multi_exposure_causal.py 删除第 7 行：import sys
```

---

## 3. 整体评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 语法正确性 | 9/10 | 无明显语法错误，可正常运行 |
| 逻辑正确性 | 8/10 | 主流程清晰，少数边界情况未处理 |
| 代码规范 | 7/10 | 命名基本规范，部分 magic number 可抽取 |
| 性能 | 7/10 | 模型训练为主耗时，可接受；无明显冗余循环 |
| 安全性 | 8/10 | 本地数据处理，无 SQL/XSS；路径硬编码可改进 |
| 可维护性 | 7/10 | 模块划分清晰，`charls_feature_lists` 集中管理；配置分散 |
| 异常处理 | 6/10 | 有 try/except，但部分过于宽泛，错误信息可更具体 |

**综合评分：7.4/10**

**简短评价**：项目结构清晰，主流程与因果分析逻辑正确，适合科研复现。主要改进点在于：边界条件防护、配置集中化、异常处理细化，以及少量冗余代码清理。

---

## 4. 最佳实践建议

1. **配置集中化**：将 `age_min`、`cesd_cutoff`、`cognition_cutoff`、`random_state`、数据路径等抽到 `config.py` 或环境变量，便于复现与调参。

2. **预处理健壮性**：在 `preprocess_charls_data` 开头增加必要列检查（`cesd10`、认知列、`ID`、`wave`），缺失时尽早返回并打日志。

3. **异常分层**：区分“可恢复”（如单模型训练失败）与“应中断”（如数据缺失、配置错误），对后者使用 `raise` 或 `sys.exit(1)`，而不是静默跳过。

4. **文件 I/O 编码**：所有读写文本文件统一使用 `encoding='utf-8'`，避免跨平台编码问题。

5. **单元测试**：为核心模块（如 `preprocess_charls_data`、`get_exclude_cols`、`get_metrics_with_ci`）增加单元测试，覆盖空数据、缺失列、边界值等场景。
