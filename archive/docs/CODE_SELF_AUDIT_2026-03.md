# 代码自查报告 2026-03-16

基于 `.remember/memory/self.md` 与全库扫描，对可能存在的问题与设置进行自查与修正。

---

## 一、已修复问题

### 1. 插补敏感性分析数据源错误（逻辑闭环）

**问题**：主分析使用插补数据时，`run_imputation_sensitivity_preprocessed(df_clean)` 收到的是已插补数据，X 无缺失，五种插补方法比较无差异。

**修正**：`run_all_charls_analyses.py` 中，当 `USE_IMPUTED_DATA=True` 且插补文件存在时，单独调用 `preprocess_charls_data` 得到带缺失的 `df_for_imp_sens`，再传入插补敏感性分析；否则传 `df_clean`。

### 2. 临床决策支持反事实与管线干预不一致

**问题**：`charls_clinical_decision_support.py` 反事实场景 B/C 使用 `sleep=7.5`，但预处理已移除 `sleep`、仅保留 `sleep_adequate`，模型特征中无 `sleep`，反事实无效。

**修正**：改为使用 `sleep_adequate=1` 表示「改善睡眠」，与预处理及因果分析一致。

### 3. to_csv 缺 encoding 导致 Windows 中文乱码

**修正**：为以下主流程相关脚本的 `to_csv` 补充 `encoding='utf-8-sig'`：
- `charls_dose_response.py`（dose_response_summary.csv）
- `charls_bias_analysis.py`（bias_sensitivity.csv）
- `charls_low_sample_optimization.py`
- `charls_cognitive_subdomain_analysis.py`
- `charls_causal_refinement.py`
- `charls_did_analysis.py`
- `charls_policy_forest_plot.py`
- `charls_cate_visualization.py`（cate_shap_values.csv）

---

## 二、已确认无问题（与 self.md 一致）

| 项目 | 状态 |
|------|------|
| confusion_matrix ravel 单类防护 | 已有 2×2 检查 |
| np.percentile 空列表防护 | 已有 len<10 防护 |
| table1 categorical astype(int) | 已有 try/except |
| LEAKAGE_KEYWORDS 拼写 | 已为 memory，无 memeory |
| USE_IMPUTED_DATA 路径不存在 | 已回退到 preprocess |
| CONTINUOUS_FOR_SCALING | 已排除 edu/gender，新增 CATEGORICAL_NO_SCALE + assert |
| 剂量反应 sleep 缺失 | 已改为 dropna 排除 + 单独标记 |

---

## 三、已知设计约定（无需修改）

1. **因果模块 Y**：硬编码为 `is_comorbidity_next`；若改 `TARGET_COL` 须同步改 `charls_recalculate_causal_impact`。
2. **exercise.fillna(0)**：主分析仍使用，Methods 中需说明；敏感性分析有完整病例（`restrict_complete_case=True`）。
3. **预处理 sleep**：移除连续变量 `sleep`，仅保留二分类 `sleep_adequate`；剂量反应在插补数据下若有 `sleep` 列则做 RCS。
4. **config.USE_IMPUTED_DATA=True**：需先运行插补脚本生成 `step1_imputed_full.csv`，否则自动回退到预处理数据。

---

## 四、建议关注（非必须修改）

| 项目 | 说明 |
|------|------|
| archive 下脚本 to_csv | 部分缺 encoding，若输出含中文可补 `encoding='utf-8-sig'` |
| external/clhls_*.py | 外部验证脚本，to_csv 可统一补 encoding |

---

## 五、本次修正文件清单

- `run_all_charls_analyses.py`：插补敏感性数据源
- `evaluation/charls_clinical_decision_support.py`：sleep_adequate 反事实
- `evaluation/charls_dose_response.py`：to_csv encoding
- `utils/charls_bias_analysis.py`：to_csv encoding
- `causal/charls_low_sample_optimization.py`：to_csv encoding
- `causal/charls_cognitive_subdomain_analysis.py`：to_csv encoding
- `causal/charls_causal_refinement.py`：to_csv encoding
- `charls_did_analysis.py`：to_csv encoding
- `charls_policy_forest_plot.py`：to_csv encoding
- `charls_cate_visualization.py`：to_csv encoding
