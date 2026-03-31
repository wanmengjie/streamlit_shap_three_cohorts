# 分析代码潜在问题检查报告 2026-03-16

基于主流程 `run_all_charls_analyses.py` 及关联模块的全面检查。

---

## 一、已确认无问题

| 模块 | 检查项 | 状态 |
|------|--------|------|
| 数据加载 | USE_IMPUTED_DATA + 路径不存在时回退预处理 | ✅ |
| 插补敏感性 | 主分析用插补时单独 preprocess 得带缺失 df | ✅ |
| 临床决策支持 | sleep_adequate 反事实与管线一致 | ✅ |
| 因果 DML | T 排除于 X，仅连续变量 StandardScaler | ✅ |
| 分类变量 | CATEGORICAL_NO_SCALE，edu/gender 不缩放 | ✅ |
| 剂量反应 | sleep 缺失 dropna 排除，exercise 二分类定序 | ✅ |
| 敏感性分析 | restrict_complete_case 时 dropna(干预) | ✅ |
| 预测模型 | Imputer/Scaler 封装在 Pipeline 内 | ✅ |
| ROC 路径 | draw_roc_combined 使用 AXIS_*_DIR，与输出一致 | ✅ |

---

## 二、潜在问题与建议

### 1. 插补敏感性：preprocess 返回 None 时未兜底 ✅ 已修复

**位置**：`run_all_charls_analyses.py` L192-202

**问题**：当 `USE_IMPUTED_DATA=True` 且 `preprocess_charls_data` 返回 None（如 CHARLS.csv 不存在）时，`df_for_imp_sens` 为 None，传入 `run_imputation_sensitivity_preprocessed` 会报错。

**修正**：preprocess 返回 None 时保持 `df_for_imp_sens = df_clean`，打 warning 说明回退到主数据、五种方法可能无差异。

---

### 2. 流失流程图与插补数据不一致

**位置**：`run_all_charls_analyses.py` L179-184

**问题**：`attrition_flow.csv` 从 `preprocessed_data/` 复制。当 `USE_IMPUTED_DATA=True` 时，主流程不执行 `preprocess_charls_data(write_output=True)`，该文件可能为旧运行结果，与当前插补队列的流失步骤不一致。

**建议**：在 Methods 中说明「流失图基于预处理流程」；或当使用插补数据时，从插补脚本输出中生成对应流失表。

---

### 3. 剂量反应 sleep 仅在插补数据下运行

**位置**：`evaluation/charls_dose_response.py`

**说明**：预处理会移除 `sleep`、仅保留 `sleep_adequate`。使用预处理数据时，`df` 无 `sleep` 列，剂量反应的 sleep RCS 块被跳过；仅在使用插补数据（含 sleep）时才会执行。此为设计选择，非错误。

**建议**：在论文/报告中注明「sleep 剂量反应分析基于插补后数据」。

---

### 4. run_ite_validation_for_axes 路径与主流程不一致

**位置**：`evaluation/charls_ite_validation.py` L95-102

**问题**：`run_ite_validation_for_axes` 使用 `os.path.join(output_root, AXIS_A_DIR)`，即 `LIU_JUE_STRATEGIC_SUMMARY/Cohort_A_...`。主流程将轴线输出写到 `./Cohort_A_...`（项目根目录）。若单独运行 `charls_ite_validation.py`，会到错误路径查找。

**说明**：主流程通过 `run_axis_protocol` 直接调用 `run_ite_stratified_validation`，传入正确 `output_dir`，故主流程不受影响。

**建议**：`run_ite_validation_for_axes` 的 `output_root` 默认改为 `'.'`，或与 config 中轴线输出路径统一。

---

### 5. estimate_causal_impact 对 Y 缺失无显式处理 ✅ 已修复

**位置**：`causal/charls_recalculate_causal_impact.py` L67-71

**说明**：`Y_series = df_sub[Y].astype(float)`，若 `is_comorbidity_next` 含 NaN，DML 可能报错或结果异常。

**修正**：在 T 检查后增加 `df_sub[Y].isna().any()` 判断，若有缺失则 `dropna(subset=[Y])` 并打日志。

---

### 6. Table1 跳过逻辑可能使用旧结果

**位置**：`run_all_charls_analyses.py` L174-177

**说明**：若 `table1_baseline_characteristics.csv` 已存在则跳过生成。若数据或队列定义已更新，可能继续使用旧表。

**建议**：按需增加「强制重算」选项，或根据数据/配置哈希决定是否跳过。

---

## 三、路径与目录约定

| 输出位置 | 说明 |
|----------|------|
| `./Cohort_A_Healthy_Prospective/` 等 | 轴线 A/B/C 输出（01_prediction, 02_shap, 03_causal 等） |
| `LIU_JUE_STRATEGIC_SUMMARY/` | 汇总表、敏感性、流失图、森林图等 |
| `preprocessed_data/` | 预处理输出（需 `preprocess_charls_data(write_output=True)`） |

`draw_roc_combined`、`charls_extra_figures.draw_combined_subgroup_cate` 等使用 `AXIS_*_DIR`，与 `./Cohort_*` 一致，运行目录需为项目根。

---

## 四、修复优先级与状态

| 优先级 | 项目 | 状态 |
|--------|------|------|
| P1 | 插补敏感性 df_for_imp_sens 为 None 时兜底 | ✅ 已修复 |
| P2 | estimate_causal_impact 对 Y 缺失防护 | ✅ 已修复 |
| P3 | run_ite_validation_for_axes 路径 | 待定（仅影响独立运行） |
| P3 | 流失图与插补数据一致性 | 方法学说明即可 |
