# charls_imputation_npj_style.py 代码审查报告

## 一、整体结论

| 项目 | 结论 |
|------|------|
| **是否符合研究设计** | 基本符合，已补充核心变量完整病例子集 |
| **是否可直接运行** | 是 |
| **核心逻辑（整体插补→队列划分）** | 正确：先全量插补，后按 baseline_group 划分 A/B/C |

---

## 二、逐维度审查结果

### 1. 数据预处理逻辑

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 先全量插补→后划分队列 | **通过** | run_full_experiment 第 564–577 行：先 df_imputed 全量插补，再按 baseline_group 划分 |
| VARS_NO_IMPUTE 完整性 | **通过** | 含 is_comorbidity_next, is_comorbidity, is_depression, is_cognitive_impairment；DEFINING_COLS 含 total_cognition/total_cog/cesd10/cesd |
| 插补前剔除定义/结局缺失 | **通过** | 数据来自 preprocess_charls_data，已剔除 cesd/cognition/is_comorbidity_next 缺失样本 |
| ANALYSIS_VARS 与 Table 1 一致 | **通过** | VARS_CONTINUOUS + VARS_CATEGORICAL 与 Table 1 一致，sleep 连续、exercise 分类 |
| 变量类型划分（连续/分类） | **通过** | 按 Table 1 格式：均值±SD=连续，n%=分类 |
| 溯源保存 | **通过** | step0_loaded.csv、step1_imputed_full.csv、step1b_complete_case_core.csv、step2_cohort_A/B/C_*.csv |
| 队列标签与论文一致 | **通过** | A=Healthy、B=Depression_only、C=Cognition_impaired_only |

---

### 2. 缺失值筛查与 MCAR 验证

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 统计所有分析变量缺失率 | **通过** | screen_missing 使用 vars_to_check=ANALYSIS_VARS+VARS_OUTCOME |
| MILD/MODERATE/SEVERE 分层 | **通过** | <5% mild, 5–10% moderate, >10% severe |
| table1_missing_distribution.csv | **通过** | 第 159 行 |
| 热力图/条形图含队列维度 | **通过** | 第 163–183 行：baseline_group 分队列热力图 |
| 仅对中度缺失变量 MCAR 验证 | **通过** | moderate_vars 来自 Tier=moderate |
| 协变量排除待验证变量 | **通过** | aux 排除 moderate_vars |
| Bonferroni 校正 | **通过** | alpha_bonf = 0.05 / n_tests |
| table2_mcar_validation.csv | **通过** | 第 263 行 |
| 极端情况处理 | **已修复** | 新增 aux 不足时跳过；y_miss.sum()<10 或 mask.sum()<50 时 continue |

---

### 3. 插补方法实现

#### 3.1 连续变量（6 种方法）

| 检查项 | 结果 | 说明 |
|--------|------|------|
| Mean/Median/LinearRegression/BayesianRidge/KNN/MissForest | **通过** | IMPUTATION_METHODS 正确调用 sklearn |
| KNN n_neighbors=5 可配置 | **通过** | impute_knn(df, cols, n_neighbors=5) |
| MissForest n_estimators=100 | **通过** | 第 301 行 |
| 迭代插补 max_iter=5 | **通过** | Linear/Bayesian/MissForest 均为 5 |
| 仅对 cols 插补 | **通过** | out[cols] = imp.fit_transform(df[cols])，其他列不变 |

#### 3.2 分类变量

| 检查项 | 结果 | 说明 |
|--------|------|------|
| Mode 众数插补 | **通过** | impute_mode 使用 strategy='most_frequent' |
| 核心干预变量完整病例子集 | **已修复** | 新增 step1b_complete_case_core.csv（exercise/smokev/drinkev 无缺失） |
| 分类变量插补后为离散值 | **通过** | SimpleImputer(most_frequent) 返回众数，无 2.5 等异常值 |

---

### 4. 性能评估（NRMSE + 辅助指标）

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 完整病例上人为引入 MCAR 缺失 | **通过** | 第 346–354 行，SIMULATE_MISSING_PCT=0.08 |
| 重复 3 次 | **通过** | n_repeat=3 |
| NRMSE 计算正确 | **通过** | RMSE/std(original)，std≈0 时用 ptp |
| SD_ratio、Mean_diff_ratio | **通过** | 第 331–332 行 |
| table3_imputation_performance.csv | **通过** | 第 403 行 |
| 自动筛选 NRMSE 最小最优方法 | **通过** | 第 404–406 行 |
| fig3_nrmse_comparison.png | **通过** | 第 407–414 行 |

---

### 5. 敏感性验证

| 检查项 | 结果 | 说明 |
|--------|------|------|
| KS 检验 p>0.05 分布相似 | **通过** | 第 391 行 |
| 均值差异率 <5% 可接受 | **通过** | MEAN_DIFF_ACCEPT=0.05 |
| 相关矩阵 Frobenius 范数 | **通过** | 第 399–402 行 |
| 密度图 + 箱线图 | **通过** | 第 416–428 行 |
| 图含核心变量 bmi/sleep/exercise | **已修复** | 优先选择 prefer=['bmi','sleep','exercise'] |

---

### 6. 队列划分与独立插补

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 主流程按 baseline_group 划分 A/B/C | **通过** | 第 428–434 行 |
| split_cohorts 为推导/地理/时间 | **通过** | 定义但未在主流程使用，主流程用 baseline_group |
| 样本量偏差提示 | **已修复** | 与论文 PAPER_N 对比，偏差>100 时 logger.warning |
| 队列 CSV 含完整分析变量 | **通过** | df_cohort 为 df_imputed 子集，含所有列 |

---

### 7. 代码完整性与鲁棒性

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 数据加载编码 | **通过** | utf-8/gbk/gb18030/latin-1 |
| province 缺失时 _region 默认 | **通过** | split_cohorts 中 fillna('E') |
| 日志输出 | **通过** | 最优方法、NRMSE、输出目录、样本量 |
| 可调参数集中 | **通过** | 第 39–52 行 |
| 语法/逻辑错误 | **通过** | 无 |

---

### 8. 与论文方法学一致性

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 一般变量插补 + 核心变量完整病例 | **已修复** | 新增 step1b_complete_case_core.csv |
| 插补方法（6 种对比） | **通过** | 符合附录 S2 补充实验 |
| 无数据泄露 | **通过** | 插补不包含结局变量，队列划分在插补后 |

---

## 三、关键修改汇总

### 修复 1：核心干预变量完整病例子集（论文附录 S2）

**问题**：未提取并保存 exercise/smokev/drinkev 无缺失子集。

**修改**：新增 Step 4b，保存 step1b_complete_case_core.csv。

```python
# 新增常量
CORE_INTERVENTION_VARS = ['exercise', 'smokev', 'drinkev']

# Step 4b
core_vars = [c for c in CORE_INTERVENTION_VARS if c in df_imputed.columns]
if core_vars:
    df_complete_core = df_imputed.dropna(subset=core_vars)
    if len(df_complete_core) >= 30:
        _save_trace(df_complete_core, 'step1b_complete_case_core', output_dir, ...)
```

### 修复 2：MCAR 协变量不足时跳过

**问题**：aux 为空时 model.fit 可能报错。

**修改**：新增 len(aux) < 1 检查。

```python
if len(aux) < 1:
    logger.warning("MCAR 协变量不足（需≥1个无缺失变量），跳过 MCAR 验证")
    return pd.DataFrame()
```

### 修复 3：队列样本量偏差提示

**修改**：与论文预期 (8828, 3123, 2435) 对比，偏差>100 时 logger.warning。

### 修复 4：分布对比图优先包含核心变量

**修改**：plot_cols 优先选择 bmi、sleep、exercise。

---

## 四、溯源文件清单（修改后）

| 文件 | 说明 |
|------|------|
| step0_loaded.csv | 加载后原始数据 |
| step1_imputed_full.csv | 全量插补后 |
| step1b_complete_case_core.csv | 核心干预变量无缺失子集（附录 S2） |
| step2_cohort_A_Healthy.csv | 队列 A |
| step2_cohort_B_Depression_only.csv | 队列 B |
| step2_cohort_C_Cognition_impaired_only.csv | 队列 C |
| README_溯源说明.txt | 步骤说明 |
