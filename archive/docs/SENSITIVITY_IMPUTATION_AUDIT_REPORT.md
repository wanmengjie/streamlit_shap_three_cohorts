# 插补后的敏感性分析 — 逻辑梳理与修改报告

## 一、逻辑分支总览

| 模块 | 功能 | 数据输入 | 输出 | 状态 |
|------|------|----------|------|------|
| `run_all_charls_analyses.py` | 主流程：数据源选择、插补敏感性调用、结果汇总 | CHARLS.csv / step1_imputed_full.csv | 07_sensitivity_imputation/*, results/tables | **已更新** |
| `charls_imputation_audit.py` | 插补敏感性分析（五法比较 AUC + Bootstrap CI） | 预处理 df（需含缺失） | imputation_sensitivity_results.csv, figS3 | **已更新** |
| `charls_sensitivity_analysis.py` | 因果敏感性（Placebo / E-Value / 噪声 / 偏倚） | res_df（含 causal_impact_*） | 07_sensitivity/* | 已确认无遗漏 |
| `run_sensitivity_scenarios.py` | 截断值敏感性 + 完整病例 | CHARLS.csv → preprocess | sensitivity_summary.csv | 与插补模块独立，无需改 |

---

## 二、数据流闭环说明

- **主分析数据**：`USE_IMPUTED_DATA=True` 且存在 `IMPUTED_DATA_PATH` 时，`df_clean` = 插补后数据；否则 `df_clean` = `preprocess_charls_data(...)`。
- **插补敏感性分析**：必须使用「带缺失」的预处理数据，否则当主分析用插补数据时，五种插补方法无缺失可填，比较无意义。
  - **已更新**：当主分析使用插补数据时，插补敏感性单独再调一次 `preprocess_charls_data` 得到 `df_for_imputation_sensitivity`，再传入 `run_imputation_sensitivity_preprocessed`；否则直接传 `df_clean`。
- **因果敏感性分析**：在 `run_axis_protocol` 内对每条轴线执行，输入为 `res_df`（来自 `estimate_causal_impact`），与主分析是否用插补数据一致（res_df 已基于当前 df_sub）。
- **结果输出**：`_copy_if_exists(..., '07_sensitivity_imputation/imputation_sensitivity_results.csv', ...)` 已存在，逻辑闭环。

---

## 三、逐模块检查结果

### 3.1 run_all_charls_analyses.py

| 位置 | 说明 | 状态 |
|------|------|------|
| L137–156 | 数据源：插补 vs 预处理 | 已更新（保持原逻辑） |
| L179–188 | 插补敏感性分析调用 | **已更新**：使用 `df_for_imputation_sensitivity`（主分析用插补时 = 预处理数据，否则 = df_clean） |
| L269 | 复制 imputation_sensitivity_results.csv 到 results/tables | 已更新（无改动，路径正确） |

**修改点**：插补敏感性分析改为始终基于「带缺失」数据——主分析用插补时额外执行一次 `preprocess_charls_data` 专供插补敏感性。

### 3.2 charls_imputation_audit.py

| 位置 | 说明 | 状态 |
|------|------|------|
| L65–69 | 入口：target_col 检查 | 已更新 |
| L77–84 | 目标缺失行剔除、特征列、样本量检查 | **已更新**：增加 `df.dropna(subset=[target_col])`，避免 y 含 NaN |
| L94–108 | _add_result：Bootstrap 空/短列表、percentile | **已更新**：过滤 nan、len<10 时 CI 用点估计，空列表写入 nan 行 |
| L52–62 | _run_single_imputation_auc | **已更新**：单类或异常时返回 np.nan，try/except 防崩 |
| L137–178 | res_df 为空检查、绘图对 NaN 的防护 | **已更新**：len(results)==0 提前 return；绘图用 fillna/nanmin/nanmax 避免 NaN 导致报错 |

**修改点**：目标缺失剔除、Bootstrap/单次 AUC 异常防护、空结果与绘图 NaN 处理。

### 3.3 charls_sensitivity_analysis.py

| 位置 | 说明 | 状态 |
|------|------|------|
| 整体 | 输入为因果结果 res_df，内部用 SimpleImputer(strategy='median') 仅做数值填充 | 已确认无遗漏 |
| 与插补关系 | 不直接读插补数据；res_df 来自当前轴线 df_sub（主分析若用插补则已含插补） | 无需改 |

### 3.4 run_sensitivity_scenarios.py

| 位置 | 说明 | 状态 |
|------|------|------|
| 整体 | 自读 CHARLS.csv，内部 preprocess_charls_data(write_output=False)，与主流程插补数据无关 | 无需改 |

### 3.5 charls_imputation_npj_style.py

| 位置 | 说明 | 状态 |
|------|------|------|
| sensitivity_validation / table4 | 插补脚本内部验证，不通过 run_all_charls_analyses 调用 | 非本次「插补后的敏感性分析」范围，未改 |

---

## 四、修改代码位置汇总（可直接对照 diff）

### run_all_charls_analyses.py

- **约 L177–188**：插补敏感性分析块改为先确定 `df_for_imputation_sensitivity`（主分析用插补时 = `preprocess_charls_data(...)`，否则 = `df_clean`），再调用 `run_imputation_sensitivity_preprocessed(df_for_imputation_sensitivity, ...)`。

### charls_imputation_audit.py

- **L52–62**：`_run_single_imputation_auc` 增加单类检查与 try/except，失败返回 np.nan。
- **L77–84**：`run_imputation_sensitivity_preprocessed` 内增加 `df = df.dropna(subset=[target_col]).copy()`。
- **L94–108**：`_add_result` 过滤 nan、len(auc_arr)<10 时 CI 用点估计、空列表写 nan 行。
- **L137–178**：`res_df` 为空时提前 return；绘图使用 `fillna`/`nanmin`/`nanmax` 及 NaN 安全标注。

---

## 五、边界与异常

- **插补敏感性无有效结果**：若五种方法均失败或 results 为空，会写 warning、不写 CSV/图、返回 None；主流程已 try/except，不会中断。
- **主分析用插补但预处理失败**：若 `preprocess_charls_data` 返回 None，不调用 `run_imputation_sensitivity_preprocessed`，同上 try/except 保护。
- **Bootstrap 中部分迭代 nan**：已过滤后算 mean/percentile，有效迭代 <10 时 CI 用点估计，与 self.md 中「np.percentile 有 len<10 防护」一致。

---

## 六、性能建议（可选）

- 主分析用插补时多调一次 `preprocess_charls_data` 会多一次预处理开销；若需可考虑在 main 中统一先算 `df_preprocessed` 再根据开关决定 `df_clean = load_imputed or df_preprocessed`，避免重复读 CHARLS。当前以逻辑正确为先，未改。
- 插补敏感性中 Bootstrap 50 次 × 5 方法，可保持现状；若将来需要可加 n_bootstrap 配置或早停逻辑。

---

## 七、结论

- **插补操作 → 敏感性分析 → 结果输出** 全流程已闭环：插补敏感性始终用带缺失的预处理数据；结果写入 `07_sensitivity_imputation/imputation_sensitivity_results.csv` 并复制到 `results/tables`。
- 所有「待更新」「需调整」点已按上表落实；`charls_sensitivity_analysis` 与 `run_sensitivity_scenarios` 已确认与插补逻辑无冲突，无需修改。
- 修改后代码可直接运行，修改点已用注释或本报告标注。
