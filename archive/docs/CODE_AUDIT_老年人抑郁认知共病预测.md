# CHARLS 老年抑郁-认知共病研究 代码审查报告

**审查时间**：基于当前代码库与论文 `PAPER_完整版_审稿修订.md` 的静态审查  
**审查范围**：数据处理→预测建模→因果推断→敏感性分析→规范性

---

## 一、数据处理层

### 1.1 队列筛选逻辑（96628→14386）✓

| 步骤 | 论文表1 | 代码输出 (table1_sample_attrition.csv) | 状态 |
|------|---------|----------------------------------------|------|
| 原始记录 | 96,628 | 96,628 | ✓ |
| 年龄≥60 | 49,015 | 49,015 | ✓ |
| CES-D-10非缺失 | 43,048 | 43,048 | ✓ |
| 认知得分非缺失 | 31,574 | 31,574 | ✓ |
| 下一波共病非缺失 | 16,983 | 16,983 | ✓ |
| 入射队列 | 14,386 | 14,386 | ✓ |

**结论**：流失流程与论文表1完全一致。

### 1.2 抑郁/认知/共病定义 ✓

- **CES-D≥10**：`charls_complete_preprocessing.py` 第60行 `cesd_cutoff=10` ✓
- **认知≤10**：第67行 `cognition_cutoff=10` ✓
- **共病**：第74行 `(is_depression==1) & (is_cognitive_impairment==1)` ✓

### 1.3 可干预因素编码 ✓

- `sleep_adequate`（≥6h）、`bmi_normal`（18.5–24）、`exercise`、`smokev`、`drinkev` 在 `run_multi_exposure_causal.py` 中正确构造 ✓

### 1.4 缺失值处理 ✓

- CES-D、认知、`is_comorbidity_next`：`dropna` ✓
- 建模：`SimpleImputer(strategy='median')` ✓
- 干预变量：`T.fillna(0)`（因果分析）——缺失当作未干预，需在论文中说明 ✓

### 1.5 三轴线分组 ✓

- A=8828、B=3123、C=2435 与 table4/table5 一致 ✓

### 1.6 问题清单

| 级别 | 问题 | 表现 | 原因 | 修正思路 |
|------|------|------|------|----------|
| **高危** | CES-D/认知列缺失时 KeyError | 若数据无 `cesd10`/`total_cog` 等列，第64/67行会报错 | 列名查找失败时未做健壮处理 | 在 `cesd_col`/`cog_col` 为 None 时提前返回并记录错误 |
| 低危 | 年龄列名依赖模糊匹配 | `age_col = next((c for c in df.columns if 'age' in c.lower()), None)` | 可能匹配到非预期列 | 优先使用 `age`，再考虑 `age_int` 等明确列名 |

---

## 二、预测建模层

### 2.1 GroupKFold 与 8:2 划分 ✓

- **训练/测试**：`GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=500)`，`groups=df['ID']` ✓
- **交叉验证**：`GroupKFold(n_splits=5)`，`groups=df.iloc[train_idx]['ID']` ✓

### 2.2 15 种模型与附录 S8

| 模型 | 附录 S8 范围 | 代码实际范围 | 一致性 |
|------|--------------|--------------|--------|
| LR | C [0.01,0.1,1,10] | [0.001,0.01,0.1,1,10,100] | 代码为超集 ✓ |
| RF | n_est [200,300,500] | [200,300,500,700,1000] | 代码为超集 ✓ |
| XGB | 无 reg_alpha/lambda | 有 reg_alpha, reg_lambda, min_child_weight | 代码为超集 ✓ |
| CatBoost | iterations [200,400] | [300,500,700,1000] | 代码为超集 ✓ |

**结论**：代码搜索空间包含并扩展了附录 S8，建议在附录中注明“实际搜索范围包含并扩展了表 S8”。

### 2.3 AUC/AUPRC 与 Bootstrap 95%CI ✓

- `charls_ci_utils.get_metrics_with_ci`：`n_bootstraps=500`，`np.percentile(arr, [2.5, 97.5])` ✓
- 单类样本：`len(np.unique(y_true_b)) < 2` 时跳过 ✓

### 2.4 类别不平衡 ✓

- `scale_pos_weight = neg / max(pos, 1)`，用于 XGB/LightGBM/CatBoost ✓
- `class_weight='balanced'` 用于 LR/RF/SVM/DT ✓

### 2.5 冠军模型选取 ✓

- `perf_df.sort_values('AUC', ascending=False)`，取第一名 ✓

### 2.6 问题清单

| 级别 | 问题 | 表现 | 原因 | 修正思路 |
|------|------|------|------|----------|
| 低危 | 轴线 A 冠军模型与论文可能不一致 | 当前输出 MLP 0.7420，论文为 LR 0.7313 | 随机搜索导致不同运行结果不同 | 固定 `random_state=500` 已设置，论文中说明复现需相同种子 |

---

## 三、因果推断层

### 3.1 Causal Forest DML ✓

- **5 折交叉拟合**：`cv=5` ✓
- **1000 棵树**：`n_estimators=1000` ✓
- **聚类稳健**：`groups=cluster_ids`（communityID 或 ID）✓

### 3.2 PSM ✓

- **1:1 最近邻**：`NearestNeighbors(n_neighbors=1)` ✓
- **卡尺 0.2×SD**：`caliper_val = 0.2 * np.std(ps)` ✓

### 3.3 PSW ⚠

- **权重**：`w = 1/ps`（干预组）、`1/(1-ps)`（对照组）✓
- **trim 1%**：论文 S7.3 要求“权重修剪（trim=1%）”，代码使用 `np.clip(w, 0.1, 50)`，**并非 1% 分位数修剪** ⚠

### 3.4 ATE 可靠性 ✓

- 判定：`reliable = 1 if -1 <= ate <= 1 else 0` ✓

### 3.5 与表 4/表 7 的数值 ✓

**表 7（轴线 B 运动）**：

| 方法 | 论文 | table7_psm_psw_dml.csv | 状态 |
|------|------|------------------------|------|
| PSM | -0.028 (-0.051, -0.005) | -0.028 (-0.051, -0.005) | ✓ |
| PSW | -0.034 (-0.058, -0.009) | -0.034 (-0.058, -0.009) | ✓ |
| DML | -0.037 (-0.103, 0.030) | -0.037 (-0.103, 0.030) | ✓ |

### 3.6 问题清单

| 级别 | 问题 | 表现 | 原因 | 修正思路 |
|------|------|------|------|----------|
| **高危** | PSW 未实现 1% trim | 论文要求 trim=1%，代码用 clip(0.1, 50) | 实现与论文描述不符 | 按 1%/99% 分位数修剪权重，或修改论文描述为“权重截断（0.1–50）” |
| 中危 | DML CI 来源 | 部分流程用 ITE 的 2.5%/97.5% 分位数 | 未使用 econml 的 `ate_interval()` | 在因果分析阶段保存 `ate_interval` 结果并直接使用 |

---

## 四、敏感性分析层

### 4.1 截断值遍历 ✓

- CES-D：8、10、12 ✓
- 认知：8、10、12 ✓

### 4.2 完整病例 ✓

- `restrict_complete_case=True` 时对干预变量 `dropna` ✓

### 4.3 偏倚分析 ✓

- `confounder_strengths=[0, 0.1, 0.2, 0.3, 0.5]` ✓

### 4.4 问题清单

| 级别 | 问题 | 修正思路 |
|------|------|----------|
| 低危 | 偏倚分析用线性回归近似 | 在附录中说明为简化敏感性分析 |

---

## 五、规范性

### 5.1 数据泄露 ✓

- `LEAKAGE_KEYWORDS` 含 `cesd`、`total_cog`、`cognition`、`memory` 等 ✓
- `get_exclude_cols` 统一排除目标、干预、因果衍生列 ✓

### 5.2 随机种子 ✓

- `RANDOM_SEED=500`，各模型与 `RandomizedSearchCV` 使用 `random_state=500` ✓

### 5.3 异常处理 ✓

- 主流程用 `try/except` 包裹子模块 ✓

---

## 六、核心数值与论文表 1–7 对比

| 表 | 论文值 | 代码/输出 | 状态 |
|----|--------|-----------|------|
| 表1 流失 | 96628→14386 | table1_sample_attrition.csv 一致 | ✓ |
| 表2 样本量 | A=8828, B=3123, C=2435 | table4/table5 一致 | ✓ |
| 表3 AUC | A LR 0.7313, B SVM 0.6834, C ExtraTrees 0.6337 | B、C 一致；A 存在运行差异 | ⚠ |
| 表4 ATE | 运动 A -0.0018, B -0.0373, C 0.0241 | table4 一致 | ✓ |
| 表6 外部验证 | 时间/区域验证 | charls_external_validation 实现 | ✓ |
| 表7 PSM/PSW/DML | 轴线 B 运动 | table7 一致 | ✓ |

---

## 七、整体可用性评分与优化建议

### 7.1 可用性评分：**7.8/10**

- 数据处理、因果推断、敏感性分析流程完整，与论文描述基本一致
- 表 4、表 7 等核心结果可复现
- 扣分点：PSW trim 与论文不符、DML CI 来源不一致、轴线 A 冠军模型存在运行差异

### 7.2 优化建议（按优先级）

| 优先级 | 建议 |
|--------|------|
| **P0** | 实现 PSW 的 1% 分位数修剪，或明确在论文中说明当前为“权重截断（0.1–50）”而非“1% trim” |
| **P1** | 在因果分析中保存 `ate_interval()` 的 CI，并在方法对比中直接使用 |
| **P2** | 在 `preprocess_charls_data` 中增加对 `cesd_col`/`cog_col` 缺失的健壮处理 |
| **P3** | 在附录 S8 中注明“实际超参搜索范围包含并扩展了表 S8” |
| **P4** | 统一文件读写使用 `encoding='utf-8'` |

---

**审查完成**
