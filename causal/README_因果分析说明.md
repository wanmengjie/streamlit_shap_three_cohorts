# 因果分析模块说明

本目录实现因果推断主流程及假设检验，供论文方法部分引用。

---

## 一、文件职责

| 文件 | 职责 |
|------|------|
| `charls_recalculate_causal_impact.py` | 主因果估计：XLearner / TLearner / CausalForestDML，含 overlap 修剪 |
| `charls_causal_assumption_checks.py` | **假设检验**：重叠、平衡、E-value |
| `charls_causal_methods_comparison.py` | PSM / PSW / XLearner 三方法对比 |
| `charls_causal_multi_method.py` | 多干预、多队列批量分析 |

---

## 二、因果推断流程（以 XLearner 为例）

```
1. 数据准备
   └─ 协变量 X、暴露 T、结局 Y
   └─ 排除泄露变量（get_exclude_cols）

2. 重叠假设检验（修剪前）  ← 论文报告用
   └─ check_overlap(..., suffix='_pre_trim')
   └─ 输出：assumption_overlap_{T}_pre_trim.txt, fig_propensity_overlap_{T}_pre_trim.png

3. Overlap 修剪
   └─ PS 超出 [0.05, 0.95] 的样本剔除
   └─ 日志：Overlap trimming: N_retained/N_total, trimmed_pct=X%

4. 因果估计
   └─ XLearner.fit(Y, T, X) → ITE, ATE, 95% CI

5. 假设检验（修剪后）
   └─ run_all_assumption_checks(...)
       ├─ 重叠（post-trim，通常 0%）
       ├─ 协变量平衡（SMD）
       └─ E-value（未测混杂敏感性）

6. 方法对比
   └─ run_causal_methods_comparison：PSM / PSW / XLearner
```

---

## 三、假设检验含义（charls_causal_assumption_checks.py）

### 1. check_overlap — 正性/重叠

**检验什么**：倾向评分 PS 在 [0.05, 0.95] 外的样本比例。

**判定**：<10% 超出 → 重叠可接受；≥10% → 需修剪，结论仅适用于修剪后人群。

**输出**：
- `assumption_overlap_{T}.txt`：N、PS 范围、超出比例
- `fig_propensity_overlap_{T}.png`：PS 分布直方图

### 2. check_balance_smd — 协变量平衡

**检验什么**：治疗组 vs 对照组各协变量的标准化均数差 |SMD|。

**判定**：|SMD|<0.1 平衡良好；0.1–0.2 需关注；>0.2 不平衡。

**输出**：
- `assumption_balance_smd_{T}.csv`：各协变量 SMD
- `assumption_balance_{T}.txt`：max SMD、不平衡变量数

### 3. check_evalue — 未测混杂敏感性

**检验什么**：需多强的未测混杂（RR）才能解释掉观测到的效应。

**判定**：E-value 越大，对未测混杂越稳健。

**输出**：`assumption_evalue_{T}.txt`

### 4. run_all_assumption_checks — 汇总

**输出**：`assumption_checks_summary.txt`，含 pre-trim overlap、post-trim、balance、E-value。

---

## 四、输出文件索引（03_causal 目录）

| 文件 | 含义 |
|------|------|
| `assumption_overlap_{T}_pre_trim.txt` | 修剪前重叠报告（论文引用） |
| `fig_propensity_overlap_{T}_pre_trim.png` | 修剪前 PS 分布图 |
| `assumption_overlap_{T}.txt` | 修剪后重叠 |
| `assumption_balance_{T}.txt` | SMD 平衡检验 |
| `assumption_balance_smd_{T}.csv` | 各协变量 SMD 明细 |
| `assumption_evalue_{T}.txt` | E-value |
| `assumption_checks_summary.txt` | 全部假设检验汇总 |
| `ATE_CI_summary_{T}.txt` | ATE、95% CI、overlap_trimmed_pct |
| `CAUSAL_ANALYSIS_{T}.csv` | 含 ITE 的完整数据 |

---

## 五、配置

- `config.CAUSAL_METHOD`：`'XLearner'`（默认）/ `'TLearner'` / `'CausalForestDML'`
- `config.RANDOM_SEED`：随机种子

---

## 六、协变量预处理（与主流程变量名单一致）

因果主路径（`charls_recalculate_causal_impact.py`）中，协变量矩阵 **X** 的构造与处理为：

1. **列筛选**：`utils.charls_feature_lists.get_exclude_cols(df, target_col=Y, treatment_col=T)`  
   - 排除结局 **Y**、处理 **T**、`EXCLUDE_COLS_BASE` 中的 ID/队列/结局衍生列等；  
   - 列名命中 **泄露关键字**（`LEAKAGE_KEYWORDS`：如含 `cesd`、`cognition`、`score` 等）的一律不进 **X**。  
2. **仅数值列**：在剩余列中取 `select_dtypes` 数值列，与预测打擂一致。  
3. **数值变换**：`utils.charls_train_only_preprocessing.fit_transform_numeric_train_only`  
   - 在**训练子集**上 fit，再 transform 全表；  
   - **`CONTINUOUS_FOR_SCALING`** 中的列：`SimpleImputer(median)` → **`StandardScaler`**；  
   - **其余数值列**（含 **`ORDINAL_COUNT_IMPUTE_ONLY`**：`adlab_c`、`iadl`、`family_size`）：**仅 `SimpleImputer(median)`**，**不** StandardScaler；  
   - 与 **CPM** 的区别：因果此处为 **SimpleImputer**；CPM Pipeline 内为 **IterativeImputer**（见 `build_numeric_column_transformer`）。数据层常用 **完成表 vs 保留缺失** 的差异见 `docs/MEDICAL_CODE_CAUTIONS.md`。  
4. **处理变量**：**`T.fillna(0)`**（如 `exercise` 缺失）。  
5. **敏感性**：当 **T = `exercise`** 且存在 **`adlab_c`** 时，可构造 **`exercise × adlab_c`** 交互项（代码内已接好）。

**完整变量表**（实质：连续 / 有序计数 / 二分类 / 多分类；管道：缩放与否）见仓库内：

→ **`docs/主流程变量类型与预处理说明.md`**

---

## 七、进一步阅读

| 文档 | 内容 |
|------|------|
| `docs/主流程变量类型与预处理说明.md` | **全变量**实质类型 + CPM/因果 Pipeline 分支 |
| `docs/全流程数据分析详细说明.md` | §11.5–11.6 预测与因果数值处理 |
| `docs/MEDICAL_CODE_CAUTIONS.md` | 预测 Pipeline 插补 vs 因果 bulk 完成表 |
| `utils/charls_feature_lists.py` | `CONTINUOUS_FOR_SCALING`、`ORDINAL_COUNT_IMPUTE_ONLY`、`CATEGORICAL_NO_SCALE` 源定义 |
