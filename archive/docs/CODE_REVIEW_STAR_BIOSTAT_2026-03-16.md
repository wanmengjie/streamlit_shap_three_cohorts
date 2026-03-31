# CHARLS 因果机器学习代码逻辑审查报告（STAR 原则）

**审查日期**：2026-03-16  
**角色**：生物统计学与因果推断专家  
**依据**：纵向队列分析、Causal Forest DML、预测模型验证最佳实践

---

## 1. 纵向数据完整性（Situation）

### 1.1 ✅ 因果时序正确

| 检查项 | 状态 | 代码定位 |
|--------|------|----------|
| Wave(t) → Wave(t+1) 前瞻性 | ✅ | `charls_complete_preprocessing.py` L84-95, L104-111 |
| 基线已患共病排除 | ✅ | L113-118 `had_comorbidity_before` |
| 显式校验 wave+1 紧邻 | ✅ | L106-111 校验 `valid_next` |

**理论依据**：Austin (2011) 强调倾向评分分析需明确暴露与结局的时间顺序；CHARLS 为 person-wave 结构，每行 = Wave(t)，Y 取自 Wave(t+1)，满足前瞻性。

---

### 1.2 🟡 主预测模型未采用时间划分

| 风险等级 | 问题 | 代码定位 |
|----------|------|----------|
| 🟡 警告 | 主 `compare_models` 使用 `GroupShuffleSplit` 随机划分，非 Wave 时间划分 | `charls_model_comparison.py` L75-77 |

**当前代码**：
```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(X, y, groups=df['ID']))
```

**影响**：若存在时间漂移（如早期波次 vs 晚期波次），主 AUC 可能偏乐观。`run_external_validation` 已单独做时间验证（wave<max_wave vs wave=max_wave），但冠军模型选择基于随机划分 AUC。

**修复建议**（可选，作为敏感性分析）：
```python
# 新增：时间划分选项（当 wave 可用且需严格时序时）
if use_temporal_split and 'wave' in df.columns:
    max_wave = df['wave'].max()
    train_idx = df[df['wave'] < max_wave].index
    test_idx = df[df['wave'] == max_wave].index
    # 需确保 train_idx/test_idx 与 X 对齐
else:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=df['ID']))
```

**理论依据**：Steyerberg et al. (TRIPOD) 建议预测模型在有时序结构时采用时间划分以评估泛化。

---

### 1.3 🔴 预处理 income 中位数存在潜在泄露

| 风险等级 | 问题 | 代码定位 |
|----------|------|----------|
| 🔴 致命 | `income_total` 缺失用全量中位数填充，中位数含未来划分的测试集信息 | `charls_complete_preprocessing.py` L55-61 |

**当前代码**：
```python
raw_inc = df['income_total'].clip(lower=0)
med = raw_inc.median()  # 全量计算
df['income_total'] = np.log1p(raw_inc.fillna(med))
```

**修复建议**：预处理阶段不做 income 插补，将插补纳入建模 Pipeline；或预处理仅输出「未插补」版本，由 `compare_models` 的 Pipeline Imputer 在训练折内 fit。

```python
# 方案 A：预处理不插补 income，由 Pipeline 处理
# 删除 L55-61 的 fillna(med)，保留 np.log1p(raw_inc)，缺失留待 Pipeline

# 方案 B：若必须预处理插补，需在 split 之后、仅对训练集计算 median
# 但预处理在 split 之前执行，故推荐方案 A
```

**理论依据**：Kaufman (2017) 指出，任何使用测试集信息的预处理都会导致乐观偏倚。

---

## 2. 预测模型合理性（Task）

### 2.1 ✅ GroupKFold 按 ID 分组

| 检查项 | 状态 | 代码定位 |
|--------|------|----------|
| 外划分用 GroupShuffleSplit + groups=ID | ✅ | `charls_model_comparison.py` L77 |
| 内 CV 用 GroupKFold + groups=ID | ✅ | L104, L169, L175-176 |
| RandomizedSearchCV 传入 groups | ✅ | L169 `search.fit(..., groups=df.iloc[train_idx]['ID'])` |

**理论依据**：同一 ID 多波次观测需保持在同一折内，避免信息泄露（Bischl et al., 2012）。

---

### 2.2 ✅ 超参搜索结构合理

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 外划分 80/20 | ✅ | GroupShuffleSplit test_size=0.2 |
| 内 CV 5 折 | ✅ | GroupKFold(n_splits=5) |
| 最优阈值在训练集内确定 | ✅ | L174-180 cross_val_predict 仅用 X_train |
| 测试集仅 predict | ✅ | L170 best_model.predict_proba(X_test) |

**说明**：RandomizedSearchCV 在训练集上做内层 CV 选超参，最终在独立测试集评估，无嵌套 CV 乐观偏倚。

---

### 2.3 🟡 插补敏感性分析未按 ID 分组

| 风险等级 | 问题 | 代码定位 |
|----------|------|----------|
| 🟡 警告 | `run_imputation_sensitivity_preprocessed` 使用 `train_test_split` 随机划分，未按 ID 分组 | `charls_imputation_audit.py` L103 |

**当前代码**：
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=500)
```

**影响**：若同一 ID 有多个 person-wave 行，可能被分入 train 与 test，导致同一个体信息泄露。

**修复建议**：
```python
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=500)
train_idx, test_idx = next(gss.split(X, y, groups=df['ID']))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

---

## 3. 因果推断假设验证（Action）

### 3.1 ✅ 倾向评分重叠检验

| 检查项 | 状态 | 代码定位 |
|--------|------|----------|
| PS 超出 [0.05, 0.95] 比例 | ✅ | `charls_causal_assumption_checks.py` L56-59, L77 |
| Trimming 比例写入 ATE_CI_summary | ✅ | `charls_recalculate_causal_impact.py` L155-160 |
| 重叠图保存 | ✅ | fig_propensity_overlap_*.png |

**理论依据**：Austin (2011) 建议检查 PS 分布重叠，超出 [0.1, 0.9] 或 [0.05, 0.95] 比例过高时需 trimming。

---

### 3.2 ✅ 连续变量标准化

| 检查项 | 状态 | 代码定位 |
|--------|------|----------|
| 仅 CONTINUOUS_FOR_SCALING 缩放 | ✅ | `charls_recalculate_causal_impact.py` L88-94 |
| 分类变量不缩放 | ✅ | `charls_feature_lists.CATEGORICAL_NO_SCALE` |

**理论依据**：Chernozhukov et al. (2018) DML 对协变量尺度敏感，Causal Forest 分裂依赖标准化。

---

### 3.3 ✅ Honest Estimation

| 检查项 | 状态 | 代码定位 |
|--------|------|----------|
| CausalForestDML honest=True | ✅ | `charls_recalculate_causal_impact.py` L105 |

**理论依据**：Wager & Athey (2018) 诚实估计将样本分为生长集与估计集，减少过拟合。

---

### 3.4 🔴 E-value 保守估计方向错误（保护效应时）

| 风险等级 | 问题 | 代码定位 |
|----------|------|----------|
| 🔴 致命 | 保护效应（ATE<0）时，应使用 CI 上限（ate_ub）作为保守端，当前误用 ate_lb | `charls_recalculate_causal_impact.py` L191; `charls_causal_assumption_checks.py` L164 |

**当前代码**：
```python
rr_conservative = (r0 + ate_lb) / r0  # 保护效应时 ate_lb 更接近 0，更保守
```

**错误**：保护效应时 ate_lb < ate < ate_ub，ate_ub 更接近 0（null），应使用 ate_ub。

**修复建议**：
```python
# VanderWeele & Ding 2017: 使用 CI 中离 null 最近的一端
if ate >= 0:  # 有害效应
    rr_conservative = (r0 + ate_lb) / r0
else:         # 保护效应
    rr_conservative = (r0 + ate_ub) / r0
```

**理论依据**：VanderWeele & Ding (2017) "The E-value for a confidence interval uses the limit closest to the null."

---

## 4. 敏感性分析完整性（Result）

### 4.1 ✅ CES-D / 认知截断值敏感性

| 检查项 | 状态 | 代码定位 |
|--------|------|----------|
| 9 种组合 (CES-D 8/10/12 × Cog 8/10/12) | ✅ | `run_sensitivity_scenarios.py` L139-141 |
| 完整病例敏感性 | ✅ | restrict_complete_case 分支 |

---

### 4.2 ✅ E-value 计算逻辑（除保守端方向外）

| 检查项 | 状态 |
|--------|------|
| 基于观察 RR 与 CI | ✅ |
| 公式 RR = (r0 + ate)/r0 | ✅ 适用于风险差 |
| _evalue_from_rr 实现 | ✅ |

---

### 4.3 🟢 建议补充的稳健性检验

| 缺失项 | 建议 | 理论依据 |
|--------|------|----------|
| 因果估计的 Bootstrap 与 解析 CI 对比 | 可对 TLearner 增加 Bootstrap CI 与 95% 解析 CI 对比 | 小样本时 Bootstrap 更稳健 |
| 不同 min_samples_leaf 敏感性 | Causal Forest min_samples_leaf 变化对 ATE 的影响 | Chernozhukov 2018 |
| 工具变量 / 断点设计 | 若有自然实验，可补充 IV/RDD | 因果识别扩展 |

---

## 5. 汇总表

| 维度 | 致命 | 警告 | 建议 |
|------|------|------|------|
| 纵向数据完整性 | 1 (income 泄露) | 1 (主模型非时间划分) | 0 |
| 预测模型合理性 | 0 | 1 (插补敏感性未按 ID) | 0 |
| 因果推断假设 | 1 (E-value 方向) | 0 | 0 |
| 敏感性分析 | 0 | 0 | 2 |

---

## 6. 修复优先级

| 优先级 | 项目 | 文件 |
|--------|------|------|
| P0 | E-value 保守端方向修正 | charls_recalculate_causal_impact.py, charls_causal_assumption_checks.py |
| P0 | income 预处理泄露 | charls_complete_preprocessing.py |
| P1 | 插补敏感性 GroupShuffleSplit | charls_imputation_audit.py |
| P2 | 主模型时间划分选项（可选） | charls_model_comparison.py |

---

## 参考文献

- Austin PC (2011). An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behavioral Research*.
- Chernozhukov V et al. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*.
- Kaufman S et al. (2017). Leakage in data mining: Formulation, detection, and avoidance. *ACM TKDD*.
- VanderWeele TJ, Ding P (2017). Sensitivity analysis in observational research: Introducing the E-value. *Annals of Internal Medicine*.
- Wager S, Athey S (2018). Estimation and inference of heterogeneous treatment effects using random forests. *JASA*.
