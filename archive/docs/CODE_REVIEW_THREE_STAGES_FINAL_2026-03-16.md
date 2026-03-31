# CHARLS 因果机器学习：三阶段严格审查报告

**审查日期**：2026-03-16  
**角色**：生物统计学家 + 因果推断专家 + AJE 审稿人

---

## 审查结果汇总表

| 等级 | 文件 | 行号 | 问题 | 理论依据 | 修复代码 |
|------|------|------|------|----------|----------|
| 🔴致命 | charls_model_comparison.py | L75-77 | 主预测使用随机划分，未保证 train_wave < test_wave，存在时序泄露风险 | TRIPOD; Steyerberg 2013 | 见下方补丁 1 |
| 🟢建议 | charls_model_comparison.py | L77 | GroupShuffleSplit 按 ID 分组，同一人不会跨集 ✅ | Bischl et al. 2012 | 无需修复 |
| 🟢建议 | charls_complete_preprocessing.py | L55-59 | income_total 已移除全量中位数，NaN 留待 Pipeline ✅ | Kaufman 2017 | 无需修复 |
| 🔴致命 | charls_recalculate_causal_impact.py | L249-254 | TLearner 在 fit 前未做 overlap trimming；当 pct_trimmed>10% 时 ATE 可能偏倚 | Austin 2011 | 见下方补丁 2 |
| 🟢建议 | charls_causal_assumption_checks.py | L163-167 | E-value 保守端方向正确（保护用 ate_ub，有害用 ate_lb）✅ | VanderWeele & Ding 2017 | 无需修复 |
| 🟢建议 | charls_recalculate_causal_impact.py | L191 | E-value 方向正确 ✅ | VanderWeele & Ding 2017 | 无需修复 |
| 🟡警告 | charls_causal_assumption_checks.py | L117-126 | 仅报告未加权 SMD，缺失 PS 加权/匹配后 SMD | Austin 2011 | 见下方补丁 3 |
| 🔴致命 | - | - | 选择偏倚：基线 ADL 困难者无法运动，T=0 组可能混入更多功能差者 | Hernán & Robins 2020 | 见下方补丁 4 |
| 🟡警告 | - | - | Cohort C n=2435 对 CATE 异质性可能不足 | Wager & Athey 2018 | 文中说明 CATE 为探索性 |
| 🟡警告 | - | - | 时变混杂：仅用 Wave(t) 运动预测 Wave(t+1) 结局，未考虑随访期运动变化 | Hernán 2010 | 敏感性分析：纳入运动习惯稳定者 |

---

## 阶段一：生物统计完整性

### 1.1 时序泄露检测

**检查点**：`charls_model_comparison.py` L75-77

```python
# 当前代码
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(X, y, groups=df['ID']))
```

**结论**：
- ✅ **ID 泄露**：`groups=df['ID']` 确保同一人不会同时出现在 train 和 test
- 🔴 **时序泄露**：随机划分不保证 `max(train_waves) < min(test_waves)`，训练集可能含晚于测试集的 wave，AUC 可能偏乐观

**补丁 1**（增加时间划分选项）：

```python
# charls_model_comparison.py，在 L74 之后插入
# 可选：use_temporal_split 由 config 或参数传入
use_temporal_split = getattr(__import__('config', fromlist=['USE_TEMPORAL_SPLIT']), 'USE_TEMPORAL_SPLIT', False)
if use_temporal_split and 'wave' in df.columns:
    max_wave = df['wave'].max()
    train_mask = df['wave'] < max_wave
    test_mask = df['wave'] == max_wave
    if train_mask.sum() >= 50 and test_mask.sum() >= 20:
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        logger.info(f"时间划分: train wave<{max_wave}, test wave={max_wave}")
    else:
        use_temporal_split = False
if not use_temporal_split:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=df['ID']))
```

### 1.2 预处理泄露

**检查点**：`charls_complete_preprocessing.py` L55-61

```python
# 当前代码（已修复）
raw_inc = df['income_total'].clip(lower=0)
df['income_total'] = np.log1p(raw_inc)  # NaN 保留，由 Pipeline Imputer 处理
```

✅ **无全局 median 填充**，符合 Kaufman 2017。

### 1.3 因果模块全局 fit

**结论**：因果推断无 train/test 概念，Imputer/Scaler 在全量 df_sub 上 fit 符合 Chernozhukov 2018 惯例。✅

---

## 阶段二：因果推断假设

### 2.1 重叠假设与 PS 修剪

**检查点**：`charls_causal_assumption_checks.py` L56-68；`charls_recalculate_causal_impact.py` L249 前

**结论**：`check_overlap` 仅报告 pct_trimmed，**未在估计前自动 trimming**。当 pct_trimmed > 10% 时，ATE 可能偏倚（Austin 2011）。

**补丁 2**（在 `estimate_causal_impact_tlearner` 中，L246 之后、L249 之前插入）：

```python
# charls_recalculate_causal_impact.py，在 est = TLearner(...) 之前
# Overlap trimming: 若 PS 超出 [0.05,0.95] 比例>10%，自动修剪 (Austin 2011)
from sklearn.linear_model import LogisticRegression
ps_model = LogisticRegression(max_iter=1000, C=1e-2, solver='lbfgs', random_state=RANDOM_SEED)
ps_model.fit(X_scaled, T_series)
ps = ps_model.predict_proba(X_scaled)[:, 1]
in_support = (ps >= 0.05) & (ps <= 0.95)
if in_support.mean() < 0.9:
    n_before = len(T_series)
    df_sub = df_sub.loc[in_support].reset_index(drop=True)
    X_scaled = X_scaled.loc[in_support].reset_index(drop=True)
    T_series = T_series.loc[in_support].reset_index(drop=True)
    Y_series = Y_series.loc[in_support].reset_index(drop=True)
    X_arr = np.asarray(X_scaled, dtype=np.float64)
    logger.warning(f"Overlap trimming: {in_support.sum()}/{n_before} retained (PS in [0.05,0.95])")
```

### 2.2 协变量平衡（SMD）

**公式**：\( \text{SMD} = \frac{\bar{x}_1 - \bar{x}_0}{\sqrt{(s_1^2 + s_0^2)/2}} \)

**结论**：当前仅未加权 SMD。Austin 2011 建议报告匹配/加权后 SMD。

**补丁 3**（扩展 `check_balance_smd` 支持 PS 权重）：

```python
# charls_causal_assumption_checks.py check_balance_smd 函数
# 新增参数 ps_weights=None
def check_balance_smd(df_sub, treatment_col, output_dir, target_col='is_comorbidity_next', ps_weights=None):
    ...
    if ps_weights is not None:
        w1 = ps_weights[mask1]
        w0 = ps_weights[mask0]
        m1 = np.average(X_arr[mask1], axis=0, weights=w1)
        m0 = np.average(X_arr[mask0], axis=0, weights=w0)
        v1 = np.average((X_arr[mask1] - m1)**2, axis=0, weights=w1)
        v0 = np.average((X_arr[mask0] - m0)**2, axis=0, weights=w0)
        pooled_std = np.sqrt((v1 + v0) / 2)
    else:
        m1 = X_arr[mask1].mean(axis=0)
        m0 = X_arr[mask0].mean(axis=0)
        ...
```

### 2.3 E-value 保守端方向

**验证**：`charls_causal_assumption_checks.py` L164；`charls_recalculate_causal_impact.py` L191

```python
rr_conservative = (r0 + ate_lb) / r0 if ate >= 0 else (r0 + ate_ub) / r0
```

✅ **正确**：有害效应用 ate_lb，保护效应用 ate_ub（VanderWeele & Ding 2017）。

### 2.4 正则化与交叉拟合

✅ TLearner RF max_depth=4；CausalForestDML cv=5, groups, honest=True。

---

## 阶段三：AJE 审稿人对抗性审查

### 漏洞 1：选择偏倚（Selection Bias）

**攻击角度**：运动可能是健康 proxy，基线 ADL 困难者无法运动，T=0 组混入更多功能差者。

**检测代码**：

```python
df_b = df[df['baseline_group']==1]
for col in ['adlab_c', 'iadl', 'disability']:
    if col in df_b.columns:
        t1 = df_b[df_b['exercise']==1][col].mean()
        t0 = df_b[df_b['exercise']==0][col].mean()
        pooled = np.sqrt((df_b[df_b['exercise']==1][col].var() + df_b[df_b['exercise']==0][col].var())/2)
        smd = abs(t1 - t0) / (pooled + 1e-9)
        if smd > 0.2:
            print(f"🔴 选择偏倚：{col} SMD={smd:.2f}>0.2")
```

**修复**（补丁 4）：增加 `exercise × adlab_c` 交互项或按基线 ADL 分层。

```python
# 在 estimate_causal_impact_tlearner 的 X 构建中
# 若 adlab_c 在协变量中且 T 为 exercise，增加交互项
if T == 'exercise' and 'adlab_c' in X_raw.columns:
    X_raw = X_raw.copy()
    X_raw['exercise_x_adl'] = df_sub[T].fillna(0).values * X_raw['adlab_c'].fillna(0).values
    # 注意：需确保 get_exclude_cols 不排除 exercise_x_adl
```

### 漏洞 2：Cohort C 样本量

**结论**：n=2435 对 ATE 可接受；对细粒度 CATE 异质性可能不足。建议文中注明「CATE 为探索性，受限于样本量」。

### 漏洞 3：时变混杂

**结论**：当前设计为 Wave(t) 运动 → Wave(t+1) 结局，未考虑随访期运动变化。建议敏感性分析：仅纳入报告「运动习惯稳定」者，或使用 IPCW 处理失访。

---

## 参考文献

- Austin PC (2011). An introduction to propensity score methods. *Multivariate Behavioral Research*.
- Chernozhukov V et al. (2018). Double/debiased machine learning. *Econometrics Journal*.
- Hernán MA, Robins JM (2020). *Causal Inference: What If*.
- Kaufman S et al. (2017). Leakage in data mining. *ACM TKDD*.
- VanderWeele TJ, Ding P (2017). Sensitivity analysis: Introducing the E-value. *Annals of Internal Medicine*.
- Wager S, Athey S (2018). Estimation and inference of heterogeneous treatment effects using random forests. *JASA*.
