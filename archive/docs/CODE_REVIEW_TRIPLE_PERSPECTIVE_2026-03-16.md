# CHARLS 因果机器学习：三重视角代码审查报告

**审查日期**：2026-03-16  
**视角**：生物统计学家 + 因果推断专家 + AJE 审稿人（对抗性）

---

## 一、生物统计学家审查清单

### 1.1 数据划分：训练集时间 < 测试集时间

| 状态 | 说明 | 代码定位 |
|------|------|----------|
| ❌ | 主预测模型使用 `GroupShuffleSplit` 随机划分，**未**按 wave 时间划分 | `charls_model_comparison.py` L75-77 |

**当前代码**：
```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(X, y, groups=df['ID']))
```

**修复**（可选，作为时间验证）：
```python
# 时间划分：训练=wave<max_wave，测试=wave=max_wave
if use_temporal_split and 'wave' in df.columns:
    max_wave = df['wave'].max()
    train_mask = df['wave'] < max_wave
    test_mask = df['wave'] == max_wave
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
else:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=df['ID']))
```

**注**：`charls_external_validation.py` L109-112 已单独做时间验证（wave<max vs wave=max），但冠军模型选择仍基于随机划分。

---

### 1.2 缺失值：训练集 fit、测试集 transform

| 状态 | 说明 | 代码定位 |
|------|------|----------|
| ✅ | 预测模型：Imputer/Scaler 在 Pipeline 内，仅 CV 训练折 fit | `charls_model_comparison.py` L94-100, L169 |
| ✅ | 预处理 income：已移除全量中位数，留待 Pipeline 处理 | `charls_complete_preprocessing.py` L55-59 |
| ⚠️ | 因果模块：Imputer/Scaler 在全量 df_sub 上 fit | `charls_recalculate_causal_impact.py` L86-94, L235-240 |

**因果模块说明**：因果推断无 train/test 划分，全量用于 ATE 估计；DML/TLearner 内部无「测试集」概念，全量 fit 符合惯例（Chernozhukov 2018）。

---

### 1.3 聚类：按个体 ID 分组

| 状态 | 说明 | 代码定位 |
|------|------|----------|
| ✅ | 外划分：`GroupShuffleSplit(..., groups=df['ID'])` | `charls_model_comparison.py` L77 |
| ✅ | 内 CV：`GroupKFold` + `groups=df.iloc[train_idx]['ID']` | L104, L169, L175-176 |
| ✅ | RandomizedSearchCV 传入 groups | L169 |

---

### 1.4 因果：倾向得分 common support

| 状态 | 说明 | 代码定位 |
|------|------|----------|
| ✅ | PS 超出 [0.05, 0.95] 比例计算并报告 | `charls_causal_assumption_checks.py` L56-59, L77 |
| ⚠️ | 当前阈值：`overlap_ok = pct_trimmed < 10`（10%），非 5% | L68 |

**Austin (2011)** 建议：PS 超出 [0.1, 0.9] 或 [0.05, 0.95] 比例过高时需 trimming。若审稿要求 <5%，可收紧：

```python
# 当前
overlap_ok = pct_trimmed < 10

# 若要求 <5%
overlap_ok = pct_trimmed < 5
```

---

### 1.5 时序：暴露早于结局至少一个 wave

| 状态 | 说明 | 代码定位 |
|------|------|----------|
| ✅ | 每行 = Wave(t)，Y 取自 Wave(t+1)，显式校验 | `charls_complete_preprocessing.py` L84-95, L104-111 |
| ✅ | docstring 明确 T/X 取自 Wave(t)，Y 取自 Wave(t+1) | `charls_recalculate_causal_impact.py` L50-52 |

---

## 二、因果推断专家审查（DML / 重叠 / 平衡 / 敏感性）

### 2.1 正则化偏倚与交叉拟合

| 检查项 | 主方法 TLearner | CausalForestDML（备用） |
|--------|-----------------|-------------------------|
| 正则化 | RF max_depth=4, n_estimators=200 | RF max_depth=4, min_samples_leaf=15 |
| 交叉拟合 | 无（TLearner 无内置 cross-fitting） | cv=5 |
| Folds 按 ID | N/A | groups=cluster_ids |

**主因果方法**：`config.CAUSAL_METHOD = 'TLearner'`，当前为 TLearner。

**TLearner 正则化**：`charls_recalculate_causal_impact.py` L249-252
```python
RandomForestRegressor(n_estimators=200, max_depth=4, random_state=RANDOM_SEED)
```
- max_depth=4 偏弱正则化，有助于减少正则化偏倚（Chernozhukov 2018）。

**CausalForestDML**（若切换）：L100-104
- cv=5 交叉拟合 ✅
- groups=cluster_ids（communityID 或 ID）✅
- min_samples_leaf=15 偏强正则化，可降低过拟合风险。

---

### 2.2 重叠假设（Overlap）

| 公式 | 说明 |
|------|------|
| \( \text{PS} = P(T=1|X) \) | 倾向评分 |
| \( \text{trimmed} = \#\{i: \text{PS}_i \notin [0.05, 0.95]\} \) | 超出支撑域样本数 |
| \( \text{pct\_trimmed} = 100 \times \text{trimmed} / n \) | 超出比例 |

**代码定位**：`charls_causal_assumption_checks.py` L56-59, L77

**当前**：计算并报告 pct_trimmed，overlap_ok = pct_trimmed < 10。

**若 pct_trimmed > 10%**：需实现 trimming 或 overlap weighting。补丁示例：

```python
# 在 estimate_causal_impact_tlearner 中，fit 之前增加：
ps_model = LogisticRegression(max_iter=1000, C=1e-2, solver='lbfgs', random_state=RANDOM_SEED)
ps_model.fit(X_scaled, T_series)
ps = ps_model.predict_proba(X_scaled)[:, 1]
in_support = (ps >= 0.05) & (ps <= 0.95)
if in_support.sum() < len(T_series) * 0.9:
    df_sub = df_sub.loc[in_support].reset_index(drop=True)
    X_scaled = X_scaled[in_support]
    T_series = T_series[in_support]
    Y_series = Y_series[in_support]
    logger.warning(f"Overlap trimming: {len(in_support)}/{len(ps)} retained")
```

---

### 2.3 协变量平衡（SMD）

**公式**（Austin 2011）：
\[
\text{SMD} = \frac{\bar{x}_1 - \bar{x}_0}{\sqrt{(s_1^2 + s_0^2)/2}}
\]

**当前实现**：`charls_causal_assumption_checks.py` L117-126
- 计算**未加权** SMD（匹配/加权前）

**缺失**：PS 加权后的 SMD（PSW 或 PSM 后）。Austin (2011) 建议报告匹配后 SMD。

**补丁**（PS 加权 SMD）：
```python
# 在 check_balance_smd 中，若传入 ps_weights：
def check_balance_smd(df_sub, treatment_col, output_dir, ps_weights=None, ...):
    ...
    if ps_weights is not None:
        w1 = ps_weights[mask1]
        w0 = ps_weights[mask0]
        m1 = np.average(X_arr[mask1], axis=0, weights=w1)
        m0 = np.average(X_arr[mask0], axis=0, weights=w0)
        m1_sq = np.average((X_arr[mask1] - m1)**2, axis=0, weights=w1)
        m0_sq = np.average((X_arr[mask0] - m0)**2, axis=0, weights=w0)
        pooled_std = np.sqrt((m1_sq + m0_sq) / 2)
    else:
        # 现有未加权逻辑
        ...
```

---

### 2.4 敏感性分析：E-value 与 Bias Curve

**E-value 公式**（VanderWeele & Ding 2017）：
\[
RR = (r_0 + \text{ATE}) / r_0, \quad E = RR + \sqrt{RR(RR-1)}
\]
保守端：取 CI 中离 null 最近的一端（有害用 ate_lb，保护用 ate_ub）。

**当前实现**：`charls_causal_assumption_checks.py` L163-167
```python
rr_conservative = (r0 + ate_lb) / r0 if ate >= 0 else (r0 + ate_ub) / r0
```
✅ 已修复（2026-03-16）

**缺失**：Bias Curve（未测量混杂强度 = 观察协变量最强者时的偏倚边界）。可补充：

```python
# 伪代码：Bias Curve
# 对每个观察协变量 X_j，计算其与 T、Y 的关联强度，得到 max_RR_observed
# 绘制：当未测量混杂 RR = max_RR_observed 时，ATE 偏倚边界
# 若 E-value > max_RR_observed，则观察协变量无法解释效应
```

---

## 三、AJE 审稿人对抗性任务：3 个可能推翻结论的漏洞

### 漏洞 1：主预测未使用时间划分，AUC 可能偏乐观

**假设情景**：若存在时间漂移（早期 Wave 1–2 vs 晚期 Wave 3–4），随机划分可能让训练集包含「未来」信息（例如测试集为 Wave 2，训练集含 Wave 3），导致 AUC 高估。

**检测代码**：
```python
# 在 run_all_charls_analyses 或 compare_models 后添加
if 'wave' in df.columns:
    max_w = df['wave'].max()
    train_waves = df.iloc[train_idx]['wave'].unique()
    test_waves = df.iloc[test_idx]['wave'].unique()
    if max(train_waves) > min(test_waves):
        logger.warning("⚠️ 时序泄露：训练集含晚于测试集的 wave")
```

**修复**：增加时间划分选项，或明确报告「主 AUC 基于随机划分，时间验证见 run_external_validation」。

---

### 漏洞 2：运动可能为健康状态 proxy，存在选择偏倚

**假设情景**：基线 ADL 困难者无法规律运动，T=0 组可能混入更多基线功能差者，导致「因健康差而不能运动」的选择偏倚。

**证据**：`adlab_c`、`iadl`、`disability` 已在协变量中（`charls_feature_lists.py` L27），用于调整。

**检测代码**：
```python
# 检查运动组 vs 对照组的基线 ADL 分布
import pandas as pd
df_b = df[df['baseline_group']==1]  # Cohort B
for col in ['adlab_c', 'iadl', 'disability']:
    if col in df_b.columns:
        t1 = df_b[df_b['exercise']==1][col].mean()
        t0 = df_b[df_b['exercise']==0][col].mean()
        smd = abs(t1 - t0) / df_b[col].std()
        print(f"{col}: SMD={smd:.3f}")  # 若 SMD>0.2 需关注
```

**修复**：增加 `exercise × adlab_c` 或 `exercise × disability` 交互项，或按基线 ADL 分层估计 ATE。

**补丁**：在 `estimate_causal_impact_tlearner` 的 X 中增加交互项：
```python
if 'adlab_c' in X_raw.columns:
    X_raw['exercise_x_adl'] = X_raw['adlab_c'] * df_sub[T].fillna(0)
```
（需同时排除 `exercise` 以避免共线）

---

### 漏洞 3：Cohort C 样本量对 CATE 异质性分析可能不足

**假设情景**：Cohort C（Cognition）n≈2435，若 Causal Forest 用于 CATE 异质性分析，样本量可能不足。

**理论依据**：Wager & Athey (2018) 未给出明确最小 n；GRF 实现中 sample.fraction×honesty.fraction 会进一步减少有效样本。Athey & Wager (2019) 建议 n 足够大以支持子群划分。

**样本量估算**（简化）：
- ATE：n>500 通常可接受
- CATE 异质性：若按 5 个协变量分层，每层约 n/32，若 n=2435 则每层约 76，可能偏少

**检测代码**：
```python
# 在因果分析后
n = len(df_sub)
n_events = int(df_sub['is_comorbidity_next'].sum())
print(f"Cohort C: n={n}, events={n_events}")
# 若 n_events < 30 或 n < 500，建议谨慎解读 CATE
```

**修复**：在论文中明确说明「CATE 异质性分析为探索性，Cohort C 样本量有限」；或仅报告 ATE，不强调细粒度 CATE。

---

## 四、汇总表

| 清单项 | 状态 | 行号 |
|--------|------|------|
| 数据划分（时间） | ❌ | charls_model_comparison L75-77 |
| 缺失值（Pipeline） | ✅ | charls_model_comparison L94-100 |
| 聚类（GroupKFold） | ✅ | charls_model_comparison L77, L104, L169 |
| 倾向得分 common support | ✅（阈值 10%） | charls_causal_assumption_checks L56-68 |
| 时序（T 早于 Y） | ✅ | charls_complete_preprocessing L84-111 |
| E-value 保守端 | ✅ | charls_causal_assumption_checks L163-167 |
| SMD 加权 | ⚠️ 仅未加权 | charls_causal_assumption_checks L117-126 |
| Bias Curve | ❌ 未实现 | - |
| Overlap trimming | ⚠️ 仅报告，未自动 trim | - |

---

## 五、参考文献

- Austin PC (2011). An introduction to propensity score methods. *Multivariate Behavioral Research*.
- Chernozhukov V et al. (2018). Double/debiased machine learning. *Econometrics Journal*.
- VanderWeele TJ, Ding P (2017). Sensitivity analysis: Introducing the E-value. *Annals of Internal Medicine*.
- Wager S, Athey S (2018). Estimation and inference of heterogeneous treatment effects using random forests. *JASA*.
