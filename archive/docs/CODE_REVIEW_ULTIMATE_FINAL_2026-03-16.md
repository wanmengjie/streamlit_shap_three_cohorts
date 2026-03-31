# CHARLS 因果机器学习：终极审查报告（三阶段整合）

**审查日期**：2026-03-16  
**角色**：生物统计学家 + 因果推断专家 + AJE 审稿人

---

## 审查结果汇总表

| 等级 | 文件 | 行号 | 问题描述 | 理论依据 | 修复状态 |
|------|------|------|----------|----------|----------|
| 🔴致命 | charls_model_comparison.py | L75-77 | 主预测随机划分，不保证 train_wave < test_wave，时序泄露风险 | TRIPOD | ✅ 已增加 USE_TEMPORAL_SPLIT 选项 |
| 🟢建议 | charls_complete_preprocessing.py | L55-59 | income_total 无全局 median，NaN 留待 Pipeline | Kaufman 2017 | ✅ 已合规 |
| 🟢建议 | charls_model_comparison.py | L77 | GroupShuffleSplit groups=ID，同一人不跨集 | Bischl 2012 | ✅ 已合规 |
| 🔴致命 | charls_recalculate_causal_impact.py | L249 前 | TLearner 未做 overlap trimming | Austin 2011 | ✅ 已增加自动修剪 |
| 🟡警告 | charls_causal_assumption_checks.py | L104-126 | 仅未加权 SMD，缺失 PS 加权 SMD | Austin 2011 | ✅ 已扩展 ps_weights 参数 |
| 🟢建议 | charls_causal_assumption_checks.py | L163-167 | E-value 保守端方向正确 | VanderWeele 2017 | ✅ 已合规 |
| 🔴致命 | charls_recalculate_causal_impact.py | L228-233 | 选择偏倚：运动为健康 proxy | Hernán 2020 | ✅ 已增加 exercise×adlab_c 交互 |
| 🟡警告 | - | - | Cohort C n=2435 对 CATE 可能不足 | Wager 2018 | 文中说明 CATE 为探索性 |
| 🟡警告 | config.py | - | 时变混杂敏感性选项 | Hernán 2010 | ✅ 已增加 EXERCISE_STABLE_ONLY |

---

## 已实施修复摘要

### 1. 时间划分选项（config.py + charls_model_comparison.py）

```python
# config.py 新增
USE_TEMPORAL_SPLIT = False  # True 时 train=wave<max, test=wave=max
EXERCISE_STABLE_ONLY = False

# charls_model_comparison.py L74 后
if USE_TEMPORAL_SPLIT and 'wave' in df.columns:
    max_wave = df['wave'].max()
    train_mask = (df['wave'] < max_wave).values
    test_mask = (df['wave'] == max_wave).values
    if train_mask.sum() >= 50 and test_mask.sum() >= 20:
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
```

### 2. Overlap 修剪（charls_recalculate_causal_impact.py）

```python
# L248 前插入：当 pct_trimmed > 10% 时自动修剪
ps_model.fit(X_scaled, T_series)
ps = ps_model.predict_proba(X_scaled)[:, 1]
in_support = (ps >= 0.05) & (ps <= 0.95)
if in_support.mean() < 0.9:
    df_sub = df_sub.loc[in_support].reset_index(drop=True)
    X_scaled = X_scaled.loc[in_support].reset_index(drop=True)
    ...
```

### 3. SMD 加权扩展（charls_causal_assumption_checks.py）

```python
def check_balance_smd(..., ps_weights=None):
    if ps_weights is not None:
        w1, w0 = ps_weights[mask1], ps_weights[mask0]
        m1 = np.average(X_arr[mask1], axis=0, weights=w1)
        ...
```

### 4. 选择偏倚交互项（charls_recalculate_causal_impact.py）

```python
if T == 'exercise' and 'adlab_c' in X_raw.columns:
    X_raw['exercise_x_adl'] = df_sub[T].fillna(0).values * X_raw['adlab_c'].fillna(0).values
```

---

## 验证清单（运行 scripts/run_verification_checklist.py）

```bash
python scripts/run_verification_checklist.py [结果目录]
```

| 验证项 | 检查内容 | 断言 |
|--------|----------|------|
| 1 时序划分 | USE_TEMPORAL_SPLIT=True 时 train_wave < test_wave | 配置检查 |
| 2 Overlap trimming | ATE_CI_summary 含 overlap_trimmed_pct | 已写入文件 |
| 3 交互项 | exercise_x_adl 在 T=exercise 时生成 | 代码检查 |
| 4 加权 SMD | assumption_balance_*_weighted.txt 或 smd_weighted | run_all_assumption_checks 自动计算 |
| 5 E-value 方向 | 保护用 ate_ub，有害用 ate_lb | 代码逻辑检查 |

---

## 参考文献

- Austin PC (2011). Propensity score methods.
- Chernozhukov V et al. (2018). Double/debiased machine learning.
- Hernán MA, Robins JM (2020). *Causal Inference: What If*.
- Kaufman S et al. (2017). Leakage in data mining.
- VanderWeele TJ, Ding P (2017). Sensitivity analysis: E-value.
- Wager S, Athey S (2018). Causal Forests.
