# 数据分析流程审稿意见（严苛版）

## 一、必须修正的错误

### 1. 拼写错误导致泄露风险：`memeory` → `memory`

**位置**：多处 `leakage_keywords` 或排除列表。

**问题**：`memeory` 无法匹配列名中的 `memory`（如认知/记忆相关变量），导致本应排除的“记忆/认知”相关列未被排除，存在**信息泄露**风险。

**涉及文件**：
- `charls_model_comparison.py`（第56行）
- `charls_recalculate_causal_impact.py`（第38行）
- `charls_ablation_study.py`（第32行）
- `charls_methodology_audit.py`（第31行）
- `charls_clinical_evaluation.py`（第26行，排除列表中的列名）
- `charls_clinical_decision_support.py`（第28行）
- `charls_cate_visualization.py`（第32行）
- `charls_sensitivity_analysis.py`（第47行）
- `charls_validation_suite.py`（第32行）
- `charls_visual_enhancement.py`（第51行）
- `replot_main_figures.py`（第20行）

**正确**：统一改为 `'memory'`。仅 `charls_shap_analysis.py` 已使用正确拼写。

---

### 2. 因果模块与决策支持中“干预”定义不一致

**位置**：`charls_clinical_decision_support.py` 第63–68行。

**问题**：全管线将 **exercise（运动）** 作为干预变量做因果估计，但决策支持里的反事实逻辑用 **is_depression / is_cognitive_impairment** 作为“治疗”并做置 0 操作；且因 `target_col` 固定为 `is_comorbidity_next`，分支永远走 `is_cognitive_impairment`，从未对抑郁轴做“移除治疗”的反事实。与“运动干预 → 共病”的因果叙事不一致，且 B/C 两轴语义不对称。

**建议**：
- 反事实应围绕 **exercise**（及可选的 sleep 等）做“若增加运动/改善睡眠”的对比；或
- 若坚持用“疾病状态”做反事实，需按当前轴线（A/B/C）或 `baseline_group` 显式选择操纵变量（抑郁轴操纵抑郁相关、认知轴操纵认知相关），并与正文“干预”定义一致。

---

### 3. 分类变量预处理与因果/预测假设不符（与 self.md 已有记录一致）

**位置**：`charls_complete_preprocessing.py` 用 `LabelEncoder` 编码；`charls_recalculate_causal_impact.py` 与 `charls_model_comparison.py` 对数值矩阵再做 `StandardScaler`。

**问题**：名义型分类变量（如 province、edu、marry）被标签编码后当作连续变量标准化，在因果森林和线性/树模型中会引入顺序假设和尺度扭曲，与“类别无自然顺序”的假设不符。

**建议**：对名义变量采用 One-Hot 或交给支持类别特征的模型（如 CatBoost）且不标准化；仅对连续变量做 StandardScaler。

---

### 4. Bootstrap 中 Youden 计算的稳健性

**位置**：`charls_ci_utils.py` 第12–16行，`calculate_youden` 内 `confusion_matrix(...).ravel()`。

**问题**：当某次 Bootstrap 子样中仅有一个类别时，`confusion_matrix` 可能返回非 2×2 矩阵，`ravel()` 后无法解包为 `tn, fp, fn, tp`，会触发 **ValueError**。当前循环中有 `if len(np.unique(y_true_b)) < 2: continue`，但未对 `y_pred_b` 做类似保护，若预测全为同一类也会出现 1×1 混淆矩阵。

**建议**：在 `calculate_youden` 内先检查 `confusion_matrix` 形状是否为 2×2，或 `len(np.unique(y_t))==2 and len(np.unique(y_p))==2`；否则返回 NaN 或 0，并在外层 bootstrap 中过滤/插补。

---

## 二、逻辑不一致与设计问题

### 5. 轴线 A 的因果结果未纳入最终汇总

**位置**：`run_all_charls_analyses.py` 中 `run_axis_A_protocol` 调用了 `estimate_causal_impact(df_a, ...)`，但只返回 `(inc_a, auc_a)`；`main()` 中仅对 B、C 绘制 ATE 对比图（`intervention_benefit_comparison.png`），**轴线 A（健康组）的 ATE 未在汇总图中展示**。

**问题**：若审稿人或读者默认“三条轴线都有因果汇报”，会认为 A 缺失；若设计上故意只报告疾病组因果，则需在方法或图注中明确说明“仅报告由抑郁/认知受损出发的亚组因果效应”。

**建议**：二选一：要么在汇总中加入轴线 A 的 ATE（并注明为“健康人群运动对共病发生的影响”）；要么在代码注释与论文方法/图注中明确“轴线 A 仅做预测与 SHAP，因果效应不在此汇报”。

---

### 6. 亚组分析未接入主管线

**位置**：`run_all_charls_analyses.py` 从 `charls_subgroup_analysis` 导入了 `run_subgroup_analysis`，但 `main()` 及 `run_axis_*` 中**从未调用** `run_subgroup_analysis`。

**问题**：亚组分析（城乡、年龄、性别）若为论文一部分，则主流程未跑会与文中“方法/结果”不一致；若为可选分析，则易造成“代码存在但未使用”的困惑。

**建议**：若为既定分析，在 B/C 某一轴或两轴后增加 `run_subgroup_analysis(df_sub, output_dir=...)`；否则在注释或文档中说明该模块为可选/独立脚本。

---

### 7. 临床评价中“模型输入”与训练时特征集的一致性

**位置**：`charls_clinical_evaluation.py` 第38–61行。

**问题**：当从模型取 `X_cols = get_feature_names_out()` 时，得到的是**变换后**的特征名（如 `num__age`），而后面又用 `X = df` 传入 `predict_proba`。Pipeline 的 ColumnTransformer 是按**拟合时的列名**从 df 中取列，因此当前写法在多数情况下可运行，但依赖“df 包含且仅多不少地包含训练时的数值列”。若 df 列顺序或列集与训练时有差异，可能静默出错或列对齐异常。

**建议**：显式用训练时使用的原始特征列（可从 `preprocessor.transformers_[0][2]` 或保存的 `feature_names_in_` 等获取）从 df 中取子集再传入 `predict_proba`，并做一次列存在性检查，避免静默不对齐。

---

## 三、前后一致性与可复现性

### 8. 排除列表与关键字在各脚本间不统一

**问题**：排除列/泄露关键字在 `charls_shap_analysis.py`、`charls_model_comparison.py`、`charls_recalculate_causal_impact.py`、`charls_clinical_evaluation.py`、`charls_clinical_decision_support.py` 等处**各自维护**，存在：
- 拼写不一致（如 memeory vs memory）；
- 有的用关键字扩展，有的用固定列表，易漏列或重复。

**建议**：在 `charls_ci_utils.py` 或单独 `charls_feature_lists.py` 中集中定义：
- `LEAKAGE_KEYWORDS`
- `EXCLUDE_COLS_BASE`
以及“由关键字扩展的排除列”的生成函数，各模块统一引用，便于审稿与修改。

---

### 9. 随机种子使用情况

**现状**：`run_all_charls_analyses.py` 未设置全局 `np.random.seed`/`random_state`；各子模块内部有 500 等固定种子，但若调用顺序或环境不同，整体流程的可复现性仍依赖各子模块是否处处固定种子。

**建议**：在 `main()` 开头统一设置 `np.random.seed(500)` 和 `random.seed(500)`，并在文档中注明“复现需使用相同 Python 版本与依赖版本”。

---

## 四、小结与优先修改顺序

| 优先级 | 类型     | 项目 |
|--------|----------|------|
| P0     | 错误     | 1. memeory → memory（全库统一） |
| P0     | 错误     | 2. 决策支持反事实与“运动干预”一致 |
| P1     | 方法     | 3. 分类变量不标准化 / One-Hot 或 CatBoost 原生类别 |
| P1     | 稳健性   | 4. Youden 的 confusion_matrix 边界处理 |
| P2     | 一致性   | 5. 轴线 A 因果是否汇报的明确化 |
| P2     | 一致性   | 6. 亚组分析是否纳入主流程的明确化 |
| P2     | 可维护性 | 7. 临床评价中特征列显式对齐 |
| P2     | 可维护性 | 8. 排除列表/关键字集中定义 |
| P2     | 可复现性 | 9. 主流程入口固定随机种子 |

建议先完成 P0，再按 P1–P2 在修订稿中逐项落实并更新方法学描述与图注。
