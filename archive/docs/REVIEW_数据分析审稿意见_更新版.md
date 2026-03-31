# 数据分析流程审稿意见（严苛版·更新）

**审稿时间**：基于当前代码库的再次严苛审阅。  
**说明**：下列“已修复”项为相对初版审稿意见的后续修正状态；“待修正/仍存在”需在定稿前处理。

---

## 一、已修复项（相对初版审稿）

| 项目 | 状态 |
|------|------|
| memeory → memory 拼写 | 已统一为 `memory`（charls_feature_lists 及引用处） |
| 决策支持反事实与干预一致 | 已改为以 exercise/sleep 为反事实（Increase Exercise / Improve Sleep / Combined） |
| Youden 的 confusion_matrix 边界 | 已加 2×2 与双类别检查，避免 ravel 解包报错 |
| 轴线 A 因果未汇报 | 已加注释说明“汇总图仅展示 B/C；A 见 03_causal” |
| 亚组分析未接入主管线 | 已在 run_axis_BC 末尾调用 run_subgroup_analysis |
| 临床评价特征列对齐 | 已从 preprocessor.transformers_ 取输入列，支持 CalibratedClassifierCV |
| 排除列表集中定义 | 已建 charls_feature_lists.get_exclude_cols，因果/模型比较已引用 |
| 主流程随机种子 | 已在 main() 开头设置 random.seed(500), np.random.seed(500) |
| 因果模块 communityID 缺失 | 已用 communityID 或 ID 兜底 |
| Bootstrap 有效迭代过少 | 已对 bootstrapped_stats 长度检查，<10 时用点估计作 CI |
| Table1 分类水平异常 | 已对 levels 计算加 try/except，避免 astype(int) 报错 |

---

## 二、仍存在或需注意的问题

### 1. 【方法学】分类变量：LabelEncoder + StandardScaler（与 self.md 一致）

**位置**：`charls_complete_preprocessing.py` 对 province/edu/marry 等做 LabelEncoder；`charls_model_comparison` / `charls_recalculate_causal_impact` 对数值列中“连续”子集做 StandardScaler，其余列仅插补不缩放。

**问题**：名义变量被编码为 0,1,2,… 后，若进入“连续”列表会被标准化，引入虚假顺序与尺度；若在 pass 列则仅插补，仍为整数，树模型可用，但线性/距离类模型仍不当。

**建议**：名义变量要么 One-Hot 后不再标准化，要么仅让树模型使用、不参与线性部分；或在配置中明确将 edu/marry/province 等从 CONTINUOUS_FOR_SCALING 排除（当前已排除，仅真正连续变量在 scale_cols）。**建议在方法部分写明**：“分类变量经标签编码后仅用于树模型；连续变量经标准化。”

---

### 2. 【一致性】汇总图输出目录与历史命名

**位置**：`run_all_charls_analyses.py` 使用 `final_dir = 'LIU_JUE_STRATEGIC_SUMMARY'`。

**问题**：若历史结果或文档中使用 `final_strategic_summary`（小写/不同名），会导致“结果目录不统一”的困惑。

**建议**：在正文或 README 中明确“所有汇总图表输出目录为 `LIU_JUE_STRATEGIC_SUMMARY`”；若需兼容旧路径，可增加一次写往 `final_strategic_summary` 的拷贝或软链。

---

### 3. 【稳健性】estimate_causal_impact 返回 None 时主流程未显式处理

**位置**：`run_axis_BC_protocol` 中 `res_df, (ate, ate_lb, ate_ub) = estimate_causal_impact(...)`，当因果拟合失败时返回 `(None, (0,0,0))`。

**问题**：此时 `res_df` 为 None，但后续使用的是 `df_sub`（未修改），且亚组分析已按“是否存在 causal_impact_* 列”跳过，故不会因 res_df 为 None 而报错。逻辑正确，但若审稿人追踪“因果失败时是否还写 CSV”，需知：失败时不会写 `CAUSAL_ANALYSIS_*.csv`，且 df_sub 无 causal_impact 列。

**建议**：在方法或补充材料中简短说明：“若某亚组因果估计未收敛，该亚组不输出因果 CSV，且不进行亚组异质性分析。”当前代码行为已符合。

---

### 4. 【表格规范】Table1 输出目录与主流程一致

**位置**：`generate_baseline_table(df_clean, output_dir=final_dir)` 已传入 `final_dir='LIU_JUE_STRATEGIC_SUMMARY'`。

**状态**：Table1 与 incidence/ATE 图同目录，一致。无需修改。

---

### 5. 【可复现性】CalibratedClassifierCV 的 cv 未按 GroupKFold

**位置**：`charls_model_comparison.py` 中对最优模型做 `CalibratedClassifierCV(best_pipe, cv=3, method='isotonic')`。

**问题**：校准时使用默认 3-fold（非按 ID 分组），同一 ID 可能同时出现在训练与校准折中，存在轻微信息泄露风险。

**建议**：若需严格按个体分组，可改为 `cv=GroupKFold(n_splits=3)` 并传入 `groups=df.iloc[train_idx]['ID']`；当前实现优先保证稳定与可运行，可在修订稿中注明“校准使用 3-fold CV，未按个体分组”。

---

## 三、优先修改顺序（更新）

| 优先级 | 类型 | 项目 | 建议 |
|--------|------|------|------|
| P0 | — | 无未解决致命错误 | 已修复项见上文 |
| P1 | 方法学 | 分类变量处理说明 | 在方法中明确“分类仅标签编码、连续变量标准化”及使用范围 |
| P2 | 一致性 | 汇总目录命名 | 文档/README 统一为 LIU_JUE_STRATEGIC_SUMMARY 或兼容旧名 |
| P2 | 可复现性 | 校准 CV 分组 | 可选：校准阶段使用 GroupKFold；或方法中说明当前 cv=3 未分组 |

---

## 四、结论

- **致命错误**：初版中的 memeory、决策支持反事实、Youden 崩溃、Bootstrap 空列表等已修复；当前未发现新的致命错误。
- **逻辑一致性**：轴线 A 因果汇报方式已注释明确；亚组分析已接入；干预与反事实已统一为 exercise/sleep。
- **建议**：在论文方法部分补充“分类变量与连续变量的处理及校准 CV 的设定”的简短说明，并统一结果目录的命名约定。其余按 P1–P2 酌情在修订稿中落实即可。
