# 医学论文与代码：易错点与可讨论点（备忘）

本文档区分 **「写法/实现上容易被判错」** 与 **「可讨论、需在正文说明」** 两类问题，便于改稿与审稿回复。实现细节以 `config.py` 注释与主流程 `run_all_charls_analyses.py` 为准。

---

## 一、务必与代码一致（避免「一定错」）

1. **因果估计失败不得当 ATE=0**  
   引擎约定：失败时 `res_df is None`，ATE 三元组为 **`(np.nan, np.nan, np.nan)`**（常量 `CAUSAL_FAILURE_ATE_TRIPLET`）。汇总与制表必须以 `res_df is None`（或等价）判失败并记 **NaN**；勿把旧版 `(0,0,0)` 或 NaN 当作真实零效应。

2. **缺失数据与插补层次（2026-03-25 起）**  
   - **CPM / compare_models**：输入为预处理宽表（**保留缺失**）；**`IterativeImputer` 嵌在 Pipeline 内**，仅在 **GroupKFold 训练折** fit，再 transform 验证/测试折。  
   - **因果 / Rubin / 多暴露等**：`USE_IMPUTED_DATA=True` 时仍可用 bulk 生成的 `step1_imputed_full`（`run_cohort_protocol(..., df_for_causal=插补队列切片)`）。  
   - 论文须区分「预测管线内插补」与「因果用单次完成集」，勿再写「全样本 bulk 后再做 CPM」作为唯一叙事。

3. **三队列命名**  
   代码与配置统一为 **Cohort A/B/C**（`COHORT_*_DIR`、`run_cohort_protocol`）。勿再用「axis」指代队列，以免与统计软件中 `axis=0/1` 或生理学「HPA axis」混淆。

4. **consolidate 输出文件名**  
   主流程会同时写入 `*_cohort*` 与 `*_axis*` 文件名（后者为兼容旧稿路径）。新文稿优先引用 **cohort** 命名。

5. **截断敏感性 vs CPM 测试划分**  
   `sensitivity_analysis_readme.txt`（由 `run_sensitivity_scenarios.py` 生成）已说明：插补分支下敏感性基表为 **按 ID 的约 80% 子集**，与 **`USE_TEMPORAL_SPLIT=True` 时 CPM 的「末波测试」** 不是同一划分；写方法或答辩时勿混为一谈。

---

## 二、可讨论、但须在正文透明说明

1. **观察性研究下的因果解释**  
   ATE 依赖无未测混杂等假设；E-value、敏感性分析可减轻但不能消除质疑。结论宜使用「与 … 一致」「提示 …」等表述。

2. **亚组与 CATE**  
   亚组多为探索性；若未做交互检验，避免写「某亚组效应更强」为确定性因果结论。

3. **外部验证**  
   时间/区域划分依赖 CHARLS 可用字段与样本量；泛化外推需对应 limitations。

4. **多重插补与 Rubin 合并**  
   若主表使用 Rubin 合并而 SHAP/部分图仍基于单次插补，应在补充材料说明原因（解释性一致 vs 点估计合并）。

---

## 三、代码中勿改动的 `axis`（非「队列」含义）

- **pandas / NumPy**：`axis=0`、`axis=1` 为 API 参数。  
- **matplotlib**：如 `fig, axes = plt.subplots(...)` 中的 `axes`。  
- **医学叙述**：如「HPA axis」为内分泌术语，与队列无关。

---

*最后更新：与 `COHORT_*` 命名及 `run_cohort_protocol` 重构同步。*
