# 审稿意见综合回应清单

**生成日期**：2026-03-14  
**对应稿件**：PAPER_完整版_2026-03-20.md

本文档按审稿人意见分类，逐项标注：**代码/数据现状**、**建议补充内容**、**优先级**。

---

## 一、方法学：因果推断假设验证

### 1.1 未测量混杂敏感性分析拓展

| 审稿意见 | 代码/数据现状 | 建议补充 | 优先级 |
|----------|---------------|----------|--------|
| 仅计算 E 值，可补充单因素/多因素未测量混杂模拟（假设存在未测量协变量，其与暴露、结局的关联强度对 ATE 的影响） | **部分已有**：`utils/charls_bias_analysis.py` 有 `fig_bias_sensitivity.png`（模拟未测量混杂强度对 ATE 的影响）；E-value 已计算 | 1) 在 Methods §2.7 补充：除 E-value 外，进行了偏倚敏感性模拟（假设存在与暴露、结局关联强度为 RR 的未测量混杂，观察 ATE 随 RR 变化）；2) 在 Results 引用 `Cohort_*/07_sensitivity/sensitivity_exercise/fig_bias_sensitivity.png` | 高 |

### 1.2 PSM/PSW 具体操作

| 审稿意见 | 代码/数据现状 | 建议补充 | 优先级 |
|----------|---------------|----------|--------|
| **PSM 卡尺**：0.024×SD of PS 的计算过程与确定依据 | **已有**：`config.CALIPER_PSM=0.024`；`charls_causal_methods_comparison._ate_psm` 中 `caliper_val = caliper * sd_ps`，`sd_ps = np.std(ps)` | 在 Methods 补充：PSM 采用 1:1 最近邻匹配，卡尺 = 0.024 × SD(倾向得分)。依据 Austin (Stat Med 2011)：常用 0.2×SD(logit PS)；本研究采用 0.024×SD(PS) 以在平衡与保留样本间折中（若 SD≈0.12 则约 0.2×SD）。敏感性分析见 PAPER_DRAFT_FINAL_SUBMISSION（0.1×SD、0.3×SD 对 SMD 影响小） | 高 |
| **PSM 匹配后 SMD**：需补充所有协变量的 SMD 值，而非仅 max SMD | **已有**：`Cohort_*/03_causal/assumption_balance_smd_exercise.csv` 含各协变量 SMD | 1) 在 Supplementary 增加 Table：各协变量匹配前后 SMD；2) 或引用 `assumption_balance_smd_*.csv` 并说明可复现 | 中 |
| **PSW 权重修剪**：[0.1, 50] 的筛选标准 | **已有**：`_ate_psw` 中先 1–99 分位数裁剪，再 `np.clip(w, 0.1, 50)`；注释引用 Sturmer et al. 2020 | 在 Methods 补充：PSW 采用 IPW，权重裁剪至 [0.1, 50]，遵循 Sturmer et al. (Epidemiology 2020) 以减少极端权重对估计的偏倚；倾向得分限制在 [0.01, 0.99] | 高 |

### 1.3 XLearner 模型调优细节

| 审稿意见 | 代码/数据现状 | 建议补充 | 优先级 |
|----------|---------------|----------|--------|
| 随机森林干扰模型超参数调优过程（网格/随机搜索范围、最优参数） | **固定参数**：`charls_recalculate_causal_impact` 中 `n_estimators=200, max_depth=4, min_samples_leaf=15`，无调优 | 在 Methods 补充：XLearner 的 nuisance 模型（μ₀、μ₁、π）采用固定超参数 Random Forest（n_estimators=200, max_depth=4, min_samples_leaf=15），基于 CHARLS 高维协变量与样本量选择的防过拟合配置，未进行网格搜索 | 高 |
| Bootstrap 重抽样具体实现（200 次、分层依据） | **已有**：`charls_recalculate_causal_impact` 与 `charls_causal_multi_method` 中 `for _ in range(200)` 非分层 bootstrap | 在 Methods 补充：ATE 的 95% CI 通过 200 次非分层 bootstrap 重抽样获得（当 closed-form 区间不可用时）；每次重抽样在完整样本上放回抽样，重新拟合 XLearner 并估计 ATE | 高 |

---

## 二、方法学：研究人群纳入/排除

### 2.1 CHARLS 波浪数据

| 审稿意见 | 代码/数据现状 | 建议补充 | 优先级 |
|----------|---------------|----------|--------|
| 明确纳入的波浪（如 2011–2013、2013–2015 等）及基线/随访波浪对应关系 | **已有**：`charls_complete_preprocessing` 中 `next_wave_val == wave + 1`，即紧邻下一波；`wave` 列存在；`archive/draw_framework_improved` 提及 "Waves 1-4" | 在 §2.1 补充：CHARLS 波次编码为 wave 1–4（对应 2011、2013、2015、2018 年）；每行 = 个体在 Wave(t)；暴露/协变量取自 Wave(t)，结局 is_comorbidity_next 取自 Wave(t+1)；仅保留 `next_wave_val == wave + 1` 的紧邻两波观测 | 高 |

### 2.2 排除标准具体执行

| 审稿意见 | 代码/数据现状 | 建议补充 | 优先级 |
|----------|---------------|----------|--------|
| “既往存在共病” 的定义（既往哪一波浪的 CESD/认知评分满足共病标准） | **已有**：`had_comorbidity_before = groupby(ID)[is_comorbidity].transform(lambda x: x.shift(1).fillna(0).cummax())`；即任一波 t 之前若任一波次 is_comorbidity=1 则排除 | 在 §2.1 或 Figure 1 脚注补充：既往共病 = 该个体在基线波次之前任一波次同时满足 CES-D≥10 且认知得分≤10 | 中 |
| “缺失结局” 的具体判定（随访波浪中 CESD/认知评分缺失比例） | **已有**：`df.dropna(subset=['is_comorbidity_next'])`；attrition 表有 "Next-wave comorbidity non-missing" | 在 Figure 1 或补充材料补充各排除步骤的 N 与占比（attrition_flow.csv 已有） | 中 |

---

## 三、结果：基线特征

### 3.1 Table 1 连续变量

| 审稿意见 | 代码/数据现状 | 建议补充 | 优先级 |
|----------|---------------|----------|--------|
| 补充中位数、四分位数（老年人数据多偏态） | **需新增**：`run_baseline_table_only` 或 `charls_table1_stats` 可能仅输出均值±SD | 修改 Table 1 生成逻辑，连续变量增加 Median (IQR)；或在表注说明“偏态变量见补充表” | 中 |
| 各队列间两两比较 P 值（A/B、A/C、B/C） | **需新增** | Kruskal-Wallis 事后检验或 χ² 分割；可写脚本输出 | 中 |
| 可干预生活方式因素基线分布（运动、饮酒、社会隔离构成比）及检验 | **部分已有**：Table 1 可能含 exercise、drinkev 等 | 在 Table 1 或补充表增加各队列 exercise/drinkev/is_socially_isolated 的构成比及队列间检验 | 中 |

---

## 四、结果：预测模型

### 4.1 Table 2 与 Figure 6

| 审稿意见 | 代码/数据现状 | 建议补充 | 优先级 |
|----------|---------------|----------|--------|
| 95% CI 计算方法（bootstrap vs 正态近似） | **已有**：`charls_cpm_evaluation._bootstrap_ci_at_threshold` 使用 1000 次分层 bootstrap | 在 Table 2 表注补充：AUC、Recall 等 95% CI 来自 1000 次分层 bootstrap | 高 |
| 校准度：ECE、Brier 分解、校准曲线 95% 置信带 | **部分已有**：Brier 已报告；校准斜率在补充材料；`fig3_clinical_evaluation_comprehensive` 含校准图 | 补充 ECE；Brier 分解（区分度、校准度、随机误差）；校准曲线加 95% 置信带（需修改绘图脚本） | 中 |
| 14 种模型 CPM 得分排序及森林图/雷达图 | **已有**：`table2_*_main_performance.csv` 含全模型；`draw_performance_radar` 有雷达图 | 在补充材料增加全模型 AUC 排序表；主文或补充增加模型性能森林图 | 低 |

---

## 五、结果：因果效应

### 5.1 Table 5 与 CATE

| 审稿意见 | 代码/数据现状 | 建议补充 | 优先级 |
|----------|---------------|----------|--------|
| 所有干预的精确 bootstrap P 值及 FDR 校正 P 值 | **已有**：`table4_ate_summary.csv` 含 `p_value_approx`、`p_adj_bonferroni`、`p_adj_fdr` | 将 table4 中 p_value_approx、p_adj_fdr 列同步到 Table 5 或补充表；表注说明主推断仍以 bootstrap P 与 95% CI 为准 | 高 |
| 亚组 CATE 交互作用检验（年龄、性别、居住地×运动 P 值） | **需新增** | 对亚组 CATE 做交互项检验（如回归中加入 subgroup×exercise）；需写分析脚本 | 中 |
| 剂量反应关系（运动频率/时长与共病风险） | **已有**：`evaluation/charls_dose_response.py` 支持 RCS；exercise 为二分类时改为定序分析 | 若 CHARLS 有运动频率/时长变量，可运行 dose_response 并补充结果；否则在 Limitations 说明 | 中 |

---

## 六、结果：外部验证

| 审稿意见 | 代码/数据现状 | 建议补充 | 优先级 |
|----------|---------------|----------|--------|
| Table S4 补充 AUPRC、Brier 的 95% CI；绘制外部验证 ROC/校准曲线 | **部分已有**：`charls_external_validation` 输出 AUC、AUPRC、Brier | 为外部验证指标加 bootstrap 95% CI；增加 ROC/校准图输出 | 中 |
| 东/中/西部、不同波浪的 AUC 差异及统计检验 | **部分已有**：地域验证（东+中 vs 西）；时间验证（wave<max vs wave=max） | 补充东/中/西三分组及波浪分层 AUC 的统计检验 | 低 |

---

## 七、讨论：机制与临床

### 7.1 机制解释

| 审稿意见 | 代码/数据现状 | 建议补充 | 优先级 |
|----------|---------------|----------|--------|
| 运动在抑郁队列有效、认知队列无效的神经生物学机制（BDNF、海马神经发生、炎症因子） | **需新增** | 引用 De la Rosa、Liu W、Schuch 等文献；可加机制示意图 | 中 |
| 低慢性病负担在认知队列增加风险的竞争风险/脆弱性指数分析 | **部分已有**：讨论中已提及竞争风险、生存偏倚 | 可计算 frailty index 与慢性病负担、共病风险的关联；引用 Zhu N 等 | 中 |
| 中国老年人群特征（运动习惯、城乡差异）与欧美研究的异同 | **需新增** | 结合 Table 1 城乡、教育分布；引用中国本土研究 | 低 |

### 7.2 临床意义

| 审稿意见 | 代码/数据现状 | 建议补充 | 优先级 |
|----------|---------------|----------|--------|
| 抑郁老年人群运动干预方案（类型、频率、时长）及指南引用 | **需新增** | 基于 DCA 阈值与现有证据，提出具体建议（如每周 3 次、30 分钟中等强度有氧）；引用 WHO/ACSM 指南 | 中 |
| 分层筛查流程（DCA 5%–35%，各队列冠军模型） | **部分已有**：DCA 阈值、冠军模型已报告 | 补充筛查流程图：抑郁人群→LightGBM；认知人群→Naive Bayes；阈值 5%–35% | 中 |
| 农村基层运动干预推广策略 | **需新增** | 结合 Table 3 农村 CATE，提出村卫生室、运动手册等建议 | 低 |

### 7.3 局限性与应对策略

| 审稿意见 | 代码/数据现状 | 建议补充 | 优先级 |
|----------|---------------|----------|--------|
| 观察性设计→未来 RCT 验证 | **需新增** | 在 Limitations 增加：建议开展 RCT 验证运动对抑郁老年人群的因果效应 | 高 |
| 51.2% 修剪→修剪前后基线对比、外推性影响 | **需新增** | 补充修剪后 vs 原样本的基线特征对比；讨论外推至“临床 equipoise”人群 | 中 |
| 认知为筛查诊断→与临床诊断的对比 | **需新增** | 若有临床诊断子集可做验证；否则说明筛查诊断的局限性及对结果的影响 | 中 |

---

## 八、优先实施顺序建议

1. **高优先级（可快速补充到论文）**
   - PSM/PSW 操作细节（§1.2）
   - XLearner 调优与 bootstrap 细节（§1.3）
   - CHARLS 波浪说明（§2.1）
   - Table 2 表注 95% CI 方法（§4.1）
   - Table 5 精确 P 值与 FDR（§5.1）
   - Limitations 应对策略（§7.3 部分）

2. **中优先级（需少量代码或数据提取）**
   - 偏倚敏感性模拟描述（§1.1）
   - SMD 全协变量表（§1.2）
   - Table 1 中位数/IQR、两两比较（§3.1）
   - 亚组交互检验（§5.1）
   - 机制与临床建议（§7.1–7.2）

3. **低优先级（可选）**
   - 14 模型森林图（§4.1）
   - 地域/波浪分层检验（§6）
   - 中国人群异同（§7.1）

---

## 九、数据文件索引（便于复现）

| 内容 | 路径 |
|------|------|
| PSM/PSW 实现 | `causal/charls_causal_methods_comparison.py` |
| SMD 明细 | `Cohort_*/03_causal/assumption_balance_smd_exercise.csv` |
| 偏倚敏感性图 | `Cohort_*/07_sensitivity/sensitivity_exercise/fig_bias_sensitivity.png` |
| ATE 汇总（含 p_adj_fdr） | `results/tables/table4_ate_summary.csv` |
| 流失表 | `preprocessed_data/attrition_flow.csv` |
| 剂量反应脚本 | `evaluation/charls_dose_response.py` |
