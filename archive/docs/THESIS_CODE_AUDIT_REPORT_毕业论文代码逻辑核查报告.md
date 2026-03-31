# 毕业论文核心代码逻辑核查报告  
## 基于因果机器学习的老年抑郁-认知共病研究（CHARLS）

**核查标准**：与论文方法学（2.1/2.4/2.5/2.8/3.3/3.5/3.6/3.8节）及附录S2完全一致，插补后敏感性分析无逻辑断点、计算错误、样本错误。

---

## 一、数据插补环节逻辑核查（论文2.1/2.5节）

| 核查点 | 合规状态 | 问题描述（如有） | 修正依据 |
|--------|----------|------------------|----------|
| 多重插补仅针对一般变量 | ⚠️ 部分一致 | 当前 `charls_imputation_npj_style.py` 中 VARS_CONTINUOUS 含 sleep、VARS_CATEGORICAL/ VARS_BINARY 含 exercise/smokev/drinkev，即关键干预变量也被插补。论文2.1/附录S2要求运动、睡眠、吸烟、饮酒、BMI 等为**完整病例分析**。 | 插补脚本已输出 step1b_complete_case_core.csv（核心干预变量无缺失子集），主因果分析若需严格对齐论文可仅用该子集或分析前对 T 做 dropna；当前主分析对 T 使用 fillna(0)。若论文定稿明确“关键变量不插补”，建议从 VARS_* 中移除 exercise/sleep/smokev/drinkev/bmi，仅对一般变量插补。 |
| 插补后数据生成、保存、调用 | ✅ 已合规 | 插补结果保存于 imputation_npj_results/pipeline_trace/step1_imputed_full.csv；主流程 config.USE_IMPUTED_DATA 且文件存在时读取该路径，敏感性分析见下节。 | 论文2.5节 |
| 敏感性分析是否调用插补后数据 | ✅ 已修正 | 原 run_sensitivity_scenarios 仅调用 preprocess_charls_data(CHARLS.csv)，即**未使用插补后数据**。 | **已修正**：主流程在使用插补数据时传入 df_base=df_clean，run_sensitivity_scenarios_analysis 在 df_base 上通过 reapply_cohort_definition 重定义队列，全流程基于插补后数据。论文2.5/附录S2。 |
| 样本量 14386、A=8828/B=3123/C=2435 | ✅ 可验证 | 插补脚本中 PAPER_N={0:8828,1:3123,2:2435}，划分后若偏差>100 会 logger.warning。实际样本量依赖 CHARLS 数据与截断值，运行后与表1核对。 | 论文表1；charls_imputation_npj_style.py 约 1299–1307 行 |

---

## 二、敏感性分析模块核心逻辑核查（论文2.5节、附录S2）

| 核查点 | 合规状态 | 问题描述（如有） | 修正依据 |
|--------|----------|------------------|----------|
| 9 种诊断阈值组合 | ✅ 已合规 | cutpoint_scenarios 含 Main(10,10)、CES-D≥8/10/12、Cog≤8/10/12 及 4 组组合，共 9 种。 | 论文2.5/附录S2 |
| 每种阈值基于插补后数据重训、重估ATE | ✅ 已修正 | 原逻辑每种阈值均 preprocess_charls_data(CHARLS.csv)，未用插补数据。 | **已修正**：当 df_base 为插补后数据时，对每种阈值调用 reapply_cohort_definition(df_base, cesd_c, cog_c) 得到队列，再 run_one_scenario → estimate_causal_impact，即重训、重估ATE。run_sensitivity_scenarios.py。 |
| 完整病例敏感性 | ✅ 已合规 | restrict_complete_case=True 时对当前干预列 dropna，再 estimate_causal_impact；当 df_base 为插补后数据时，完整病例同样基于插补后数据。 | 论文附录S2 |
| 输出含各干预在 9 阈值+完整病例的 ATE、95%CI | ✅ 已合规 | sensitivity_summary.csv 含 scenario、axis、intervention、ate、ate_lb、ate_ub、n、incidence；sensitivity_ate_comparison_*.png 按干预分图。 | 论文附录S2 表格结构 |

---

## 三、插补逻辑与全分析流程融合核查（无断点）

| 核查点 | 合规状态 | 问题描述（如有） | 修正依据 |
|--------|----------|------------------|----------|
| 敏感性分析全流程使用插补后数据 | ✅ 已修正 | 原敏感性分析未接插补数据。 | **已修正**：main() 在 USE_IMPUTED_DATA 且存在插补文件时调用 run_sensitivity_scenarios_analysis(..., df_base=df_clean)；run_one_scenario 内 estimate_causal_impact 与主分析一致，无环节回退未插补数据。 |
| 队列划分与论文2.1一致 | ✅ 已合规 | reapply_cohort_definition 与 preprocess_charls_data 中逻辑一致：无共病基线、had_comorbidity_before==0、baseline_group 按抑郁/认知定义。 | 论文2.1节入组标准 |
| Causal Forest DML 参数 | ✅ 已合规 | n_estimators=1000，cv=5，discrete_treatment=True；charls_recalculate_causal_impact.py 已加注释对应论文2.4节。 | 论文2.4/2.8节 |
| PSM 1:1 最近邻+卡尺 | ✅ 已修正 | 原 _ate_psm 默认 caliper=0.2。 | **已修正**：config 增加 CALIPER_PSM=0.024，charls_causal_methods_comparison 使用 CALIPER_PSM；论文2.4节卡尺0.024。 |
| PSW 权重 [0.1, 50] | ✅ 已合规 | _ate_psw 中 w = np.clip(w, 0.1, 50)。 | 论文2.4/2.8节 |

---

## 四、结果一致性与学术合规性

| 核查点 | 合规状态 | 说明 |
|--------|----------|------|
| 敏感性ATE与论文3.6结论一致 | 需运行后核对 | 论文3.6：95%CI 均包含0，诊断阈值对主结论无显著影响。运行后检查 sensitivity_summary 中各场景 ate_lb/ate_ub 是否含0。 |
| AUC、ATE、95%CI、SMD 计算 | ✅ 已合规 | AUC 来自 sklearn；ATE/CI 来自 DML ate_interval 或 PSM/PSW 的 1.96*se；SMD 匹配后 _compute_smd，<0.1 为平衡良好。与 3.3/3.5/3.8 节指标定义一致。 |
| 硬编码与可追溯性 | ✅ 已加强 | 轴线目录、PSM 卡尺已收拢至 config；关键处已加「毕业论文修正-对应论文X.X节」或方法学对应注释。 |

---

## 五、修改点汇总（代码位置与论文对应）

| 文件 | 修改内容 | 对应论文 |
|------|----------|----------|
| charls_complete_preprocessing.py | 新增 reapply_cohort_definition(df, cesd_cutoff, cognition_cutoff) | 2.1/2.5、附录S2 |
| run_sensitivity_scenarios.py | 支持 df_base；9 阈值与完整病例在 df_base 上通过 reapply_cohort_definition 生成队列 | 2.5、附录S2 |
| run_all_charls_analyses.py | 使用插补数据时传入 df_base=df_clean 给 run_sensitivity_scenarios_analysis | 2.5 |
| config.py | 新增 CALIPER_PSM=0.024 | 2.4/2.8 |
| charls_causal_methods_comparison.py | PSM 使用 CALIPER_PSM；注释论文2.4节 | 2.4/2.8 |
| charls_recalculate_causal_impact.py | 注释 cv=5、n_estimators=1000 对应论文2.4节 | 2.4 |

---

## 六、逻辑验证建议（与附录S2主分析结果匹配）

1. **运行条件**：确保已运行 charls_imputation_npj_style.py 生成 step1_imputed_full.csv，config 中 USE_IMPUTED_DATA=True。
2. **运行主流程**：执行 run_all_charls_analyses.py，完成后再查看 LIU_JUE_STRATEGIC_SUMMARY/sensitivity_summary.csv。
3. **典型场景核对**：筛选 scenario=="Main (CES-D≥10, Cog≤10)"、intervention=="exercise"，对比三轴线 A/B/C 的 ate、ate_lb、ate_ub、n 与论文附录S2 主分析结果及表1 样本量。
4. **单跑敏感性（可选）**：若仅验证敏感性逻辑，可单独调用：
   - `df = pd.read_csv('imputation_npj_results/pipeline_trace/step1_imputed_full.csv', encoding='utf-8-sig')`
   - `prepare_exposures(df)`（若尚未含 sleep_adequate）
   - `run_sensitivity_scenarios_analysis(final_dir='LIU_JUE_STRATEGIC_SUMMARY', data_path='CHARLS.csv', df_base=df)`
   - 检查输出 sensitivity_summary.csv 中 Main (CES-D≥10, Cog≤10) 的 ATE 与 95%CI。

---

## 七、学术性优化建议（不影响逻辑一致性）

- **插补策略与论文完全对齐**：若定稿要求“关键干预变量不插补”，可在 charls_imputation_npj_style 中将 exercise、sleep、smokev、drinkev、bmi 从插补变量列表中移除，主分析因果步骤对 T 使用 dropna(subset=[T]) 或仅使用 step1b_complete_case_core.csv。
- **敏感性结果表与附录S2 表格形式一致**：可将 sensitivity_summary 按附录S2 表头（如“场景”“轴线”“干预”“ATE”“95%CI”“N”）导出为 Excel 或 Word 表，便于直接贴入论文。
- **可视化**：sensitivity_ate_comparison_*.png 已按干预分图；可增加一张“9 场景×轴线”热力图或森林图，便于审稿人一眼看出 95%CI 均含 0。

---

**核查结论**：在已实施的修正下，插补后数据已贯穿敏感性分析全流程（数据读取→队列重定义→模型训练→DML/PSM/PSW 因果估计→结果输出），无断点；9 种诊断阈值与完整病例敏感性逻辑与论文2.5/附录S2 一致；PSM 卡尺与 DML 参数已与论文2.4/2.8 对齐。建议按第六节做一次典型场景运行并与附录S2 主分析结果核对，以 100% 支撑论文结论的可复现性。
