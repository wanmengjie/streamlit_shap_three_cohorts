# 论文写作前最终检查清单

**更新日期**：2026-03-19（2026-03-14 增补 Methods 与代码一致性项）  
**目的**：确保代码与输出无误，避免写论文后推倒重来。

---

## 一、已修复的致命问题（本次检查）

| 问题 | 影响 | 修复 |
|------|------|------|
| **亚组分析始终跳过** | 06_subgroup 从未生成，Table 3 无数据 | `run_cohort_protocol` 改用 res_df（含 causal_impact）传入 run_subgroup_analysis |
| SHAP 模型列表不完整 | Cohort_B 无 SHAP 图 | 扩展 TreeExplainer 支持 ExtraTrees/GBDT/HistGBM/DT |
| run_subgroup_and_joint_causal 仅 B/C | 亚组 ATE 缺 Cohort A | 已扩展为三队列 |
| 路径/异常处理 | 部分脚本路径错误 | visualize_causal_forest、charls_external_validation、check_data_quality 已修正 |

---

## 二、运行前必查

### 1. 数据文件

- [ ] `CHARLS.csv` 存在于项目根目录
- [ ] 若 `config.USE_IMPUTED_DATA=True`：`imputation_npj_results/pipeline_trace/step1_imputed_full.csv` 需存在（否则自动回退预处理数据）

### 1b. Methods 表述勿与代码矛盾（审稿雷区）

- [ ] **禁止**写「所有 MICE/插补均在训练折内完成」：`USE_IMPUTED_DATA=True` 时为**全样本先插补**再划分 train/test（见 `config.py` 方法学事实注释、`PAPER_完整版` Supplementary Text S1）。
- [ ] 应区分：**bulk MICE**（全分析队列）vs **`Pipeline` 预处理**（仅训练折）vs **CPM 阈值**（不用最终测试集标签）。

### 2. 配置锁定（论文可重复）

- [ ] `config.RANDOM_SEED = 500`（已固定）
- [ ] `config.ANALYSIS_LOCK = True`
- [ ] `config.CAUSAL_METHOD` 与论文 2.4 一致（当前为 'XLearner'）
- [ ] `config.COLS_TO_DROP` 含 rgrip, grip_strength_avg, psyche, puff；sleep_adequate 在 EXCLUDE_COLS_BASE（预测排除、因果干预用）

### 3. 主流程运行

```bash
cd C:\Users\lenovo\Desktop\因果机器学习
python run_all_charls_analyses.py
```

运行后检查：
- [ ] `Cohort_A_Healthy_Prospective/06_subgroup/subgroup_analysis_results.csv` 存在
- [ ] `Cohort_B_Depression_to_Comorbidity/06_subgroup/subgroup_analysis_results.csv` 存在
- [ ] `Cohort_C_Cognition_to_Comorbidity/06_subgroup/subgroup_analysis_results.csv` 存在
- [ ] 三队列 `02_shap/` 下均有 `fig5a_shap_summary_*.png`
- [ ] `LIU_JUE_FINAL_FIXED.log` 无 ERROR（可搜索 "ERROR" 或 "失败"）

---

## 三、论文引用数据源速查

| 论文位置 | 数据来源 | 路径 |
|----------|----------|------|
| Table 1 基线 | table1_baseline_characteristics.csv | results/tables/ 或 LIU_JUE_STRATEGIC_SUMMARY/ |
| Table 2 预测 | table2_*_main_performance.csv（CPM 主表） | **首选** `results/tables/table2_prediction_cohortA/B/C.csv`；旧名 `table2_prediction_axis*.csv` 为双写同内容；或 `Cohort_*/01_prediction/` |
| Table 3 亚组 CATE | subgroup_analysis_results.csv | **首选** `results/tables/table3_subgroup_cohortA/B/C.csv`；旧名 `table3_subgroup_axis*.csv` 同内容；或 `Cohort_*/06_subgroup/` |
| Table 4 多干预 ATE | table4_ate_summary.csv | results/tables/ |
| Table 5 截断值敏感性 | sensitivity_summary.csv | results/tables/table5_sensitivity_summary.csv 或 LIU_JUE_STRATEGIC_SUMMARY/ |
| Table 6 外部验证 | **首选** `table6_external_validation_cohort*.csv`；旧名 `table6_external_validation_axis*.csv` 同内容 | results/tables/ |
| Table 7 PSM/PSW/XLearner | table7_psm_psw_dml.csv | results/tables/ |
| Table 4b X-Learner+PSM+PSW | table4_xlearner_psm_psw_wide.csv | results/tables/ |
| 因果假设检验 | assumption_checks_summary.txt | Cohort_A/B/C 的 03_causal/（三队列均有） |
| E-value | assumption_evalue_exercise.txt | Cohort_A/B/C 的 03_causal/（三队列均有） |

---

## 四、方法学表述核对

| 方法点 | 代码实现 | 论文应写 |
|--------|----------|----------|
| 阈值选择 | Internal CV (训练集) Youden 最大化 | 禁止在测试集寻优（TRIPOD） |
| 冠军选择 | CPM 评估：验证集 Youden 最优阈值下 AUC 最高 | 同左 |
| 因果主方法 | XLearner (RF nuisance) | XLearner；PSM/PSW 验证 |
| Overlap | PS∈[0.05,0.95] 外 >10% 时 trim | 写入 ATE_CI_summary |
| 选择偏倚 | exercise×adlab_c 交互项 | Hernán 2020 |
| E-value | 保护用 ate_ub，有害用 ate_lb | VanderWeele & Ding 2017 |
| 敏感性分析 | 仅用训练集 | 先分割再传递 |

---

## 五、验证脚本

```bash
python scripts/run_verification_checklist.py
```

通过后即可安心写论文。
