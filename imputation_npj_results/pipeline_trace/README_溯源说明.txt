插补实验数据溯源说明
==================================================
step0_loaded.csv         - 加载后的原始/预处理数据（未插补）
step0b_aligned.csv       - 个体历史对齐后（gender/edu/rural 等 ffill/bfill）
step1_imputed_full.csv   - 全量插补后数据（含队列特征+纵向影响）
step1b_complete_case_core.csv - 核心干预变量(exercise/drinkev)无缺失子集(附录S2)
step2_cohort_A_*.csv     - 队列A（健康组）划分后
step2_cohort_B_*.csv     - 队列B（仅抑郁组）划分后
step2_cohort_C_*.csv     - 队列C（仅认知受损组）划分后

变量身份与插补策略：
  结局/定义变量(is_comorbidity_next等) → 不插补，缺失剔除
  协变量(age/bmi/income/exercise等) → 插补对象

变量类型（按 Table 1）：
  连续（均值±SD）→ NRMSE 选优+物理边界+辅助变量(province/wave/age)
  有序分类(edu/srh/satlife) → 贝叶斯回归+取整
  二分类(exercise/drinkev等) → RF 预测
  名义分类 → Mode 众数插补

流程：加载 → 个体历史对齐 → 全量插补(队列+轨迹+边界) → 划分队列
多重插补：N_MULTIPLE_IMPUTATIONS>0 时生成 step1_imputed_m1..mN.csv
