# -*- coding: utf-8 -*-
"""
科研管线全局配置文件：一处修改，全库生效。
"""

# --- 核心科研定义 ---
TARGET_COL = 'is_comorbidity_next'
TREATMENT_COL = 'exercise'
AGE_MIN = 60

# --- 干预因素英文标签（图表统一使用英文）---
INTERVENTION_LABELS_EN = {
    'exercise': 'Exercise',
    'drinkev': 'Drinking',
    'is_socially_isolated': 'Social isolation',
    'bmi_normal': 'Normal BMI (18.5-24)',
    'chronic_low': 'Low chronic disease burden (≤1)',
    'sleep_adequate': 'Adequate sleep (≥6h)',
    'smokev': 'Current smoking',
}

# --- 可干预因素（SHAP 中应优先关注：因果分析的前提是这些因素在预测中排名靠前）---
# 列名包含以下任一关键词即视为可干预因素；用于验证「SHAP 排名 + 可干预性」逻辑闭环
INTERVENABLE_KEYWORDS = ['exercise', 'sleep', 'drinkev', 'social']
# 主干预因素在 SHAP 中的期望最低排名（若超出则提示，论文叙述时需额外说明）
TOP_N_FOR_STRONG_NARRATIVE = 15
RANDOM_SEED = 500

# --- 预处理截断值 ---
CESD_CUTOFF = 10
COGNITION_CUTOFF = 10

# --- 输出目录定义 ---
OUTPUT_ROOT = 'LIU_JUE_STRATEGIC_SUMMARY'
# 三队列输出根目录名（与论文 Cohort A/B/C 一致；勿称 axis）
COHORT_A_DIR = 'Cohort_A_Healthy_Prospective'
COHORT_B_DIR = 'Cohort_B_Depression_to_Comorbidity'
COHORT_C_DIR = 'Cohort_C_Cognition_to_Comorbidity'
# 各队列内步骤子目录（一处修改，全流程生效）
COHORT_STEP_DIRS = {
    'prediction': '01_prediction',
    'shap': '02_shap',
    'shap_stratified': '02_shap_stratified',
    'causal': '03_causal',
    'eval': '04_eval',
    'decision': '05_decision',
    'subgroup': '06_subgroup',
    'sensitivity': '07_sensitivity',
    'external_validation': '04b_external_validation',
    'dose_response': '04c_dose_response',
}

# --- GitHub 开源适配：统一结果输出目录 ---
RESULTS_ROOT = 'results'
RESULTS_TABLES = 'results/tables'
RESULTS_FIGURES = 'results/figures'
RESULTS_MODELS = 'results/models'

# --- 运行开关 ---
SAVE_INTERMEDIATE = True  # 是否保存中间过程的 CSV 文件
BBOX_INCHES = 'tight'     # 绘图保存参数
PARALLEL_COHORTS = False  # False 串行运行三队列，避免 Windows 下 joblib 并行 + GUI 冲突导致 worker 崩溃

# --- 崩溃续跑（仅重跑部分 Cohort，避免整管线重来）---
# None 或 []：三队列全跑。设为 ['B','C'] 等时只跑所列队列；未跑的队列从磁盘读取 table2_* / ATE_CI_summary_* 拼汇总图（需该队列曾成功跑完过）。
RUN_COHORTS_ONLY = None  # 例: ['B', 'C']
# True：跳过概念框架图、Table1、流失图、插补敏感性，直接从三队列开始（续跑省时间；首次全流程请 False）。
MAIN_SKIP_STEPS_BEFORE_COHORTS = False
# True：三队列内仅跑因果模块及因果相关扩展（ITE/列线图/时序/插补敏感性/亚组），跳过 compare_models（CPM）、SHAP、
# 临床评价、决策支持、外部验证、剂量反应。数据仍按主流程加载（插补表用于因果）；不写新的 Table2/冠军模型。
MAIN_COHORT_CAUSAL_ONLY = True
# True：「因果轻量重跑」— 用磁盘上**已有**插补表与 CPM 结果，不重跑因果前置与队列后大块分析。
# 效果：跳过本轮 npj 前置插补、跳过概念图/Table1/流失/插补敏感性、跳过 Table1b 发病密度与组合发病图、
# 跳过 Rubin/截断敏感性/多暴露/扩展干预/XLearner全干预/PSM全队列交叉/低样本优化/生理因果；仍加载插补数据并跑三队列因果（隐含 causal_only）。
# False：队列结束后执行上述扩展模块（耗时显著增加）。若只需扩展分析、不重跑 bulk MICE，可设 RUN_IMPUTATION_BEFORE_MAIN=False。
MAIN_MINIMAL_CAUSAL_RERUN = False
PARALLEL_IMPUTATION_BOOTSTRAP = True  # True 时插补敏感性 Bootstrap 循环并行
USE_GPU = True            # True 时 XGB/LGBM/CatBoost 优先使用 CUDA 加速（需 NVIDIA GPU + 对应库的 GPU 版）

# --- 分析锁（审稿修正：确保全流程可重复）---
# 审稿期间建议固定 RANDOM_SEED，创建 Release Tag 后不再改动
ANALYSIS_LOCK = True  # True 时全流程使用 RANDOM_SEED，保证可重复

# --- 主分析数据源：插补数据 vs 预处理数据 ---
# True: 插补 CSV 用于**因果/多暴露/Rubin 等**主分析表；**监督学习（CPM）**在 run_all 中自动改用
#      预处理宽表（保留缺失）+ Pipeline 内 IterativeImputer，避免「全样本插补后再划分」与 CV 内 Imputer 叙事冲突。
# False: 全流程使用 preprocess_charls_data（缺失未插补）。
USE_IMPUTED_DATA = True
# sklearn IterativeImputer（嵌入 compare_models 的 Pipeline，仅训练折 fit）；越大越准但越慢
ITERATIVE_IMPUTER_MAX_ITER = 10
# 插补脚本输出根目录（其下 pipeline_trace/ 含 step1_imputed_full.csv 与 m1..mN）
IMPUTATION_OUTPUT_ROOT = 'imputation_npj_results'
IMPUTED_DATA_PATH = 'imputation_npj_results/pipeline_trace/step1_imputed_full.csv'
# True：每次跑 run_all_charls_analyses 时**先**执行 npj 插补，**覆盖** step1_imputed_full → 主分析必定用**本轮最新**插补。
# False：直接读已有 step1_imputed_full（快）；若预处理表比该 CSV 新，会打 WARNING（见下项）。扩展分析重跑、插补已锁定时用 False。
RUN_IMPUTATION_BEFORE_MAIN = False
# 当 RUN_IMPUTATION_BEFORE_MAIN=False 时，若 preprocessed_data/CHARLS_final_preprocessed.csv 比 step1_imputed_full 新，记录 WARNING，避免误用旧插补。
WARN_IMPUTED_OLDER_THAN_PREPROCESSED = True
# --- 方法学事实（写论文时勿与代码矛盾）---
# 【sklearn 路径】禁止在 train/test 划分前对特征矩阵单独 fit SimpleImputer/StandardScaler（全局泄露）。
# Imputer 与 Scaler 必须放在 sklearn Pipeline / ColumnTransformer 内；开发集上仅用「训练子集」fit，再 transform 全量或测试集
# （与 compare_models、因果模块 utils.charls_train_only_preprocessing 一致）。
# 【bulk 层】USE_IMPUTED_DATA=True 时：step1_imputed_full 仍可用于因果推断、敏感性等；**不得**再作为 CPM 的唯一输入。
# CPM 输入为预处理缺失宽表 + Pipeline 内 IterativeImputer（与论文「imputation strictly inside CV」一致）。
# --- 终稿跑批推荐：时间划分 + 多重插补 + Rubin（下列已按终稿开启；快速迭代可改回 0 / False / False）---
N_MULTIPLE_IMPUTATIONS = 5  # >0 时 archive 插补写出 step1_imputed_m1..mN；run_all 启动前会同步到 npj 脚本
IMPUTED_MI_DIR = 'imputation_npj_results/pipeline_trace'
USE_RUBIN_POOLING = True    # True 且 n_mi>=2 时主流程尝试 Rubin 合并 AUC/ATE（需 m1..mN 文件齐全）
# 原始数据路径（预处理、敏感性等统一引用）
RAW_DATA_PATH = 'CHARLS.csv'
# 加载数据时需删除的列：puff 高缺失且吸烟用 smokev；sleep 保留供预测，sleep_adequate 仅用于因果干预
COLS_TO_DROP = ['rgrip', 'grip_strength_avg', 'psyche', 'puff']

# --- 预测模型划分（审稿修正）---
# True: 时间划分 train=wave<max_wave, test=wave=max_wave，避免时序泄露（终稿推荐）
# False: 随机划分 GroupShuffleSplit（与部分历史结果一致）
USE_TEMPORAL_SPLIT = True
# 时变混杂敏感性：仅纳入运动习惯稳定者（需数据含 exercise_stable 列）
EXERCISE_STABLE_ONLY = False

# --- 因果方法参数（论文2.4/2.8节）---
# 主因果方法：'TLearner'（流行病学常用）、'XLearner'（治疗/对照组不平衡时通常优于 TLearner）、'CausalForestDML' 为原方法
CAUSAL_METHOD = 'XLearner'  # 可改为 'TLearner' 或 'CausalForestDML'
# PSM：1:1 最近邻匹配，卡尺为倾向得分标准差的倍数（论文卡尺 0.024）
CALIPER_PSM = 0.024
# PSW 权重截断范围 [0.1, 50] 在 charls_causal_methods_comparison 中硬编码一致
# PSM 匹配后若 max|SMD|≥0.1，在匹配样本上对结局做 Logit，调整仍未平衡的协变量（双重调整）
PSM_DOUBLE_ADJUST_LOGIT = True
# 倾向得分重叠修剪（Austin 2011）：主分析区间；TLearner/XLearner 内 PS 修剪与此一致
PS_TRIM_LOW = 0.05
PS_TRIM_HIGH = 0.95
# --- 投稿补充（不改变 CPM / compare_models；主流程末尾可选执行）---
# 阴性对照结局：数据集中无统一 accidental_injury 列时常用「下期跌倒」作逻辑弱相关对照；改列名时请核对 utils/charls_feature_lists.EXCLUDE_COLS_BASE
NEGATIVE_CONTROL_OUTCOME_COL = 'is_fall_next'
RUN_NEGATIVE_CONTROL_OUTCOME = True  # → results/tables/negative_control_results.csv（三队列，XLearner 与主分析同设置）
RUN_ATE_METHOD_SENSITIVITY = True  # Cohort B：XLearner vs CausalForestDML（PS 修剪与主 XLearner 一致）→ results/tables/ate_method_sensitivity.csv

# 敏感性分析：见 scripts/run_ate_ps_trim_sensitivity.py → results/tables/ate_sensitivity_trimming.csv
# 每项: trim_lo, trim_high, label, force_trim（True=凡在区间外即剔除；False=与主流程一致：仅当区间内占比<90% 时才剔除）
PS_TRIM_SENSITIVITY_SCENARIOS = [
    (0.05, 0.95, 'main_same_band_legacy_rule', False),
    (0.01, 0.99, 'sensitivity_01_99', True),
    (0.03, 0.97, 'sensitivity_03_97', True),
]
# 是否对 CPM 冠军模型做事后概率校准（Platt/sigmoid）；True 会改变 champion_model.joblib 结构（CalibratedClassifierCV 包装）
CALIBRATE_CHAMPION_PROBA = True

# --- 绘图字体（解决英文/中文方框显示问题）---
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号不显示为方框

# --- 旧名称兼容（archive/外部笔记本若仍 from config import AXIS_* 可继续运行）---
AXIS_A_DIR = COHORT_A_DIR
AXIS_B_DIR = COHORT_B_DIR
AXIS_C_DIR = COHORT_C_DIR
AXIS_STEP_DIRS = COHORT_STEP_DIRS
PARALLEL_AXES = PARALLEL_COHORTS
