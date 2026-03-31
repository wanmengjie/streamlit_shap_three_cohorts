# -*- coding: utf-8 -*-
"""
集中定义：泄露关键字、排除列基准、由关键字扩展的排除列。
各模块统一引用，避免拼写不一致与漏列。
"""
import pandas as pd

# 泄露风险关键字（列名包含任一则排除，注意拼写为 memory 非 memeory）
LEAKAGE_KEYWORDS = [
    'cesd', 'total_cog', 'cognition', 'memory', 'executive', 'score', 'test'
]

# 分类/名义变量：不得加入 CONTINUOUS_FOR_SCALING，否则会错误标准化（LabelEncoder 后数值无尺度意义）
CATEGORICAL_NO_SCALE = [
    'edu', 'gender', 'rural', 'marry', 'province', 'retire', 'ins',
    'hibpe', 'lunge', 'cancre', 'diabe', 'hearte', 'stroke', 'arthre',
    'exercise', 'drinkev', 'fall_down', 'disability', 'srh', 'satlife',
    'is_socially_isolated', 'pension',
]

# 有序计数 / 家庭人数 / ADL·IADL 困难项计分：Pipeline 中仅 IterativeImputer，不做 StandardScaler
# （保持整数档位语义；见 build_numeric_column_transformer 的 pass 分支）
ORDINAL_COUNT_IMPUTE_ONLY = ('adlab_c', 'iadl', 'family_size')

# 真正连续型变量：仅对这些列做 StandardScaler（与 pass 分支互斥）
CONTINUOUS_FOR_SCALING = [
    'age', 'bmi', 'pulse',
    'mwaist', 'systo', 'diasto', 'lgrip', 'wspeed', 'sleep',
    'income_total',
]
# 运行时校验：防止误将分类变量加入
assert not (set(CONTINUOUS_FOR_SCALING) & set(CATEGORICAL_NO_SCALE)), \
    "CONTINUOUS_FOR_SCALING 不得包含分类变量 (edu/gender 等)，请检查 charls_feature_lists"

# 建模/因果分析时一律排除的列（不含目标、干预、因果衍生列）
# 含：慢性病负担（保留8种慢病明细）、二次加工代理特征（与原始变量共线）
# sleep_adequate：仅用于因果干预分析，预测模型用 sleep（连续）而非二分类
# had_comorbidity_before：队列定义变量（incident 样本中恒为 0），非预测特征
EXCLUDE_COLS_BASE = [
    'ID', 'wave', 'province', 'communityID',
    'is_comorbidity', 'is_comorbidity_next', 'is_fall_next',
    'baseline_group', 'first_com_wave', 'causal_impact',
    'had_comorbidity_before',
    'is_depression', 'is_cognitive_impairment',
    'is_depression_next', 'is_cognitive_impairment_next',
    'chronic_burden',  # 保留7种慢病明细，排除综合负担
    'risk_age_chronic', 'lifestyle_active_sleep', 'frailty_proxy',  # 二次加工特征，与原始变量共线
    'psyche',  # 精神疾病与抑郁定义重叠，已从数据移除
    'grip_strength_avg',  # 握力均值，已从数据移除
    'rgrip',  # 右手握力，已从数据移除
    'sleep_adequate',  # 仅用于因果干预，预测用 sleep（连续）
]


def get_exclude_cols(df, target_col=None, treatment_col=None, include_causal_impact_prefix=True):
    """
    得到当前 df 上应排除的列名列表（用于特征矩阵 X 的构建）。
    - target_col: 目标列，加入排除列表
    - treatment_col: 干预列，因果分析时加入排除列表
    - include_causal_impact_prefix: 是否排除所有 causal_impact_ 开头的列
    """
    out = list(EXCLUDE_COLS_BASE)
    if target_col and target_col not in out:
        out.append(target_col)
    if treatment_col and treatment_col not in out:
        out.append(treatment_col)
    if include_causal_impact_prefix and df is not None:
        out.extend([c for c in df.columns if c.startswith('causal_impact_')])
    for kw in LEAKAGE_KEYWORDS:
        out.extend([c for c in df.columns if kw in c.lower()])
    # 联合干预 exercise_sleep_both 由 exercise 与 sleep 构成
    if treatment_col == 'exercise_sleep_both' and df is not None:
        for c in ('exercise', 'sleep'):
            if c in df.columns:
                out.append(c)
    # bmi_normal 由 bmi 二值化而来，排除 bmi
    if treatment_col == 'bmi_normal' and df is not None and 'bmi' in df.columns:
        out.append('bmi')
    # chronic_low 由 chronic_burden 二值化而来，排除 chronic_burden
    if treatment_col == 'chronic_low' and df is not None and 'chronic_burden' in df.columns:
        out.append('chronic_burden')
    # puff_low 由 puff 二值化而来，排除 puff
    if treatment_col == 'puff_low' and df is not None and 'puff' in df.columns:
        out.append('puff')
    # sleep_adequate 由 sleep 二值化而来，排除 sleep
    if treatment_col == 'sleep_adequate' and df is not None and 'sleep' in df.columns:
        out.append('sleep')
    return list(set(out))
