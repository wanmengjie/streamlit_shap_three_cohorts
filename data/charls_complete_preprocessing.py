import pandas as pd
import numpy as np
import os
import logging
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 避免预处理中多次新增列触发的 DataFrame 碎片化性能警告刷屏（逻辑与结果不变）
warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_charls_data(input_path='CHARLS.csv', cesd_cutoff=10, cognition_cutoff=10, age_min=60, write_output=True):
    """
    科研级预处理：引入交叉驱动结局 (Directional Outcomes)。

    因果时序（P0）：每行 = 个体在 Wave(t)。暴露/协变量取自 Wave(t)；结局 is_comorbidity_next 取自 Wave(t+1)。
    cesd_cutoff: CES-D-10 ≥ 该值定义为抑郁阳性；cognition_cutoff: 认知得分 ≤ 该值定义为认知受损。
    age_min: 年龄下限（默认60）；write_output: 若为 False，不写入 CSV/流失表。
    """
    if not os.path.exists(input_path):
        logger.error(f"找不到文件: {input_path}")
        return None

    logger.info(">>> 启动顶级期刊级预处理 (引入交叉驱动结局)...")
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        # 常见于国内导出的 CSV 使用 GBK/GB18030
        for enc in ('gb18030', 'gbk', 'latin-1'):
            try:
                df = pd.read_csv(input_path, encoding=enc)
                logger.info(f"已使用编码 {enc} 读取: {input_path}")
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            raise ValueError(f"无法解码 {input_path}，请将文件另存为 UTF-8 或确认编码后传入 encoding 参数")

    if 'ID' not in df.columns or 'wave' not in df.columns:
        logger.error("预处理需要列 ID 与 wave，当前数据缺失其一。")
        return None

    # 流失记录（STROBE/公共卫生报告用，与论文表1对齐）
    attrition = [('Raw records (person-waves)', len(df))]

    # 1. 基础过滤（age_min 可放宽至 55/50 以增大样本）
    age_col = next((c for c in df.columns if 'age' in c.lower()), None)
    if age_col:
        df = df[df[age_col] >= age_min]
        attrition.append((f'Age ≥ {age_min} years', len(df)))
    if 'income_total' in df.columns:
        # 缺失值不在此处插补，留待建模 Pipeline 在训练折内 fit Imputer，避免数据泄露
        # （全量中位数含未来划分的测试集信息，会导致乐观偏倚；Kaufman 2017）
        raw_inc = df['income_total'].clip(lower=0)
        df['income_total'] = np.log1p(raw_inc)  # NaN 保留，由 Pipeline Imputer 处理

    # 2. 状态定义（截断值可调，用于敏感性分析）
    cesd_col = next((c for c in df.columns if 'cesd' in c.lower() and '10' in c.lower()), None)
    if cesd_col is None:
        cesd_col = next((c for c in df.columns if 'cesd' in c.lower()), None)
    if cesd_col:
        df = df.dropna(subset=[cesd_col])
        df['is_depression'] = (df[cesd_col] >= cesd_cutoff).astype(int)
        attrition.append(('CES-D-10 non-missing', len(df)))
    
    cog_col = next((c for c in df.columns if 'total_cognition' in c.lower() or 'total_cog' in c.lower()), None)
    if cog_col:
        df = df.dropna(subset=[cog_col])
        df['is_cognitive_impairment'] = (df[cog_col] <= cognition_cutoff).astype(int)
        attrition.append(('Cognition score non-missing', len(df)))
    
    if 'is_depression' not in df.columns or 'is_cognitive_impairment' not in df.columns:
        logger.error("预处理失败：缺少 is_depression 或 is_cognitive_impairment，请检查数据是否含 cesd10 及认知列。")
        return None

    df['is_comorbidity'] = ((df['is_depression'] == 1) & (df['is_cognitive_impairment'] == 1)).astype(int)
    
    # 3. [核心升级] 构建定向驱动结局
    df = df.sort_values(by=['ID', 'wave'])
    df['next_wave_val'] = df.groupby('ID')['wave'].shift(-1)
    
    # 定义紧邻下一波的各状态（仅处理存在的列，避免 fall_down 等缺失时 KeyError）
    next_cols = [c for c in ['is_depression', 'is_cognitive_impairment', 'is_comorbidity', 'fall_down'] if c in df.columns]
    for col in next_cols:
        raw_col = f'{col}_next_raw'
        df[raw_col] = df.groupby('ID')[col].shift(-1)
        final_col = f'{col}_next' if 'fall' not in col else 'is_fall_next'
        mask_valid = (df['next_wave_val'] == df['wave'] + 1)
        df[final_col] = np.where(mask_valid, df[raw_col], np.nan)

    # 剔除结局缺失的行
    if 'is_comorbidity_next' not in df.columns:
        logger.error("预处理失败：未生成 is_comorbidity_next，请检查数据是否含 is_depression/is_cognitive_impairment 及波次信息。")
        return None
    df = df.dropna(subset=['is_comorbidity_next'])
    attrition.append(('Next-wave comorbidity non-missing', len(df)))

    # 【P0 因果时序显式检查】暴露/协变量取自 Wave(t)，结局取自 Wave(t+1)
    # 若数据结构不符（如 outcome 与 exposure 同波），需用滞后变量：T_lag = groupby(ID)[T].shift(1)
    if 'wave' in df.columns and 'next_wave_val' in df.columns:
        valid_next = (df['next_wave_val'] == df['wave'] + 1).sum()
        if valid_next < len(df) * 0.9:
            logger.warning(f"⚠️ 因果时序预警：仅 {valid_next}/{len(df)} 行满足 wave+1 紧邻下一波，请检查 wave 编码或使用滞后变量。")
        else:
            logger.info(f"✅ 因果时序校验通过：{valid_next}/{len(df)} 行满足 暴露/协变量(Wave_t) → 结局(Wave_t+1)")
    
    # 4. 入射队列逻辑（首次发病：一旦任一波次发生共病，之后所有波次均排除）
    df = df.sort_values(by=['ID', 'wave'])
    df['had_comorbidity_before'] = df.groupby('ID')['is_comorbidity'].transform(
        lambda x: x.shift(1).fillna(0).cummax()
    ).astype(int)
    df = df[(df['is_comorbidity'] == 0) & (df['had_comorbidity_before'] == 0)]
    attrition.append(('Incident cohort (baseline free of comorbidity)', len(df)))
    df['baseline_group'] = 0
    df.loc[(df['is_depression'] == 1) & (df['is_cognitive_impairment'] == 0), 'baseline_group'] = 1
    df.loc[(df['is_depression'] == 0) & (df['is_cognitive_impairment'] == 1), 'baseline_group'] = 2

    # 5. [新增学术级合成特征] 暴力提分手段
    # A. 慢性病负担指数 (Multimorbidity Burden)，不含 psyche（已移除）
    chronic_cols = ['hibpe', 'diabe', 'cancre', 'lunge', 'hearte', 'stroke', 'arthre']
    available_chronic = [c for c in chronic_cols if c in df.columns]
    if available_chronic:
        df['chronic_burden'] = df[available_chronic].sum(axis=1)

    # B. 社会脆弱性 (Social Vulnerability)
    if 'marry' in df.columns and 'family_size' in df.columns:
        # 独居或分居的老人更脆弱；family_size 缺失时用中位数插补，避免 NaN<=1 误判为非隔离
        fs_med = df['family_size'].median()
        fs = df['family_size'].fillna(fs_med if not pd.isna(fs_med) else 2)
        df['is_socially_isolated'] = ((df['marry'] != 1) & (fs <= 1)).astype(int)

    # C. BMI 合理化（防止列错位或异常值导致 Table 1 显示 13 kg/m² 等不合理值）
    # CHARLS mheight 单位为米(m)，mweight 为 kg；若存在则用公式 bmi=weight/height² 重算；最后裁剪 15–50
    if 'mweight' in df.columns and 'mheight' in df.columns:
        valid = df['mweight'].notna() & df['mheight'].notna() & (df['mheight'] > 0.5) & (df['mheight'] < 2.5)
        if valid.any():
            df.loc[valid, 'bmi'] = df.loc[valid, 'mweight'] / (df.loc[valid, 'mheight'] ** 2)
    if 'bmi' in df.columns:
        df['bmi'] = df['bmi'].clip(15, 50)

    # 二次加工特征（risk_age_chronic, lifestyle_active_sleep, frailty_proxy）已移除，避免与原始变量共线

    semantic_features = [
        'ID', 'wave', 'province', 'age', 'gender', 'rural', 'edu', 'marry', 'bmi', 
        'mwaist', 'puff', 'systo', 'diasto', 'pulse', 'lgrip',
        'exercise', 'sleep', 'drinkev', 'chronic_burden', 'is_socially_isolated',
        'hibpe', 'lunge', 'cancre', 'diabe', 'hearte', 'stroke', 'arthre', 
        'total_cognition', 'total_cog', 'cesd10', 'cesd', 'srh', 'satlife', 'income_total', 'family_size',
        # 新增：功能、社会经济、躯体指标（来自 CHARLS.csv）
        'adlab_c', 'iadl', 'pension', 'ins', 'disability', 'fall_down', 'retire', 'wspeed',
        'is_depression', 'is_cognitive_impairment', 'baseline_group',
        'is_comorbidity', 'is_comorbidity_next', 'is_fall_next', 'communityID',
        'is_depression_next', 'is_cognitive_impairment_next'
    ]
    available_cols = [c for c in semantic_features if c in df.columns]
    df = df[available_cols].copy()
    
    # sleep 保留连续变量（小时），sleep_adequate 已移除

    # 缺失值说明：中位数插补适用于 MCAR 或轻度 MAR；运动等高缺失变量在主分析中按 0 处理（与因果分析 T.fillna(0) 一致）
    if 'exercise' in df.columns:
        df['exercise'] = df['exercise'].fillna(0)

    # 敏感性分析见 run_sensitivity_scenarios（完整病例）
    # 编码
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        if col not in ['ID', 'communityID']:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    if write_output:
        os.makedirs('preprocessed_data', exist_ok=True)
        df.to_csv('preprocessed_data/CHARLS_final_preprocessed.csv', index=False, encoding='utf-8-sig')
        attrition_df = pd.DataFrame(attrition, columns=['Step', 'N'])
        attrition_path = 'preprocessed_data/attrition_flow.csv'
        attrition_df.to_csv(attrition_path, index=False, encoding='utf-8-sig')
        # 缺失率汇总（便于方法学报告与高缺失变量识别）
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        miss_rows = [{'Variable': c, 'Missing_pct': round(df[c].isna().mean() * 100, 2), 'N_non_missing': int(df[c].notna().sum())}
                     for c in num_cols if df[c].isna().any()]
        if miss_rows:
            pd.DataFrame(miss_rows).to_csv('preprocessed_data/preprocessing_missing_summary.csv', index=False, encoding='utf-8-sig')
            logger.info(f"变量缺失汇总已写: preprocessed_data/preprocessing_missing_summary.csv")
        logger.info(f"预处理完成。样本量: {len(df)}；流失表已写: {attrition_path}")
    else:
        logger.info(f"预处理完成（未写盘）。样本量: {len(df)}")
    return df


def reapply_cohort_definition(df, cesd_cutoff, cognition_cutoff):
    """
    在已有数据（如插补后数据）上按给定诊断阈值重算队列定义，与论文2.1节入组标准一致。
    用于敏感性分析（论文2.5/附录S2）：基于插补后数据做9种诊断阈值+完整病例时，仅重定义
    is_depression/is_cognitive_impairment/入射队列/baseline_group，不重新读盘。
    要求 df 含 ID、wave、cesd 相关列、total_cog/total_cognition、is_comorbidity_next。
    """
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    cesd_col = next((c for c in df.columns if 'cesd' in c.lower() and '10' in c.lower()), None) or next((c for c in df.columns if 'cesd' in c.lower()), None)
    cog_col = next((c for c in df.columns if 'total_cognition' in c.lower() or 'total_cog' in c.lower()), None)
    if not cesd_col or not cog_col:
        return None
    if 'is_comorbidity_next' not in df.columns:
        return None
    df = df.dropna(subset=[cesd_col, cog_col, 'is_comorbidity_next'])
    df['is_depression'] = (df[cesd_col] >= cesd_cutoff).astype(int)
    df['is_cognitive_impairment'] = (df[cog_col] <= cognition_cutoff).astype(int)
    df['is_comorbidity'] = ((df['is_depression'] == 1) & (df['is_cognitive_impairment'] == 1)).astype(int)
    df = df.sort_values(by=['ID', 'wave'])
    df['had_comorbidity_before'] = df.groupby('ID')['is_comorbidity'].transform(lambda x: x.shift(1).fillna(0).cummax()).astype(int)
    df = df[(df['is_comorbidity'] == 0) & (df['had_comorbidity_before'] == 0)]
    df['baseline_group'] = 0
    df.loc[(df['is_depression'] == 1) & (df['is_cognitive_impairment'] == 0), 'baseline_group'] = 1
    df.loc[(df['is_depression'] == 0) & (df['is_cognitive_impairment'] == 1), 'baseline_group'] = 2
    return df
