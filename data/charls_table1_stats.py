# -*- coding: utf-8 -*-
"""
基线表 Table 1：按生物-心理-社会模型（Biopsychosocial Model）组织。
由 BPS_SECTIONS 驱动，变量按生物、心理、社会、生活方式、定义变量、结局顺序呈现。
"""
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# ============== 生物-心理-社会模型配置 ==============
GROUP_COL = 'baseline_group'
GROUP_LABELS = {0: 'Healthy', 1: 'Depression only', 2: 'Cognition impaired only'}

# 分类变量水平标签（level 0/1/2 → 可读标签，用于 Table 1 行名）
# 若某变量未配置，则仍显示 "level k"
CATEGORICAL_LEVEL_LABELS = {
    'marry': {0: 'Unmarried/Divorced/Widowed', 1: 'Married'},  # 婚姻：0=未婚/离异/丧偶, 1=在婚
    'edu': {1: 'Below primary school', 2: 'Primary school', 3: 'Middle school', 4: 'Senior high school and above'},  # 教育 1–4 级
    'srh': {1: 'Very bad', 2: 'Bad', 3: 'Average', 4: 'Good', 5: 'Very good'},  # 自评健康
    'satlife': {1: 'Very bad', 2: 'Bad', 3: 'Average', 4: 'Good', 5: 'Very good'},  # 生活满意度
}

# 生物-心理-社会模型各维度及变量（按呈现顺序）
# 格式：section_label, 然后为该维度的变量列表
BPS_SECTIONS = [
    # 1. 生物因素 (Biological)
    ('Biological factors', {
        'continuous': [('age', 'Age, years'), ('bmi', 'BMI'), ('mwaist', 'Waist circumference, cm')],
        'sex_col': 'gender',
        'binary': [('fall_down', 'Fall in past year', 1), ('disability', 'Has disability', 1)],
        'physical': [
            ('pulse', 'Pulse, /min'),
            ('systo', 'Systolic BP, mmHg'),
            ('diasto', 'Diastolic BP, mmHg'),
            ('lgrip', 'Grip strength (max), kg'),
            ('wspeed', 'Walking speed, m/s'),
        ],
        'chronic_disease_cols': [
            ('hibpe', 'Hypertension'),
            ('diabe', 'Diabetes'),
            ('cancre', 'Cancer'),
            ('lunge', 'Lung disease'),
            ('hearte', 'Heart disease'),
            ('stroke', 'Stroke'),
            ('arthre', 'Arthritis'),
        ],
        'optional_continuous': [('adlab_c', 'ADL difficulties'), ('iadl', 'IADL difficulties')],
    }),
    # 2. 心理因素 (Psychological)
    ('Psychological factors', {
        'categorical': [('srh', 'Self-rated health'), ('satlife', 'Life satisfaction')],
    }),
    # 3. 社会因素 (Social)：居住地、教育、婚姻、家庭、经济
    ('Social factors', {
        'binary': [
            ('rural', 'Rural residence', 1),
            ('is_socially_isolated', 'Socially isolated', 1),
            ('pension', 'Has pension', 1),
            ('ins', 'Has insurance', 1),
            ('retire', 'Retired', 1),
        ],
        'categorical': [('edu', 'Education'), ('marry', 'Marital status')],
        'optional_continuous': [('family_size', 'Family size'), ('income_total', 'Log(income+1)')],
    }),
    # 4. 生活方式（可干预）：smokev 吸烟，sleep/sleep_adequate 睡眠（puff 已移除）
    ('Lifestyle (intervenable)', {
        'lifestyle_binary': [
            ('exercise', 'Regular exercise', 1),
            ('drinkev', 'Current drinking', 1),
            ('smokev', 'Current smoking', 1),
            ('sleep_adequate', 'Adequate sleep (≥6h)', 1),
        ],
        'lifestyle_continuous': [('sleep', 'Sleep hours')],
    }),
    # 5. 定义变量
    ('Defining variables', {
        'defining_continuous': [
            ('total_cognition', 'Cognition score (defining)'),
            ('cesd10', 'CES-D-10 (defining)'),
        ],
    }),
    # 6. 结局
    ('Outcome', {
        'outcome_col': 'is_comorbidity_next',
        'outcome_label': 'Incident comorbidity, follow-up',
    }),
]

# 用于缺失汇总：汇总所有变量列名（传入 df 以解析 total_cognition 列名）
def _get_all_cols_from_bps(df=None):
    cols = []
    for _, sec_cfg in BPS_SECTIONS:
        for k, v in sec_cfg.items():
            if k == 'continuous' or k == 'optional_continuous' or k == 'lifestyle_continuous':
                cols.extend([(c, c) for c, _ in v])
            elif k == 'binary' or k == 'lifestyle_binary':
                cols.extend([(t[0], t[0]) for t in v])
            elif k == 'sex_col':
                cols.append((v, v))
            elif k == 'categorical':
                cols.extend([(c, c) for c, _ in v])
            elif k == 'physical':
                cols.extend([(c, c) for c, _ in v])
            elif k == 'chronic_disease_cols':
                cols.extend([(c, c) for c, _ in v])
            elif k == 'grip_col':
                cols.append((v[0], v[0]))
            elif k == 'defining_continuous':
                for col_or_key, label in v:
                    c = _cog_col(df) if col_or_key == 'total_cognition' and df is not None else col_or_key
                    if c:
                        cols.append((c, label))
            elif k == 'outcome_col':
                cols.append((v, v))
    return cols


def _safe_mean_std(series):
    """连续变量：mean ± SD"""
    m, s = series.mean(), series.std()
    if pd.isna(s): s = 0
    return f"{m:.4f} ± {s:.4f}"


def _safe_n_pct(series, value=1):
    """二值/水平：n (%)"""
    n = (series == value).sum()
    pct = 100 * n / len(series.dropna()) if len(series.dropna()) else 0
    return f"{int(n)} ({pct:.4f}%)"


def _format_pvalue(p):
    """p<0.001 显示为 <0.001"""
    if np.isnan(p):
        return ''
    return '<0.001' if p < 0.001 else f"{p:.4f}"


def _pvalue_continuous(df, col, group_col):
    try:
        from scipy.stats import kruskal
    except ImportError:
        return np.nan
    try:
        groups = [df.loc[df[group_col] == g, col].dropna() for g in [0, 1, 2] if (df[group_col] == g).any()]
        if len(groups) < 2 or any(len(x) < 2 for x in groups):
            return np.nan
        _, p = kruskal(*groups)
        return p
    except Exception:
        return np.nan


def _pvalue_categorical(df, col, group_col):
    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        return np.nan
    try:
        ct = pd.crosstab(df[group_col], df[col].fillna(-1))
        if ct.size < 2:
            return np.nan
        _, p, _, _ = chi2_contingency(ct)
        return p
    except Exception:
        return np.nan


def _cog_col(df):
    return next((c for c in df.columns if 'total_cognition' in c.lower() or 'total_cog' in c.lower()), None)


def tabulate_baseline_table_bps(
    df,
    group_col=GROUP_COL,
    group_labels=None,
    add_pvalues=True,
    p_col_name='P',
    continuous_pvalue_fn=None,
):
    """
    按 BPS_SECTIONS 生成 Table 1 行（不写盘）。
    group_labels: {0: str, 1: str, 2: str}，列顺序 0→1→2。
    continuous_pvalue_fn: (df, col, group_col) -> float；默认 Kruskal-Wallis（与主流程 Table 1 一致）。
    """
    if group_col not in df.columns:
        logger.warning("无 %s，无法生成 Table 1。", group_col)
        return None
    if group_labels is None:
        group_labels = GROUP_LABELS
    p_fn = continuous_pvalue_fn if continuous_pvalue_fn is not None else _pvalue_continuous

    rows = []
    g0, g1, g2 = (df[group_col] == 0), (df[group_col] == 1), (df[group_col] == 2)
    n0, n1, n2 = g0.sum(), g1.sum(), g2.sum()
    cols = [group_labels[0], group_labels[1], group_labels[2]]
    rows.append({
        'Variable': 'N',
        **{cols[0]: str(int(n0)), cols[1]: str(int(n1)), cols[2]: str(int(n2))},
        p_col_name: '',
    })

    def add_continuous(col, label):
        if col not in df.columns:
            return
        s0 = df.loc[g0, col].copy()
        s1 = df.loc[g1, col].copy()
        s2 = df.loc[g2, col].copy()
        if col == 'bmi':
            s0, s1, s2 = s0.clip(15, 50), s1.clip(15, 50), s2.clip(15, 50)
            bmi_mean_all = pd.concat([s0, s1, s2]).mean()
            if bmi_mean_all < 16 or bmi_mean_all > 35:
                logger.warning(
                    f"⚠️ Table 1 BMI 均值 {bmi_mean_all:.1f} 超出合理范围(18.5–24)，请检查数据源或重新运行预处理+插补"
                )
        r0 = _safe_mean_std(s0)
        r1 = _safe_mean_std(s1)
        r2 = _safe_mean_std(s2)
        p = p_fn(df, col, group_col) if add_pvalues else np.nan
        rows.append({'Variable': label, **{cols[0]: r0, cols[1]: r1, cols[2]: r2}, p_col_name: _format_pvalue(p)})

    def add_binary(col, label, value=1):
        if col not in df.columns:
            return
        r0 = _safe_n_pct(df.loc[g0, col], value=value)
        r1 = _safe_n_pct(df.loc[g1, col], value=value)
        r2 = _safe_n_pct(df.loc[g2, col], value=value)
        p = _pvalue_categorical(df, col, group_col) if add_pvalues else np.nan
        rows.append({'Variable': label, **{cols[0]: r0, cols[1]: r1, cols[2]: r2}, p_col_name: _format_pvalue(p)})

    for section_label, sec_cfg in BPS_SECTIONS:
        rows.append({'Variable': f'—— {section_label} ——', **{cols[0]: '', cols[1]: '', cols[2]: ''}, p_col_name: ''})
        _process_section(
            df, sec_cfg, rows, cols, g0, g1, g2, group_col, add_pvalues, add_continuous, add_binary,
            p_col_name=p_col_name,
        )
    return pd.DataFrame(rows)


def _process_section(df, cfg, rows, cols, g0, g1, g2, group_col, add_pvalues, add_continuous, add_binary, p_col_name='P'):
    """处理单个 BPS 维度的变量；p_col_name 与 tabulate_baseline_table_bps 一致（如 'P' 或 'P-value'）。"""
    p_key = p_col_name

    def append_row(variable, r0, r1, r2, p_str):
        rows.append({'Variable': variable, **{cols[0]: r0, cols[1]: r1, cols[2]: r2}, p_key: p_str})

    # continuous
    for col, label in cfg.get('continuous', []):
        add_continuous(col, label)
    # sex_col
    if cfg.get('sex_col') and cfg['sex_col'] in df.columns:
        for label, val in [('Female', 1), ('Male', 0)]:
            r0 = _safe_n_pct(df.loc[g0, cfg['sex_col']], value=val)
            r1 = _safe_n_pct(df.loc[g1, cfg['sex_col']], value=val)
            r2 = _safe_n_pct(df.loc[g2, cfg['sex_col']], value=val)
            p = _pvalue_categorical(df, cfg['sex_col'], group_col) if add_pvalues and label == 'Female' else np.nan
            append_row(label, r0, r1, r2, _format_pvalue(p) if label == 'Female' else '')
    # binary
    for t in cfg.get('binary', []):
        add_binary(t[0], t[1], value=t[2])
    # categorical
    for col, var_label in cfg.get('categorical', []):
        if col not in df.columns:
            continue
        try:
            levels = sorted(pd.Series(df[col].dropna().unique()).astype(int).tolist())
        except (ValueError, TypeError):
            continue
        if len(levels) <= 1:
            continue
        p = _pvalue_categorical(df, col, group_col) if add_pvalues else np.nan
        lev_map = CATEGORICAL_LEVEL_LABELS.get(col, {})
        for i, lev in enumerate(levels):
            r0 = _safe_n_pct(df.loc[g0, col], value=lev)
            r1 = _safe_n_pct(df.loc[g1, col], value=lev)
            r2 = _safe_n_pct(df.loc[g2, col], value=lev)
            lev_label = lev_map.get(lev, f"level {lev}")
            row_label = f"  {var_label}: {lev_label}"
            append_row(row_label, r0, r1, r2, _format_pvalue(p) if (i == 0) else '')
    # lifestyle_continuous
    for col, label in cfg.get('lifestyle_continuous', []):
        add_continuous(col, label)
    # lifestyle_binary
    for t in cfg.get('lifestyle_binary', []):
        add_binary(t[0], t[1], value=t[2])
    # physical
    for col, label in cfg.get('physical', []):
        add_continuous(col, label)
    # chronic_disease_cols
    for col, label in cfg.get('chronic_disease_cols', []):
        add_binary(col, label, value=1)
    # grip_col
    if cfg.get('grip_col'):
        c, l = cfg['grip_col']
        add_continuous(c, l)
    # optional_continuous
    for col, label in cfg.get('optional_continuous', []):
        add_continuous(col, label)
    # defining_continuous
    for col_or_key, label in cfg.get('defining_continuous', []):
        col = _cog_col(df) if col_or_key == 'total_cognition' else (col_or_key if col_or_key in df.columns else None)
        if col is not None:
            add_continuous(col, label)
    # outcome
    if cfg.get('outcome_col') and cfg['outcome_col'] in df.columns:
        add_binary(cfg['outcome_col'], cfg.get('outcome_label', 'Outcome'), value=1)


def generate_baseline_table(df, output_dir='evaluation_results', add_pvalues=True):
    """按生物-心理-社会模型生成 Table 1。"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"正在生成 Table 1（生物-心理-社会模型）到: {output_dir}")

    group_col = GROUP_COL
    if group_col not in df.columns:
        logger.warning("无 baseline_group，跳过 Table 1。")
        return None

    table1 = tabulate_baseline_table_bps(df, group_col=group_col, add_pvalues=add_pvalues, p_col_name='P')
    if table1 is None:
        return None
    out_path = os.path.join(output_dir, 'table1_baseline_characteristics.csv')
    table1.to_csv(out_path, index=False, encoding='utf-8-sig')
    try:
        table1.to_excel(os.path.join(output_dir, 'table1_baseline_characteristics.xlsx'), index=False, engine='openpyxl')
    except Exception as e:
        logger.warning(f"Excel 导出跳过 (需 openpyxl): {e}")
    logger.info(f"✅ Table 1 已生成: {out_path}")
    table1.to_csv(os.path.join(output_dir, 'table1_academic_final.csv'), index=False, encoding='utf-8-sig')

    # STROBE：Table 1 涉及变量的缺失比例（便于方法/补充表）
    collect_cols = _get_all_cols_from_bps(df)
    missing_rows = []
    seen = set()
    for col, label in collect_cols:
        if col not in df.columns or col in seen:
            continue
        seen.add(col)
        pct = df[col].isna().mean() * 100
        missing_rows.append({'Variable': label, 'Missing_pct': round(pct, 4), 'N_non_missing': int((df[col].notna()).sum())})
    if missing_rows:
        miss_df = pd.DataFrame(missing_rows)
        miss_df.to_csv(os.path.join(output_dir, 'table1_missing_summary.csv'), index=False, encoding='utf-8-sig')
        try:
            miss_df.to_excel(os.path.join(output_dir, 'table1_missing_summary.xlsx'), index=False, engine='openpyxl')
        except Exception:
            pass
        logger.info(f"✅ Table 1 缺失汇总已生成: {output_dir}/table1_missing_summary.csv")
    return table1
