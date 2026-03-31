# -*- coding: utf-8 -*-
"""
老年抑郁-认知共病研究 — 缺失值插补实验（npj 文献风格）

参考 npj Digital Medicine 等文献的插补实验设计：
1. 缺失值筛查与可视化
2. MCAR 缺失机制验证（单变量逻辑回归 + Bonferroni 校正）
3. 6 种插补方法对比（均值、中位数、线性回归、贝叶斯回归、KNN、MissForest）
4. 性能评估：NRMSE、标准差比率、均值差异率
5. 敏感性验证：分布、相关性
6. 按队列独立插补（推导/地理/时间），无数据泄露
"""
import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from scipy import stats

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 可调参数
# =============================================================================
RANDOM_SEED = 500
OUTPUT_DIR = 'imputation_npj_results'
# 缺失率分层阈值（%）
MISSING_MILD = 5      # 轻度 <5%
MISSING_MODERATE = 10 # 中度 5%-10%
MISSING_SEVERE = 10   # 重度 >10%，评估后剔除
# NRMSE 模拟缺失比例（在完整病例上人为引入）
SIMULATE_MISSING_PCT = 0.08
# 均值差异率可接受阈值
MEAN_DIFF_ACCEPT = 0.05
# 队列定义
COHORT_DERIVATION = 'derivation'   # 推导队列（东中部 或 全部）
COHORT_GEOGRAPHIC = 'geographic'   # 地理外部（西部）
COHORT_TEMPORAL = 'temporal'       # 时间外部（wave=4，保留原始缺失）
# 是否启用宽表纵向插补（利用同人历史波次填补当前波次缺失）
USE_WIDE_IMPUTATION = True
# 多重插补：N>0 时生成 m1..mN（供 Rubin 等）；默认与 config 一致（run_all 仍会再次同步）。
try:
    from config import N_MULTIPLE_IMPUTATIONS as N_MULTIPLE_IMPUTATIONS
except ImportError:
    N_MULTIPLE_IMPUTATIONS = 0
# 快速模式：MissForest 树数减半(50)，便于调试；若卡住可设 True 或 N_MULTIPLE_IMPUTATIONS=0
FAST_MODE = False


# =============================================================================
# 变量定义：按功能身份区分插补策略（与 Table 1 呈现一致）
# 功能维度：1) 结局→不插补剔除  2) 核心定义→不插补剔除  3) 协变量→插补
# =============================================================================
# 不可插补：结局变量、核心定义变量（缺失则剔除，不参与插补）
VARS_NO_IMPUTE = ['is_comorbidity_next', 'is_comorbidity', 'is_depression', 'is_cognitive_impairment']
DEFINING_COLS = ['total_cognition', 'total_cog', 'cesd10', 'cesd']  # 划分人群用，缺失剔除

# 协变量/预测因子（插补对象）：连续变量（与 config.COLS_TO_DROP 一致，已移除 rgrip/grip_strength_avg）
VARS_CONTINUOUS = [
    'age', 'bmi', 'mwaist',
    'puff', 'systo', 'diasto', 'pulse', 'lgrip', 'wspeed',
    'adlab_c', 'iadl',
    'family_size', 'income_total', 'sleep',
]

# 协变量/预测因子（插补对象）：分类变量（已移除 psyche/smokev，与论文基线表一致）
VARS_CATEGORICAL = [
    'gender', 'fall_down', 'disability',
    'hibpe', 'diabe', 'cancre', 'lunge', 'hearte', 'stroke', 'arthre',
    'srh', 'satlife',
    'rural', 'is_socially_isolated', 'pension', 'ins', 'retire', 'edu', 'marry',
    'exercise', 'drinkev',
]
# 核心干预变量（论文附录 S2：需保留完整病例子集用于敏感性分析；smokev 已移除）
CORE_INTERVENTION_VARS = ['exercise', 'drinkev']
# 不随时间变化的变量（按 ID 组内 ffill/bfill 利用同人其他波次填补）
VARS_TIME_INVARIANT = ['gender', 'edu', 'rural']
# 有序分类变量（贝叶斯回归+取整，非众数）
VARS_ORDINAL = ['edu', 'srh', 'satlife']
# 二分类变量（Exercise/Drinking 等）：随机森林分类器预测，非众数（smokev/psyche 已移除）
VARS_BINARY = ['exercise', 'drinkev', 'fall_down', 'disability',
               'hibpe', 'diabe', 'cancre', 'lunge', 'hearte', 'stroke', 'arthre',
               'rural', 'is_socially_isolated', 'pension', 'ins', 'retire', 'gender', 'marry']
# 插补值物理边界（防止 BMI 负数、年龄超范围等）
IMPUTATION_BOUNDS = {
    'age': (60, 120), 'bmi': (10, 60), 'mwaist': (40, 200),
    'systo': (60, 250), 'diasto': (30, 180), 'pulse': (30, 200),
    'sleep': (0, 24), 'family_size': (0, 30), 'adlab_c': (0, 6), 'iadl': (0, 8),
    'income_total': (0, 1e10),
}
# 扩展辅助变量（队列、地理位置、波次等，作为插补预测因子）
AUX_COLS_EXTRA = ['baseline_group', 'province', 'wave', 'age']

# 合并为基线表全部特征（去重、保序）
ANALYSIS_VARS = list(dict.fromkeys(VARS_CONTINUOUS + VARS_CATEGORICAL))
VARS_OUTCOME = ['is_comorbidity_next']


# =============================================================================
# 1. 数据加载
# =============================================================================
def load_data(path=None, use_preprocessed=True):
    """
    加载数据，支持 CSV/Excel，自定义路径。
    path: 自定义文件路径；若为 None，则使用预处理后的 CHARLS 或原始 CHARLS.csv
    """
    if path and os.path.exists(path):
        if path.endswith('.xlsx') or path.endswith('.xls'):
            df = pd.read_excel(path)
        else:
            for enc in ('utf-8', 'gbk', 'gb18030', 'latin-1'):
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                df = pd.read_csv(path, encoding='utf-8', errors='ignore')
        logger.info(f"已加载: {path}, n={len(df)}")
        return df

    # 默认：使用项目预处理
    if use_preprocessed and os.path.exists('preprocessed_data/CHARLS_final_preprocessed.csv'):
        df = pd.read_csv('preprocessed_data/CHARLS_final_preprocessed.csv', encoding='utf-8-sig')
        logger.info(f"已加载预处理数据, n={len(df)}")
        return df

    try:
        from charls_complete_preprocessing import preprocess_charls_data
        from run_multi_exposure_causal import prepare_exposures
        df = preprocess_charls_data('CHARLS.csv', age_min=60, write_output=False)
        if df is not None:
            prepare_exposures(df)
        return df
    except Exception as e:
        logger.error(f"加载失败: {e}")
        return None


# =============================================================================
# 1a. 辅助变量数值化（province/wave 等需参与插补，须为 float64/int64）
# =============================================================================
def ensure_aux_cols_numeric(df, cols=None):
    """
    将 province、wave 等辅助变量转为数值型，以便参与 IterativeImputer。
    CHARLS 地域聚集性强，province 作为插补特征可显著提升真实感。
    """
    cols = cols or ['province', 'wave']
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        if df[c].dtype in [np.float64, np.int64]:
            continue
        # 尝试直接转数值
        conv = pd.to_numeric(df[c], errors='coerce')
        if conv.notna().sum() >= len(df) * 0.9:
            df[c] = conv.astype(np.float64)
        else:
            # 对象/字符串：用类别编码（CHARLS 省份代码等）
            codes = pd.Categorical(df[c].astype(str)).codes
            df[c] = codes.astype(np.int64)
    return df


# =============================================================================
# 1b. 个体历史对齐（不随时间变化变量按 ID 组内 ffill/bfill）
# =============================================================================
def align_individual_history(df, cols, id_col='ID', wave_col='wave'):
    """
    针对教育、性别等不随时间变化的变量，利用 ID 进行组内前后填充（ffill/bfill）。
    在 ML 插补前执行，减少需插补的缺失量。
    """
    if id_col not in df.columns or wave_col not in df.columns:
        return df
    cols_avail = [c for c in cols if c in df.columns and df[c].isna().any()]
    if not cols_avail:
        return df
    df = df.copy()
    df = df.sort_values([id_col, wave_col])
    for c in cols_avail:
        df[c] = df.groupby(id_col)[c].transform(lambda x: x.ffill().bfill())
    return df


# =============================================================================
# 2. 缺失值筛查与可视化
# =============================================================================
def screen_missing(df, vars_to_check=None, output_dir=OUTPUT_DIR):
    """
    统计各变量缺失率，生成分布表、热力图、条形图。
    """
    vars_to_check = vars_to_check or [c for c in ANALYSIS_VARS + VARS_OUTCOME if c in df.columns]
    vars_to_check = [c for c in vars_to_check if c in df.columns]

    rows = []
    for c in vars_to_check:
        n_miss = df[c].isna().sum()
        pct = 100 * n_miss / len(df)
        if pct < MISSING_MILD:
            tier = 'mild'
        elif pct < MISSING_MODERATE:
            tier = 'moderate'
        else:
            tier = 'severe'
            rows.append({
                'Variable': c,
                'N_missing': int(n_miss),
                'Missing_pct': round(pct, 4),
                'Tier': tier,
                'N_complete': int(df[c].notna().sum()),
            })
    tbl = pd.DataFrame(rows)
    tbl = tbl.sort_values('Missing_pct', ascending=False)

    os.makedirs(output_dir, exist_ok=True)
    tbl.to_csv(os.path.join(output_dir, 'table1_missing_distribution.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"表1 缺失分布已保存")

    # 热力图（变量 x 队列，仅当存在多队列时）
    if 'baseline_group' in df.columns:
        groups_present = df['baseline_group'].dropna().unique()
        if len(groups_present) >= 2:
            heat_data = []
            for v in vars_to_check:
                row = {'Variable': v}
                for g in sorted(groups_present):
                    sub = df[df['baseline_group'] == g]
                    if len(sub) > 0:
                        pct = 100 * sub[v].isna().mean()
                        row[f'Cohort_{int(g)}'] = round(pct, 4)
                heat_data.append(row)
            if heat_data:
                heat_df = pd.DataFrame(heat_data).set_index('Variable')
                plt.figure(figsize=(6, max(6, len(heat_df) * 0.3)))
                sns.heatmap(heat_df, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Missing %'})
                plt.title('Missing Rate by Variable and Cohort')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'fig1_missing_heatmap.png'), dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("图1 缺失热力图已保存")

    # 条形图
    plt.figure(figsize=(10, max(5, len(tbl) * 0.35)))
    colors = ['#2ecc71' if t == 'mild' else '#f39c12' if t == 'moderate' else '#e74c3c' for t in tbl['Tier']]
    plt.barh(tbl['Variable'], tbl['Missing_pct'], color=colors)
    plt.axvline(MISSING_MILD, color='gray', linestyle='--', alpha=0.7, label=f'Mild <{MISSING_MILD}%')
    plt.axvline(MISSING_MODERATE, color='gray', linestyle=':', alpha=0.7, label=f'Moderate {MISSING_MILD}-{MISSING_MODERATE}%')
    plt.xlabel('Missing rate (%)')
    plt.title('Missing Value Screening by Variable')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_missing_barchart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("图2 缺失条形图已保存")

    return tbl


# =============================================================================
# 3. MCAR 验证（单变量逻辑回归 + Bonferroni）
# =============================================================================
def test_mcar(df, moderate_vars=None, output_dir=OUTPUT_DIR):
    """
    对中度缺失变量，用逻辑回归预测缺失与否，Bonferroni 校正。
    若所有变量 p > α/n，则支持 MCAR。
    """
    moderate_vars = moderate_vars or []
    if not moderate_vars:
        tbl = pd.read_csv(os.path.join(output_dir, 'table1_missing_distribution.csv'), encoding='utf-8-sig')
        moderate_vars = tbl[tbl['Tier'] == 'moderate']['Variable'].tolist()

    moderate_vars = [v for v in moderate_vars if v in df.columns and df[v].isna().any()]
    if not moderate_vars:
        logger.info("无中度缺失变量，跳过 MCAR 验证")
        return pd.DataFrame()

    # 辅助变量（预测缺失的协变量，排除待验证变量，需无缺失）
    aux = [c for c in ANALYSIS_VARS if c not in moderate_vars and c in df.columns and df[c].notna().all()]
    aux = [c for c in aux if df[c].dtype in [np.float64, np.int64]][:15]
    if len(aux) < 1:
        logger.warning("MCAR 协变量不足（需≥1个无缺失变量），跳过 MCAR 验证")
        return pd.DataFrame()

    rows = []
    n_tests = len(moderate_vars)
    alpha_bonf = 0.05 / max(1, n_tests)

    for var in moderate_vars:
        y_miss = df[var].isna().astype(int)
        if y_miss.sum() < 10 or y_miss.sum() == len(df):
            continue
        X = df[aux].fillna(df[aux].median())
        mask = X.notna().all(axis=1)
        if mask.sum() < 50:
            continue
        try:
            model = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
            model.fit(X[mask], y_miss[mask])
            pred = model.predict_proba(X[mask])[:, 1]
            # 使用卡方检验：预测缺失 vs 观测缺失的独立性
            from scipy.stats import chi2_contingency
            pred_bin = (pred > 0.5).astype(int)
            tab = pd.crosstab(y_miss[mask], pred_bin)
            if tab.size >= 4:
                _, p, _, _ = chi2_contingency(tab)
            else:
                p = 0.5
            rows.append({
                'Variable': var,
                'N_missing': int(y_miss.sum()),
                'P_value': round(p, 4),
                'Alpha_Bonferroni': round(alpha_bonf, 4),
                'MCAR_supported': p > alpha_bonf,
            })
        except Exception as e:
            rows.append({'Variable': var, 'N_missing': int(y_miss.sum()), 'P_value': np.nan, 'Alpha_Bonferroni': alpha_bonf, 'MCAR_supported': False})

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(output_dir, 'table2_mcar_validation.csv'), index=False, encoding='utf-8-sig')
    n_support = tbl['MCAR_supported'].sum() if 'MCAR_supported' in tbl.columns else 0
    logger.info(f"表2 MCAR验证已保存; {n_support}/{len(tbl)} 变量支持 MCAR (p>{alpha_bonf:.4f})")
    return tbl


# =============================================================================
# 3b. Little's MCAR 检验（p<0.05 拒绝 MCAR，支持使用多元插补）
# =============================================================================
def test_littles_mcar(df, cols=None, output_dir=OUTPUT_DIR):
    """
    Little's MCAR 检验。H0: 数据为 MCAR。
    p<0.05 拒绝 H0 → 非 MCAR（多为 MAR），此时多元插补（MICE/MissForest）更合理。
    """
    cols = cols or [c for c in ANALYSIS_VARS if c in df.columns and df[c].dtype in [np.float64, np.int64]]
    cols = [c for c in cols if df[c].isna().any()][:20]
    if len(cols) < 2:
        return np.nan
    X = df[cols].copy()
    try:
        global_mean = X.mean()
        global_cov = X.cov()
        if global_cov.isna().any().any() or np.linalg.det(global_cov.values) == 0:
            return np.nan
        r = 1 * X.isnull()
        n_features = len(cols)
        mdp = np.dot(r, [2**i for i in range(n_features)])
        sorted_mdp = sorted(np.unique(mdp))
        n_pat = len(sorted_mdp)
        mdp_map = {v: i for i, v in enumerate(sorted_mdp)}
        X['_mdp'] = [mdp_map.get(x, 0) for x in mdp]
        pj, d2 = 0, 0
        for i in range(n_pat):
            sub = X.loc[X['_mdp'] == i, cols]
            select_mask = ~sub.isnull().any()
            sv = [c for c in cols if select_mask.get(c, False)]
            pj += len(sv)
            if not sv:
                continue
            means = sub[sv].mean() - global_mean[sv]
            scov = global_cov.loc[sv, sv].values
            mj = len(sub)
            try:
                parta = np.dot(means.T, np.linalg.solve(scov, np.eye(len(sv))))
                d2 += mj * np.dot(parta, means)
            except np.linalg.LinAlgError:
                continue
        df_stat = max(1, pj - n_features)
        p_value = 1 - stats.chi2.cdf(d2, df_stat)
        tbl = pd.DataFrame([{'Test': "Little's MCAR", 'Chi2': round(d2, 4), 'df': df_stat, 'P_value': round(p_value, 4),
                             'MCAR_rejected': p_value < 0.05, 'Interpretation': 'Non-MCAR (MAR likely)' if p_value < 0.05 else 'MCAR plausible'}])
        tbl.to_csv(os.path.join(output_dir, 'table2b_littles_mcar.csv'), index=False, encoding='utf-8-sig')
        logger.info(f"Little's MCAR: p={p_value:.4f} ({'拒绝 MCAR，支持多元插补' if p_value < 0.05 else '未拒绝 MCAR'})")
        return p_value
    except Exception as e:
        logger.warning(f"Little's MCAR 检验跳过: {e}")
        return np.nan


def _write_methodology_justification(output_dir, littles_p):
    """生成论文方法学论证文本，供 Methods 引用。"""
    path = os.path.join(output_dir, 'methodology_justification.txt')
    if pd.isna(littles_p):
        narrative = "Little's MCAR test was not performed (insufficient data)."
    elif littles_p < 0.05:
        narrative = (
            "Little's MCAR test rejected the null hypothesis (p={:.4f}), indicating that missingness "
            "is not completely at random (MCAR). This supports the use of multivariate imputation "
            "(MICE/MissForest) that accounts for MAR (Missing At Random) mechanisms, thereby reducing bias."
        ).format(littles_p)
    else:
        narrative = (
            "Little's MCAR test did not reject the null (p={:.4f}). We nevertheless used "
            "multivariate imputation (MICE/MissForest) for robustness, as CHARLS missingness "
            "often exhibits MAR patterns (e.g., health-related non-response)."
        ).format(littles_p)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("Imputation Methodology Justification (for Methods section)\n")
        f.write("=" * 60 + "\n\n")
        f.write(narrative + "\n\n")
        f.write("Suggested Methods text:\n")
        f.write('"Due to rejected MCAR assumption (Little\'s test p<0.05), we employed ')
        f.write('chain-equation multiple imputation (MICE) / machine-learning imputation (MissForest) ')
        f.write('to reduce bias under MAR."\n')
    logger.info(f"  方法学论证已保存: methodology_justification.txt")


def _append_methodology_method_selection(output_dir, best_method, best_nrmse):
    """Append NRMSE-based method selection rationale to methodology_justification.txt (English)."""
    path = os.path.join(output_dir, 'methodology_justification.txt')
    with open(path, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("Method Selection (NRMSE-based)\n")
        f.write("=" * 60 + "\n\n")
        f.write(
            "We evaluated six imputation methods (Mean, Median, LinearRegression, BayesianRidge, "
            "KNN, MissForest) on continuous variables by introducing 8%% MCAR missingness on complete "
            "cases and computing NRMSE (Normalized Root Mean Square Error). The method with the "
            "lowest mean NRMSE across three replicates was selected for the final imputation.\n\n"
        )
        f.write(f"Selected method: {best_method} (NRMSE = {best_nrmse})\n\n")
        f.write(
            "Suggested Methods text:\n"
            '"Continuous covariates were imputed using [method] (selected by lowest NRMSE among '
            'six candidate methods). Ordinal and binary categorical variables were imputed via '
            'Bayesian regression with rounding and random forest classification, respectively."\n'
        )


# =============================================================================
# 4. 6 种插补方法（支持 aux_cols、bounds、队列/轨迹特征）
# =============================================================================
def _cols_for_imputer(df, cols, aux_cols=None):
    """将 aux_cols 加入特征矩阵供多元插补使用，aux 本身不缺失故不会被改写。"""
    if not aux_cols:
        return cols
    aux_valid = [c for c in aux_cols if c in df.columns and df[c].notna().all() and df[c].dtype in [np.float64, np.int64]]
    if not aux_valid:
        return cols
    return aux_valid + [c for c in cols if c not in aux_valid]


def _apply_imputation_bounds(out, cols):
    """对插补结果施加物理边界，防止 BMI 负数、年龄超范围等。"""
    for c in cols:
        if c in IMPUTATION_BOUNDS and c in out.columns:
            lo, hi = IMPUTATION_BOUNDS[c]
            out[c] = out[c].clip(lo, hi)


def _safe_assign_imputed(out, X_imp, cols, cols_use):
    """安全地将 X_imp 列赋给 out，避免 cols_use 与 X_imp 列数不一致时的 IndexError；
    目标列为 int 时取整，避免 LossySetitemError。"""
    n_out = X_imp.shape[1]
    target_cols = [c for c in cols if c in cols_use]
    skipped = []
    for c in target_cols:
        i = cols_use.index(c)
        if i < n_out:
            vals = X_imp[:, i].copy()
            if c in out.columns and np.issubdtype(out[c].dtype, np.integer):
                vals = np.round(vals).astype(np.int64)
            out[c] = vals
        else:
            skipped.append(c)
    if skipped:
        logger.warning(f"  插补跳过 {len(skipped)} 列（索引越界）: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")


def post_imputation_cleanup(df):
    """
    插补后逻辑一致性检查与校准。
    1) 年龄、BMI 等连续变量：物理边界，防止负数或超出生理常识
    2) 教育程度、生活满意度等有序分类：clip 后四舍五入取整，保持量表离散特性
    """
    out = df.copy()
    # 连续变量物理边界（防止负数、超出生理范围）
    for c, (lo, hi) in IMPUTATION_BOUNDS.items():
        if c in out.columns:
            out[c] = out[c].clip(lo, hi)
    # 有序分类变量：clip 到合法水平后四舍五入取整
    if 'edu' in out.columns:
        out['edu'] = out['edu'].clip(1, 4)
        out['edu'] = np.round(out['edu']).astype(int)
    if 'srh' in out.columns:
        out['srh'] = out['srh'].clip(1, 5)
        out['srh'] = np.round(out['srh']).astype(int)
    if 'satlife' in out.columns:
        out['satlife'] = out['satlife'].clip(1, 5)
        out['satlife'] = np.round(out['satlife']).astype(int)
    # 二分类变量
    for c in ['exercise', 'drinkev', 'gender', 'marry', 'rural', 'fall_down', 'disability',
              'hibpe', 'diabe', 'cancre', 'lunge', 'hearte', 'stroke', 'arthre',
              'is_socially_isolated', 'pension', 'ins', 'retire']:
        if c in out.columns:
            out[c] = out[c].clip(0, 1)
            out[c] = np.round(out[c]).astype(int)
    return out

def impute_mean(df, cols, aux_cols=None, seed=None):
    imp = SimpleImputer(strategy='mean')
    out = df.copy()
    X_imp = imp.fit_transform(df[cols])
    for j, c in enumerate(cols):
        if c not in out.columns:
            continue
        vals = X_imp[:, j]
        if np.issubdtype(out[c].dtype, np.integer):
            vals = np.round(vals).astype(np.int64)
        out[c] = vals
    return out

def impute_median(df, cols, aux_cols=None, seed=None):
    imp = SimpleImputer(strategy='median')
    out = df.copy()
    X_imp = imp.fit_transform(df[cols])
    for j, c in enumerate(cols):
        if c not in out.columns:
            continue
        vals = X_imp[:, j]
        if np.issubdtype(out[c].dtype, np.integer):
            vals = np.round(vals).astype(np.int64)
        out[c] = vals
    return out

def _get_iterative_bounds(cols):
    """为 IterativeImputer 生成 min_value/max_value 数组。"""
    min_v, max_v = np.full(len(cols), -np.inf), np.full(len(cols), np.inf)
    for i, c in enumerate(cols):
        if c in IMPUTATION_BOUNDS:
            min_v[i], max_v[i] = IMPUTATION_BOUNDS[c]
    return min_v, max_v

def _imputer_with_bounds(estimator, cols_use, seed=None):
    """带边界约束的 IterativeImputer（兼容旧版 sklearn）。"""
    rs = seed if seed is not None else RANDOM_SEED
    try:
        min_v, max_v = _get_iterative_bounds(cols_use)
        return IterativeImputer(estimator=estimator, max_iter=5, random_state=rs, min_value=min_v, max_value=max_v)
    except TypeError:
        return IterativeImputer(estimator=estimator, max_iter=5, random_state=rs)

def impute_linear(df, cols, aux_cols=None, seed=None):
    cols_use = _cols_for_imputer(df, cols, aux_cols)
    imp = _imputer_with_bounds(LinearRegression(), cols_use, seed=seed)
    out = df.copy()
    X_imp = imp.fit_transform(df[cols_use])
    _safe_assign_imputed(out, X_imp, cols, cols_use)
    _apply_imputation_bounds(out, cols)
    return out

def impute_bayesian(df, cols, aux_cols=None, seed=None):
    cols_use = _cols_for_imputer(df, cols, aux_cols)
    imp = _imputer_with_bounds(BayesianRidge(), cols_use, seed=seed)
    out = df.copy()
    X_imp = imp.fit_transform(df[cols_use])
    _safe_assign_imputed(out, X_imp, cols, cols_use)
    _apply_imputation_bounds(out, cols)
    return out

def impute_knn(df, cols, aux_cols=None, n_neighbors=5, seed=None):
    cols_use = _cols_for_imputer(df, cols, aux_cols)
    imp = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    out = df.copy()
    X_imp = imp.fit_transform(df[cols_use])
    _safe_assign_imputed(out, X_imp, cols, cols_use)
    _apply_imputation_bounds(out, cols)
    return out

def impute_missforest(df, cols, aux_cols=None, n_estimators=None, seed=None):
    cols_use = _cols_for_imputer(df, cols, aux_cols)
    rs = seed if seed is not None else RANDOM_SEED
    n_est = (50 if FAST_MODE else 100) if n_estimators is None else n_estimators
    imp = _imputer_with_bounds(RandomForestRegressor(n_estimators=n_est, random_state=rs), cols_use, seed=rs)
    out = df.copy()
    X_imp = imp.fit_transform(df[cols_use])
    _safe_assign_imputed(out, X_imp, cols, cols_use)
    _apply_imputation_bounds(out, cols)
    return out

def impute_mode(df, cols, aux_cols=None, seed=None):
    """分类变量插补：众数（名义变量）。Mean/Median 为单变量方法，无法利用 aux_cols。"""
    imp = SimpleImputer(strategy='most_frequent')
    out = df.copy()
    X_imp = imp.fit_transform(df[cols])
    for j, c in enumerate(cols):
        if c not in out.columns:
            continue
        vals = X_imp[:, j]
        if np.issubdtype(out[c].dtype, np.integer):
            vals = np.round(vals).astype(np.int64)
        out[c] = vals
    return out

def impute_binary_rf(df, cols, aux_cols=None, seed=None):
    """二分类变量：随机森林分类器预测，避免众数压缩方差。"""
    from sklearn.ensemble import RandomForestClassifier
    out = df.copy()
    rs = seed if seed is not None else RANDOM_SEED
    n_cols = len([c for c in cols if c in df.columns and df[c].isna().sum() > 0])
    done = 0
    for c in cols:
        if c not in df.columns or df[c].isna().sum() == 0:
            continue
        feat_cols = [x for x in (aux_cols or []) + [y for y in ANALYSIS_VARS if y != c and y in df.columns]
                     if x in df.columns and df[x].notna().all() and df[x].dtype in [np.float64, np.int64]][:20]
        if len(feat_cols) < 2:
            out[c] = df[c].fillna(df[c].mode().iloc[0] if len(df[c].mode()) > 0 else 0)
            continue
        mask = df[c].notna()
        if mask.sum() < 30:
            continue
        X_tr, y_tr = df.loc[mask, feat_cols].fillna(df[feat_cols].median()), df.loc[mask, c].astype(int)
        X_miss = df.loc[~mask, feat_cols].fillna(df[feat_cols].median())
        try:
            clf = RandomForestClassifier(n_estimators=50, random_state=rs)
            clf.fit(X_tr, y_tr)
            out.loc[~mask, c] = clf.predict(X_miss)
            done += 1
            if n_cols >= 10 and done % 5 == 0:
                logger.info(f"      二分类插补进度: {done}/{n_cols}")
        except Exception:
            out[c] = df[c].fillna(df[c].mode().iloc[0] if len(df[c].mode()) > 0 else 0)
    return out

def impute_ordinal(df, cols, aux_cols=None, seed=None):
    """有序分类变量：贝叶斯回归+取整到合法水平，避免众数压缩方差。"""
    cols_use = _cols_for_imputer(df, cols, aux_cols)
    rs = seed if seed is not None else RANDOM_SEED
    imp = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=rs)
    out = df.copy()
    X_imp = imp.fit_transform(df[cols_use])
    for j, c in enumerate(cols):
        if c not in cols_use:
            continue
        idx = cols_use.index(c)
        vals = X_imp[:, idx]
        valid = df[c].dropna().unique()
        if len(valid) > 0:
            valid = np.sort(valid.astype(float))
            vals = np.clip(vals, valid.min(), valid.max())
            nearest = np.array([valid[np.argmin(np.abs(valid - v))] for v in vals])
        else:
            nearest = vals
        if c in out.columns and np.issubdtype(out[c].dtype, np.integer):
            nearest = np.round(nearest).astype(np.int64)
        out[c] = nearest
    return out

IMPUTATION_METHODS = {
    'Mean': impute_mean,
    'Median': impute_median,
    'LinearRegression': impute_linear,
    'BayesianRidge': impute_bayesian,
    'KNN': impute_knn,
    'MissForest': impute_missforest,
}


# =============================================================================
# 5. NRMSE 与辅助指标
# =============================================================================
def compute_nrmse(original, imputed, mask_missing):
    """NRMSE = RMSE / std(original)，若 std≈0 则用 ptp"""
    o = original[mask_missing].values.astype(float)
    i = imputed[mask_missing].values.astype(float)
    rmse = np.sqrt(mean_squared_error(o, i))
    denom = np.std(o)
    if denom < 1e-10:
        denom = np.ptp(o)
    if denom < 1e-10:
        denom = 1.0
    return rmse / denom

def compute_sd_ratio(original, imputed, mask_missing):
    return np.std(imputed[mask_missing].values) / max(1e-10, np.std(original[mask_missing].values))

def compute_mean_diff_ratio(original, imputed, mask_missing):
    return np.abs(imputed[mask_missing].mean() - original[mask_missing].mean()) / max(1e-10, np.abs(original[mask_missing].mean()))


def evaluate_imputation_nrmse(df, cols, aux_cols=None, pct_missing=SIMULATE_MISSING_PCT, n_repeat=3):
    """
    在完整病例上人为引入 MCAR 缺失，插补后计算 NRMSE、SD 比率、均值差异率。
    aux_cols: 队列等已知变量，作为预测特征参与多元插补（不参与插补）。
    """
    complete = df.dropna(subset=cols)
    if len(complete) < 100:
        return pd.DataFrame()

    np.random.seed(RANDOM_SEED)
    results = []
    for method_name, impute_fn in IMPUTATION_METHODS.items():
        logger.info(f"  NRMSE 评估: {method_name} (共{len(IMPUTATION_METHODS)}种方法，各{n_repeat}次重复)")
        nrmse_list, sd_ratio_list, mean_diff_list = [], [], []
        for rep in range(n_repeat):
            df_sim = complete.copy()
            # 对每列随机引入 pct_missing 比例的缺失
            for c in cols:
                n_miss = max(5, int(len(df_sim) * pct_missing))
                idx = np.random.choice(df_sim.index, size=min(n_miss, len(df_sim)), replace=False)
                df_sim.loc[idx, c] = np.nan

            try:
                df_imp = impute_fn(df_sim, cols, aux_cols=aux_cols)
            except Exception:
                continue
            for c in cols:
                m = df_sim[c].isna()
                if m.sum() < 5:
                    continue
                nrmse_list.append(compute_nrmse(complete[c], df_imp[c], m))
                sd_ratio_list.append(compute_sd_ratio(complete[c], df_imp[c], m))
                mean_diff_list.append(compute_mean_diff_ratio(complete[c], df_imp[c], m))

        if nrmse_list:
            results.append({
                'Method': method_name,
                'NRMSE_mean': round(np.mean(nrmse_list), 4),
                'NRMSE_sd': round(np.std(nrmse_list), 4),
                'SD_ratio_mean': round(np.mean(sd_ratio_list), 4),
                'Mean_diff_ratio_mean': round(np.mean(mean_diff_list), 4),
                'Mean_diff_acceptable': np.mean(mean_diff_list) < MEAN_DIFF_ACCEPT,
            })
    return pd.DataFrame(results)


# =============================================================================
# 5b. 分类变量插补验证（一致性 / Kappa）
# =============================================================================
def evaluate_categorical_imputation(df, cols, aux_cols=None, pct_missing=0.08, n_repeat=3):
    """
    对分类变量人为引入缺失，按类型使用实际插补策略（ordinal/binary/mode），计算准确率、Kappa、F1。
    """
    from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
    cols = [c for c in cols if c in df.columns and df[c].notna().sum() >= 50]
    if not cols:
        return pd.DataFrame()
    complete = df.dropna(subset=cols)
    if len(complete) < 100:
        return pd.DataFrame()
    np.random.seed(RANDOM_SEED)
    results = []
    for c in cols:
        acc_list, kappa_list, f1_list = [], [], []
        for rep in range(n_repeat):
            df_sim = complete.copy()
            n_miss = max(5, int(len(df_sim) * pct_missing))
            idx = np.random.choice(df_sim.index, size=min(n_miss, len(df_sim)), replace=False)
            df_sim.loc[idx, c] = np.nan
            if c in VARS_ORDINAL:
                df_imp = impute_ordinal(df_sim, [c], aux_cols=aux_cols)
            elif c in VARS_BINARY:
                df_imp = impute_binary_rf(df_sim, [c], aux_cols=aux_cols)
            else:
                df_imp = impute_mode(df_sim, [c])
            y_true = complete.loc[idx, c].astype(int)
            y_pred = df_imp.loc[idx, c].astype(int)
            if len(np.unique(y_true)) >= 2 and len(np.unique(y_pred)) >= 2:
                acc_list.append(accuracy_score(y_true, y_pred))
                kappa_list.append(cohen_kappa_score(y_true, y_pred))
                try:
                    n_classes = len(np.unique(np.concatenate([y_true.values, y_pred.values])))
                    avg = 'binary' if n_classes == 2 else 'weighted'
                    f1_list.append(f1_score(y_true, y_pred, average=avg, zero_division=0))
                except Exception:
                    f1_list.append(np.nan)
        if acc_list:
            row = {'Variable': c, 'Accuracy_mean': round(np.mean(acc_list), 4), 'Kappa_mean': round(np.mean(kappa_list), 4)}
            if f1_list and not all(pd.isna(f1_list)):
                row['F1_mean'] = round(np.nanmean(f1_list), 4)
            results.append(row)
    return pd.DataFrame(results)


# =============================================================================
# 6. 敏感性验证（含诊断：观测数、插补后仍缺失数、仅插补格均值）
# =============================================================================
def sensitivity_validation(original, imputed, cols, output_dir=OUTPUT_DIR):
    """
    分布对比、相关性、统计量。
    Original_mean = 插补前该列「有观测」的均值；Imputed_mean = 插补后该列「非缺失」的均值。
    若插补后仍有缺失（如宽表跳过列），两者可能基于不同样本，需结合 table4_diagnostics 解读。
    """
    cols = [c for c in cols if c in original.columns and c in imputed.columns]
    if len(cols) < 1:
        return pd.DataFrame()

    rows = []
    diag_rows = []
    for c in cols:
        o = original[c].dropna()
        i = imputed[c]
        if len(o) < 10:
            continue
        n_orig_obs = int(o.shape[0])
        n_imp_valid = int(i.notna().sum())
        n_still_missing = int(original[c].notna().shape[0]) - n_imp_valid if original[c].notna().sum() >= n_imp_valid else 0
        n_still_missing = max(0, int(imputed[c].isna().sum()))

        # 仅被插补的格子：原始为缺失、插补后非缺失
        mask_filled = original[c].isna() & imputed[c].notna()
        mean_imputed_cells = float(imputed.loc[mask_filled, c].mean()) if mask_filled.sum() > 0 else np.nan

        i_valid = i.dropna()
        if len(i_valid) < 10:
            continue
        _, ks_p = stats.ks_2samp(o, i_valid)
        mean_diff = (i_valid.mean() - o.mean()) / max(1e-10, abs(o.mean())) if o.mean() != 0 else 0
        rows.append({
            'Variable': c,
            'N_observed': n_orig_obs,
            'N_valid_after_imputation': n_imp_valid,
            'N_still_missing': n_still_missing,
            'Original_mean': round(o.mean(), 4),
            'Imputed_mean': round(i_valid.mean(), 4),
            'Mean_diff_pct': round(100 * mean_diff, 4),
            'KS_p': round(ks_p, 4),
            'Distribution_similar': ks_p > 0.05,
        })
        # 诊断：仅插补格均值、异常标记
        abs_diff_pct = abs(100 * mean_diff)
        flag = 'OK'
        if n_still_missing > 100:
            flag = 'remaining_NaN'
        elif abs_diff_pct > 50:
            flag = 'large_shift'
        elif abs_diff_pct > 20:
            flag = 'moderate_shift'
        diag_rows.append({
            'Variable': c,
            'N_orig_observed': n_orig_obs,
            'N_valid_after_imputation': n_imp_valid,
            'N_still_missing': n_still_missing,
            'Original_mean': round(o.mean(), 4),
            'Imputed_mean': round(i_valid.mean(), 4),
            'Mean_of_imputed_cells_only': round(mean_imputed_cells, 4) if pd.notna(mean_imputed_cells) else np.nan,
            'Mean_diff_pct': round(100 * mean_diff, 4),
            'Flag': flag,
        })
    tbl = pd.DataFrame(rows)
    if diag_rows:
        pd.DataFrame(diag_rows).to_csv(os.path.join(output_dir, 'table4_diagnostics.csv'), index=False, encoding='utf-8-sig')
        # 说明文件：便于核查敏感性异常
        readme_path = os.path.join(output_dir, 'table4_sensitivity_readme.txt')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("table4_sensitivity_validation.csv / table4_diagnostics.csv 解读\n")
            f.write("============================================================\n")
            f.write("Original_mean: 插补前该列「有观测」的均值\n")
            f.write("Imputed_mean: 插补后该列「非缺失」的均值（若 N_still_missing>0 则与 Original 非同一批样本）\n")
            f.write("Mean_of_imputed_cells_only: 仅「原缺失、插补后填入」的格子均值，用于判断插补值是否偏倚\n")
            f.write("Flag: OK=正常; remaining_NaN=插补后仍缺失>100; moderate_shift=均值差异 20-50%%; large_shift=>50%%\n")
            f.write("若 large_shift 且 Mean_of_imputed_cells_only 与 Original_mean 差异大，建议检查该变量单位/波次一致性或宽表跳过列。\n")
        logger.info("  敏感性诊断已保存: table4_diagnostics.csv, table4_sensitivity_readme.txt")
    if len(cols) >= 2:
        corr_orig = original[cols].corr()
        corr_imp = imputed[cols].corr()
        frob = np.linalg.norm(corr_orig.values - corr_imp.values, 'fro')
        tbl = pd.concat([tbl, pd.DataFrame([{'Variable': 'Corr_Frobenius_diff', 'N_observed': np.nan, 'N_valid_after_imputation': np.nan, 'N_still_missing': np.nan, 'Original_mean': np.nan, 'Imputed_mean': np.nan, 'Mean_diff_pct': round(frob, 4), 'KS_p': np.nan, 'Distribution_similar': np.nan}])], ignore_index=True)
    tbl.to_csv(os.path.join(output_dir, 'table4_sensitivity_validation.csv'), index=False, encoding='utf-8-sig')

    # 分布对比图（密度 + 箱线，优先包含核心变量 bmi/sleep/exercise）
    n_plot = min(6, len([c for c in cols if c != 'Corr_Frobenius_diff']))
    if n_plot < 1:
        return tbl
    prefer = [c for c in ['bmi', 'sleep', 'exercise'] if c in cols and c in original.columns]
    rest = [c for c in cols if c in original.columns and c not in prefer][:n_plot - len(prefer)]
    plot_cols = (prefer + rest)[:n_plot]
    fig, axes = plt.subplots(2, n_plot, figsize=(4 * n_plot, 8))
    if n_plot == 1:
        axes = np.array([axes]).T
    for j, c in enumerate(plot_cols):
        ax = axes[0, j]
        o = original[c].dropna()
        i = imputed[c]
        ax.hist(o, bins=20, alpha=0.5, label='Original', density=True, color='blue')
        ax.hist(i, bins=20, alpha=0.5, label='Imputed', density=True, color='orange')
        ax.set_title(c)
        ax.legend(fontsize=8)
        ax = axes[1, j]
        ax.boxplot([o, i], labels=['Original', 'Imputed'])
        ax.set_title(c)
    plt.suptitle('Before vs After Imputation: Distribution Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_distribution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("图4 分布对比已保存")
    return tbl


# =============================================================================
# 6a. KDE 密度分布对比图（保存至 pipeline_trace）
# =============================================================================
def plot_kde_before_after(original, imputed, cols, output_dir=OUTPUT_DIR):
    """
    插补前后 KDE 密度分布对比图，保存至 pipeline_trace 文件夹。
    """
    cols = [c for c in cols if c in original.columns and c in imputed.columns]
    if not cols:
        return
    trace_dir = os.path.join(output_dir, 'pipeline_trace')
    os.makedirs(trace_dir, exist_ok=True)
    n_plot = min(6, len(cols))
    prefer = [c for c in ['bmi', 'sleep', 'exercise', 'age'] if c in cols]
    rest = [c for c in cols if c not in prefer][:n_plot - len(prefer)]
    plot_cols = (prefer + rest)[:n_plot]
    fig, axes = plt.subplots(2, (n_plot + 1) // 2, figsize=(5 * ((n_plot + 1) // 2), 8))
    if n_plot == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()[:n_plot]
    for j, c in enumerate(plot_cols):
        ax = axes[j]
        o = original[c].dropna()
        i = imputed[c]
        if len(o) >= 5:
            try:
                sns.kdeplot(o, ax=ax, label='Before imputation', color='blue', linewidth=2)
            except Exception:
                ax.hist(o, bins=20, alpha=0.5, density=True, label='Before', color='blue')
        if len(i) >= 5:
            try:
                sns.kdeplot(i, ax=ax, label='After imputation', color='orange', linewidth=2)
            except Exception:
                ax.hist(i, bins=20, alpha=0.5, density=True, label='After', color='orange')
        ax.set_title(c)
        ax.legend(fontsize=8)
        ax.set_xlim(left=0)
    for k in range(j + 1, len(axes)):
        axes[k].set_visible(False)
    plt.suptitle('Before vs After Imputation: KDE Density Comparison')
    plt.tight_layout()
    path = os.path.join(trace_dir, 'fig_kde_before_after_imputation.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  KDE 密度对比图已保存: pipeline_trace/fig_kde_before_after_imputation.png")


# =============================================================================
# 6b. 插补收敛图（Convergence Plot）
# =============================================================================
def plot_imputation_convergence(df, cols, aux_cols=None, max_iter=10, output_dir=OUTPUT_DIR):
    """
    展示 IterativeImputer 在多次迭代中均值和标准差趋于稳定的过程。
    证明插补算法收敛且可靠。
    """
    cols = [c for c in cols if c in df.columns and df[c].isna().any()][:4]
    if not cols:
        return
    cols_use = _cols_for_imputer(df, cols, aux_cols)
    means, stds = {c: [] for c in cols}, {c: [] for c in cols}
    for it in range(1, max_iter + 1):
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=it, random_state=RANDOM_SEED)
        X_imp = imp.fit_transform(df[cols_use])
        for j, c in enumerate(cols):
            if c in cols_use:
                idx = cols_use.index(c)
                vals = X_imp[:, idx]
                means[c].append(np.nanmean(vals))
                stds[c].append(np.nanstd(vals))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    x = range(1, max_iter + 1)
    for c in cols:
        if means[c]:
            axes[0].plot(x, means[c], '-o', label=c, markersize=4)
            axes[1].plot(x, stds[c], '-o', label=c, markersize=4)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Mean')
    axes[0].set_title('Imputation Convergence: Mean by Iteration')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Std')
    axes[1].set_title('Imputation Convergence: Std by Iteration')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_imputation_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("图5 插补收敛图已保存")


# =============================================================================
# 7. 按队列独立插补（无泄露）
# =============================================================================
def split_cohorts(df):
    """划分推导、地理外部、时间外部队列。各队列独立插补，无数据泄露。"""
    df = df.copy()
    PROVINCE_TO_REGION = {1: 'E', 2: 'E', 4: 'E', 5: 'E', 6: 'E', 16: 'E', 18: 'E', 24: 'E', 25: 'E',
                          3: 'C', 7: 'C', 8: 'C', 9: 'C', 17: 'C', 19: 'C', 21: 'C', 23: 'C',
                          10: 'W', 11: 'W', 12: 'W', 13: 'W', 14: 'W', 15: 'W', 22: 'W', 26: 'W', 27: 'W', 28: 'W'}
    if 'province' in df.columns:
        prov = pd.to_numeric(df['province'], errors='coerce')
        df['_region'] = prov.map(PROVINCE_TO_REGION).fillna('E')
    else:
        df['_region'] = 'E'
    if 'wave' not in df.columns:
        df['wave'] = 1
    max_wave = df['wave'].max()
    geo = df[df['_region'] == 'W'].copy()
    if '_region' in geo.columns:
        geo = geo.drop(columns=['_region'])
    temp = df[df['wave'] == max_wave].copy()
    if '_region' in temp.columns:
        temp = temp.drop(columns=['_region'])
    deriv = df[(df['_region'].isin(['E', 'C'])) | (df['wave'] < max_wave)].copy()
    if '_region' in deriv.columns:
        deriv = deriv.drop(columns=['_region'])
    if len(deriv) == 0:
        deriv = df.copy()
        if '_region' in deriv.columns:
            deriv = deriv.drop(columns=['_region'])
    return {
        COHORT_DERIVATION: deriv,
        COHORT_GEOGRAPHIC: geo,
        COHORT_TEMPORAL: temp,
    }


# =============================================================================
# 7b. 宽表纵向插补（利用同人历史波次填补当前波次缺失）
# =============================================================================
def build_wide_format(df_long, vars_to_pivot, id_col='ID', wave_col='wave', aux_cols=None):
    """
    将长表转为宽表：每人一行，列为 var_w1, var_w2, var_w3 等。
    若某人某波次无记录，则为 NaN。aux_cols（如 baseline_group）取每人首行值。
    """
    if id_col not in df_long.columns or wave_col not in df_long.columns:
        return None, None
    vars_avail = [c for c in vars_to_pivot if c in df_long.columns]
    if not vars_avail:
        return None, None

    wide_list = []
    for pid, grp in df_long.groupby(id_col):
        row = {id_col: pid}
        for c in (aux_cols or []):
            if c in grp.columns:
                row[c] = grp[c].iloc[0]
        for _, r in grp.iterrows():
            w = int(r[wave_col]) if pd.notna(r[wave_col]) else 0
            for v in vars_avail:
                row[f'{v}_w{w}'] = r[v]
        wide_list.append(row)
    df_wide = pd.DataFrame(wide_list)
    return df_wide, vars_avail


def _prefill_all_nan_wide_cols(df_wide, wide_cols):
    """
    插补前对全 NaN 列填占位值，避免 sklearn 丢弃导致列错位（根因修复）。
    用同变量其他 wave 的中位数占位，如 bmi_w4 全空则用 bmi_w1/w2/w3 的中位数。
    """
    out = df_wide.copy()
    filled = []
    for c in wide_cols:
        if out[c].isna().all():
            base = c.rsplit('_w', 1)[0] if '_w' in c else c
            related = [col for col in wide_cols if col.startswith(base + '_w') and col != c and out[col].notna().any()]
            if related:
                med = out[related].stack().median()
            else:
                med = IMPUTATION_BOUNDS.get(base, (0, 100))[0] if base in IMPUTATION_BOUNDS else 0
            out[c] = out[c].fillna(med if not pd.isna(med) else 0)
            filled.append(c)
    if filled:
        logger.info(f"  宽表全 NaN 列占位（防列错位）: {len(filled)} 列")
    return out


def impute_wide_and_map_back(df_long, cols_imp, impute_fn, aux_cols=None, id_col='ID', wave_col='wave', seed=None):
    """
    宽表纵向插补：构建宽表 → 插补（同人 var_w1, var_w2, var_w3 可互相借用）→ 映射回长表。
    seed: 用于多重插补时产生不同结果。
    """
    df_wide, vars_pivot = build_wide_format(df_long, cols_imp, id_col, wave_col, aux_cols)
    if df_wide is None or len(df_wide) < 50:
        logger.warning("宽表构建失败或样本不足，回退至普通插补")
        return None

    wide_cols = [c for c in df_wide.columns if c != id_col and not (aux_cols and c in aux_cols) and '_w' in str(c)]
    if not wide_cols:
        return None

    # 根因修复：全 NaN 列占位，避免 sklearn 丢弃导致列错位
    df_wide = _prefill_all_nan_wide_cols(df_wide, wide_cols)

    # 宽表列数多时，仅用 baseline_group 等关键 aux，避免 cols_use 过长导致索引越界
    aux_for_wide = [c for c in (aux_cols or []) if c in df_wide.columns and c in ['baseline_group', 'province', 'wave']][:3]
    df_wide_imp = impute_fn(df_wide, wide_cols, aux_cols=aux_for_wide if aux_for_wide else None, seed=seed)
    wide_lookup = df_wide_imp.set_index(id_col)
    df_long_out = df_long.copy()
    for idx, r in df_long_out.iterrows():
        pid = r[id_col]
        w = r[wave_col]
        if pid not in wide_lookup.index or pd.isna(w):
            continue
        w = int(w)
        for v in vars_pivot:
            c_wide = f'{v}_w{w}'
            if c_wide in wide_lookup.columns:
                val = wide_lookup.loc[pid, c_wide]
                if pd.notna(val):
                    # 目标列为 int 时需取整，避免 LossySetitemError
                    if v in df_long_out.columns and np.issubdtype(df_long_out[v].dtype, np.integer):
                        val = int(round(float(val)))
                    df_long_out.loc[idx, v] = val
    return df_long_out


# =============================================================================
# 8. 溯源保存：每步预处理后保存 CSV
# =============================================================================
def _save_trace(df, step_name, output_dir, desc=''):
    """保存当前数据状态，便于溯源。"""
    trace_dir = os.path.join(output_dir, 'pipeline_trace')
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, f'{step_name}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    logger.info(f"  [溯源] {step_name}.csv 已保存 (n={len(df)}) {desc}")
    return path


# =============================================================================
# 9. 主流程：整体插补 → 再划分队列，每步保存 CSV 溯源
# =============================================================================
def run_full_experiment(data_path=None, output_dir=OUTPUT_DIR):
    """
    整体插补流程：先对全量数据插补，再按 baseline_group 划分 A/B/C 队列。
    每步数据预处理均保存 CSV 至 pipeline_trace/，便于溯源。
    """
    os.makedirs(output_dir, exist_ok=True)
    trace_dir = os.path.join(output_dir, 'pipeline_trace')
    os.makedirs(trace_dir, exist_ok=True)
    logger.info("=" * 60)
    logger.info(">>> 启动 npj 风格缺失值插补实验（整体插补 → 划分队列，每步溯源）")
    logger.info("=" * 60)

    # ---------- Step 0: 加载数据 ----------
    df = load_data(path=data_path)
    if df is None or len(df) < 100:
        logger.error("数据加载失败或样本不足")
        return None
    _save_trace(df, 'step0_loaded', output_dir, desc='加载后原始/预处理数据')

    if 'baseline_group' not in df.columns:
        logger.error("数据缺少 baseline_group 列")
        return None

    # ---------- Step 0b: 个体历史对齐（不随时间变化变量 ffill/bfill） ----------
    cols_time_inv = [c for c in VARS_TIME_INVARIANT if c in df.columns]
    if cols_time_inv and 'ID' in df.columns and 'wave' in df.columns:
        n_before = df[cols_time_inv].isna().sum().sum()
        df = align_individual_history(df, cols_time_inv)
        n_after = df[cols_time_inv].isna().sum().sum()
        filled = n_before - n_after
        logger.info(f"  Step 0b 个体历史对齐: {cols_time_inv}，填补 {filled} 个缺失")
        if filled > 0:
            _save_trace(df, 'step0b_aligned', output_dir, desc='不随时间变化变量 ffill/bfill 后')

    # ---------- Step 0c: 辅助变量数值化（province/wave 参与插补须为数值型） ----------
    df = ensure_aux_cols_numeric(df, cols=['province', 'wave'])

    # 按 Table 1 类型划分：连续变量（均值±SD）vs 分类变量（n%）
    vars_in_data = [c for c in ANALYSIS_VARS if c in df.columns]
    vars_num = [c for c in vars_in_data if df[c].dtype in [np.float64, np.int64]]
    cols_continuous = [c for c in vars_num if c in VARS_CONTINUOUS]
    cols_categorical = [c for c in vars_num if c in VARS_CATEGORICAL]
    cols_imp = cols_continuous + cols_categorical
    logger.info(f"  连续变量 {len(cols_continuous)} 个: {cols_continuous[:5]}...")
    logger.info(f"  分类变量 {len(cols_categorical)} 个: {cols_categorical[:5]}...")

    # ---------- Step 1: 缺失筛查（不改变数据） ----------
    screen_missing(df, vars_to_check=vars_num + [c for c in VARS_OUTCOME if c in df.columns], output_dir=output_dir)

    # ---------- Step 2: MCAR 验证 ----------
    moderate = []
    if os.path.exists(os.path.join(output_dir, 'table1_missing_distribution.csv')):
        tbl1 = pd.read_csv(os.path.join(output_dir, 'table1_missing_distribution.csv'), encoding='utf-8-sig')
        moderate = tbl1[tbl1['Tier'] == 'moderate']['Variable'].tolist()
    test_mcar(df, moderate_vars=moderate, output_dir=output_dir)
    littles_p = test_littles_mcar(df, cols=None, output_dir=output_dir)
    _write_methodology_justification(output_dir, littles_p)

    # ---------- Step 3: NRMSE 评估（仅连续变量，分类变量用 Mode） ----------
    cols_eval = [c for c in cols_continuous if df[c].notna().sum() >= 50]
    aux_cols = []
    for c in AUX_COLS_EXTRA:
        if c in df.columns and df[c].notna().all() and df[c].dtype in [np.float64, np.int64]:
            aux_cols.append(c)
    aux_cols = aux_cols if aux_cols else None
    if aux_cols:
        logger.info(f"  辅助变量（队列+轨迹）: {aux_cols}")
    logger.info(f"  Step 3 NRMSE 评估中（6 种方法×3 次重复，MissForest/KNN 较慢，请耐心等待）...")
    tbl3 = evaluate_imputation_nrmse(df, cols_eval, aux_cols=aux_cols, pct_missing=SIMULATE_MISSING_PCT, n_repeat=3)
    if len(tbl3) > 0:
        tbl3.to_csv(os.path.join(output_dir, 'table3_imputation_performance.csv'), index=False, encoding='utf-8-sig')
        best_idx = tbl3['NRMSE_mean'].idxmin()
        best_method = tbl3.loc[best_idx, 'Method']
        best_nrmse = tbl3.loc[best_idx, 'NRMSE_mean']
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Method', y='NRMSE_mean', data=tbl3, palette='Set2')
        plt.xticks(rotation=20)
        plt.ylabel('NRMSE (mean)')
        plt.title('Imputation Method Performance: NRMSE Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig3_nrmse_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        best_method = 'Median'
        best_nrmse = np.nan

    # ---------- Step 4: 全量插补（连续→最优方法+队列特征，分类→Ordinal+Binary+Mode） ----------
    # df 已含 Step 0b 个体历史对齐结果
    logger.info("  Step 4 开始全量插补（连续变量可能较慢，请耐心等待）...")
    df_imputed = df.copy()
    if cols_continuous:
        impute_fn = IMPUTATION_METHODS.get(best_method, impute_median)
        logger.info(f"  Step 4 全量插补中：连续变量 ({best_method})，n={len(df)} 行，请耐心等待...")
        if best_method == 'MissForest':
            logger.info("    MissForest 较慢（每列拟合随机森林），约需数分钟")
        if USE_WIDE_IMPUTATION and 'ID' in df.columns and 'wave' in df.columns:
            logger.info("  使用宽表纵向插补（利用同人历史波次）")
            df_wide_imp = impute_wide_and_map_back(df, cols_continuous, impute_fn, aux_cols=aux_cols, seed=None)
            if df_wide_imp is not None:
                df_imputed = df_wide_imp.copy()
                # 宽表可能因索引越界跳过部分列，对仍含缺失的连续列用长表补插一次，便于敏感性对比
                still_missing = [c for c in cols_continuous if c in df_imputed.columns and df_imputed[c].isna().any()]
                if still_missing:
                    logger.warning(f"  宽表回填后仍有 {len(still_missing)} 列含缺失，长表补插: {still_missing[:5]}{'...' if len(still_missing) > 5 else ''}")
                    df_imputed = impute_fn(df_imputed, still_missing, aux_cols=aux_cols)
            else:
                df_imputed = impute_fn(df_imputed, cols_continuous, aux_cols=aux_cols)
        else:
            df_imputed = impute_fn(df_imputed, cols_continuous, aux_cols=aux_cols)
        logger.info("    连续变量插补完成")
    if cols_categorical:
        cols_ordinal = [c for c in cols_categorical if c in VARS_ORDINAL]
        cols_binary = [c for c in cols_categorical if c in VARS_BINARY and c not in VARS_ORDINAL]
        cols_nominal = [c for c in cols_categorical if c not in VARS_ORDINAL and c not in VARS_BINARY]
        if cols_ordinal:
            df_imputed = impute_ordinal(df_imputed, cols_ordinal, aux_cols=aux_cols)
        if cols_binary:
            df_imputed = impute_binary_rf(df_imputed, cols_binary, aux_cols=aux_cols)
        if cols_nominal:
            df_imputed = impute_mode(df_imputed, cols_nominal)
        logger.info("    分类变量插补完成")
    n_mi = N_MULTIPLE_IMPUTATIONS
    if n_mi > 0:
        logger.info(f"  Step 4: 多重插补中（共 {n_mi} 份，每份约 2–5 分钟，请耐心等待）...")
        mi_seed_base = RANDOM_SEED
        for m in range(1, n_mi + 1):
            logger.info(f"    Step 4: 多重插补 第 {m}/{n_mi} 份...")
            seed_m = mi_seed_base + m
            df_m = df.copy()
            if cols_continuous:
                if USE_WIDE_IMPUTATION and 'ID' in df.columns and 'wave' in df.columns:
                    df_wide_m = impute_wide_and_map_back(df, cols_continuous, impute_fn, aux_cols=aux_cols, seed=seed_m)
                    if df_wide_m is not None:
                        df_m = df_wide_m.copy()
                        still_missing_m = [c for c in cols_continuous if c in df_m.columns and df_m[c].isna().any()]
                        if still_missing_m:
                            df_m = impute_fn(df_m, still_missing_m, aux_cols=aux_cols, seed=seed_m)
                    else:
                        df_m = impute_fn(df_m, cols_continuous, aux_cols=aux_cols, seed=seed_m)
                else:
                    df_m = impute_fn(df_m, cols_continuous, aux_cols=aux_cols, seed=seed_m)
            if cols_categorical:
                cols_ordinal = [c for c in cols_categorical if c in VARS_ORDINAL]
                cols_binary = [c for c in cols_categorical if c in VARS_BINARY and c not in VARS_ORDINAL]
                cols_nominal = [c for c in cols_categorical if c not in VARS_ORDINAL and c not in VARS_BINARY]
                if cols_ordinal:
                    df_m = impute_ordinal(df_m, cols_ordinal, aux_cols=aux_cols, seed=seed_m)
                if cols_binary:
                    df_m = impute_binary_rf(df_m, cols_binary, aux_cols=aux_cols, seed=seed_m)
                if cols_nominal:
                    df_m = impute_mode(df_m, cols_nominal, seed=seed_m)
            df_m = post_imputation_cleanup(df_m)
            _save_trace(df_m, f'step1_imputed_m{m}', output_dir, desc=f'MI 第{m}份 (seed={seed_m})')
            logger.info(f"    多重插补 第{m}/{n_mi} 份已完成")
        logger.info(f"  多重插补: 已生成 {n_mi} 份数据集")
    df_imputed = post_imputation_cleanup(df_imputed)
    _save_trace(df_imputed, 'step1_imputed_full', output_dir,
                desc=f'连续={best_method}, 分类=Ordinal+Binary+Mode+逻辑校准')
    _append_methodology_method_selection(output_dir, best_method, best_nrmse)

    # ---------- Step 4b: 核心干预变量完整病例子集（论文附录 S2 敏感性分析用） ----------
    core_vars = [c for c in CORE_INTERVENTION_VARS if c in df_imputed.columns]
    if core_vars:
        df_complete_core = df_imputed.dropna(subset=core_vars)
        if len(df_complete_core) >= 30:
            _save_trace(df_complete_core, 'step1b_complete_case_core', output_dir,
                        desc=f'核心变量{core_vars}无缺失子集 n={len(df_complete_core)}')

    # ---------- Step 4c: 分类变量插补验证（Kappa/准确率/F1，按类型用实际插补策略） ----------
    cols_cat_eval = [c for c in cols_categorical if df[c].notna().sum() >= 50]
    if cols_cat_eval:
        tbl_cat = evaluate_categorical_imputation(df, cols_cat_eval, aux_cols=aux_cols, pct_missing=0.08, n_repeat=3)
        if len(tbl_cat) > 0:
            tbl_cat.to_csv(os.path.join(output_dir, 'table3b_categorical_imputation_validation.csv'), index=False, encoding='utf-8-sig')
            logger.info(f"  分类变量插补验证已保存: table3b_categorical_imputation_validation.csv")

    # ---------- Step 5: 敏感性验证 ----------
    tbl4 = sensitivity_validation(df, df_imputed, cols_imp, output_dir=output_dir)

    # ---------- Step 5a: KDE 密度分布对比图（保存至 pipeline_trace） ----------
    plot_kde_before_after(df, df_imputed, cols_imp, output_dir=output_dir)

    # ---------- Step 5b: 插补收敛图（IterativeImputer 均值/标准差随迭代趋于稳定） ----------
    cols_conv = [c for c in cols_continuous if c in df.columns and df[c].isna().any()]
    if cols_conv:
        plot_imputation_convergence(df, cols_conv, aux_cols=aux_cols, max_iter=10, output_dir=output_dir)

    # ---------- Step 6: 按队列划分并保存 ----------
    COHORT_SPEC = [(0, 'A', 'Healthy'), (1, 'B', 'Depression_only'), (2, 'C', 'Cognition_impaired_only')]
    PAPER_N = {0: 8828, 1: 3123, 2: 2435}
    for group_val, label, name in COHORT_SPEC:
        df_cohort = df_imputed[df_imputed['baseline_group'] == group_val].copy()
        if len(df_cohort) >= 30:
            _save_trace(df_cohort, f'step2_cohort_{label}_{name}', output_dir, desc=f'n={len(df_cohort)}')
            n_paper = PAPER_N.get(group_val)
            if n_paper and abs(len(df_cohort) - n_paper) > 100:
                logger.warning(f"队列{label} 样本量 {len(df_cohort)} 与论文预期 {n_paper} 偏差>100，请检查数据")

    # ---------- 溯源说明文件 ----------
    trace_readme = os.path.join(trace_dir, 'README_溯源说明.txt')
    with open(trace_readme, 'w', encoding='utf-8') as f:
        f.write("插补实验数据溯源说明\n")
        f.write("=" * 50 + "\n")
        f.write("step0_loaded.csv         - 加载后的原始/预处理数据（未插补）\n")
        f.write("step0b_aligned.csv       - 个体历史对齐后（gender/edu/rural 等 ffill/bfill）\n")
        f.write("step1_imputed_full.csv   - 全量插补后数据（含队列特征+纵向影响）\n")
        f.write("step1b_complete_case_core.csv - 核心干预变量(exercise/drinkev)无缺失子集(附录S2)\n")
        f.write("step2_cohort_A_*.csv     - 队列A（健康组）划分后\n")
        f.write("step2_cohort_B_*.csv     - 队列B（仅抑郁组）划分后\n")
        f.write("step2_cohort_C_*.csv     - 队列C（仅认知受损组）划分后\n")
        f.write("\n变量身份与插补策略：\n")
        f.write("  结局/定义变量(is_comorbidity_next等) → 不插补，缺失剔除\n")
        f.write("  协变量(age/bmi/income/exercise等) → 插补对象\n")
        f.write("\n变量类型（按 Table 1）：\n")
        f.write("  连续（均值±SD）→ NRMSE 选优+物理边界+辅助变量(province/wave/age)\n")
        f.write("  有序分类(edu/srh/satlife) → 贝叶斯回归+取整\n")
        f.write("  二分类(exercise/drinkev等) → RF 预测\n")
        f.write("  名义分类 → Mode 众数插补\n")
        f.write("\n流程：加载 → 个体历史对齐 → 全量插补(队列+轨迹+边界) → 划分队列\n")
        f.write("多重插补：N_MULTIPLE_IMPUTATIONS>0 时生成 step1_imputed_m1..mN.csv\n")
    logger.info(f"  溯源说明已保存: pipeline_trace/README_溯源说明.txt")

    # ---------- 打印关键结果 ----------
    logger.info("\n" + "=" * 60)
    logger.info("【实验关键结果】")
    logger.info("=" * 60)
    logger.info(f"连续变量最优方法: {best_method} (NRMSE={best_nrmse})")
    logger.info(f"分类变量: 有序=贝叶斯+取整, 二分类=RF, 名义=Mode")
    logger.info(f"全量插补 n={len(df_imputed)}，划分后 A/B/C 见 pipeline_trace/")
    logger.info(f"输出目录: {os.path.abspath(output_dir)}")
    logger.info("=" * 60)

    return {
        'best_method': best_method,
        'best_nrmse': best_nrmse,
        'df_loaded': df,
        'df_imputed': df_imputed,
        'table3': tbl3,
    }


# =============================================================================
# 入口
# =============================================================================
if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    out_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    result = run_full_experiment(data_path=data_path, output_dir=out_dir)
    if result:
        print(f"\n最优插补方法: {result.get('best_method', 'N/A')}")
        print(f"NRMSE: {result.get('best_nrmse', 'N/A')}")
        print(f"溯源文件见: {out_dir}/pipeline_trace/")
