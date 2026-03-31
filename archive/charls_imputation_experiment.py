# -*- coding: utf-8 -*-
"""
老年抑郁-认知共病因果机器学习研究 — 缺失值插补实验

实验设计：
- 分组：实验组（分层针对性插补）+ 3个对照组（统一MICE、统一简单插补、完整病例）
- 变量分层：Tier0不可插补，Tier1/2/3按缺失率与类型分层
- 验证：数据层面（分布、相关性、异常值）+ 分析层面（AUC、ATE）

依赖：pandas, numpy, scikit-learn, scipy
可选：fancyimpute（若需 SoftImpute 等高级方法）
"""
import os
import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_SEED = 500

# =============================================================================
# 1. 变量分层配置（与 CHARLS 列名对应）
# =============================================================================

# Tier0：不可插补，缺失则剔除
TIER0 = ['is_comorbidity_next', 'is_depression', 'is_cognitive_impairment',
         'is_comorbidity']  # cesd10/total_cognition 在预处理中已剔除

# Tier1：核心暴露，缺失率 3%-20%，MAR
TIER1 = ['exercise', 'sleep', 'smokev', 'drinkev', 'bmi']
# 衍生：sleep_adequate(≥6h)、bmi_normal(18.5-24)

# Tier2：关键协变量，缺失率 5%-15%
TIER2 = ['age', 'systo', 'edu', 'lgrip']

# Tier3：普通协变量，缺失率 <5%
TIER3 = ['pulse', 'family_size', 'marry']

# 所有可插补变量（与 CHARLS 列名对应，缺失则自动跳过）
IMPUTABLE_COLS = list(dict.fromkeys(TIER1 + TIER2 + TIER3))

# MICE 参数
MICE_MAX_ITER = 5
# 随机森林插补参数（stratified_impute_rf）
RF_N_ESTIMATORS = 100
RF_TOP_K_AUX = 5


# =============================================================================
# 2. 分层插补函数
# =============================================================================

def stratified_impute_mice(df_sub, cols=None, max_iter=5, random_state=RANDOM_SEED):
    """
    按队列分层 MICE 插补。
    连续变量用线性回归，分类变量用逻辑回归（迭代器）。
    输入：单队列 df_sub，cols 指定要插补的列（默认 IMPUTABLE_COLS）
    输出：插补后 df
    """
    df = df_sub.copy()
    cols = cols or IMPUTABLE_COLS
    cols_to_imp = [c for c in cols if c in df.columns and df[c].isna().any()]
    if not cols_to_imp:
        return df

    X_num = df[cols_to_imp].select_dtypes(include=[np.number])
    if X_num.empty or X_num.shape[1] == 0:
        return df

    # 连续型用 LinearRegression，sklearn IterativeImputer 默认
    imp = IterativeImputer(
        estimator=LinearRegression(),
        max_iter=max_iter,
        random_state=random_state,
        sample_posterior=False
    )
    filled = imp.fit_transform(X_num)
    df[X_num.columns] = filled
    return df


def stratified_impute_rf(df_sub, n_estimators=100, top_k_aux=5, random_state=RANDOM_SEED):
    """
    随机森林插补：对每个缺失变量，纳入相关性前 top_k 的辅助变量。
    输入：单队列 df_sub
    输出：插补后 df
    """
    df = df_sub.copy()
    cols_to_imp = [c for c in IMPUTABLE_COLS if c in df.columns and df[c].isna().any()]
    if not cols_to_imp:
        return df

    aux = [c for c in IMPUTABLE_COLS if c in df.columns and c not in cols_to_imp]
    aux = [c for c in aux if df[c].notna().all()]  # 辅助变量需无缺失

    for col in cols_to_imp:
        if df[col].notna().all():
            continue
        if col not in df.columns:
            continue
        # 计算与其余变量的相关性，选 top_k
        cand = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != col and c not in TIER0 and df[c].notna().all()]
        if len(cand) < 2:
            continue
        corr = df[cand].corrwith(df[col].fillna(df[col].median()))
        corr = corr.abs().sort_values(ascending=False)
        use_cols = corr.head(top_k_aux).index.tolist()

        mask = df[col].isna()
        if mask.sum() == 0:
            continue
        X_train = df.loc[~mask, use_cols]
        y_train = df.loc[~mask, col]
        X_miss = df.loc[mask, use_cols]

        if X_train.isna().any().any() or X_miss.isna().any().any():
            continue
        # 数值型用 RF 回归
        if df[col].dtype in [np.float64, np.int64] and df[col].nunique() > 5:
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
            model.fit(X_train, y_train)
            df.loc[mask, col] = model.predict(X_miss)
        else:
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            model.fit(X_train, y_train.astype(int))
            df.loc[mask, col] = model.predict(X_miss)
    return df


def run_experimental_stratified(df_full, cohort_col='baseline_group'):
    """
    实验组：分层针对性插补。
    Tier1（核心暴露）用 MICE；Tier2/3 用中位数插补。
    """
    out_dfs = []
    for g in [0, 1, 2]:
        df_sub = df_full[df_full[cohort_col] == g].copy()
        if len(df_sub) < 50:
            continue
        # Tier1 用 MICE（仅对 Tier1 列）
        tier1_cols = [c for c in TIER1 if c in df_sub.columns]
        if tier1_cols:
            df_sub = stratified_impute_mice(df_sub, cols=tier1_cols, max_iter=5)
        # Tier2/3 用中位数
        tier23 = [c for c in TIER2 + TIER3 if c in df_sub.columns]
        if tier23:
            imp = SimpleImputer(strategy='median')
            df_sub[tier23] = imp.fit_transform(df_sub[tier23])
        out_dfs.append(df_sub)
    return pd.concat(out_dfs, ignore_index=True) if out_dfs else df_full.copy()


def run_unified_mice(df_full):
    """对照组1：统一 MICE，不按队列分层"""
    cols = [c for c in IMPUTABLE_COLS if c in df_full.columns and df_full[c].isna().any()]
    if not cols:
        return df_full.copy()
    df = df_full.copy()
    X = df[cols].select_dtypes(include=[np.number])
    if X.empty:
        return df
    imp = IterativeImputer(estimator=LinearRegression(), max_iter=5, random_state=RANDOM_SEED)
    df[X.columns] = imp.fit_transform(X)
    return df


def run_unified_simple(df_full):
    """对照组2：统一简单插补（中位数）"""
    df = df_full.copy()
    cols = [c for c in IMPUTABLE_COLS if c in df.columns and df[c].isna().any()]
    if not cols:
        return df
    imp = SimpleImputer(strategy='median')
    df[cols] = imp.fit_transform(df[cols])
    return df


def run_complete_case(df_full):
    """对照组3：完整病例，剔除任何可插补变量缺失的行"""
    cols = [c for c in IMPUTABLE_COLS if c in df_full.columns]
    if not cols:
        return df_full.copy()
    return df_full.dropna(subset=cols).copy()


# =============================================================================
# 3. 插补效果验证
# =============================================================================

def validate_distribution(original, imputed, cols):
    """分布一致性：KS 检验 p 值（仅对非缺失部分比较插补值 vs 原始分布）"""
    results = []
    for c in cols:
        if c not in original.columns or c not in imputed.columns:
            continue
        orig = original[c].dropna()
        imp = imputed[c]
        if len(orig) < 10 or len(imp) < 10:
            continue
        try:
            _, p = stats.ks_2samp(orig, imp)
            results.append({'Variable': c, 'KS_p': p, 'Pass': p > 0.05})
        except Exception:
            results.append({'Variable': c, 'KS_p': np.nan, 'Pass': False})
    return pd.DataFrame(results)


def validate_correlation(original, imputed, cols):
    """相关性保持：插补前后 Pearson 相关矩阵的 Frobenius 范数差异"""
    cols = [c for c in cols if c in original.columns and c in imputed.columns]
    if len(cols) < 2:
        return np.nan
    try:
        corr_orig = original[cols].corr().fillna(0).values
        corr_imp = imputed[cols].corr().fillna(0).values
        diff = np.linalg.norm(corr_orig - corr_imp, 'fro')
        return diff
    except Exception:
        return np.nan


def validate_outliers(imputed, cols):
    """异常值检测：IQR 法超出 [Q1-1.5*IQR, Q3+1.5*IQR] 的比例"""
    results = []
    for c in cols:
        if c not in imputed.columns:
            continue
        s = imputed[c].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out_pct = ((s < low) | (s > high)).mean() * 100
        results.append({'Variable': c, 'Outlier_pct': out_pct})
    return pd.DataFrame(results) if results else pd.DataFrame()


def compute_auc(df, target_col='is_comorbidity_next', n_splits=5):
    """分析层面：5 折交叉验证预测 AUC"""
    exclude = TIER0 + [target_col]
    X_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    X = df[X_cols].copy()
    y = df[target_col].astype(int)
    if X.shape[1] < 3 or len(X) < 50:
        return np.nan
    X = X.fillna(X.median())
    aucs = []
    for seed in range(n_splits):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED + seed)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)
        rf.fit(X_tr, y_tr)
        aucs.append(roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1]))
    return np.mean(aucs)


def compute_ate_simple(df, treatment_col='exercise', target_col='is_comorbidity_next'):
    """
    分析层面：简化 ATE 估计（回归法，供插补效果对比）。
    完整因果推断建议调用 charls_recalculate_causal_impact。
    """
    if treatment_col not in df.columns or target_col not in df.columns:
        return np.nan
    from sklearn.linear_model import LinearRegression
    exclude = TIER0 + [target_col, treatment_col]
    W = [c for c in df.columns if c not in exclude and c in df.columns and df[c].dtype in [np.float64, np.int64]]
    use_cols = [treatment_col] + W[:10]  # 限制协变量避免过拟合
    if treatment_col not in df.columns:
        return np.nan
    X = df[use_cols].copy().fillna(df[use_cols].median())
    y = df[target_col].astype(float)
    mask = X.notna().all(axis=1) & y.notna()
    if mask.sum() < 50:
        return np.nan
    model = LinearRegression().fit(X[mask], y[mask])
    return model.coef_[0]  # ATE 近似


# =============================================================================
# 4. 主实验流程
# =============================================================================

def run_imputation_experiment(df_full, output_dir='imputation_experiment_results'):
    """
    运行完整插补实验，输出 4 组验证指标汇总表。
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(">>> 启动缺失值插补实验...")

    # 保留有完整 Tier0 的样本（结局等不可插补变量非缺失）
    tier0_in_data = [c for c in TIER0 if c in df_full.columns]
    df = df_full.dropna(subset=tier0_in_data).copy() if tier0_in_data else df_full.copy()
    n_orig = len(df)
    logger.info(f"基线样本量: {n_orig}")

    # 4 组插补
    groups = {
        'Experimental': run_experimental_stratified(df),
        'Unified_MICE': run_unified_mice(df),
        'Unified_Simple': run_unified_simple(df),
        'Complete_Case': run_complete_case(df),
    }

    # 验证指标汇总
    summary_rows = []
    cols_valid = [c for c in IMPUTABLE_COLS if c in df.columns]

    for group_name, df_imp in groups.items():
        n = len(df_imp)
        # 分布一致性
        dist_df = validate_distribution(df, df_imp, cols_valid)
        ks_pass_mean = dist_df['Pass'].mean() if len(dist_df) > 0 else np.nan
        # 相关性
        corr_diff = validate_correlation(df, df_imp, cols_valid)
        # 异常值
        out_df = validate_outliers(df_imp, cols_valid)
        out_mean = out_df['Outlier_pct'].mean() if len(out_df) > 0 else np.nan
        # AUC
        auc = compute_auc(df_imp, target_col='is_comorbidity_next')
        # ATE（简化回归法，供插补效果对比）
        ate = compute_ate_simple(df_imp, treatment_col='exercise', target_col='is_comorbidity_next')

        summary_rows.append({
            'Group': group_name,
            'N': n,
            'KS_pass_rate': round(ks_pass_mean, 3) if not np.isnan(ks_pass_mean) else np.nan,
            'Corr_diff_Frobenius': round(corr_diff, 4) if not np.isnan(corr_diff) else np.nan,
            'Outlier_pct_mean': round(out_mean, 2) if not np.isnan(out_mean) else np.nan,
            'AUC_mean': round(auc, 4) if not np.isnan(auc) else np.nan,
            'ATE_exercise': round(ate, 4) if not np.isnan(ate) else np.nan,
        })

        # 保存各组插补后数据（可选）
        if output_dir:
            out_path = os.path.join(output_dir, f'imputed_{group_name}.csv')
            df_imp.to_csv(out_path, index=False, encoding='utf-8-sig')
            logger.info(f"  已保存: {out_path}")

    summary_df = pd.DataFrame(summary_rows)

    # 保存汇总表
    summary_path = os.path.join(output_dir, 'imputation_experiment_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    logger.info(f"✅ 插补实验汇总表已保存: {summary_path}")

    return summary_df, groups


# =============================================================================
# 5. 指标汇总表模板（供论文补充方法使用）
# =============================================================================

SUMMARY_TEMPLATE = """
| 分组 | N | 分布一致性(KS通过率) | 相关性差异(Frobenius) | 异常值比例(%) | AUC | ATE(运动) |
|------|---|----------------------|----------------------|---------------|-----|-----------|
| 实验组(分层) |  |  |  |  |  |  |
| 对照组1(统一MICE) |  |  |  |  |  |  |
| 对照组2(统一简单) |  |  |  |  |  |  |
| 对照组3(完整病例) |  |  |  |  |  |  |
"""


# =============================================================================
# 6. 入口
# =============================================================================

if __name__ == '__main__':
    from charls_complete_preprocessing import preprocess_charls_data
    from run_multi_exposure_causal import prepare_exposures

    # 加载预处理数据
    df = preprocess_charls_data('CHARLS.csv', age_min=60, write_output=False)
    if df is None:
        logger.error("预处理失败，退出")
        exit(1)
    prepare_exposures(df)

    summary_df, groups = run_imputation_experiment(df, output_dir='imputation_experiment_results')
    print("\n" + "=" * 60)
    print("插补实验汇总表")
    print("=" * 60)
    print(summary_df.to_string(index=False))
