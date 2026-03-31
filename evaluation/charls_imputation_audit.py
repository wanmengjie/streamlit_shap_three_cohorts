import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def run_imputation_sensitivity(output_dir='evaluation_results/missing_audit', data_path=None):
    """插补方法对比 (Figure S3)。注意：使用 cesd10 目标，与主流程 is_comorbidity_next 不同；主流程使用 run_imputation_sensitivity_preprocessed。"""
    os.makedirs(output_dir, exist_ok=True)
    if data_path is None:
        try:
            from config import RAW_DATA_PATH
            data_path = RAW_DATA_PATH
        except ImportError:
            data_path = 'CHARLS.csv'
    df = pd.read_csv(data_path)
    
    # 模拟插补灵敏度：Mean vs Median vs Mode 对建模结局的影响
    target = 'cesd10'
    if target not in df.columns:
        logger.warning(f"缺失目标变量 {target}，跳过插补审计")
        return None
        
    df = df.dropna(subset=[target])
    y = (df[target] >= 10).astype(int)
    X = df[['age', 'gender', 'edu']].copy()
    
    methods = ['mean', 'median', 'most_frequent']
    auc_results = []
    
    for method in methods:
        imputer = SimpleImputer(strategy=method)
        X_imp = imputer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=500)
        
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_prob = lr.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        auc_results.append({'Method': method, 'AUC': auc})
        
    auc_df = pd.DataFrame(auc_results)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Method', y='AUC', data=auc_df, hue='Method', palette='Set2', legend=False)
    plt.ylim(0.5, 0.7)
    plt.title('Imputation Sensitivity Analysis (Figure S3)')
    plt.savefig(os.path.join(output_dir, 'figS3_imputation_sensitivity.png'), dpi=300)
    plt.close()
    logger.info(f"✅ 插补敏感性分析已完成：{output_dir}")
    return auc_df


def _run_single_imputation_auc(X_train, X_test, y_train, y_test, imp, method_label, random_state=500):
    """单次插补+建模，返回 AUC；若仅单类或计算失败则返回 np.nan。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    try:
        X_tr = imp.fit_transform(X_train)
        X_te = imp.transform(X_test)
        if len(np.unique(y_test)) < 2:
            return np.nan
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)
        rf.fit(X_tr, y_train)
        return roc_auc_score(y_test, rf.predict_proba(X_te)[:, 1])
    except Exception:
        return np.nan


def run_imputation_sensitivity_preprocessed(df, output_dir, target_col='is_comorbidity_next', n_bootstrap=50, n_jobs_bootstrap=-1):
    """
    基于预处理数据（主流程目标 is_comorbidity_next）的插补敏感性分析。
    比较 Median / Mean / Mode / KNN / MICE 五种方法对预测 AUC 的影响。
    采用 Bootstrap 重采样计算 95% CI，输出更详细的图表。
    n_jobs_bootstrap: Bootstrap 循环并行数，-1 用全部核心，1 为串行；由 config.PARALLEL_IMPUTATION_BOOTSTRAP 控制
    """
    import matplotlib
    matplotlib.use('Agg')
    os.makedirs(output_dir, exist_ok=True)
    from sklearn.model_selection import GroupShuffleSplit
    from utils.charls_feature_lists import get_exclude_cols

    if target_col not in df.columns:
        logger.warning(f"目标列 {target_col} 不存在，跳过插补敏感性。")
        return None

    # 剔除目标缺失行，避免 y 含 NaN 导致建模/指标异常
    df = df.dropna(subset=[target_col]).copy()
    if 'ID' not in df.columns:
        logger.warning("插补敏感性：无 ID 列，无法按个体分组划分，回退至随机划分。")
    exclude = get_exclude_cols(df, target_col=target_col)
    X_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    X = df[X_cols].copy()
    y = df[target_col].astype(int)
    if X.shape[1] < 3 or len(X) < 50:
        logger.warning("插补敏感性：特征或样本过少，跳过。")
        return None

    # 按 ID 分组划分，避免同一个体多波次观测泄露到 train/test（Bischl et al. 2012）
    if 'ID' in df.columns and df['ID'].nunique() >= 5:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=500)
        train_idx, test_idx = next(gss.split(X, y, groups=df['ID']))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=500)
    n_train, n_test = len(X_train), len(X_test)
    results = []

    def _add_result(label, method_cn, auc_list):
        # 过滤掉 nan（单类或异常迭代），避免 percentile 报错
        valid = [a for a in auc_list if not (isinstance(a, float) and np.isnan(a))]
        if not valid:
            results.append({'Method': label, 'Method_CN': method_cn, 'AUC': np.nan, 'AUC_lb': np.nan, 'AUC_ub': np.nan})
            return
        auc_arr = np.array(valid)
        mean_auc = np.mean(auc_arr)
        if len(auc_arr) < 10:
            ci_l = ci_u = mean_auc
        else:
            ci_l = np.percentile(auc_arr, 2.5)
            ci_u = np.percentile(auc_arr, 97.5)
        results.append({'Method': label, 'Method_CN': method_cn, 'AUC': mean_auc, 'AUC_lb': ci_l, 'AUC_ub': ci_u})

    def _one_bootstrap_simple(b, method):
        idx = np.random.RandomState(b).choice(len(X_train), size=len(X_train), replace=True)
        X_b, y_b = X_train.iloc[idx], y_train.iloc[idx]
        imp = SimpleImputer(strategy=method)
        return _run_single_imputation_auc(X_b, X_test, y_b, y_test, imp, method, random_state=500 + b)

    def _one_bootstrap_knn(b):
        idx = np.random.RandomState(b).choice(len(X_train), size=len(X_train), replace=True)
        X_b, y_b = X_train.iloc[idx], y_train.iloc[idx]
        from sklearn.impute import KNNImputer
        imp = KNNImputer(n_neighbors=5, weights='distance')
        return _run_single_imputation_auc(X_b, X_test, y_b, y_test, imp, 'KNN', random_state=500 + b)

    def _one_bootstrap_mice(b):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        idx = np.random.RandomState(b).choice(len(X_train), size=len(X_train), replace=True)
        X_b, y_b = X_train.iloc[idx], y_train.iloc[idx]
        imp = IterativeImputer(max_iter=5, random_state=500 + b)
        return _run_single_imputation_auc(X_b, X_test, y_b, y_test, imp, 'MICE', random_state=500 + b)

    try:
        from config import PARALLEL_IMPUTATION_BOOTSTRAP
        _n_jobs = -1 if PARALLEL_IMPUTATION_BOOTSTRAP else 1
    except ImportError:
        _n_jobs = n_jobs_bootstrap if n_jobs_bootstrap is not None else 1

    if _n_jobs != 1:
        from joblib import Parallel, delayed

    # 1. SimpleImputer: Median, Mean, Most frequent (Mode) — Bootstrap（可并行）
    for method, label in [('median', 'Median'), ('mean', 'Mean'), ('most_frequent', 'Mode')]:
        if _n_jobs != 1:
            aucs = Parallel(n_jobs=_n_jobs, backend='loky')(delayed(_one_bootstrap_simple)(b, method) for b in range(n_bootstrap))
        else:
            aucs = [_one_bootstrap_simple(b, method) for b in range(n_bootstrap)]
        _add_result(label, {'Median': 'Median imputation', 'Mean': 'Mean imputation', 'Mode': 'Mode imputation'}[label], aucs)

    # 2. KNN 插补 — Bootstrap（可并行）
    try:
        if _n_jobs != 1:
            aucs = Parallel(n_jobs=_n_jobs, backend='loky')(delayed(_one_bootstrap_knn)(b) for b in range(n_bootstrap))
        else:
            aucs = [_one_bootstrap_knn(b) for b in range(n_bootstrap)]
        _add_result('KNN', 'K-Nearest Neighbors', aucs)
    except Exception as e:
        logger.warning(f"KNN 插补跳过: {e}")

    # 3. MICE — Bootstrap（可并行）
    try:
        if _n_jobs != 1:
            aucs = Parallel(n_jobs=_n_jobs, backend='loky')(delayed(_one_bootstrap_mice)(b) for b in range(n_bootstrap))
        else:
            aucs = [_one_bootstrap_mice(b) for b in range(n_bootstrap)]
        _add_result('MICE', 'Multivariate imputation', aucs)
    except Exception as e:
        logger.warning(f"MICE 插补跳过: {e}")

    res_df = pd.DataFrame(results)
    if len(res_df) == 0:
        logger.warning("插补敏感性：无有效结果，跳过 CSV 与图表。")
        return None
    out_csv = os.path.join(output_dir, 'imputation_sensitivity_results.csv')
    res_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    logger.info(f"✅ 插补敏感性结果已保存: {out_csv}")

    # 绘图：更详细的设计（NaN 用点估计代替 CI 显示）
    fig, ax = plt.subplots(figsize=(12, 6))
    methods_order = res_df['Method'].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods_order)))
    median_idx = methods_order.index('Median') if 'Median' in methods_order else -1
    if median_idx >= 0:
        colors[median_idx] = (0.2, 0.6, 0.4, 1.0)  # 主分析方法高亮

    auc_vals = res_df['AUC'].fillna(0.5).values
    lb_vals = res_df['AUC_lb'].fillna(res_df['AUC']).values
    ub_vals = res_df['AUC_ub'].fillna(res_df['AUC']).values
    x_pos = np.arange(len(res_df))
    bars = ax.bar(x_pos, auc_vals, color=colors, edgecolor='black', linewidth=0.8)
    ax.errorbar(x_pos, auc_vals, yerr=[auc_vals - lb_vals, ub_vals - auc_vals],
                fmt='none', color='black', capsize=5, capthick=1.5)

    # 主分析参考线
    if 'Median' in res_df['Method'].values:
        med_row = res_df.loc[res_df['Method'] == 'Median', 'AUC'].values
        if len(med_row) > 0 and not np.isnan(med_row[0]):
            med_auc = med_row[0]
            ax.axhline(y=med_auc, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Main analysis (Median, AUC={med_auc:.3f})')

    # 柱顶标注 AUC 及 95% CI
    for i, (_, row) in enumerate(res_df.iterrows()):
        a, lb, ub = row['AUC'], row['AUC_lb'], row['AUC_ub']
        lb = lb if not (isinstance(lb, float) and np.isnan(lb)) else a
        ub = ub if not (isinstance(ub, float) and np.isnan(ub)) else a
        txt = f"{a:.4f}\n(95% CI: {lb:.4f}, {ub:.4f})" if not np.isnan(a) else "N/A"
        ax.text(i, (ub if not np.isnan(ub) else a) + 0.02, txt, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{m}\n({c})" for m, c in zip(res_df['Method'], res_df['Method_CN'])], fontsize=10)
    ax.set_ylabel('AUC (95% CI)', fontsize=12)
    ax.set_xlabel('Imputation method', fontsize=12)
    y_min = np.nanmin(np.r_[res_df['AUC_lb'].values, res_df['AUC'].values])
    y_max = np.nanmax(np.r_[res_df['AUC_ub'].values, res_df['AUC'].values])
    ax.set_ylim(max(0.45, y_min - 0.08), min(1.0, y_max + 0.15))
    ax.set_title('Figure S3. Imputation Sensitivity Analysis\n'
                 f'Five methods compared by AUC (Bootstrap n={n_bootstrap}); RF classifier; N_train={n_train}, N_test={n_test}',
                 fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figS3_imputation_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ 插补敏感性（主目标）已完成：{output_dir}")
    return res_df
