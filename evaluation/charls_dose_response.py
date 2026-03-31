# -*- coding: utf-8 -*-
"""
剂量反应关系分析（审稿意见 P2）
对运动、睡眠等可量化干预，采用限制性立方样条（RCS）分析剂量反应关系。
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from utils.charls_feature_lists import get_exclude_cols
from config import BBOX_INCHES

logger = logging.getLogger(__name__)


def _rcs_basis(x, knots):
    """4 节点限制性立方样条基函数（Harrell）。knots = [k1,k2,k3,k4] 百分位。"""
    def _tp(z, k):
        return np.maximum(z - k, 0) ** 3
    k1, k2, k3, k4 = knots
    d43, d42, d32 = k4 - k3, k4 - k2, k3 - k2
    if abs(d32) < 1e-9:
        d32 = 1e-9
    v1 = _tp(x, k1) - _tp(x, k3)
    v2 = _tp(x, k2) - _tp(x, k3)
    f1 = x
    f2 = v1 * (k4 - k1) / (d43 * (k4 - k2)) - v2 * (k4 - k1) / (d43 * d32)
    return np.column_stack([np.ones(len(x)), f1, f2])


def _fit_rcs(df, x_col, y_col, cov_cols, n_knots=4):
    """4 节点 RCS 逻辑回归（连续变量，如 sleep 小时数）"""
    X = df[[x_col]].dropna()
    valid_idx = X.index.intersection(df[y_col].dropna().index)
    for c in cov_cols:
        if c in df.columns:
            valid_idx = valid_idx.intersection(df[c].dropna().index)
    df_sub = df.loc[valid_idx].copy()
    if len(df_sub) < 100:
        return None, None, None

    x_vals = df_sub[x_col].values.astype(float)
    knots = np.percentile(x_vals, [5, 35, 65, 95])
    knots = np.clip(knots, x_vals.min(), x_vals.max())
    X_rcs = _rcs_basis(x_vals, knots)
    cov_mat = df_sub[cov_cols].fillna(df_sub[cov_cols].median()).values if cov_cols else np.zeros((len(df_sub), 0))
    if cov_mat.size > 0:
        X_rcs = np.column_stack([X_rcs, cov_mat])
    y = df_sub[y_col].astype(int).values

    imp = SimpleImputer(strategy='median')
    X_rcs = imp.fit_transform(X_rcs)
    try:
        lr = LogisticRegression(max_iter=1000, C=0.1)
        lr.fit(X_rcs, y)
        x_grid = np.linspace(x_vals.min(), x_vals.max(), 50)
        X_grid = _rcs_basis(x_grid, knots)
        cov_median = np.median(cov_mat, axis=0) if cov_mat.size > 0 else np.array([])
        if len(cov_median) > 0:
            X_grid = np.column_stack([X_grid, np.tile(cov_median, (50, 1))])
        X_grid = imp.transform(X_grid)
        pred = lr.predict_proba(X_grid)[:, 1]
        x_plot = np.linspace(df_sub[x_col].min(), df_sub[x_col].max(), 50)
        return x_plot, pred, (x_vals, None)
    except Exception as e:
        logger.warning(f"RCS 拟合失败: {e}")
        return None, None, None


def run_dose_response(df, output_dir='dose_response', target_col='is_comorbidity_next'):
    """
    对运动频率、睡眠时长等可量化暴露做剂量反应分析。
    - sleep（小时）：连续变量，采用 RCS（限制性立方样条）。
    - exercise：若为二分类(0/1)，无法做 RCS，改为定序分析（各水平发生率）；若为多水平频率(≥4类)，则用 RCS。
    """
    logger.info(f">>> 启动剂量反应分析 (Target: {target_col})...")
    os.makedirs(output_dir, exist_ok=True)

    exclude_cols = get_exclude_cols(df, target_col)
    cov_cols = [c for c in ['age', 'gender', 'rural', 'hibpe', 'hearte'] if c in df.columns and c not in exclude_cols][:4]

    results = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. 睡眠时长 (小时) + 分箱 <5h/5-6h/>=6h
    # 缺失不做 fillna(0) 归入 <5h，而是排除并单独标记，避免扭曲剂量反应
    if 'sleep' in df.columns:
        df_s = df.copy()
        df_s['sleep'] = df_s['sleep'].clip(0, 24)
        n_sleep_missing = df_s['sleep'].isna().sum()
        if n_sleep_missing > 0:
            results.append({'exposure': 'sleep', 'level': 'Missing (excluded)', 'rate': np.nan, 'n': n_sleep_missing})
        df_s_valid = df_s[df_s['sleep'].notna()].copy()
        df_s_valid['sleep_bin'] = pd.cut(df_s_valid['sleep'], bins=[0, 5, 6, 24], labels=['<5h', '5-6h', '>=6h'])
        for lb in ['<5h', '5-6h', '>=6h']:
            m = (df_s_valid['sleep_bin'] == lb)
            if m.sum() >= 30:
                results.append({'exposure': 'sleep', 'level': lb, 'rate': df_s_valid.loc[m, target_col].mean(), 'n': m.sum()})
        x_plot, pred, _ = _fit_rcs(df_s_valid, 'sleep', target_col, cov_cols)
        if x_plot is not None and pred is not None:
            axes[0].plot(x_plot, pred, color='steelblue', linewidth=2)
            axes[0].set_xlabel('Sleep (hours)')
            axes[0].set_ylabel('P(Comorbidity)')
            axes[0].set_title('Dose-Response: Sleep Duration')
            axes[0].set_ylim(0, 1)
            results.append({'exposure': 'sleep', 'type': 'continuous', 'status': 'ok'})

    # 2. 运动：二分类(0/1)无法做RCS，改为定序分析；多水平(≥4)才用RCS
    if 'exercise' in df.columns:
        ex_vals = df['exercise'].dropna()
        if ex_vals.nunique() >= 4:
            x_plot, pred, _ = _fit_rcs(df, 'exercise', target_col, cov_cols)
            if x_plot is not None:
                axes[1].plot(x_plot, pred, color='coral', linewidth=2)
                axes[1].set_xlabel('Exercise (frequency)')
                axes[1].set_ylabel('P(Comorbidity)')
                axes[1].set_title('Dose-Response: Exercise')
                axes[1].set_ylim(0, 1)
                results.append({'exposure': 'exercise', 'type': 'continuous', 'status': 'ok'})
        else:
            # 二分类或 3 水平：定序分析 + 趋势检验（Cochran-Armitage 等价于 logistic 回归系数）
            levels = sorted(df['exercise'].dropna().unique())
            n_levels = len(levels)
            results.append({'exposure': 'exercise', 'type': 'ordinal', 'note': 'Categorical: level-wise rates + trend test'})
            for v in levels:
                mask = df['exercise'] == v
                if mask.sum() >= 20:
                    rate = df.loc[mask, target_col].mean()
                    results.append({'exposure': 'exercise', 'level': v, 'rate': rate, 'n': mask.sum()})
            # 趋势检验：outcome ~ exposure_level (ordinal)
            if n_levels >= 2:
                try:
                    from scipy import stats
                    df_ex = df[['exercise', target_col]].dropna()
                    df_ex['ex_ordinal'] = df_ex['exercise'].astype(int)
                    X_t = df_ex[['ex_ordinal']].values
                    y_t = df_ex[target_col].astype(int).values
                    lr_trend = LogisticRegression(max_iter=1000, C=0.1)
                    lr_trend.fit(X_t, y_t)
                    from sklearn.metrics import log_loss
                    ll_full = -log_loss(y_t, lr_trend.predict_proba(X_t)[:, 1])
                    lr_null = LogisticRegression(max_iter=1000, C=0.1)
                    lr_null.fit(np.ones((len(y_t), 1)), y_t)
                    ll_null = -log_loss(y_t, lr_null.predict_proba(np.ones((len(y_t), 1)))[:, 1])
                    chi2 = 2 * (ll_full - ll_null)
                    p_trend = 1 - stats.chi2.cdf(chi2, 1) if chi2 > 0 else 1.0
                    results.append({'exposure': 'exercise', 'type': 'trend_test', 'p_trend': round(p_trend, 4)})
                except Exception as e:
                    logger.debug(f"趋势检验跳过: {e}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_dose_response_rcs.png'), dpi=200, bbox_inches=BBOX_INCHES)
    plt.close()

    if results:
        pd.DataFrame(results).to_csv(os.path.join(output_dir, 'dose_response_summary.csv'), index=False, encoding='utf-8-sig')
    # 数据来源说明：sleep 保留为连续变量（小时）
    with open(os.path.join(output_dir, 'dose_response_readme.txt'), 'w', encoding='utf-8') as f:
        if 'sleep' in df.columns:
            f.write("数据含 sleep 列（连续变量，小时）。剂量反应分析包含：sleep RCS + exercise 定序/趋势。\n")
        else:
            f.write("数据无 sleep 列，剂量反应仅分析 exercise。\n")
    return True


if __name__ == '__main__':
    from config import COHORT_A_DIR, COHORT_B_DIR, COHORT_C_DIR
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    from data.charls_complete_preprocessing import preprocess_charls_data
    df = preprocess_charls_data('CHARLS.csv', age_min=60)
    if df is not None:
        for cohort_id, bg, adir in [('A', 0, COHORT_A_DIR), ('B', 1, COHORT_B_DIR), ('C', 2, COHORT_C_DIR)]:
            df_sub = df[df['baseline_group'] == bg]
            if len(df_sub) > 100:
                run_dose_response(df_sub, output_dir=os.path.join('LIU_JUE_STRATEGIC_SUMMARY', adir, 'dose_response'))
