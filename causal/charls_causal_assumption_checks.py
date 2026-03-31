# -*- coding: utf-8 -*-
"""
因果推断假设检验模块

【检验内容】
1. 正性/重叠 (Positivity/Overlap)：PS 超出 [0.05,0.95] 比例
2. 协变量平衡 (Covariate Balance)：SMD 标准化均数差
3. E-value：未测量混杂敏感性

【调用方式】
- 主流程：charls_recalculate_causal_impact 在 XLearner/TLearner 中自动调用 run_all_assumption_checks
- 单独调用：from causal.charls_causal_assumption_checks import run_all_assumption_checks, check_overlap

【输出】03_causal/ 下 assumption_*.txt、fig_propensity_overlap_*.png
"""
import os
import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression

from utils.charls_feature_lists import get_exclude_cols
from utils.charls_train_only_preprocessing import fit_transform_numeric_train_only

logger = logging.getLogger(__name__)

try:
    from config import RANDOM_SEED, BBOX_INCHES
except ImportError:
    RANDOM_SEED = 500
    BBOX_INCHES = 'tight'


def _prepare_covariates(df_sub, treatment_col, target_col='is_comorbidity_next'):
    """准备协变量矩阵 X（训练子集 fit Imputer/Scaler，与 compare_models 一致）"""
    exclude_cols = get_exclude_cols(df_sub, target_col=target_col, treatment_col=treatment_col)
    W_cols = [c for c in df_sub.columns if c not in exclude_cols]
    X_raw = df_sub[W_cols].select_dtypes(include=[np.number])
    if X_raw.shape[1] == 0:
        return None, None
    X_filled = fit_transform_numeric_train_only(df_sub, X_raw, y_col=target_col)
    return X_filled, X_raw.columns.tolist()


def check_overlap(df_sub, treatment_col, output_dir, target_col='is_comorbidity_next', suffix=''):
    """
    正性/重叠假设检验 (Positivity/Overlap)

    检验内容：倾向评分 PS 在 [0.05, 0.95] 外的样本比例。
    判定：<10% 超出 → 重叠可接受；≥10% → 需修剪。

    参数:
        suffix: 文件名后缀。'_pre_trim' = 修剪前（论文报告用），'' = 修剪后。

    输出:
        assumption_overlap_{T}{suffix}.txt
        fig_propensity_overlap_{T}{suffix}.png
    """
    X, col_names = _prepare_covariates(df_sub, treatment_col, target_col)
    if X is None:
        return {}
    T_vals = df_sub[treatment_col].fillna(0).astype(int)
    if T_vals.nunique() < 2:
        return {}
    os.makedirs(output_dir, exist_ok=True)

    ps_model = LogisticRegression(max_iter=5000, C=1e-2, solver='lbfgs', random_state=RANDOM_SEED)
    ps_model.fit(X, T_vals)
    ps = ps_model.predict_proba(X)[:, 1]

    trim_lo, trim_hi = 0.05, 0.95
    in_support = (ps >= trim_lo) & (ps <= trim_hi)
    n_trimmed = len(T_vals) - in_support.sum()
    pct_trimmed = 100 * n_trimmed / len(T_vals)
    ps_min, ps_max = float(ps.min()), float(ps.max())

    report = {
        'n_total': len(T_vals),
        'n_trimmed': int(n_trimmed),
        'pct_trimmed': round(pct_trimmed, 2),
        'ps_min': round(ps_min, 4),
        'ps_max': round(ps_max, 4),
        'overlap_ok': pct_trimmed < 10,  # <10% 超出为可接受
    }

    # 保存文本报告
    fname = f'assumption_overlap_{treatment_col}{suffix}.txt'
    with open(os.path.join(output_dir, fname), 'w', encoding='utf-8') as f:
        label = '（修剪前）' if suffix == '_pre_trim' else ''
        f.write(f"=== 正性/重叠假设检验 (Positivity/Overlap) {label}===\n")
        f.write(f"Treatment: {treatment_col}\n")
        f.write(f"N: {len(T_vals)}, T=1: {(T_vals==1).sum()}, T=0: {(T_vals==0).sum()}\n")
        f.write(f"PS range: [{ps_min:.4f}, {ps_max:.4f}]\n")
        f.write(f"Samples with PS outside [{trim_lo},{trim_hi}]: {n_trimmed} ({pct_trimmed:.1f}%)\n")
        f.write(f"Interpretation: <10% excluded suggests adequate overlap.\n")

    # 绘图
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        mask0, mask1 = (T_vals == 0), (T_vals == 1)
        if mask0.sum() > 5 and mask1.sum() > 5:
            plt.hist(ps[mask0], bins=25, alpha=0.5, color='#d9534f', label='Control (T=0)', density=True, edgecolor='black')
            plt.hist(ps[mask1], bins=25, alpha=0.5, color='#5cb85c', label='Treated (T=1)', density=True, edgecolor='black')
        plt.axvline(trim_lo, color='gray', linestyle='--', alpha=0.7)
        plt.axvline(trim_hi, color='gray', linestyle='--', alpha=0.7)
        plt.xlabel('Propensity Score')
        plt.ylabel('Density')
        title = f'Positivity Check: {treatment_col} (Pre-trim)' if suffix == '_pre_trim' else f'Positivity Check: {treatment_col}'
        plt.title(title)
        plt.legend()
        plt.xlim(0, 1)
        plt.savefig(os.path.join(output_dir, f'fig_propensity_overlap_{treatment_col}{suffix}.png'), dpi=300, bbox_inches=BBOX_INCHES)
        plt.close()
    except Exception as e:
        logger.debug(f"Overlap plot skip: {e}")

    return report


def check_balance_smd(df_sub, treatment_col, output_dir, target_col='is_comorbidity_next', ps_weights=None):
    """
    协变量平衡检验 (Covariate Balance / SMD)

    检验内容：治疗组 vs 对照组各协变量的标准化均数差 |SMD|。
    判定：|SMD|<0.1 平衡良好；0.1-0.2 需关注；>0.2 不平衡。

    参数:
        ps_weights: 若提供（IPW 权重），计算加权 SMD（Austin 2011）。

    输出:
        assumption_balance_smd_{T}.csv, assumption_balance_{T}.txt
    """
    X, col_names = _prepare_covariates(df_sub, treatment_col, target_col)
    if X is None:
        return {}
    T_vals = df_sub[treatment_col].fillna(0).astype(int)
    if T_vals.nunique() < 2:
        return {}

    X_arr = np.asarray(X)
    mask1 = (T_vals == 1).values
    mask0 = (T_vals == 0).values
    if ps_weights is not None:
        w1 = np.asarray(ps_weights)[mask1]
        w0 = np.asarray(ps_weights)[mask0]
        m1 = np.average(X_arr[mask1], axis=0, weights=w1)
        m0 = np.average(X_arr[mask0], axis=0, weights=w0)
        v1 = np.average((X_arr[mask1] - m1)**2, axis=0, weights=w1)
        v0 = np.average((X_arr[mask0] - m0)**2, axis=0, weights=w0)
        pooled_std = np.sqrt((v1 + v0) / 2)
    else:
        m1 = X_arr[mask1].mean(axis=0)
        m0 = X_arr[mask0].mean(axis=0)
        s1 = X_arr[mask1].std(axis=0)
        s0 = X_arr[mask0].std(axis=0)
        pooled_std = np.sqrt((s1**2 + s0**2) / 2)
    pooled_std[pooled_std < 1e-9] = 1e-9
    smds = np.abs((m1 - m0) / pooled_std)

    smd_df = pd.DataFrame({
        'covariate': col_names,
        'mean_T1': m1,
        'mean_T0': m0,
        'smd': smds,
        'balanced': np.abs(smds) < 0.1,
    })
    smd_df = smd_df.sort_values('smd', ascending=False)

    max_smd = float(np.max(smds))
    n_imbalanced = (np.abs(smds) >= 0.1).sum()

    os.makedirs(output_dir, exist_ok=True)
    suffix = "_weighted" if ps_weights is not None else ""
    smd_df.to_csv(os.path.join(output_dir, f'assumption_balance_smd_{treatment_col}{suffix}.csv'), index=False, encoding='utf-8-sig')

    with open(os.path.join(output_dir, f'assumption_balance_{treatment_col}{suffix}.txt'), 'w', encoding='utf-8') as f:
        f.write(f"=== 协变量平衡检验 (Covariate Balance / SMD{suffix}) ===\n")
        f.write(f"Treatment: {treatment_col}\n")
        f.write(f"Max |SMD|: {max_smd:.4f}\n")
        f.write(f"Covariates with |SMD|>=0.1: {n_imbalanced} / {len(col_names)}\n")
        f.write(f"Interpretation: |SMD|<0.1 balanced; 0.1-0.2 moderate; >0.2 substantial imbalance.\n")

    out = {'max_smd': round(max_smd, 4), 'n_imbalanced': int(n_imbalanced), 'n_covariates': len(col_names)}
    if ps_weights is not None:
        out['smd_weighted'] = round(max_smd, 4)
    return out


def compute_evalue(ate, ate_lb, ate_ub, r0):
    """E-value: 未测量混杂需多强才能解释掉效应 (VanderWeele & Ding 2017)"""
    r0 = np.clip(r0, 1e-6, 1 - 1e-6)

    def _evalue_from_rr(rr):
        if rr <= 1:
            rr = 1 / (rr + 1e-9)
        return rr + np.sqrt(rr * (rr - 1)) if rr > 1 else np.nan

    rr_point = (r0 + ate) / r0
    # VanderWeele & Ding 2017: 保守端取 CI 中离 null 最近的一端
    if ate_lb is not None and ate_ub is not None:
        rr_conservative = (r0 + ate_lb) / r0 if ate >= 0 else (r0 + ate_ub) / r0
    else:
        rr_conservative = rr_point
    e_point = _evalue_from_rr(rr_point)
    e_conservative = _evalue_from_rr(rr_conservative)
    return e_point, e_conservative


def check_evalue(df_sub, treatment_col, ate, ate_lb, ate_ub, output_dir, target_col='is_comorbidity_next'):
    """
    E-value：未测量混杂敏感性分析 (VanderWeele & Ding 2017)

    含义：需多强的未测混杂（RR≥E 且同时影响 T 和 Y）才能解释掉观测效应。
    E 越大，对未测混杂越稳健。

    输出: assumption_evalue_{T}.txt
    """
    if ate is None or np.isnan(ate):
        return {}
    T_vals = df_sub[treatment_col].fillna(0).astype(int)
    Y_vals = df_sub[target_col].astype(float)
    r0 = Y_vals[T_vals == 0].mean()
    e_point, e_conservative = compute_evalue(ate, ate_lb, ate_ub, r0)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'assumption_evalue_{treatment_col}.txt'), 'w', encoding='utf-8') as f:
        f.write(f"=== E-Value: 未测量混杂敏感性 (Unmeasured Confounding) ===\n")
        f.write(f"Treatment: {treatment_col}\n")
        f.write(f"ATE: {ate:.4f}, 95%% CI: ({ate_lb:.4f}, {ate_ub:.4f})\n")
        f.write(f"E-Value (point): {e_point:.2f}\n")
        f.write(f"E-Value (conservative, from 95%% CI): {e_conservative:.2f}\n")
        f.write(f"Interpretation: An unmeasured confounder would need RR>=E with both T and Y to explain away the effect.\n")

    return {'e_value_point': round(e_point, 2), 'e_value_conservative': round(e_conservative, 2)}


def run_all_assumption_checks(df_sub, treatment_col, output_dir, ate=None, ate_lb=None, ate_ub=None,
                               target_col='is_comorbidity_next', pre_trim_overlap=None):
    """
    运行全部假设检验：重叠、平衡、E-value。

    执行顺序：1) 重叠 2) 平衡（含加权 SMD）3) E-value 4) 写入 assumption_checks_summary.txt

    参数:
        pre_trim_overlap: 修剪前 overlap 报告（dict），若提供则汇总中优先报告，便于论文引用。

    返回: summary dict
    """
    summary = {}
    try:
        overlap = check_overlap(df_sub, treatment_col, output_dir, target_col)
        summary['overlap'] = overlap
        if pre_trim_overlap:
            summary['overlap_pre_trim'] = pre_trim_overlap
    except Exception as e:
        logger.warning(f"Overlap check failed: {e}")
    try:
        balance = check_balance_smd(df_sub, treatment_col, output_dir, target_col)
        summary['balance'] = balance
        # 加权 SMD（Austin 2011）：用 PS 权重计算平衡
        try:
            X, _ = _prepare_covariates(df_sub, treatment_col, target_col)
            if X is not None:
                T_vals = df_sub[treatment_col].fillna(0).astype(int)
                if T_vals.nunique() >= 2:
                    ps_model = LogisticRegression(max_iter=5000, C=1e-2, solver='lbfgs', random_state=RANDOM_SEED)
                    ps_model.fit(X, T_vals)
                    ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)
                    w = np.where(T_vals == 1, 1 / ps, 1 / (1 - ps))
                    w = np.clip(w, 0.1, 50)
                    bal_w = check_balance_smd(df_sub, treatment_col, output_dir, target_col, ps_weights=w)
                    summary['balance']['smd_weighted'] = bal_w.get('smd_weighted', bal_w.get('max_smd'))
        except Exception as ew:
            logger.debug(f"Weighted SMD skip: {ew}")
    except Exception as e:
        logger.warning(f"Balance check failed: {e}")
    if ate is not None:
        try:
            evalue = check_evalue(df_sub, treatment_col, ate, ate_lb, ate_ub, output_dir, target_col)
            summary['evalue'] = evalue
        except Exception as e:
            logger.warning(f"E-value check failed: {e}")

    # 写入汇总
    if summary:
        with open(os.path.join(output_dir, 'assumption_checks_summary.txt'), 'w', encoding='utf-8') as f:
            f.write("=== Causal Inference Assumption Checks (Overlap, Balance, E-Value) ===\n\n")
            # 优先报告修剪前 overlap（论文方法部分引用）
            if 'overlap_pre_trim' in summary:
                o = summary['overlap_pre_trim']
                f.write(f"Overlap (pre-trim): {o.get('pct_trimmed', '')}% PS outside [0.05,0.95]; OK={o.get('overlap_ok', '')}\n")
                f.write(f"  -> N excluded: {o.get('n_trimmed', '')}; N retained: {summary.get('overlap', {}).get('n_total', 'N/A')}\n")
            if 'overlap' in summary:
                o = summary['overlap']
                f.write(f"Overlap (post-trim): {o.get('pct_trimmed', '')}% PS outside [0.05,0.95]; OK={o.get('overlap_ok', '')}\n")
            if 'balance' in summary:
                b = summary['balance']
                f.write(f"Balance: max|SMD|={b.get('max_smd', '')}; imbalanced vars={b.get('n_imbalanced', '')}\n")
            if 'evalue' in summary:
                e = summary['evalue']
                f.write(f"E-Value: point={e.get('e_value_point', '')}; conservative={e.get('e_value_conservative', '')}\n")

    return summary
