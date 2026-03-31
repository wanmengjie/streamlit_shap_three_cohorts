# -*- coding: utf-8 -*-
"""
因果推断方法交叉验证（审稿意见 P1）
对运动、睡眠等可靠干预，补充 PSM、PSW 方法重估 ATE，与 Causal Forest DML 对比。
"""
import os
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from utils.charls_feature_lists import get_exclude_cols
from utils.charls_train_only_preprocessing import fit_transform_numeric_train_only
from config import BBOX_INCHES, TARGET_COL

logger = logging.getLogger(__name__)

# 论文2.4/2.8节：PSM 卡尺（倾向得分标准差的倍数）；PSW 权重 [0.1, 50] 见 _ate_psw
try:
    from config import CALIPER_PSM
except ImportError:
    CALIPER_PSM = 0.024

RELIABLE_INTERVENTIONS = ['exercise', 'drinkev', 'is_socially_isolated', 'bmi_normal', 'chronic_low']


def load_ml_ate_ci_from_summary_txt(output_dir, treatment_col):
    """
    Read population ATE and 95% CI from ATE_CI_summary_{treatment}.txt written by
    charls_recalculate_causal_impact (XLearner / TLearner / DML). This is the same
    estimand and uncertainty as table4 / main-text contrasts.

    Returns (ate, lb, ub) or (None, None, None) if missing or unparseable.
    """
    if not output_dir or not treatment_col:
        return None, None, None
    path = os.path.join(output_dir, f'ATE_CI_summary_{treatment_col}.txt')
    if not os.path.isfile(path):
        return None, None, None
    try:
        with open(path, encoding='utf-8') as f:
            text = f.read()
    except OSError:
        return None, None, None

    # XLearner / TLearner: one line starting with "ATE" (not "ATT") contains ATE and 95% CI
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('ATT'):
            continue
        if not s.startswith('ATE'):
            continue
        if '95% CI:' not in s:
            continue
        m = re.search(
            r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*95%\s*CI:\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)',
            s,
        )
        if m:
            return float(m.group(1)), float(m.group(2)), float(m.group(3))

    # DML: "ATE (point estimate): x" on one line, "95% CI: (lb, ub)" on the next
    ate_v = lb_v = ub_v = None
    for line in text.splitlines():
        if 'ATE (point estimate):' in line:
            m = re.search(r'ATE \(point estimate\):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
            if m:
                ate_v = float(m.group(1))
        ls = line.strip()
        if ls.startswith('95% CI:'):
            m = re.search(r'95%\s*CI:\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)', ls)
            if m:
                lb_v, ub_v = float(m.group(1)), float(m.group(2))
    if ate_v is not None and lb_v is not None and ub_v is not None:
        return ate_v, lb_v, ub_v

    return None, None, None


def calc_ate(df, treatment, outcome, confounders=None, method='ipw', **kwargs):
    """
    Estimate ATE using IPW or matching. Returns (effect, ci_lower, ci_upper).

    Parameters
    ----------
    df : pd.DataFrame
    treatment : str, column name of binary treatment
    outcome : str, column name of outcome
    confounders : list of str or None
        Covariate column names. If None, inferred via get_exclude_cols.
    method : str, 'ipw'|'psw' or 'matching'|'psm'
    **kwargs : caliper (for PSM), random_state

    Returns
    -------
    tuple : (effect, ci_lower, ci_upper)
    """
    if confounders is None:
        exclude_cols = get_exclude_cols(df, target_col=outcome, treatment_col=treatment)
        X, _ = _prepare_covariates(df, treatment, outcome, exclude_cols)
    else:
        cols = [c for c in confounders if c in df.columns]
        X_raw = df[cols].select_dtypes(include=[np.number])
        if X_raw.shape[1] == 0:
            return np.nan, np.nan, np.nan
        X = fit_transform_numeric_train_only(df, X_raw, y_col=outcome)
    if X is None:
        return np.nan, np.nan, np.nan
    method = method.lower()
    if method in ('ipw', 'psw'):
        ate, lb, ub = _ate_psw(df, treatment, outcome, X, random_state=kwargs.get('random_state'))
    elif method in ('matching', 'psm'):
        ate, lb, ub, _ = _ate_psm(df, treatment, outcome, X,
                                   caliper=kwargs.get('caliper'),
                                   random_state=kwargs.get('random_state'))
    else:
        raise ValueError(f"method must be 'ipw'|'psw' or 'matching'|'psm', got '{method}'")
    return ate, lb, ub


def _prepare_covariates(df, T, Y, exclude_cols):
    """准备协变量矩阵（训练子集 fit Imputer/Scaler，与 compare_models 一致）"""
    W_cols = [c for c in df.columns if c not in exclude_cols and c not in [T, Y]]
    X_raw = df[W_cols].select_dtypes(include=[np.number])
    if X_raw.shape[1] == 0:
        return None, None
    X_filled = fit_transform_numeric_train_only(df, X_raw, y_col=Y)
    return X_filled, None


def _compute_smd(X_treated, X_control):
    """计算匹配后协变量标准化均数差 (SMD)，|SMD|<0.1 表示平衡良好"""
    smds = []
    for j in range(X_treated.shape[1]):
        m1, m0 = X_treated[:, j].mean(), X_control[:, j].mean()
        s1, s0 = X_treated[:, j].std(), X_control[:, j].std()
        pooled_std = np.sqrt((s1**2 + s0**2) / 2) if (s1 > 0 or s0 > 0) else 1e-9
        smds.append(abs((m1 - m0) / pooled_std))
    return np.array(smds)


def _ate_psm(df, T, Y, X, caliper=None, random_state=None, return_match_diag=False):
    """
    倾向得分匹配 (PSM)：1:1 最近邻匹配，估计 ATE（论文2.4节：卡尺 0.024*std(PS)）
    return_match_diag=True 时额外返回 dict：匹配索引与逐列 SMD，供双重调整使用。
    """
    if caliper is None:
        caliper = CALIPER_PSM
    if random_state is None:
        try:
            from config import RANDOM_SEED
            random_state = RANDOM_SEED
        except ImportError:
            random_state = 500
    T_vals = df[T].fillna(0).astype(int)
    Y_vals = df[Y].astype(float)
    if T_vals.nunique() < 2 or Y_vals.nunique() < 2:
        if return_match_diag:
            return np.nan, np.nan, np.nan, np.nan, None
        return np.nan, np.nan, np.nan, np.nan

    ps_model = LogisticRegression(max_iter=5000, C=1e-2, solver='lbfgs', random_state=random_state)
    ps_model.fit(X, T_vals)
    ps = ps_model.predict_proba(X)[:, 1]

    treated_idx = np.where(T_vals == 1)[0]
    control_idx = np.where(T_vals == 0)[0]
    if len(treated_idx) < 20 or len(control_idx) < 20:
        if return_match_diag:
            return np.nan, np.nan, np.nan, np.nan, None
        return np.nan, np.nan, np.nan, np.nan

    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(ps[control_idx].reshape(-1, 1))
    distances, matched_control_idx = nn.kneighbors(ps[treated_idx].reshape(-1, 1))
    matched_control_idx = control_idx[matched_control_idx.flatten()]
    sd_ps = np.std(ps)
    caliper_val = caliper * sd_ps if sd_ps > 0 else 0.1
    keep = distances.flatten() <= caliper_val
    if keep.sum() < 10:
        keep = np.ones(len(treated_idx), dtype=bool)
    t_matched = treated_idx[keep]
    c_matched = matched_control_idx[keep]
    ate = Y_vals.iloc[t_matched].mean() - Y_vals.iloc[c_matched].mean()
    se = np.sqrt(Y_vals.iloc[t_matched].var() / len(t_matched) + Y_vals.iloc[c_matched].var() / len(c_matched) + 1e-9)
    lb, ub = ate - 1.96 * se, ate + 1.96 * se
    # Round to 4 decimal places
    ate = round(ate, 4)
    lb = round(lb, 4)
    ub = round(ub, 4)
    # 匹配后 SMD 验证
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    smds = _compute_smd(X_arr[t_matched], X_arr[c_matched])
    max_smd = float(np.max(smds)) if len(smds) > 0 else np.nan
    max_smd = round(max_smd, 4)
    diag = {
        't_matched': t_matched,
        'c_matched': c_matched,
        'smds': smds,
        'columns': list(X.columns),
    }
    if return_match_diag:
        return ate, lb, ub, max_smd, diag
    return ate, lb, ub, max_smd


def _psm_double_adjust_logit(df_sub, T, Y, X, diag, smd_threshold=0.1):
    """
    匹配后双重调整：在 1:1 匹配集上拟合 Logit(Y ~ T + 未配平协变量)。
    返回 (coef_T, se_T, pvalue_T) 或 (None, None, None)。
    """
    if diag is None:
        return None, None, None
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("statsmodels 未安装，跳过 PSM 双重调整 Logit")
        return None, None, None
    t_matched = diag['t_matched']
    c_matched = diag['c_matched']
    smds = np.asarray(diag['smds'])
    cols = diag['columns']
    imbalanced = [cols[j] for j in range(len(smds)) if j < len(cols) and smds[j] >= smd_threshold]
    # 匹配集：处理组 + 对照组各一行
    idx = np.unique(np.concatenate([t_matched, c_matched]))
    matched = df_sub.iloc[idx].copy()
    yv = matched[Y].astype(float).values
    if len(np.unique(yv)) < 2:
        return None, None, None
    use_cols = [T] + [c for c in imbalanced if c in matched.columns and c != T]
    use_cols = list(dict.fromkeys(use_cols))
    try:
        Xd = matched[use_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        Xd = sm.add_constant(Xd, has_constant='add')
        model = sm.Logit(yv, Xd.astype(float)).fit(disp=False, maxiter=200)
        if T not in model.params.index:
            return None, None, None
        coef = float(model.params[T])
        se = float(model.bse[T]) if T in model.bse.index else np.nan
        pv = float(model.pvalues[T]) if T in model.pvalues.index else np.nan
        return coef, se, pv
    except Exception as ex:
        logger.debug(f"PSM 双重调整 Logit 失败: {ex}")
        return None, None, None


def _ate_psw(df, T, Y, X, random_state=None):
    """
    倾向得分加权 (PSW / IPW)：逆概率加权估计 ATE
    """
    if random_state is None:
        try:
            from config import RANDOM_SEED
            random_state = RANDOM_SEED
        except ImportError:
            random_state = 500
    T_vals = df[T].fillna(0).astype(int)
    Y_vals = df[Y].astype(float)
    if T_vals.nunique() < 2 or Y_vals.nunique() < 2:
        return np.nan, np.nan, np.nan

    ps_model = LogisticRegression(max_iter=5000, C=1e-2, solver='lbfgs', random_state=random_state)
    ps_model.fit(X, T_vals)
    ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)
    w = np.where(T_vals == 1, 1 / ps, 1 / (1 - ps))
    # Weight trimming: 截断极值避免逆概率加权爆炸，对应 Sturmer et al. 2020 建议
    # 先限制在 1–99 分位数，再兜底 [0.1, 50]
    p1, p99 = np.percentile(w, [1, 99])
    w = np.clip(w, p1, p99)
    w = np.clip(w, 0.1, 50)
    y1_mean = (Y_vals * (T_vals == 1) * w).sum() / ((T_vals == 1) * w).sum()
    y0_mean = (Y_vals * (T_vals == 0) * w).sum() / ((T_vals == 0) * w).sum()
    ate = y1_mean - y0_mean
    se = np.sqrt(np.var(Y_vals * w) / len(Y_vals) + 1e-9)
    lb, ub = ate - 1.96 * se, ate + 1.96 * se
    # Round to 4 decimal places
    ate = round(ate, 4)
    lb = round(lb, 4)
    ub = round(ub, 4)
    return ate, lb, ub


def run_causal_methods_comparison(df_sub, treatment_col, output_dir, target_col='is_comorbidity_next',
                                  dml_ate=None, dml_lb=None, dml_ub=None, ml_method_name='TLearner'):
    """
    对指定干预运行 PSM、PSW，与因果 ML 方法对比。
    dml_ate, dml_lb, dml_ub: 因果 ML 方法（如 TLearner）的 ATE 与 95% CI
    ml_method_name: ML 方法名称，默认 'TLearner'；亦可为 'Causal Forest DML'
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f">>> 因果方法交叉验证: {treatment_col} (n={len(df_sub)})")

    T = treatment_col
    Y = target_col
    exclude_cols = get_exclude_cols(df_sub, target_col=Y, treatment_col=T)
    X, _ = _prepare_covariates(df_sub, T, Y, exclude_cols)
    if X is None:
        logger.warning("无可用协变量，跳过")
        return pd.DataFrame()

    psm_ate, psm_lb, psm_ub, psm_max_smd, psm_diag = _ate_psm(
        df_sub, T, Y, X, return_match_diag=True
    )
    psw_ate, psw_lb, psw_ub = _ate_psw(df_sub, T, Y, X)

    results = [
        {'Method': 'PSM', 'ATE': psm_ate, 'ATE_lb': psm_lb, 'ATE_ub': psm_ub, 'PSM_max_SMD': psm_max_smd},
        {'Method': 'PSW', 'ATE': psw_ate, 'ATE_lb': psw_lb, 'ATE_ub': psw_ub, 'PSM_max_SMD': np.nan},
    ]
    try:
        from config import PSM_DOUBLE_ADJUST_LOGIT
    except ImportError:
        PSM_DOUBLE_ADJUST_LOGIT = True
    if PSM_DOUBLE_ADJUST_LOGIT and psm_diag is not None:
        coef_t, se_t, pv_t = _psm_double_adjust_logit(df_sub, T, Y, X, psm_diag)
        if coef_t is not None and se_t is not None and not np.isnan(se_t) and se_t > 0:
            adj_lb = coef_t - 1.96 * se_t
            adj_ub = coef_t + 1.96 * se_t
            with open(os.path.join(output_dir, f'psm_double_adjustment_{T}.txt'), 'w', encoding='utf-8') as f:
                f.write(
                    "PSM matched-sample logistic adjustment for residual |SMD|>=0.1 covariates.\n"
                    "Outcome: binary; scale: LOG-ODDS (not risk difference). Compare to PSM risk-diff ATE separately.\n"
                )
                f.write(f"coef({T})={coef_t:.6f}, SE={se_t:.6f}, 95% CI=({adj_lb:.6f},{adj_ub:.6f}), p={pv_t}\n")
            logger.info(
                "PSM double-adjustment Logit %s: coef=%.4f (log-OR scale), p=%s — see psm_double_adjustment_%s.txt",
                T, coef_t, pv_t, T,
            )
    if dml_ate is not None and dml_lb is not None and dml_ub is not None:
        results.append({'Method': ml_method_name, 'ATE': dml_ate, 'ATE_lb': dml_lb, 'ATE_ub': dml_ub, 'PSM_max_SMD': np.nan})

    res_df = pd.DataFrame(results)
    if not np.isnan(psm_max_smd):
        logger.info(f"PSM 匹配后 max_SMD={psm_max_smd:.4f} {'(<0.1 平衡良好)' if psm_max_smd < 0.1 else '(≥0.1 建议关注)'}")
    res_df.to_csv(os.path.join(output_dir, f'causal_methods_comparison_{T}.csv'), index=False, encoding='utf-8-sig')
    try:
        res_df.to_excel(os.path.join(output_dir, f'causal_methods_comparison_{T}.xlsx'), index=False)
    except Exception as ex:
        logger.debug(f"Excel 导出跳过 (需 openpyxl): {ex}")

    # 森林图
    if len(res_df) > 0 and res_df['ATE'].notna().any():
        fig, ax = plt.subplots(figsize=(8, 4))
        y_pos = np.arange(len(res_df))[::-1]
        colors = ['#3498db', '#e74c3c', '#2ecc71']  # PSM, PSW, DML
        for i, row in res_df.iterrows():
            ate = row['ATE']
            lb = row.get('ATE_lb', ate - 0.05)
            ub = row.get('ATE_ub', ate + 0.05)
            if np.isnan(ate):
                continue
            ax.barh(y_pos[i], ate, color=colors[i % 3], alpha=0.8)
            ax.errorbar(ate, y_pos[i], xerr=[[ate - lb], [ub - ate]], fmt='none', color='black', capsize=3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(res_df['Method'].values)
        ax.set_xlabel('ATE (Risk Difference)')
        ax.set_title(f'Causal Methods Comparison: {treatment_col}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fig_causal_methods_forest_{T}.png'), dpi=150, bbox_inches=BBOX_INCHES)
        plt.close()

    return res_df


def run_all_cohorts_comparison(df_clean, output_root='LIU_JUE_STRATEGIC_SUMMARY'):
    """
    对三队列（Cohort A/B/C）、可靠干预运行 ML（XLearner/TLearner/DML）vs PSM vs PSW 对比。

    ML 行的 ATE 与 95% CI 优先从同目录下 ATE_CI_summary_{T}.txt 读取（与主分析 / table4 同源）；
    若缺失则回退为 CAUSAL_ANALYSIS 中 causal_impact 列的分位数（仅为异质性 spread，非有效 ATE CI，
    且会打 WARNING）。
    """
    from scripts.run_all_interventions_analysis import prepare_interventions
    from config import COHORT_A_DIR, COHORT_B_DIR, COHORT_C_DIR
    df_clean = prepare_interventions(df_clean.copy())
    df_a = df_clean[df_clean['baseline_group'] == 0]
    df_b = df_clean[df_clean['baseline_group'] == 1]
    df_c = df_clean[df_clean['baseline_group'] == 2]

    cohort_dirs = {'Cohort_A': COHORT_A_DIR, 'Cohort_B': COHORT_B_DIR, 'Cohort_C': COHORT_C_DIR}
    all_results = []
    for cohort_name, df_sub in [('Cohort_A', df_a), ('Cohort_B', df_b), ('Cohort_C', df_c)]:
        if len(df_sub) < 50:
            continue
        for T in RELIABLE_INTERVENTIONS:
            if T not in df_sub.columns:
                continue
            out_dir = os.path.join(output_root, 'all_interventions_causal', cohort_name, T)
            causal_csv = os.path.join(out_dir, f'CAUSAL_ANALYSIS_{T}.csv')
            if not os.path.exists(causal_csv):
                # 03_causal 在项目根目录 Cohort_* 下，非 output_root 下
                alt_csv = os.path.join(cohort_dirs[cohort_name], '03_causal', f'CAUSAL_ANALYSIS_{T}.csv')
                if os.path.exists(alt_csv):
                    causal_csv = alt_csv
                    out_dir = os.path.join(cohort_dirs[cohort_name], '03_causal')
                else:
                    # 多暴露分析结果在 08_multi_exposure 下（output_root 内）
                    multi_csv = os.path.join(output_root, '08_multi_exposure', cohort_name, T, f'CAUSAL_ANALYSIS_{T}.csv')
                    if os.path.exists(multi_csv):
                        causal_csv = multi_csv
                        out_dir = os.path.join(output_root, '08_multi_exposure', cohort_name, T)
                    else:
                        # X-Learner 全干预结果（主流程新增）
                        xl_csv = os.path.join(output_root, 'xlearner_all_interventions', cohort_name, T, f'CAUSAL_ANALYSIS_{T}.csv')
                        if os.path.exists(xl_csv):
                            causal_csv = xl_csv
                            out_dir = os.path.join(output_root, 'xlearner_all_interventions', cohort_name, T)
                        else:
                            out_dir = os.path.join(output_root, 'causal_methods_comparison', cohort_name, T)
            dml_ate = dml_lb = dml_ub = None
            summary_dirs = []
            if os.path.exists(causal_csv):
                summary_dirs.append(os.path.dirname(os.path.abspath(causal_csv)))
            if out_dir and os.path.isdir(out_dir):
                summary_dirs.append(os.path.abspath(out_dir))
            seen = set()
            for sd in summary_dirs:
                if not sd or sd in seen:
                    continue
                seen.add(sd)
                ate_s, lb_s, ub_s = load_ml_ate_ci_from_summary_txt(sd, T)
                if ate_s is not None and lb_s is not None and ub_s is not None:
                    dml_ate = round(float(ate_s), 4)
                    dml_lb = round(float(lb_s), 4)
                    dml_ub = round(float(ub_s), 4)
                    logger.info(
                        "%s %s: ML row uses ATE_CI_summary_%s.txt in %s (same as primary ATE CI).",
                        cohort_name,
                        T,
                        T,
                        sd,
                    )
                    break

            if dml_ate is None and os.path.exists(causal_csv):
                try:
                    df_causal = pd.read_csv(causal_csv)
                    causal_col = f'causal_impact_{T}'
                    if causal_col in df_causal.columns:
                        dml_ate = df_causal[causal_col].mean()
                        lb, ub = np.percentile(df_causal[causal_col].dropna(), [2.5, 97.5])
                        dml_lb, dml_ub = lb, ub
                        dml_ate = round(dml_ate, 4)
                        dml_lb = round(dml_lb, 4)
                        dml_ub = round(dml_ub, 4)
                        logger.warning(
                            "%s %s: no ATE_CI_summary_%s.txt next to CAUSAL_ANALYSIS; "
                            "using 2.5/97.5%% of per-row %s (spread of predicted effects, not a valid ATE CI).",
                            cohort_name,
                            T,
                            T,
                            causal_col,
                        )
                except Exception as ex:
                    logger.debug(f"读取 ML 因果 CSV 失败 {cohort_name} {T}: {ex}")
            try:
                try:
                    from config import CAUSAL_METHOD
                    ml_name = {'TLearner': 'TLearner', 'XLearner': 'XLearner'}.get(CAUSAL_METHOD, 'Causal Forest DML')
                except ImportError:
                    ml_name = 'Causal Forest DML'
                res_df = run_causal_methods_comparison(
                    df_sub, T, out_dir, target_col=TARGET_COL,
                    dml_ate=dml_ate, dml_lb=dml_lb, dml_ub=dml_ub,
                    ml_method_name=ml_name
                )
                for _, r in res_df.iterrows():
                    all_results.append({
                        'cohort': cohort_name, 'exposure': T, 'method': r['Method'],
                        'ate': r['ATE'], 'ate_lb': r.get('ATE_lb'), 'ate_ub': r.get('ATE_ub')
                    })
            except Exception as e:
                logger.warning(f"{cohort_name} {T} 方法对比失败: {e}")

    if all_results:
        summary = pd.DataFrame(all_results)
        # 多重检验校正（6暴露×3队列×3方法=54次比较）
        try:
            from utils.multiplicity_correction import add_multiplicity_columns
            summary = add_multiplicity_columns(summary, ate_col='ate', lb_col='ate_lb', ub_col='ate_ub')
        except Exception as ex:
            logger.debug(f"多重检验校正跳过: {ex}")
        # 一致性评价：按 cohort+exposure 分组，检查 XLearner/PSM/PSW 方向一致且 95% CI 重叠
        summary = _add_consistency_column(summary)
        out_path = os.path.join(output_root, 'causal_methods_comparison_summary.csv')
        summary.to_csv(out_path, index=False, encoding='utf-8-sig')
        logger.info(f"因果方法对比汇总已保存: {out_path}")
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()


def _add_consistency_column(summary):
    """
    为 causal_methods_comparison_summary 增加 Consistency 列。
    按 cohort+exposure 分组，对每组内三种方法（XLearner/TLearner、PSM、PSW）判断：
    - Consistent: 方向一致且 95% CI 有重叠
    - Direction_only: 方向一致但 CI 不重叠
    - Inconsistent: 方向不一致
    """
    if summary.empty or 'exposure' not in summary.columns:
        return summary
    s = summary.copy()
    if 'cohort' not in s.columns and 'axis' in s.columns:
        s = s.rename(columns={'axis': 'cohort'})
    if 'cohort' not in s.columns:
        return summary
    consistency_map = {}
    for (cohort, exposure), grp in s.groupby(['cohort', 'exposure']):
        if len(grp) < 2:
            consistency_map[(cohort, exposure)] = 'N/A'
            continue
        valid = grp[['ate', 'ate_lb', 'ate_ub']].dropna(how='any')
        if len(valid) < 2:
            consistency_map[(cohort, exposure)] = 'N/A'
            continue
        lbs = valid['ate_lb'].values
        ubs = valid['ate_ub'].values
        # 方向：正(1)、负(-1)、含零(0)
        signs = []
        for _, row in valid.iterrows():
            lb, ub = row['ate_lb'], row['ate_ub']
            if lb > 0:
                signs.append(1)
            elif ub < 0:
                signs.append(-1)
            else:
                signs.append(0)
        direction_ok = len(set(signs)) == 1
        # CI 重叠：max(lb) <= min(ub)
        ci_overlap = float(np.nanmax(lbs)) <= float(np.nanmin(ubs))
        if direction_ok and ci_overlap:
            consistency_map[(cohort, exposure)] = 'Consistent'
        elif direction_ok:
            consistency_map[(cohort, exposure)] = 'Direction_only'
        else:
            consistency_map[(cohort, exposure)] = 'Inconsistent'
    s['Consistency'] = s.apply(
        lambda r: consistency_map.get((r['cohort'], r['exposure']), 'N/A'), axis=1
    )
    return s


run_all_axes_comparison = run_all_cohorts_comparison  # 旧函数名兼容


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    from data.charls_complete_preprocessing import preprocess_charls_data
    df = preprocess_charls_data('CHARLS.csv', age_min=60)
    if df is not None:
        run_all_cohorts_comparison(df, output_root='LIU_JUE_STRATEGIC_SUMMARY')
