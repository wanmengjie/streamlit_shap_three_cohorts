import pandas as pd
import numpy as np
import os
import shutil
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import CausalForestDML
from sklearn.linear_model import LogisticRegression
import warnings

try:
    from config import BBOX_INCHES
except ImportError:
    BBOX_INCHES = 'tight'

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=UserWarning, module="loky")

from utils.charls_feature_lists import get_exclude_cols
from utils.charls_train_only_preprocessing import fit_transform_numeric_train_only
from utils.charls_ci_utils import cluster_bootstrap_indices_once
from config import RANDOM_SEED

logger = logging.getLogger(__name__)

# 估计失败或跳过：与 (0,0,0) 区分，避免汇总图 / Rubin 合并误读为「零效应」
CAUSAL_FAILURE_ATE_TRIPLET = (np.nan, np.nan, np.nan)


def _p_value_two_sided_from_ci(ate, ate_lb, ate_ub):
    """由 ATE 与 95% CI 反推近似双侧 P（正态近似，与 table4 口径可对照）。"""
    try:
        from scipy.stats import norm

        lb, ub = float(ate_lb), float(ate_ub)
        se = (ub - lb) / 3.92
        if se <= 0 or np.isnan(se) or np.isnan(float(ate)):
            return np.nan
        z = abs(float(ate) / se)
        return float(2 * (1 - norm.cdf(z)))
    except Exception:
        return np.nan


def _apply_ps_overlap_trim(
    df_sub,
    X_scaled,
    T_series,
    Y_series,
    trim_lo,
    trim_hi,
    *,
    force_trim: bool,
):
    """
    倾向得分修剪。
    force_trim=True：始终只保留 PS∈[trim_lo, trim_hi]。
    force_trim=False：与历史主流程一致——仅当「落在区间内」比例 <90% 时才按该区间剔除；否则不剔除。
    返回修剪后的 df_sub, X_scaled, T_series, Y_series 及诊断字典。
    """
    ps_model = LogisticRegression(max_iter=5000, C=1e-2, solver="lbfgs", random_state=RANDOM_SEED)
    ps_model.fit(X_scaled, T_series)
    ps = ps_model.predict_proba(X_scaled)[:, 1]
    in_band = (ps >= trim_lo) & (ps <= trim_hi)
    n_before_trim = int(len(df_sub))
    pct_in_band = float(np.mean(in_band))
    overlap_trimmed_pct = float(100 * (1 - pct_in_band))

    if force_trim:
        do_subset = True
        mask = in_band
    else:
        do_subset = pct_in_band < 0.9
        mask = in_band

    if do_subset:
        df_sub = df_sub.loc[mask].reset_index(drop=True)
        X_scaled = X_scaled.loc[mask].reset_index(drop=True)
        T_series = T_series.loc[mask].reset_index(drop=True)
        Y_series = Y_series.loc[mask].reset_index(drop=True)

    n_after_trim = int(len(df_sub))
    logger.info(
        "PS trim [%.4f,%.4f] force_trim=%s | N_before_trim=%d N_after_trim=%d | "
        "pct_in_band=%.2f%% overlap_trimmed_pct(全量)=%.2f%% | applied_subset=%s",
        trim_lo,
        trim_hi,
        force_trim,
        n_before_trim,
        n_after_trim,
        100 * pct_in_band,
        overlap_trimmed_pct,
        do_subset,
    )
    info = {
        "n_before_trim": n_before_trim,
        "n_after_trim": n_after_trim,
        "overlap_trimmed_pct": overlap_trimmed_pct,
        "pct_in_band_before_subset": 100 * pct_in_band,
        "force_trim": force_trim,
        "applied_subset": do_subset,
    }
    return df_sub, X_scaled, T_series, Y_series, info


def _write_ite_export_csv(df_sub, ite, treatment_col, outcome_col):
    """将个体 ITE 写入 results/tables/ite_xlearner_{treatment}_cohort_{A|B|C}.csv（每队列单行 baseline）。"""
    try:
        from config import RESULTS_TABLES
    except ImportError:
        RESULTS_TABLES = "results/tables"
    if "baseline_group" not in df_sub.columns or "ID" not in df_sub.columns:
        logger.debug("ITE 导出跳过：缺少 ID 或 baseline_group")
        return
    bg = df_sub["baseline_group"].dropna()
    if len(bg) == 0 or bg.nunique() != 1:
        logger.debug("ITE 导出跳过：baseline_group 不唯一")
        return
    g = int(bg.iloc[0])
    letter = {0: "A", 1: "B", 2: "C"}.get(g, "X")
    os.makedirs(RESULTS_TABLES, exist_ok=True)
    out = pd.DataFrame(
        {
            "ID": df_sub["ID"].values,
            "baseline_group": g,
            "Baseline_Cohort": f"Cohort {letter}",
            "tau_hat": np.asarray(ite, dtype=float).ravel(),
            "treatment_col": treatment_col,
            "outcome_col": outcome_col,
        }
    )
    if "wave" in df_sub.columns:
        out["wave"] = df_sub["wave"].values
    path = os.path.join(RESULTS_TABLES, f"ite_xlearner_{treatment_col}_cohort_{letter}.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("ITE 导出: %s (n=%s)", path, len(out))

def cleanup_temp_cat_dirs():
    """清理历史遗留的 temp_cat_* 临时目录（CatBoost 旧版，可单独调用）"""
    import glob
    cwd = os.getcwd()
    removed = 0
    for d in glob.glob(os.path.join(cwd, 'temp_cat_*')):
        if os.path.isdir(d):
            for _ in range(3):
                try:
                    shutil.rmtree(d)
                    removed += 1
                    break
                except (OSError, PermissionError) as e:
                    time.sleep(1)
            else:
                logger.debug(f"跳过无法删除: {d}")
    if removed > 0:
        logger.info(f"已清理 {removed} 个历史临时目录")
    return removed
def estimate_causal_impact(df_sub, treatment_col='exercise', output_dir='causal_results'):
    """
    【通用因果验证版】直接对传入的子集进行因果评估，不进行二次过滤。

    因果时序（P0）：数据为 person-wave 结构。每行 = 个体在 Wave(t)。
    - 暴露 T（如 exercise）、协变量 X：取自 Wave(t) 或更早。
    - 结局 Y（is_comorbidity_next）：取自 Wave(t+1)。
    满足前瞻性因果推断要求。
    """
    logger.info(f"\n>>> [因果验证] 正在针对干预变量 [{treatment_col}] 进行因果归因 (n={len(df_sub)})...")
    os.makedirs(output_dir, exist_ok=True)
    
    Y = 'is_comorbidity_next'
    T = treatment_col
    
    # 检查 T 是否存在于数据集中
    if T not in df_sub.columns:
        logger.warning(f"⚠️ 处理变量 {T} 不在数据集中，跳过因果分析。")
        return None, CAUSAL_FAILURE_ATE_TRIPLET
    # 防护：Y 缺失会导致 DML 异常，显式剔除
    if df_sub[Y].isna().any():
        n_before = len(df_sub)
        df_sub = df_sub.dropna(subset=[Y]).copy()
        logger.info(f"剔除 Y 缺失行: {n_before} -> {len(df_sub)}")

    # 1. 彻底排除泄露与干扰（与 charls_feature_lists 集中定义一致）
    exclude_cols = get_exclude_cols(df_sub, target_col=Y, treatment_col=T)
    W_cols = [c for c in df_sub.columns if c not in exclude_cols]
    X_raw = df_sub[W_cols].select_dtypes(include=[np.number])
    if X_raw.shape[1] == 0 or len(X_raw) < 10:
        logger.warning(f"⚠️ 有效特征数为 0 或样本过少 (n={len(X_raw)})，跳过因果分析。")
        return None, CAUSAL_FAILURE_ATE_TRIPLET

    # 聚类/群组 ID：若无 communityID 则用个体 ID
    if 'communityID' in df_sub.columns:
        cluster_ids = df_sub['communityID'].fillna(df_sub['ID']).astype(str)
    else:
        cluster_ids = df_sub['ID'].astype(str)

    # 2. 预处理：Imputer/Scaler 与 compare_models 一致，仅在开发集训练子集上 fit（见 utils.charls_train_only_preprocessing）
    X_scaled = fit_transform_numeric_train_only(df_sub, X_raw, y_col=Y)
    
    T_series = df_sub[T].fillna(0).astype(int) # 确保干预变量是 0/1
    Y_series = df_sub[Y].astype(float)

    # 3. 拟合因果森林 (Causal Forest DML)，论文2.4/2.8节：1000棵决策树、五折交叉拟合
    # Nuisance 模型：RF + min_samples_leaf 增强正则化，防止 CHARLS 高维协变量过拟合
    dml = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=15, random_state=RANDOM_SEED, n_jobs=-1),
        model_t=RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=15, random_state=RANDOM_SEED, n_jobs=-1),
        discrete_treatment=True, cv=5, n_estimators=1000, random_state=RANDOM_SEED,
        honest=True  # 诚实估计：生长与叶节点估计用不同子样本，减少过拟合
    )

    try:
        dml.fit(Y=Y_series, T=T_series, X=X_scaled, groups=cluster_ids)
        ite = dml.effect(X_scaled)
        ate = ite.mean()
        ate_lb, ate_ub = dml.ate_interval(X_scaled)

        logger.info(f"🏆 干预 [{T}] 的平均因果效应 (ATE): {ate:.4f} (95% CI: {ate_lb:.4f}, {ate_ub:.4f})")

        # 保存结果
        df_sub[f'causal_impact_{T}'] = ite
        out_file = os.path.join(output_dir, f'CAUSAL_ANALYSIS_{T}.csv')
        df_sub.to_csv(out_file, index=False, encoding='utf-8-sig')

        # ITE 分布直方图：观察因果效应异质性
        try:
            plt.figure(figsize=(8, 5))
            plt.hist(ite, bins=40, color='steelblue', alpha=0.8, edgecolor='white')
            plt.axvline(ate, color='red', linestyle='--', linewidth=2, label=f'ATE = {ate:.4f}')
            plt.axvline(0, color='gray', linestyle='-', alpha=0.5)
            plt.xlabel('Individual Treatment Effect (ITE)')
            plt.ylabel('Frequency')
            plt.title(f'ITE Distribution: {T} → {Y}')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'fig_ite_distribution_{T}.png'), dpi=300, bbox_inches=BBOX_INCHES)
            plt.close()
            logger.info(f"ITE 分布直方图已保存: fig_ite_distribution_{T}.png")
        except Exception as ex:
            logger.debug(f"ITE 直方图跳过: {ex}")
        # 保存 ATE 与 95% CI 汇总，便于论文报告与敏感性表格使用
        with open(os.path.join(output_dir, f'ATE_CI_summary_{T}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Treatment: {T}\n")
            f.write(f"ATE (point estimate): {ate:.4f}\n")
            f.write(f"95% CI: ({ate_lb:.4f}, {ate_ub:.4f})\n")
            f.write(f"N: {len(df_sub)}\n")

        # 重叠假设（Overlap）：倾向评分分布图 + Trimming 报告
        try:
            ps_model = LogisticRegression(max_iter=5000, C=1e-2, solver='lbfgs', random_state=RANDOM_SEED)
            ps_model.fit(X_scaled, T_series)
            ps = ps_model.predict_proba(X_scaled)[:, 1]
            # Trimming：PS 极端值样本（重叠差），界定结论适用范围
            n_before = len(T_series)
            trim_lo, trim_hi = 0.05, 0.95
            in_support = (ps >= trim_lo) & (ps <= trim_hi)
            n_trimmed = n_before - in_support.sum()
            pct_trimmed = 100 * n_trimmed / n_before if n_before > 0 else 0
            with open(os.path.join(output_dir, f'ATE_CI_summary_{T}.txt'), 'a', encoding='utf-8') as f:
                f.write(f"Overlap trimming (PS not in [{trim_lo},{trim_hi}]): {n_trimmed} excluded ({pct_trimmed:.1f}%)\n")
            logger.info(f"重叠假设：{n_trimmed} 样本 PS 超出 [{trim_lo},{trim_hi}] 已标注 ({pct_trimmed:.1f}%)")
            plt.figure(figsize=(8, 5))
            mask0, mask1 = (T_series == 0), (T_series == 1)
            if mask0.sum() > 5 and mask1.sum() > 5:
                plt.hist(ps[mask0], bins=25, alpha=0.5, color='#d9534f', label='Control', density=True, edgecolor='black')
                plt.hist(ps[mask1], bins=25, alpha=0.5, color='#5cb85c', label='Treated', density=True, edgecolor='black')
            plt.axvline(trim_lo, color='gray', linestyle='--', alpha=0.7, label=f'Trim boundary')
            plt.axvline(trim_hi, color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('Propensity Score')
            plt.ylabel('Density')
            plt.title(f'Propensity Score Overlap ({T}): Positivity Assumption Check')
            plt.legend()
            plt.xlim(0, 1)
            plt.savefig(os.path.join(output_dir, f'fig_propensity_overlap_{T}.png'), dpi=300, bbox_inches=BBOX_INCHES)
            plt.close()
            logger.info(f"倾向评分重叠图已保存: fig_propensity_overlap_{T}.png")
        except Exception as ex:
            logger.debug(f"倾向评分图跳过: {ex}")

        # E-value：未测量混杂敏感性（VanderWeele & Ding 2017），二手数据必报项
        try:
            r0 = Y_series[T_series == 0].mean()
            r0 = np.clip(r0, 1e-6, 1 - 1e-6)
            # 基于 ATE 与 95% CI 下限计算 E-value（保守估计）
            def _evalue_from_rr(rr):
                if rr <= 1:
                    rr = 1 / (rr + 1e-9)
                return rr + np.sqrt(rr * (rr - 1))
            rr_point = (r0 + ate) / r0
            # VanderWeele & Ding 2017: 保守端取 CI 中离 null 最近的一端
            rr_conservative = (r0 + ate_lb) / r0 if ate >= 0 else (r0 + ate_ub) / r0
            e_point = _evalue_from_rr(rr_point)
            e_conservative = _evalue_from_rr(rr_conservative)
            with open(os.path.join(output_dir, f'ATE_CI_summary_{T}.txt'), 'a', encoding='utf-8') as f:
                f.write(f"E-Value (point): {e_point:.2f}\n")
                f.write(f"E-Value (conservative, from 95%% CI): {e_conservative:.2f}\n")
                f.write(f"  (Unmeasured confounder would need RR≥E with both T and Y to explain away the effect)\n")
            logger.info(f"E-Value: {e_point:.2f} (point), {e_conservative:.2f} (conservative)")
        except Exception as ex:
            logger.debug(f"E-value 计算跳过: {ex}")

        return df_sub, (ate, ate_lb, ate_ub)
    except Exception as e:
        logger.error(f"因果引擎失败: {e}")
        return None, CAUSAL_FAILURE_ATE_TRIPLET


def estimate_causal_impact_tlearner(df_sub, treatment_col='exercise', output_dir='causal_results'):
    """
    TLearner 因果估计 + PSM/PSW 验证，与 estimate_causal_impact 接口一致。
    返回 (df_sub, (ate, ate_lb, ate_ub))，df_sub 含 causal_impact_{T} 列。
    """
    from sklearn.ensemble import RandomForestRegressor
    from econml.metalearners import TLearner

    logger.info(f"\n>>> [因果验证 TLearner] 正在针对干预变量 [{treatment_col}] 进行因果归因 (n={len(df_sub)})...")
    os.makedirs(output_dir, exist_ok=True)
    Y = 'is_comorbidity_next'
    T = treatment_col

    if T not in df_sub.columns:
        logger.warning(f"⚠️ 处理变量 {T} 不在数据集中，跳过因果分析。")
        return None, CAUSAL_FAILURE_ATE_TRIPLET
    if df_sub[Y].isna().any():
        df_sub = df_sub.dropna(subset=[Y]).copy()

    exclude_cols = get_exclude_cols(df_sub, target_col=Y, treatment_col=T)
    W_cols = [c for c in df_sub.columns if c not in exclude_cols]
    X_raw = df_sub[W_cols].select_dtypes(include=[np.number])
    # 选择偏倚：增加 exercise×adlab_c 交互项（Hernán 2020），基线 ADL 困难者无法运动
    if T == 'exercise' and 'adlab_c' in X_raw.columns:
        X_raw = X_raw.copy()
        X_raw['exercise_x_adl'] = df_sub[T].fillna(0).values * X_raw['adlab_c'].fillna(0).values
    if X_raw.shape[1] == 0 or len(X_raw) < 10:
        logger.warning(f"⚠️ 有效特征数为 0 或样本过少，跳过因果分析。")
        return None, CAUSAL_FAILURE_ATE_TRIPLET

    X_scaled = fit_transform_numeric_train_only(df_sub, X_raw, y_col=Y)

    T_series = df_sub[T].fillna(0).astype(int)
    Y_series = df_sub[Y].astype(float)
    X_arr = np.asarray(X_scaled, dtype=np.float64)

    # ---------- 步骤 1：修剪前 overlap 检验 ----------
    from causal.charls_causal_assumption_checks import check_overlap
    pre_trim_overlap = check_overlap(df_sub, T, output_dir, target_col=Y, suffix='_pre_trim')

    # ---------- 步骤 2：Overlap 修剪 (Austin 2011)，区间见 config.PS_TRIM_* ----------
    try:
        from config import PS_TRIM_LOW as _PS_LO, PS_TRIM_HIGH as _PS_HI
    except ImportError:
        _PS_LO, _PS_HI = 0.05, 0.95
    df_sub, X_scaled, T_series, Y_series, _trim_info = _apply_ps_overlap_trim(
        df_sub, X_scaled, T_series, Y_series, _PS_LO, _PS_HI, force_trim=False
    )
    X_arr = np.asarray(X_scaled, dtype=np.float64)
    overlap_trimmed_pct = _trim_info["overlap_trimmed_pct"]

    try:
        est = TLearner(models=(
            RandomForestRegressor(n_estimators=200, max_depth=4, random_state=RANDOM_SEED, n_jobs=-1),
            RandomForestRegressor(n_estimators=200, max_depth=4, random_state=RANDOM_SEED + 1, n_jobs=-1)
        ))
        est.fit(Y=Y_series.values, T=T_series.values, X=X_arr)
        ite = est.effect(X_arr)
        ate = float(np.mean(ite))

        rng = np.random.RandomState(RANDOM_SEED)
        if 'ID' in df_sub.columns:
            id_arr = df_sub['ID'].to_numpy()
            n_clust = len(np.unique(id_arr))
        else:
            id_arr, n_clust = None, None
            logger.warning("TLearner: 无 ID 列，ATE bootstrap 退化为行级有放回抽样")
        ates_boot = []
        boot_fail = 0
        boot_skip_short = 0
        for b in range(200):
            if id_arr is not None:
                idx = cluster_bootstrap_indices_once(id_arr, rng, n_clusters_draw=n_clust)
            else:
                idx = rng.choice(len(Y_series), size=len(Y_series), replace=True)
            if len(idx) < 5:
                boot_skip_short += 1
                boot_fail += 1
                continue
            try:
                est_b = TLearner(models=(
                    RandomForestRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_SEED + b, n_jobs=-1),
                    RandomForestRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_SEED + b + 1, n_jobs=-1)
                ))
                est_b.fit(Y=Y_series.values[idx], T=T_series.values[idx], X=X_arr[idx])
                ates_boot.append(float(np.mean(est_b.effect(X_arr))))
            except Exception as ex:
                boot_fail += 1
                logger.debug("TLearner ATE bootstrap replicate %s failed: %s", b, ex)
        ate_lb = float(np.percentile(ates_boot, 2.5)) if len(ates_boot) >= 20 else np.nan
        ate_ub = float(np.percentile(ates_boot, 97.5)) if len(ates_boot) >= 20 else np.nan
        msg = (
            f"TLearner [{T}] cluster bootstrap (ATE): planned=200, successful_draws={len(ates_boot)}, "
            f"failed_or_skipped={boot_fail} (includes idx_len<5: {boot_skip_short}); "
            f"CI percentile requires >=20 successful draws (got {len(ates_boot)})."
        )
        if boot_fail > 0:
            logger.warning(msg)
        else:
            logger.info(msg)
        print(msg, flush=True)

        logger.info(f"🏆 干预 [{T}] TLearner ATE: {ate:.4f} (95% CI: {ate_lb:.4f}, {ate_ub:.4f})")

        df_sub = df_sub.copy()
        df_sub[f'causal_impact_{T}'] = ite
        df_sub.to_csv(os.path.join(output_dir, f'CAUSAL_ANALYSIS_{T}.csv'), index=False, encoding='utf-8-sig')

        with open(os.path.join(output_dir, f'ATE_CI_summary_{T}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Treatment: {T} (TLearner)\n")
            f.write(f"ATE: {ate:.4f}, 95% CI: ({ate_lb:.4f}, {ate_ub:.4f})\n")
            f.write(f"N: {len(df_sub)}\n")
            f.write(f"overlap_trimmed_pct: {overlap_trimmed_pct:.2f}\n")

        try:
            plt.figure(figsize=(8, 5))
            plt.hist(ite, bins=40, color='steelblue', alpha=0.8, edgecolor='white')
            plt.axvline(ate, color='red', linestyle='--', linewidth=2, label=f'ATE = {ate:.4f}')
            plt.axvline(0, color='gray', linestyle='-', alpha=0.5)
            plt.xlabel('Individual Treatment Effect (ITE)')
            plt.ylabel('Frequency')
            plt.title(f'ITE Distribution (TLearner): {T} → {Y}')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'fig_ite_distribution_{T}.png'), dpi=300, bbox_inches=BBOX_INCHES)
            plt.close()
        except Exception:
            pass

        from causal.charls_causal_assumption_checks import run_all_assumption_checks
        from causal.charls_causal_methods_comparison import run_causal_methods_comparison

        run_all_assumption_checks(df_sub, T, output_dir, ate=ate, ate_lb=ate_lb, ate_ub=ate_ub,
                                  pre_trim_overlap=pre_trim_overlap)
        run_causal_methods_comparison(df_sub, T, output_dir, dml_ate=ate, dml_lb=ate_lb, dml_ub=ate_ub, ml_method_name='TLearner')

        return df_sub, (ate, ate_lb, ate_ub)
    except Exception as e:
        logger.error(f"TLearner 因果引擎失败: {e}")
        return None, CAUSAL_FAILURE_ATE_TRIPLET


def estimate_causal_impact_xlearner(
    df_sub,
    treatment_col='exercise',
    output_dir='causal_results',
    *,
    outcome_col=None,
    ps_trim_low=None,
    ps_trim_high=None,
    trim_force: bool = False,
    run_auxiliary_steps: bool = True,
    bootstrap_replicates: int = 200,
    bootstrap_min_for_ci: int = 20,
):
    """
    XLearner 因果估计（Künzel et al.）

    流程：1) 修剪前 overlap 检验 2) PS 修剪 3) XLearner 拟合 4) 假设检验 5) PSM/PSW 对比

    outcome_col：结局列；默认 config.TARGET_COL。阴性对照等敏感性可传入如 is_fall_next。
    ps_trim_low / ps_trim_high：默认读 config.PS_TRIM_LOW/HIGH。
    trim_force：True 时凡 PS 在区间外即剔除；False 时与主流程历史一致（区间内占比≥90% 则不剔除）。
    run_auxiliary_steps：False 时仅拟合与返回 ATE/CI，不写图、不写 PSM/PSW、不写 assumption（供修剪敏感性批量调用）。
    bootstrap_replicates：ate_interval 不可用时的聚类 bootstrap 次数；主分析保持 200，批量敏感性可酌减以省时间。
    bootstrap_min_for_ci：成功 bootstrap 次数达到该值才计算 ATE 的 2.5/97.5% 分位 CI。

    返回: (df_sub, (ate, ate_lb, ate_ub))，df_sub 含 causal_impact_{T} 列（auxiliary 时）。
    """
    from sklearn.ensemble import RandomForestRegressor
    from econml.metalearners import XLearner

    try:
        from config import TARGET_COL as _TARGET_COL
    except ImportError:
        _TARGET_COL = 'is_comorbidity_next'
    Y = outcome_col if outcome_col is not None else _TARGET_COL
    T = treatment_col

    logger.info(
        f"\n>>> [因果验证 XLearner] 干预 [{treatment_col}] → 结局 [{Y}] (n={len(df_sub)})..."
    )
    os.makedirs(output_dir, exist_ok=True)

    if T not in df_sub.columns:
        logger.warning(f"⚠️ 处理变量 {T} 不在数据集中，跳过因果分析。")
        return None, CAUSAL_FAILURE_ATE_TRIPLET
    if Y not in df_sub.columns:
        logger.warning(f"⚠️ 结局列 {Y} 不在数据集中，跳过因果分析。")
        return None, CAUSAL_FAILURE_ATE_TRIPLET
    if df_sub[Y].isna().any():
        df_sub = df_sub.dropna(subset=[Y]).copy()

    exclude_cols = get_exclude_cols(df_sub, target_col=Y, treatment_col=T)
    W_cols = [c for c in df_sub.columns if c not in exclude_cols]
    X_raw = df_sub[W_cols].select_dtypes(include=[np.number])
    if T == 'exercise' and 'adlab_c' in X_raw.columns:
        X_raw = X_raw.copy()
        X_raw['exercise_x_adl'] = df_sub[T].fillna(0).values * X_raw['adlab_c'].fillna(0).values
    if X_raw.shape[1] == 0 or len(X_raw) < 10:
        logger.warning(f"⚠️ 有效特征数为 0 或样本过少，跳过因果分析。")
        return None, CAUSAL_FAILURE_ATE_TRIPLET

    X_scaled = fit_transform_numeric_train_only(df_sub, X_raw, y_col=Y)

    T_series = df_sub[T].fillna(0).astype(int)
    Y_series = df_sub[Y].astype(float)
    X_arr = np.asarray(X_scaled, dtype=np.float64)

    # ---------- 步骤 1：修剪前 overlap 检验 ----------
    from causal.charls_causal_assumption_checks import check_overlap

    pre_trim_overlap = None
    if run_auxiliary_steps:
        pre_trim_overlap = check_overlap(df_sub, T, output_dir, target_col=Y, suffix='_pre_trim')

    # ---------- 步骤 2：Overlap 修剪（config.PS_TRIM_* 或参数覆盖）----------
    try:
        from config import PS_TRIM_LOW as _PSL, PS_TRIM_HIGH as _PSH
    except ImportError:
        _PSL, _PSH = 0.05, 0.95
    tlo = float(ps_trim_low) if ps_trim_low is not None else float(_PSL)
    thi = float(ps_trim_high) if ps_trim_high is not None else float(_PSH)
    df_sub, X_scaled, T_series, Y_series, trim_info = _apply_ps_overlap_trim(
        df_sub, X_scaled, T_series, Y_series, tlo, thi, force_trim=trim_force
    )
    X_arr = np.asarray(X_scaled, dtype=np.float64)
    overlap_trimmed_pct = trim_info["overlap_trimmed_pct"]
    n_before_trim = trim_info["n_before_trim"]
    n_after_trim = trim_info["n_after_trim"]

    # ---------- 步骤 3：XLearner 因果估计 ----------
    try:
        est = XLearner(models=(
            RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=15, random_state=RANDOM_SEED, n_jobs=-1),
            RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=15, random_state=RANDOM_SEED + 1, n_jobs=-1)
        ))
        est.fit(Y=Y_series.values, T=T_series.values, X=X_arr)
        ite = est.effect(X_arr)
        ate = float(np.mean(ite))
        # ATT：仅在真实接受干预者上平均 ITE（与 PSM/ATT 叙述对齐；ATE 为全样本平均）
        tmask = T_series.values == 1
        att = float(np.mean(ite[tmask])) if np.any(tmask) else np.nan

        # XLearner 可能有 ate_interval，优先使用；否则 Bootstrap
        ate_lb, ate_ub = np.nan, np.nan
        ate_ci_source = "unset"
        bootstrap_ate_successful_draws = np.nan
        if hasattr(est, 'ate_interval'):
            try:
                lb, ub = est.ate_interval(X=X_arr, alpha=0.05)
                ate_lb = float(lb) if np.isscalar(lb) else float(lb[0])
                ate_ub = float(ub) if np.isscalar(ub) else float(ub[0])
                if not (np.isnan(ate_lb) or np.isnan(ate_ub)):
                    ate_ci_source = "econml_ate_interval"
            except Exception as ex:
                logger.debug("XLearner ate_interval failed, will use bootstrap: %s", ex)
        att_lb, att_ub = np.nan, np.nan
        if np.isnan(ate_lb) or np.isnan(ate_ub):
            rng = np.random.RandomState(RANDOM_SEED)
            n_boot_plan = max(1, int(bootstrap_replicates))
            n_boot_min = max(1, int(bootstrap_min_for_ci))
            if 'ID' in df_sub.columns:
                id_arr = df_sub['ID'].to_numpy()
                n_clust = len(np.unique(id_arr))
            else:
                id_arr, n_clust = None, None
                logger.warning("XLearner: 无 ID 列，ATE/ATT bootstrap 退化为行级有放回抽样")
            ates_boot = []
            atts_boot = []
            boot_fail = 0
            boot_skip_short = 0
            for b in range(n_boot_plan):
                if id_arr is not None:
                    idx = cluster_bootstrap_indices_once(id_arr, rng, n_clusters_draw=n_clust)
                else:
                    idx = rng.choice(len(Y_series), size=len(Y_series), replace=True)
                if len(idx) < 5:
                    boot_skip_short += 1
                    boot_fail += 1
                    continue
                try:
                    est_b = XLearner(models=(
                        RandomForestRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_SEED + b, n_jobs=-1),
                        RandomForestRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_SEED + b + 1, n_jobs=-1)
                    ))
                    est_b.fit(Y=Y_series.values[idx], T=T_series.values[idx], X=X_arr[idx])
                    ite_b = est_b.effect(X_arr)
                    ates_boot.append(float(np.mean(ite_b)))
                    tm = T_series.values == 1
                    if np.any(tm):
                        atts_boot.append(float(np.mean(ite_b[tm])))
                except Exception as ex:
                    boot_fail += 1
                    logger.debug("XLearner ATE bootstrap replicate %s failed: %s", b, ex)
            msg = (
                f"XLearner [{T}] cluster bootstrap (ATE/ATT): planned={n_boot_plan}, successful_ATE_draws={len(ates_boot)}, "
                f"successful_ATT_draws={len(atts_boot)}, failed_or_skipped={boot_fail} (idx_len<5: {boot_skip_short}); "
                f"ATE CI percentiles require >={n_boot_min} successful ATE draws (got {len(ates_boot)})."
            )
            if boot_fail > 0:
                logger.warning(msg)
            else:
                logger.info(msg)
            print(msg, flush=True)
            bootstrap_ate_successful_draws = int(len(ates_boot))
            if len(ates_boot) >= n_boot_min:
                ate_lb = float(np.percentile(ates_boot, 2.5))
                ate_ub = float(np.percentile(ates_boot, 97.5))
                ate_ci_source = "cluster_bootstrap_percentile"
            else:
                ate_ci_source = "cluster_bootstrap_insufficient"
            if len(atts_boot) >= n_boot_min:
                att_lb = float(np.percentile(atts_boot, 2.5))
                att_ub = float(np.percentile(atts_boot, 97.5))
        else:
            if ate_ci_source == "unset":
                ate_ci_source = "econml_ate_interval"
            inf_msg = f"XLearner [{T}] ATE 95% CI from econml ate_interval (no cluster bootstrap)."
            logger.info(inf_msg)
            print(inf_msg, flush=True)

        logger.info(
            f"🏆 干预 [{T}] XLearner ATE: {ate:.4f} (95% CI: {ate_lb:.4f}, {ate_ub:.4f}); "
            f"ATT (treated-only mean ITE): {att:.4f} (95% CI: {att_lb:.4f}, {att_ub:.4f})"
        )

        df_sub = df_sub.copy()
        df_sub[f'causal_impact_{T}'] = ite

        if run_auxiliary_steps:
            df_sub.to_csv(os.path.join(output_dir, f'CAUSAL_ANALYSIS_{T}.csv'), index=False, encoding='utf-8-sig')

            with open(os.path.join(output_dir, f'ATE_CI_summary_{T}.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Treatment: {T} (XLearner)\n")
                f.write(f"PS trim bounds: [{tlo:.4f}, {thi:.4f}], trim_force={trim_force}\n")
                f.write(f"N_before_trim: {n_before_trim}\n")
                f.write(f"N_after_trim (analysis): {n_after_trim}\n")
                f.write(f"ATE (mean ITE over full sample): {ate:.4f}, 95% CI: ({ate_lb:.4f}, {ate_ub:.4f})\n")
                f.write(
                    f"ATT (mean ITE among treated T=1, comparable in spirit to matching estimands): "
                    f"{att:.4f}, 95% CI: ({att_lb:.4f}, {att_ub:.4f})\n"
                )
                f.write(f"N: {len(df_sub)}\n")
                f.write(f"overlap_trimmed_pct (pre-subset): {overlap_trimmed_pct:.2f}\n")

            _write_ite_export_csv(df_sub, ite, T, Y)

            try:
                plt.figure(figsize=(8, 5))
                plt.hist(ite, bins=40, color='steelblue', alpha=0.8, edgecolor='white')
                plt.axvline(ate, color='red', linestyle='--', linewidth=2, label=f'ATE = {ate:.4f}')
                plt.axvline(0, color='gray', linestyle='-', alpha=0.5)
                plt.xlabel('Individual Treatment Effect (ITE)')
                plt.ylabel('Frequency')
                plt.title(f'ITE Distribution (XLearner): {T} → {Y}')
                plt.legend()
                plt.savefig(os.path.join(output_dir, f'fig_ite_distribution_{T}.png'), dpi=300, bbox_inches=BBOX_INCHES)
                plt.close()
            except Exception:
                pass

            # ---------- 步骤 4：假设检验（重叠、平衡、E-value）----------
            from causal.charls_causal_assumption_checks import run_all_assumption_checks
            from causal.charls_causal_methods_comparison import run_causal_methods_comparison

            run_all_assumption_checks(df_sub, T, output_dir, ate=ate, ate_lb=ate_lb, ate_ub=ate_ub,
                                      pre_trim_overlap=pre_trim_overlap)

            # ---------- 步骤 5：PSM/PSW 方法对比 ----------
            run_causal_methods_comparison(df_sub, T, output_dir, dml_ate=ate, dml_lb=ate_lb, dml_ub=ate_ub, ml_method_name='XLearner')

        try:
            df_sub.attrs["ps_trim_info"] = {
                "trim_lo": tlo,
                "trim_hi": thi,
                "trim_force": trim_force,
                "n_before_trim": n_before_trim,
                "n_after_trim": n_after_trim,
                "overlap_trimmed_pct": overlap_trimmed_pct,
                "pct_in_band_before_subset": trim_info["pct_in_band_before_subset"],
                "applied_subset": trim_info["applied_subset"],
                "bootstrap_replicates_planned": int(bootstrap_replicates),
                "bootstrap_min_for_ci": int(bootstrap_min_for_ci),
                "ate_ci_source": ate_ci_source,
                "bootstrap_ate_successful_draws": bootstrap_ate_successful_draws,
            }
        except Exception:
            pass
        return df_sub, (ate, ate_lb, ate_ub)
    except Exception as e:
        logger.error(f"XLearner 因果引擎失败: {e}")
        return None, CAUSAL_FAILURE_ATE_TRIPLET


def estimate_causal_impact_dml_sensitivity(
    df_sub,
    treatment_col='exercise',
    *,
    outcome_col=None,
    ps_trim_low=None,
    ps_trim_high=None,
    trim_force: bool = False,
):
    """
    与 XLearner 主流程一致的 PS 重叠修剪 + CausalForestDML，仅返回 ATE/CI（不写 auxiliary 文件）。
    用于 Cohort B 上「估计器鲁棒性」与 XLearner 并列对比。
    """
    try:
        from config import TARGET_COL as _TARGET_COL, PS_TRIM_LOW as _PSL, PS_TRIM_HIGH as _PSH
    except ImportError:
        _TARGET_COL, _PSL, _PSH = 'is_comorbidity_next', 0.05, 0.95
    Y = outcome_col if outcome_col is not None else _TARGET_COL
    T = treatment_col
    if T not in df_sub.columns or Y not in df_sub.columns:
        return CAUSAL_FAILURE_ATE_TRIPLET, {}
    df_work = df_sub.copy()
    if df_work[Y].isna().any():
        df_work = df_work.dropna(subset=[Y]).copy()

    exclude_cols = get_exclude_cols(df_work, target_col=Y, treatment_col=T)
    W_cols = [c for c in df_work.columns if c not in exclude_cols]
    X_raw = df_work[W_cols].select_dtypes(include=[np.number])
    if T == 'exercise' and 'adlab_c' in X_raw.columns:
        X_raw = X_raw.copy()
        X_raw['exercise_x_adl'] = df_work[T].fillna(0).values * X_raw['adlab_c'].fillna(0).values
    if X_raw.shape[1] == 0 or len(X_raw) < 10:
        return CAUSAL_FAILURE_ATE_TRIPLET, {}

    X_scaled = fit_transform_numeric_train_only(df_work, X_raw, y_col=Y)
    T_series = df_work[T].fillna(0).astype(int)
    Y_series = df_work[Y].astype(float)
    tlo = float(ps_trim_low) if ps_trim_low is not None else float(_PSL)
    thi = float(ps_trim_high) if ps_trim_high is not None else float(_PSH)
    df_work, X_scaled, T_series, Y_series, trim_info = _apply_ps_overlap_trim(
        df_work, X_scaled, T_series, Y_series, tlo, thi, force_trim=trim_force
    )
    trim_info = dict(trim_info)
    if 'communityID' in df_work.columns:
        cluster_ids = df_work['communityID'].fillna(df_work['ID']).astype(str)
    else:
        cluster_ids = df_work['ID'].astype(str)

    try:
        dml = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=200, max_depth=4, min_samples_leaf=15, random_state=RANDOM_SEED, n_jobs=-1
            ),
            model_t=RandomForestClassifier(
                n_estimators=200, max_depth=4, min_samples_leaf=15, random_state=RANDOM_SEED, n_jobs=-1
            ),
            discrete_treatment=True,
            cv=5,
            n_estimators=1000,
            random_state=RANDOM_SEED,
            honest=True,
        )
        dml.fit(Y=Y_series, T=T_series, X=X_scaled, groups=cluster_ids)
        ite = dml.effect(X_scaled)
        ate = float(np.mean(ite))
        lb, ub = dml.ate_interval(X_scaled)
        ate_lb = float(lb) if np.isscalar(lb) else float(np.asarray(lb).ravel()[0])
        ate_ub = float(ub) if np.isscalar(ub) else float(np.asarray(ub).ravel()[0])
        trim_info['method'] = 'CausalForestDML'
        return (ate, ate_lb, ate_ub), trim_info
    except Exception as e:
        logger.error("CausalForestDML 敏感性估计失败: %s", e)
        return CAUSAL_FAILURE_ATE_TRIPLET, trim_info


def run_negative_control_outcome_summary(df_clean):
    """
    三队列阴性对照结局：与主分析相同的 XLearner + PS 修剪设置，结局换为 NEGATIVE_CONTROL_OUTCOME_COL。
    输出 results/tables/negative_control_results.csv（不写各队列 03_causal 主文件）。
    """
    import tempfile

    try:
        from config import (
            NEGATIVE_CONTROL_OUTCOME_COL,
            RESULTS_TABLES,
            TREATMENT_COL,
        )
    except ImportError:
        NEGATIVE_CONTROL_OUTCOME_COL = 'is_fall_next'
        RESULTS_TABLES = 'results/tables'
        TREATMENT_COL = 'exercise'

    rows = []
    for bg, letter in [(0, 'A'), (1, 'B'), (2, 'C')]:
        df_sub = df_clean[df_clean['baseline_group'] == bg].copy()
        if len(df_sub) < 20:
            rows.append({
                'cohort': letter,
                'negative_control_outcome': NEGATIVE_CONTROL_OUTCOME_COL,
                'treatment': TREATMENT_COL,
                'n_after_trim': np.nan,
                'ate': np.nan,
                'ate_lb': np.nan,
                'ate_ub': np.nan,
                'p_two_sided_approx': np.nan,
                'ate_ci_source': 'skipped_small_n',
            })
            continue
        with tempfile.TemporaryDirectory() as td:
            res_df, trip = estimate_causal_impact_xlearner(
                df_sub,
                treatment_col=TREATMENT_COL,
                output_dir=td,
                outcome_col=NEGATIVE_CONTROL_OUTCOME_COL,
                run_auxiliary_steps=False,
            )
        ate, lb, ub = trip
        ti = res_df.attrs.get('ps_trim_info', {}) if res_df is not None else {}
        p = _p_value_two_sided_from_ci(ate, lb, ub)
        rows.append({
            'cohort': letter,
            'negative_control_outcome': NEGATIVE_CONTROL_OUTCOME_COL,
            'treatment': TREATMENT_COL,
            'n_after_trim': ti.get('n_after_trim', np.nan),
            'n_before_trim': ti.get('n_before_trim', np.nan),
            'ate': ate,
            'ate_lb': lb,
            'ate_ub': ub,
            'p_two_sided_approx': p,
            'ate_ci_source': ti.get('ate_ci_source', ''),
        })
    os.makedirs(RESULTS_TABLES, exist_ok=True)
    out_path = os.path.join(RESULTS_TABLES, 'negative_control_results.csv')
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')
    logger.info("阴性对照结局汇总已写入: %s", out_path)

    # PSM/PSW 平行估计（与稿件 Table S6b / negative_control_psm_psw.csv 一致）
    try:
        from causal.charls_causal_methods_comparison import run_causal_methods_comparison

        prows = []
        for bg, letter in [(0, 'A'), (1, 'B'), (2, 'C')]:
            df_sub = df_clean[df_clean['baseline_group'] == bg].copy()
            df_sub = df_sub.dropna(subset=[NEGATIVE_CONTROL_OUTCOME_COL, TREATMENT_COL])
            if len(df_sub) < 50:
                continue
            with tempfile.TemporaryDirectory() as td:
                res = run_causal_methods_comparison(
                    df_sub,
                    TREATMENT_COL,
                    td,
                    target_col=NEGATIVE_CONTROL_OUTCOME_COL,
                    dml_ate=None,
                    dml_lb=None,
                    dml_ub=None,
                    ml_method_name='XLearner',
                )
            for _, r in res[res['Method'].isin(['PSM', 'PSW'])].iterrows():
                lb, ub = float(r['ATE_lb']), float(r['ATE_ub'])
                prows.append({
                    'cohort': letter,
                    'method': r['Method'],
                    'ate': r['ATE'],
                    'ate_lb': r['ATE_lb'],
                    'ate_ub': r['ATE_ub'],
                    'ci_includes_null': 'Yes' if (lb < 0 < ub) else 'No',
                })
        if prows:
            ps_path = os.path.join(RESULTS_TABLES, 'negative_control_psm_psw.csv')
            pd.DataFrame(prows).to_csv(ps_path, index=False, encoding='utf-8-sig')
            logger.info("阴性对照 PSM/PSW 已写入: %s", ps_path)
    except Exception as ex:
        logger.warning("阴性对照 PSM/PSW 导出跳过: %s", ex)


def run_ate_method_sensitivity_cohort_b(df_clean):
    """
    Cohort B 主结局：XLearner 与 CausalForestDML（PS 修剪与 XLearner 一致）并列写入 ate_method_sensitivity.csv。
    """
    import tempfile

    try:
        from config import RESULTS_TABLES, TARGET_COL, TREATMENT_COL
    except ImportError:
        RESULTS_TABLES = 'results/tables'
        TARGET_COL = 'is_comorbidity_next'
        TREATMENT_COL = 'exercise'

    df_b = df_clean[df_clean['baseline_group'] == 1].copy()
    rows = []
    if len(df_b) < 20:
        logger.warning("Cohort B 样本过少，跳过 ate_method_sensitivity 估计")
        for m in ('XLearner', 'CausalForestDML'):
            rows.append({
                'cohort': 'B',
                'method': m,
                'outcome': TARGET_COL,
                'treatment': TREATMENT_COL,
                'n_after_trim': np.nan,
                'ate': np.nan,
                'ate_lb': np.nan,
                'ate_ub': np.nan,
                'p_two_sided_approx': np.nan,
                'ate_ci_source': 'skipped_small_n',
            })
    else:
        with tempfile.TemporaryDirectory() as td:
            res_df, (ate_x, lb_x, ub_x) = estimate_causal_impact_xlearner(
                df_b,
                treatment_col=TREATMENT_COL,
                output_dir=td,
                outcome_col=TARGET_COL,
                run_auxiliary_steps=False,
            )
        ti_x = res_df.attrs.get('ps_trim_info', {}) if res_df is not None else {}
        rows.append({
            'cohort': 'B',
            'method': 'XLearner',
            'outcome': TARGET_COL,
            'treatment': TREATMENT_COL,
            'n_after_trim': ti_x.get('n_after_trim', np.nan),
            'ate': ate_x,
            'ate_lb': lb_x,
            'ate_ub': ub_x,
            'p_two_sided_approx': _p_value_two_sided_from_ci(ate_x, lb_x, ub_x),
            'ate_ci_source': ti_x.get('ate_ci_source', ''),
        })

        (ate_d, lb_d, ub_d), ti_d = estimate_causal_impact_dml_sensitivity(
            df_b,
            treatment_col=TREATMENT_COL,
            outcome_col=TARGET_COL,
        )
        rows.append({
            'cohort': 'B',
            'method': 'CausalForestDML',
            'outcome': TARGET_COL,
            'treatment': TREATMENT_COL,
            'n_after_trim': ti_d.get('n_after_trim', np.nan),
            'ate': ate_d,
            'ate_lb': lb_d,
            'ate_ub': ub_d,
            'p_two_sided_approx': _p_value_two_sided_from_ci(ate_d, lb_d, ub_d),
            'ate_ci_source': 'econml_ate_interval',
        })

    os.makedirs(RESULTS_TABLES, exist_ok=True)
    out_path = os.path.join(RESULTS_TABLES, 'ate_method_sensitivity.csv')
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')
    logger.info("估计器敏感性汇总已写入: %s", out_path)


def get_estimate_causal_impact():
    """根据 config.CAUSAL_METHOD 返回因果估计函数。

    约定：失败或跳过时返回 ``(None, CAUSAL_FAILURE_ATE_TRIPLET)``（即 ``(np.nan, np.nan, np.nan)``）。
    调用方须用 ``res_df is None`` 判定失败；勿将 NaN 三元组与「真实零效应」混淆（成功时 res_df 非空且 CI 有限）。
    """
    try:
        from config import CAUSAL_METHOD
        if CAUSAL_METHOD == 'TLearner':
            return estimate_causal_impact_tlearner
        if CAUSAL_METHOD == 'XLearner':
            return estimate_causal_impact_xlearner
    except ImportError:
        pass
    return estimate_causal_impact
