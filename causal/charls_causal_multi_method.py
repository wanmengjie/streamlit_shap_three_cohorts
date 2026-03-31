# -*- coding: utf-8 -*-
"""
多方法因果估计：Causal Forest DML、LinearDML、OrthoForest、ForestDRLearner、S-Learner、T-Learner、X-Learner。
对同一暴露运行全部方法，输出 ATE 与 95% CI 对比表，便于选择最合适的方法。
X-Learner (Künzel et al.) 在治疗/对照组不平衡时通常优于 T-Learner。
"""
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from utils.charls_feature_lists import get_exclude_cols, CONTINUOUS_FOR_SCALING
from config import RANDOM_SEED

logger = logging.getLogger(__name__)


def _prepare_data(df_sub, treatment_col, target_col='is_comorbidity_next'):
    """准备 X, T, Y 及预处理，与 charls_recalculate_causal_impact 一致"""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    exclude_cols = get_exclude_cols(df_sub, target_col=target_col, treatment_col=treatment_col)
    W_cols = [c for c in df_sub.columns if c not in exclude_cols]
    X_raw = df_sub[W_cols].select_dtypes(include=[np.number])
    if X_raw.shape[1] == 0 or len(X_raw) < 10:
        return None, None, None, None

    imputer = SimpleImputer(strategy='median')
    X_filled = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)
    scale_cols = [c for c in X_filled.columns if c in CONTINUOUS_FOR_SCALING]
    if scale_cols:
        X_scaled = X_filled.copy()
        X_scaled[scale_cols] = StandardScaler().fit_transform(X_filled[scale_cols])
    else:
        X_scaled = X_filled

    T_series = df_sub[treatment_col].fillna(0).astype(int)
    Y_series = df_sub[target_col].astype(float)
    X_arr = np.asarray(X_scaled, dtype=np.float64)
    return X_arr, T_series.values, Y_series.values, df_sub


def _run_estimator(name, est, X, T, Y, groups=None):
    """运行单个估计器，返回 (ate, ate_lb, ate_ub) 或 (nan, nan, nan)。部分估计器不支持 groups，失败时回退到无 groups"""
    try:
        try:
            if groups is not None:
                est.fit(Y=Y, T=T, X=X, groups=groups)
            else:
                est.fit(Y=Y, T=T, X=X)
        except TypeError:
            est.fit(Y=Y, T=T, X=X)

        # 获取 ATE
        if hasattr(est, 'ate'):
            ate = est.ate(X=X)
        elif hasattr(est, 'effect'):
            ate = float(np.mean(est.effect(X)))
        elif hasattr(est, 'const_marginal_ate'):
            ate = float(est.const_marginal_ate(X=X))
        else:
            return np.nan, np.nan, np.nan

        # 获取 95% CI
        if hasattr(est, 'ate_interval'):
            lb, ub = est.ate_interval(X=X, alpha=0.05)
            ate_lb = float(lb) if np.isscalar(lb) else float(lb[0])
            ate_ub = float(ub) if np.isscalar(ub) else float(ub[0])
        elif hasattr(est, 'ate_inference'):
            inf = est.ate_inference(X=X)
            ate_lb = float(inf.conf_int()[0])
            ate_ub = float(inf.conf_int()[1])
        else:
            ate_lb, ate_ub = np.nan, np.nan

        return float(ate), ate_lb, ate_ub
    except Exception as e:
        logger.debug(f"  {name} 失败: {e}")
        return np.nan, np.nan, np.nan


def estimate_causal_multi_method(df_sub, treatment_col='exercise', output_dir=None, target_col='is_comorbidity_next', methods=None):
    """
    对同一暴露运行因果估计方法，返回对比 DataFrame。
    methods: 可选列表，如 ['LinearDML','TLearner','XLearner']；None 时运行全部 7 种方法。
    全部方法：CausalForestDML, LinearDML, DMLOrthoForest, ForestDRLearner, SLearner, TLearner, XLearner
    """
    X, T, Y, df_sub = _prepare_data(df_sub, treatment_col, target_col)
    if X is None:
        return pd.DataFrame()

    if methods is not None:
        methods = [m if m != 'T-Learner' else 'TLearner' for m in methods]
        methods = [m if m != 'S-Learner' else 'SLearner' for m in methods]
        methods = [m if m != 'X-Learner' else 'XLearner' for m in methods]

    if 'communityID' in df_sub.columns:
        groups = df_sub['communityID'].fillna(df_sub['ID']).astype(str).values
    else:
        groups = df_sub['ID'].astype(str).values

    rf_y = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=15, random_state=RANDOM_SEED, n_jobs=-1)
    rf_t = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=15, random_state=RANDOM_SEED, n_jobs=-1)

    results = []

    def _run(name):
        return (methods is None) or (name in methods)

    # 1. Causal Forest DML
    if _run('CausalForestDML'):
        try:
            from econml.dml import CausalForestDML
            est = CausalForestDML(
                model_y=rf_y, model_t=rf_t, discrete_treatment=True, cv=5,
                n_estimators=1000, random_state=RANDOM_SEED, honest=True
            )
            ate, lb, ub = _run_estimator('CausalForestDML', est, X, T, Y, groups)
            results.append({'method': 'CausalForestDML', 'ate': ate, 'ate_lb': lb, 'ate_ub': ub})
        except Exception as e:
            logger.debug(f"CausalForestDML 导入/运行失败: {e}")
            results.append({'method': 'CausalForestDML', 'ate': np.nan, 'ate_lb': np.nan, 'ate_ub': np.nan})

    # 2. LinearDML
    if _run('LinearDML'):
        try:
            from econml.dml import LinearDML
            est = LinearDML(
                model_y=rf_y, model_t=rf_t, discrete_treatment=True, cv=5, random_state=RANDOM_SEED
            )
            ate, lb, ub = _run_estimator('LinearDML', est, X, T, Y, None)  # LinearDML 不支持 groups
            results.append({'method': 'LinearDML', 'ate': ate, 'ate_lb': lb, 'ate_ub': ub})
        except Exception as e:
            logger.debug(f"LinearDML 失败: {e}")
            results.append({'method': 'LinearDML', 'ate': np.nan, 'ate_lb': np.nan, 'ate_ub': np.nan})

    # 3. DML OrthoForest（不支持 groups，传 None）
    if _run('DMLOrthoForest'):
        try:
            from econml.orf import DMLOrthoForest
            est = DMLOrthoForest(
                n_trees=500, min_leaf_size=10, max_depth=5, random_state=RANDOM_SEED
            )
            ate, lb, ub = _run_estimator('DMLOrthoForest', est, X, T, Y, None)
            results.append({'method': 'DMLOrthoForest', 'ate': ate, 'ate_lb': lb, 'ate_ub': ub})
        except Exception as e:
            logger.debug(f"DMLOrthoForest 失败: {e}")
            results.append({'method': 'DMLOrthoForest', 'ate': np.nan, 'ate_lb': np.nan, 'ate_ub': np.nan})

    # 4. Forest DR Learner（多数 DR 估计器不支持 groups）
    if _run('ForestDRLearner'):
        try:
            from econml.dr import ForestDRLearner
            est = ForestDRLearner(
                model_propensity=rf_t, model_regression=rf_y,
                n_estimators=500, min_samples_leaf=10, honest=True, random_state=RANDOM_SEED
            )
            ate, lb, ub = _run_estimator('ForestDRLearner', est, X, T, Y, None)
            results.append({'method': 'ForestDRLearner', 'ate': ate, 'ate_lb': lb, 'ate_ub': ub})
        except Exception as e:
            logger.debug(f"ForestDRLearner 失败: {e}")
            results.append({'method': 'ForestDRLearner', 'ate': np.nan, 'ate_lb': np.nan, 'ate_ub': np.nan})

    # 5. S-Learner
    if _run('SLearner'):
        try:
            from econml.metalearners import SLearner
            est = SLearner(overall_model=RandomForestRegressor(n_estimators=200, max_depth=4, random_state=RANDOM_SEED, n_jobs=-1))
            est.fit(Y=Y, T=T, X=X)
            ate = float(np.mean(est.effect(X)))
            from sklearn.utils import resample
            np.random.seed(RANDOM_SEED)
            ates_boot = []
            for _ in range(200):
                idx = resample(np.arange(len(Y)), n_samples=len(Y))
                try:
                    est_b = SLearner(overall_model=RandomForestRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_SEED + _, n_jobs=-1))
                    est_b.fit(Y=Y[idx], T=T[idx], X=X[idx])
                    ates_boot.append(float(np.mean(est_b.effect(X))))
                except Exception:
                    pass
            if len(ates_boot) >= 20:
                lb = float(np.percentile(ates_boot, 2.5))
                ub = float(np.percentile(ates_boot, 97.5))
            else:
                lb, ub = np.nan, np.nan
            results.append({'method': 'SLearner', 'ate': ate, 'ate_lb': lb, 'ate_ub': ub})
        except Exception as e:
            logger.debug(f"SLearner 失败: {e}")
            results.append({'method': 'SLearner', 'ate': np.nan, 'ate_lb': np.nan, 'ate_ub': np.nan})

    # 6. T-Learner
    if _run('TLearner'):
        try:
            from econml.metalearners import TLearner
            est = TLearner(
                models=(
                    RandomForestRegressor(n_estimators=200, max_depth=4, random_state=RANDOM_SEED, n_jobs=-1),
                    RandomForestRegressor(n_estimators=200, max_depth=4, random_state=RANDOM_SEED + 1, n_jobs=-1)
                )
            )
            est.fit(Y=Y, T=T, X=X)
            ate = float(np.mean(est.effect(X)))
            from sklearn.utils import resample
            np.random.seed(RANDOM_SEED)
            ates_boot = []
            boot_fail = 0
            for b in range(200):
                idx = resample(np.arange(len(Y)), n_samples=len(Y))
                try:
                    est_b = TLearner(models=(
                        RandomForestRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_SEED + b, n_jobs=-1),
                        RandomForestRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_SEED + b + 1, n_jobs=-1)
                    ))
                    est_b.fit(Y=Y[idx], T=T[idx], X=X[idx])
                    ates_boot.append(float(np.mean(est_b.effect(X))))
                except Exception as ex:
                    boot_fail += 1
                    logger.debug("TLearner multi_method bootstrap replicate %s failed: %s", b, ex)
            planned = 200
            msg = (
                f"TLearner [multi_method] row bootstrap: planned={planned}, successful={len(ates_boot)}, failed={boot_fail}"
            )
            logger.info(msg)
            print(msg, flush=True)
            if len(ates_boot) >= 20:
                lb = float(np.percentile(ates_boot, 2.5))
                ub = float(np.percentile(ates_boot, 97.5))
            else:
                lb, ub = np.nan, np.nan
            results.append({'method': 'TLearner', 'ate': ate, 'ate_lb': lb, 'ate_ub': ub})
        except Exception as e:
            logger.debug(f"TLearner 失败: {e}")
            results.append({'method': 'TLearner', 'ate': np.nan, 'ate_lb': np.nan, 'ate_ub': np.nan})

    # 7. X-Learner（治疗/对照组不平衡时通常优于 T-Learner，Künzel et al.）
    if _run('XLearner'):
        try:
            from econml.metalearners import XLearner
            est = XLearner(
                models=(
                    RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=15, random_state=RANDOM_SEED, n_jobs=-1),
                    RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=15, random_state=RANDOM_SEED + 1, n_jobs=-1)
                )
            )
            ate, lb, ub = _run_estimator('XLearner', est, X, T, Y, None)
            results.append({'method': 'XLearner', 'ate': ate, 'ate_lb': lb, 'ate_ub': ub})
        except Exception as e:
            logger.debug(f"XLearner 失败: {e}")
            # XLearner 可能无 ate_interval，回退至 Bootstrap
            try:
                from econml.metalearners import XLearner
                from sklearn.utils import resample
                est = XLearner(
                    models=(
                        RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=15, random_state=RANDOM_SEED, n_jobs=-1),
                        RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=15, random_state=RANDOM_SEED + 1, n_jobs=-1)
                    )
                )
                est.fit(Y=Y, T=T, X=X)
                ate = float(np.mean(est.effect(X)))
                np.random.seed(RANDOM_SEED)
                ates_boot = []
                boot_fail = 0
                for b in range(200):
                    idx = resample(np.arange(len(Y)), n_samples=len(Y))
                    try:
                        est_b = XLearner(models=(
                            RandomForestRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_SEED + b, n_jobs=-1),
                            RandomForestRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_SEED + b + 1, n_jobs=-1)
                        ))
                        est_b.fit(Y=Y[idx], T=T[idx], X=X[idx])
                        ates_boot.append(float(np.mean(est_b.effect(X))))
                    except Exception as ex:
                        boot_fail += 1
                        logger.debug("XLearner multi_method bootstrap replicate %s failed: %s", b, ex)
                planned = 200
                msg = (
                    f"XLearner [multi_method] row bootstrap: planned={planned}, successful={len(ates_boot)}, failed={boot_fail}"
                )
                logger.info(msg)
                print(msg, flush=True)
                lb = float(np.percentile(ates_boot, 2.5)) if len(ates_boot) >= 20 else np.nan
                ub = float(np.percentile(ates_boot, 97.5)) if len(ates_boot) >= 20 else np.nan
                results.append({'method': 'XLearner', 'ate': ate, 'ate_lb': lb, 'ate_ub': ub})
            except Exception as e2:
                logger.debug(f"XLearner Bootstrap 回退失败: {e2}")
                results.append({'method': 'XLearner', 'ate': np.nan, 'ate_lb': np.nan, 'ate_ub': np.nan})

    res_df = pd.DataFrame(results)
    res_df['significant'] = res_df.apply(
        lambda r: 1 if (not np.isnan(r['ate_lb']) and not np.isnan(r['ate_ub']) and (r['ate_lb'] > 0 or r['ate_ub'] < 0)) else 0,
        axis=1
    )

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f'multi_method_comparison_{treatment_col}.csv')
        res_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        logger.info(f"多方法对比已保存: {out_path}")

    return res_df
