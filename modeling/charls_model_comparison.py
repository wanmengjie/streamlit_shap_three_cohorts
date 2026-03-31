import pandas as pd
import numpy as np
import os
import logging
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              GradientBoostingClassifier, AdaBoostClassifier,
                              HistGradientBoostingClassifier, BaggingClassifier)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
except ImportError:
    LGBMClassifier = None
    CatBoostClassifier = None

from utils.charls_ci_utils import get_metrics_with_ci
from utils.charls_feature_lists import get_exclude_cols
from utils.charls_sklearn_preprocess_pipelines import build_numeric_column_transformer
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer, recall_score
from sklearn.base import clone

try:
    from config import RANDOM_SEED, USE_GPU
except ImportError:
    RANDOM_SEED = 500
    USE_GPU = True


def _gpu_available():
    """检测是否有可用的 NVIDIA GPU（用于 XGB/LGBM/CatBoost 加速）"""
    if not USE_GPU:
        return False
    try:
        import subprocess
        r = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        pass
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    return False


def _xgb_gpu_kwargs():
    if _gpu_available():
        try:
            # XGBoost 2.0+ 使用 device; 旧版用 tree_method
            return {'device': 'cuda', 'tree_method': 'hist'}
        except Exception:
            pass
    return {'n_jobs': -1}


def _lgbm_gpu_kwargs():
    if _gpu_available():
        return {'device': 'gpu', 'n_jobs': -1}
    return {'n_jobs': -1}


def _catboost_gpu_kwargs():
    if _gpu_available():
        return {'task_type': 'GPU', 'thread_count': -1}
    return {'thread_count': -1}


# 屏蔽干扰
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=UserWarning, module="loky")

logger = logging.getLogger(__name__)

# 仅用于「OOF Youden 最优阈值 → 测试集 Recall」门槛；**禁止**用概率 0.5 下的 Recall 做筛选（极度不平衡时会误杀高 AUC 模型）
CPM_MIN_RECALL_THRESHOLD = 0.05


def parse_table2_recall_value(val):
    """Table2 df_main['Recall']：数值或 '0.12 (95% CI: ...)'。"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float, np.floating)):
        return float(val)
    try:
        return float(str(val).strip().split()[0])
    except (ValueError, IndexError):
        return np.nan


def select_champion_from_df_main(df_main, model_names=None, log=None):
    """
    **Table 2 权威**：与论文 Table 2 完全一致。先筛 Recall（Youden 最优阈值下测试集灵敏度）>= CPM_MIN_RECALL_THRESHOLD，
    再取 AUC 最高；若无满足者则全表 AUC 最高并 used_recall_fallback=True（由调用方打醒目 WARNING）。

    Returns
    -------
    (best_row: pd.Series | None, used_recall_fallback: bool)
    """
    if df_main is None or len(df_main) == 0:
        return None, False
    dm = df_main.copy()
    if model_names is not None:
        dm = dm[dm['Model'].isin(list(model_names))]
    if len(dm) == 0:
        return None, False
    if 'Recall' not in dm.columns or 'AUC' not in dm.columns:
        if log:
            log.warning('select_champion_from_df_main: df_main 缺 Recall/AUC 列，跳过')
        return None, False
    recalls = dm['Recall'].map(parse_table2_recall_value)
    eligible = dm[recalls.fillna(0) >= CPM_MIN_RECALL_THRESHOLD]
    used_fallback = len(eligible) == 0
    if used_fallback:
        eligible = dm
    eligible = eligible.sort_values('AUC', ascending=False, na_position='last')
    return eligible.iloc[0], used_fallback


def reorder_perf_df_champion_first(perf_df, champion_model_name: str) -> pd.DataFrame:
    """将冠军模型行置于 perf_df 顶部（与 Table2 选定冠军一致，便于 CSV/审阅）。"""
    if perf_df is None or len(perf_df) == 0 or not champion_model_name:
        return perf_df
    pf = perf_df.copy()
    if 'Model' not in pf.columns or champion_model_name not in set(pf['Model'].astype(str).values):
        return pf
    pf['_champ'] = pf['Model'].astype(str) == str(champion_model_name)
    sort_col = 'AUC_raw' if 'AUC_raw' in pf.columns else 'AUC'
    return pf.sort_values(['_champ', sort_col], ascending=[False, False]).drop(columns=['_champ'])


def rewrite_model_performance_full_csv(output_dir, target_col, perf_df, champion_model_name):
    """在选定冠军后重写 model_performance_full_*.csv，避免与 Table2/SHAP 不一致。"""
    ordered = reorder_perf_df_champion_first(perf_df, champion_model_name)
    path = os.path.join(output_dir, f'model_performance_full_{target_col}.csv')
    _to_csv_with_retry(ordered.round(4), path, index=False, encoding='utf-8-sig')
    logger.info('已按 Table2 冠军重排并保存: %s', path)


def select_champion_from_perf_df(perf_df, model_names=None):
    """
    统一冠军规则：**仅**用 Recall_at_opt_t_raw（内部 CV 所得 Youden 最优阈值在**测试集**上的 Recall，
    与 Table2 主 Recall / df_main['Recall'] 同定义）与 CPM_MIN_RECALL_THRESHOLD 比较；
    通过者中取 AUC_raw 最高。不得使用 get_metrics_with_ci 的 Recall_raw（固定 0.5 阈值）作任何淘汰。

    若缺少 Recall_at_opt_t_raw 列（旧结果表）：记 ERROR，视为无人通过门槛，退回全表 AUC 最高（used_recall_fallback=True）。
    """
    if perf_df is None or len(perf_df) == 0:
        return None, False
    pf = perf_df.copy()
    if model_names is not None:
        pf = pf[pf['Model'].isin(model_names)]
    if len(pf) == 0:
        return None, False
    if 'Recall_at_opt_t_raw' not in pf.columns:
        logger.error(
            'select_champion_from_perf_df: 缺少 Recall_at_opt_t_raw，无法按 Youden 最优阈值 Recall 筛选；'
            '已禁止回退到 0.5 阈值的 Recall_raw。将退回「全模型中测试集 AUC 最高」并标记 used_recall_fallback。'
            '请重新运行 compare_models 生成完整 perf_df。'
        )
        used_fallback = True
        valid = pf
    else:
        recall_series = pd.to_numeric(pf['Recall_at_opt_t_raw'], errors='coerce')
        valid = pf[recall_series.fillna(0) >= CPM_MIN_RECALL_THRESHOLD]
        used_fallback = len(valid) == 0
        if used_fallback:
            valid = pf
    sort_col = 'AUC_raw' if 'AUC_raw' in valid.columns else 'AUC'
    best_row = valid.sort_values(sort_col, ascending=False, na_position='last').iloc[0]
    return best_row, used_fallback


def _to_csv_with_retry(df, path, *, max_attempts=6, sleep_s=2.5, **kwargs):
    """Windows 下 CSV 被 Excel/WPS 占用时常引发 PermissionError，稍作重试并提示续跑命令。"""
    last_err = None
    for attempt in range(max_attempts):
        try:
            df.to_csv(path, **kwargs)
            return
        except PermissionError as e:
            last_err = e
            logger.warning(
                "写入 %s 被拒（可能被 Excel/WPS 打开），第 %s/%s 次，%.1fs 后重试…",
                path,
                attempt + 1,
                max_attempts,
                sleep_s,
            )
            if attempt + 1 == max_attempts:
                logger.error(
                    "仍无法写入。请关闭占用该文件的程序后重跑；或运行 "
                    "`python scripts/resume_charls_cohorts.py B,C` 从 B 起续跑（勿开占用输出 CSV）。"
                )
                raise
            time.sleep(sleep_s)
    if last_err:
        raise last_err


def compare_models(df, output_dir='evaluation_results', n_iter=25, target_col='is_comorbidity_next', save_roc_path=None, return_search_objects=False, return_xy_test=False):
    """
    【调优加强版】扩展搜索空间、类别不平衡处理、5 折 CV、可选集成，提升 AUC。

    TRIPOD 合规：
    - **IterativeImputer**（MICE 风格）与 Scaler 均在 Pipeline 内，仅在 CV 训练折上 fit，再 transform 验证/测试折。
    - 输入数据应保留真实缺失（勿先对全样本做 bulk 插补再传入本函数），否则 Imputer 仅作数值稳定。
    - 最优阈值在 development set 内通过 Internal CV 确定，禁止在测试集寻优。

    save_roc_path: 若提供，保存 y_true/y_prob 供后续绘制 ROC 叠加图。
    return_search_objects: 若为 True，额外返回 RandomizedSearchCV 对象字典，供 CPM 附录表提取 best_params_/best_score_。
    return_xy_test: 若为 True，额外返回 (X_test, y_test, X_train, y_train, groups_train, groups_test)；
        groups_train / groups_test 为对应划分的 ID 向量，供 CPM、事后校准（GroupKFold）与 **cluster bootstrap** 使用。
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f">>> [全量打擂台·调优版] 正在识别未来共病的最佳预警信号 (Target: {target_col})...")

    Y = target_col
    exclude = get_exclude_cols(df, target_col=Y)
    W_cols = [col for col in df.columns if col not in exclude]
    X = df[W_cols].select_dtypes(include=[np.number])
    y = df[Y].astype(int)
    if X.shape[1] == 0:
        logger.error("排除后无可用数值特征，请检查数据与排除列表。")
        raise ValueError("无可用特征列，compare_models 中止。")
    if len(X) < 30 or df['ID'].nunique() < 5:
        logger.error("样本量或个体数过少，无法进行分组划分。")
        raise ValueError("样本量过少，compare_models 中止。")

    # 【TRIPOD 防泄露】必须先 split，严禁在 split 之前对 X 调用 Imputer/Scaler
    try:
        from config import USE_TEMPORAL_SPLIT
    except ImportError:
        USE_TEMPORAL_SPLIT = False
    use_temporal = USE_TEMPORAL_SPLIT and 'wave' in df.columns
    if use_temporal:
        max_wave = df['wave'].max()
        train_mask = (df['wave'] < max_wave).values
        test_mask = (df['wave'] == max_wave).values
        if train_mask.sum() >= 50 and test_mask.sum() >= 20:
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            logger.info(f"时间划分: train wave<{max_wave} (n={len(train_idx)}), test wave={max_wave} (n={len(test_idx)})")
        else:
            logger.warning("时间划分样本量不足，回退到随机划分")
            use_temporal = False
    if not use_temporal:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
        train_idx, test_idx = next(gss.split(X, y, groups=df['ID']))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # TRIPOD：开发集（训练池）事件数/候选预测因子数，供文稿与方法学记录
    n_events_train = int((y_train == 1).sum())
    n_features_cpm = int(X_train.shape[1])
    epv = n_events_train / max(n_features_cpm, 1)
    logger.info(
        "[TRIPOD] EPV (development train pool): n_events=%d, n_predictors=%d, EPV=%.3f "
        "(events per candidate predictor; positive class = outcome event)",
        n_events_train,
        n_features_cpm,
        epv,
    )

    # 类别不平衡：用于 XGB/LGBM/CatBoost 的 scale_pos_weight（仅训练集）
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0
    # 4.1% 患病率对应正负比约 23.4，作为 XGB scale_pos_weight 候选
    scale_pos_weight_approx = round(scale_pos_weight, 1)
    logger.info(f"训练集正负样本: {pos}/{neg}, scale_pos_weight={scale_pos_weight:.2f}")

    num_cols = X.columns.tolist()
    # 【TRIPOD】IterativeImputer + Scaler 仅在 CV 训练折 fit（build_numeric_column_transformer）
    preprocessor = build_numeric_column_transformer(num_cols)

    # 5 折分组交叉验证，更稳的超参选择
    cv = GroupKFold(n_splits=5)
    n_iter_tuning = max(n_iter, 80)   # 调优加强：80 次随机搜索，提升 AUC
    n_iter_slow_models = 50           # XGB/LGBM/CatBoost 50 次，争取突破 0.75

    # 【差异化参数空间】n_jobs=-1 用满所有 CPU 核心；XGB/LGBM/CatBoost 有 GPU 时优先 CUDA
    xgb_kw = _xgb_gpu_kwargs()
    lgbm_kw = _lgbm_gpu_kwargs()
    cb_kw = _catboost_gpu_kwargs()
    if _gpu_available():
        logger.info("检测到 GPU，XGB/LGBM/CatBoost 将使用 CUDA 加速")
    model_configs = [
        # --- Cost-Sensitive: class_weight ---
        {'name': 'LR', 'model': LogisticRegression(max_iter=5000), 'params': {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100], 'clf__solver': ['lbfgs', 'saga'], 'clf__tol': [1e-4, 1e-3], 'clf__class_weight': ['balanced', None]}, 'imbalance_param': 'clf__class_weight'},
        {'name': 'RF', 'model': RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1), 'params': {'clf__n_estimators': [200, 300, 500, 700, 1000], 'clf__max_depth': [6, 10, 14, 18, None], 'clf__min_samples_leaf': [2, 5, 10], 'clf__min_samples_split': [2, 5, 10], 'clf__max_features': ['sqrt', 'log2', 0.5], 'clf__class_weight': ['balanced', None]}, 'imbalance_param': 'clf__class_weight'},
        {'name': 'XGB', 'model': XGBClassifier(eval_metric='logloss', random_state=RANDOM_SEED, **xgb_kw), 'params': {'clf__n_estimators': [200, 300, 500, 700], 'clf__max_depth': [4, 6, 8, 10], 'clf__learning_rate': [0.02, 0.03, 0.05, 0.1], 'clf__subsample': [0.6, 0.7, 0.85], 'clf__colsample_bytree': [0.6, 0.7, 0.85], 'clf__scale_pos_weight': [1, 10, 23.4, 30], 'clf__reg_alpha': [0.01, 0.1, 1], 'clf__reg_lambda': [0.1, 1, 10], 'clf__min_child_weight': [1, 3, 5]}, 'imbalance_param': 'clf__scale_pos_weight'},
        {'name': 'GBDT', 'model': GradientBoostingClassifier(random_state=RANDOM_SEED), 'params': {'clf__n_estimators': [150, 250, 350, 500], 'clf__max_depth': [4, 6, 8], 'clf__learning_rate': [0.03, 0.05, 0.1], 'clf__min_samples_leaf': [2, 5, 10], 'clf__subsample': [0.7, 0.85, 1.0], 'clf__max_features': ['sqrt', 'log2', None]}},
        {'name': 'ExtraTrees', 'model': ExtraTreesClassifier(random_state=RANDOM_SEED, n_jobs=-1), 'params': {'clf__n_estimators': [200, 400, 600, 800], 'clf__max_depth': [6, 10, 14, None], 'clf__min_samples_leaf': [2, 5, 10], 'clf__min_samples_split': [2, 5], 'clf__max_features': ['sqrt', 'log2', 0.5], 'clf__class_weight': ['balanced', None]}, 'imbalance_param': 'clf__class_weight'},
        {'name': 'AdaBoost', 'model': AdaBoostClassifier(random_state=RANDOM_SEED), 'params': {'clf__n_estimators': [50, 100, 150, 200, 250], 'clf__learning_rate': [0.03, 0.05, 0.1, 0.15]}},
        {'name': 'DT', 'model': DecisionTreeClassifier(random_state=RANDOM_SEED), 'params': {'clf__max_depth': [4, 6, 8, 10, 12, None], 'clf__min_samples_leaf': [5, 10, 20], 'clf__min_samples_split': [2, 5, 10], 'clf__criterion': ['gini', 'entropy'], 'clf__class_weight': ['balanced', None]}, 'imbalance_param': 'clf__class_weight'},
        {'name': 'Bagging', 'model': BaggingClassifier(random_state=RANDOM_SEED, n_jobs=-1), 'params': {'clf__n_estimators': [30, 80, 120, 200], 'clf__max_samples': [0.5, 0.7, 1.0], 'clf__max_features': [0.5, 0.7, 1.0]}},
        {'name': 'KNN', 'model': KNeighborsClassifier(weights='distance', n_jobs=-1), 'params': {'clf__n_neighbors': [5, 11, 21, 31, 51, 71], 'clf__p': [1, 2], 'clf__algorithm': ['auto', 'ball_tree', 'kd_tree']}},
        {'name': 'MLP', 'model': MLPClassifier(max_iter=800, early_stopping=True, random_state=RANDOM_SEED), 'params': {'clf__hidden_layer_sizes': [(64,), (64, 32), (128, 64), (128, 64, 32), (256, 128)], 'clf__alpha': [1e-5, 1e-4, 1e-3], 'clf__learning_rate_init': [0.001, 0.01], 'clf__activation': ['relu', 'tanh']}},
        {'name': 'NB', 'model': GaussianNB(), 'params': {'clf__var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]}},
        {'name': 'SVM', 'model': SVC(probability=True, random_state=RANDOM_SEED), 'params': {'clf__C': [0.01, 0.1, 1, 10, 100], 'clf__gamma': ['scale', 'auto', 0.001, 0.01, 0.1], 'clf__kernel': ['rbf', 'linear'], 'clf__tol': [1e-4, 1e-3], 'clf__class_weight': ['balanced', None]}, 'imbalance_param': 'clf__class_weight'},
        {'name': 'HistGBM', 'model': HistGradientBoostingClassifier(random_state=RANDOM_SEED), 'params': {'clf__max_iter': [150, 250, 350, 500], 'clf__learning_rate': [0.02, 0.03, 0.05, 0.1], 'clf__max_depth': [4, 6, 8], 'clf__min_samples_leaf': [5, 15, 25], 'clf__l2_regularization': [0, 0.1, 1], 'clf__class_weight': ['balanced', None]}, 'imbalance_param': 'clf__class_weight'}
    ]
    if LGBMClassifier:
        lgbm_params = {'clf__n_estimators': [200, 400, 600], 'clf__max_depth': [4, 6, 8, 10], 'clf__learning_rate': [0.02, 0.03, 0.05, 0.1], 'clf__scale_pos_weight': [1, 10, scale_pos_weight_approx, 30], 'clf__num_leaves': [31, 63, 127], 'clf__min_child_samples': [10, 20, 50], 'clf__reg_alpha': [0.01, 0.1, 1], 'clf__reg_lambda': [0.1, 1, 10], 'clf__subsample': [0.7, 0.85, 1.0]}
        model_configs.append({'name': 'LightGBM', 'model': LGBMClassifier(verbosity=-1, random_state=RANDOM_SEED, **lgbm_kw), 'params': lgbm_params, 'imbalance_param': 'clf__scale_pos_weight'})
    if CatBoostClassifier:
        # CatBoost 参数：不同版本可能用 iterations/depth 或 n_estimators/max_depth，兜底多组
        cb_params = {'clf__iterations': [300, 500, 700], 'clf__depth': [4, 6, 8], 'clf__learning_rate': [0.03, 0.05, 0.1], 'clf__l2_leaf_reg': [1, 3, 5], 'clf__min_data_in_leaf': [5, 15]}
        model_configs.append({'name': 'CatBoost', 'model': CatBoostClassifier(verbose=0, random_state=RANDOM_SEED, **cb_kw), 'params': cb_params, 'imbalance_param': None})

    metrics = []
    trained_models = {}
    search_objects = {} if return_search_objects else None
    n_models = len(model_configs)

    for i, config in enumerate(model_configs):
        name = config['name']
        t0 = time.time()
        slow_models = ('LightGBM', 'CatBoost', 'XGB')
        n_iter_this = n_iter_slow_models if name in slow_models else n_iter_tuning
        n_jobs_this = -1  # 用满所有 CPU 核心
        logger.info(f"[{i+1}/{n_models}] 开始训练: {name}（{n_iter_this}×5 折 RandomizedSearchCV，n_jobs={n_jobs_this}）...")
        pipe = Pipeline([('preprocessor', preprocessor), ('clf', config['model'])])
        param_grid = config['params']
        # 鲁棒性：仅保留模型支持的参数，避免 InvalidParameterError
        supported = set(pipe.get_params().keys())
        param_grid = {k: v for k, v in param_grid.items() if k in supported}
        # CatBoost 兜底：新版本可能用 n_estimators/max_depth，旧版用 iterations/depth
        if not param_grid and name == 'CatBoost':
            for alt in [{'clf__n_estimators': [500], 'clf__max_depth': [6]}, {'clf__iterations': [500], 'clf__depth': [6]}]:
                alt = {k: v for k, v in alt.items() if k in supported}
                if alt:
                    param_grid = alt
                    logger.info(f"[{i+1}/{n_models}] CatBoost 使用兜底参数: {list(param_grid.keys())}")
                    break
        if not param_grid:
            logger.warning(f"[{i+1}/{n_models}] 模型 {name} 无有效参数，跳过。")
            continue
        try:
            search = RandomizedSearchCV(pipe, param_grid, n_iter=n_iter_this, cv=cv, scoring='roc_auc', n_jobs=n_jobs_this, random_state=RANDOM_SEED)
            search.fit(X_train, y_train, groups=df.iloc[train_idx]['ID'])
            best_model = search.best_estimator_
            y_prob = best_model.predict_proba(X_test)[:, 1]
            grp_test = df.iloc[test_idx]['ID'].to_numpy()
            ci_res = get_metrics_with_ci(y_test, y_prob, groups=grp_test)
            # 【TRIPOD Step 1】Internal CV 确定最优阈值，供 CPM 在测试集上使用固定阈值（禁止在测试集寻优）
            try:
                oof_prob = cross_val_predict(clone(best_model), X_train, y_train, cv=cv, method='predict_proba',
                                             groups=df.iloc[train_idx]['ID'], n_jobs=-1)[:, 1]
                fpr, tpr, thresh = roc_curve(y_train, oof_prob)
                youden = tpr - fpr
                opt_t = float(thresh[np.argmax(youden)]) if len(thresh) > 0 else 0.5
                ci_res['_opt_threshold'] = opt_t
            except Exception as ex:
                logger.debug(f"模型 {name} Internal CV 阈值计算跳过: {ex}")
                ci_res['_opt_threshold'] = 0.5
            opt_t_f = float(ci_res.get('_opt_threshold', 0.5))
            recall_at_opt_t = recall_score(
                np.asarray(y_test),
                (np.asarray(y_prob) >= opt_t_f).astype(int),
                zero_division=0,
            )
            # 保留所有指标，包括 _raw 后缀的数值型指标，用于后续排序和筛选
            res_entry = {'Model': name}
            res_entry.update(ci_res)
            # 冠军筛选唯一依据：OOF Youden 阈值在测试集上的 Recall（非下方 ci_res 中 0.5 阈值的 Recall_raw）
            res_entry['Recall_at_opt_t_raw'] = recall_at_opt_t
            # 确保 AUC 列存在（如果 get_metrics_with_ci 返回了 AUC_raw 但 AUC 是字符串）
            # 这里我们保留 AUC 为格式化字符串（用于展示），AUC_raw 为数值（用于排序）
            if 'AUC_raw' not in res_entry and 'AUC' in res_entry:
                 # 如果只有 AUC 且是数值，则复制给 AUC_raw
                 if isinstance(res_entry['AUC'], (int, float)):
                     res_entry['AUC_raw'] = res_entry['AUC']
            
            metrics.append(res_entry)
            trained_models[name] = best_model
            if search_objects is not None:
                search_objects[name] = search
            elapsed = time.time() - t0
            logger.info(f"[{i+1}/{n_models}] 完成: {name} AUC={ci_res['AUC_raw']:.4f} 耗时 {elapsed:.1f}s")
        except Exception as e:
            logger.warning(f"[{i+1}/{n_models}] 模型 {name} 训练失败: {e}")

    perf_df = pd.DataFrame(metrics)
    # 优先使用 AUC_raw 排序，如果没有则尝试使用 AUC
    sort_col = 'AUC_raw' if 'AUC_raw' in perf_df.columns else 'AUC'
    perf_df = perf_df.sort_values(sort_col, ascending=False)
    if len(perf_df) == 0:
        logger.error("所有模型均训练失败，无法得到最优模型。")
        raise ValueError("无有效模型结果，compare_models 中止。")

    # 冠军不在此敲定：Table 2 (evaluate_and_report 的 df_main) 为唯一权威，由 run_cohort_protocol._select_champion 决定。
    # 此处仅按测试集 AUC 排序；model_performance_full CSV 会在选定冠军后由 rewrite_model_performance_full_csv 重排首行。
    logger.info(
        "compare_models: 擂台 perf_df 已按 AUC 排序；最终冠军以 Table2(df_main) 规则在 _select_champion 中确定并重写 CSV。"
    )

    if save_roc_path:
        logger.warning(
            "compare_models(save_roc_path=…): ROC JSON 使用当前 perf_df 首行（AUC 排序），"
            "与 Table2/_select_champion 冠军可能不一致；主流程请勿传入 save_roc_path。"
        )
        import json
        best_name = perf_df.iloc[0]['Model']
        best_model = trained_models[best_name]
        y_prob = best_model.predict_proba(X_test)[:, 1]
        with open(save_roc_path, 'w', encoding='utf-8') as f:
            json.dump({'y_true': y_test.tolist(), 'y_prob': y_prob.tolist()}, f)
        logger.info(f"ROC 数据已保存至 {save_roc_path}")

    # Round all numeric columns to 4 decimal places
    perf_df = perf_df.round(4)
    # 保存前可以考虑去掉 _raw 列，或者保留以便查看。这里保留。
    _to_csv_with_retry(
        perf_df,
        os.path.join(output_dir, f'model_performance_full_{target_col}.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    # 审稿修正：多重检验说明（15模型×3队列=45次比较，因果5暴露×3队列=15次）
    with open(os.path.join(output_dir, 'multiple_testing_note.txt'), 'w', encoding='utf-8') as f:
        f.write("Multiple Testing Note (审稿修正):\n")
        f.write("- Model comparison: 15 models × 3 cohorts = 45 comparisons (exploratory).\n")
        f.write("- Causal inference: 5 exposures × 3 cohorts = 15 ATE estimates.\n")
        f.write("- For confirmatory claims, consider FDR (Benjamini-Hochberg) or Bonferroni (α=0.05/15≈0.0033).\n")
        f.write("- Current analysis: primary results reported with 95%% CI; sensitivity analyses support robustness.\n")

    # 冠军模型与复杂度：主流程 CPM 集成时由 run_cohort_protocol 负责保存，此处跳过避免重复
    if not return_xy_test:
        logger.warning(
            "compare_models(return_xy_test=False): 此处 champion 为 AUC 排序首行，非 Table2(df_main) 规则；"
            "正式分析请仅通过 run_cohort_protocol 运行 CPM。"
        )
        best_name = perf_df.iloc[0]['Model']
        best_model = trained_models[best_name]
        n_feat = X_train.shape[1]
        try:
            import time as _t
            t0 = _t.time()
            _ = best_model.predict_proba(X_test.head(100))
            pred_time_ms = (_t.time() - t0) * 1000 / min(100, len(X_test))
        except Exception:
            pred_time_ms = np.nan
        depth_str = ''
        _m = best_model.estimator if hasattr(best_model, 'estimator') else best_model
        if hasattr(_m, 'named_steps') and 'clf' in _m.named_steps:
            clf = _m.named_steps['clf']
            if hasattr(clf, 'max_depth') and clf.max_depth is not None:
                depth_str = f', max_depth={clf.max_depth}'
            if hasattr(clf, 'n_estimators'):
                depth_str += f', n_estimators={getattr(clf, "n_estimators", "N/A")}'
        if not depth_str:
            depth_str = 'see model object'
        with open(os.path.join(output_dir, 'model_complexity_efficiency.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Champion Model: {best_name}\n")
            f.write(f"n_features: {n_feat}\n")
            f.write(f"features: {', '.join(X_train.columns.tolist())}\n")
            f.write(f"prediction_time_per_100_samples_ms: {pred_time_ms:.2f}\n")
            f.write(f"hyperparams: {depth_str}\n")
        try:
            import joblib
            joblib.dump(best_model, os.path.join(output_dir, 'champion_model.joblib'))
            logger.info(f"冠军模型已保存: {os.path.join(output_dir, 'champion_model.joblib')}")
        except Exception as e:
            logger.debug(f"冠军模型保存跳过: {e}")

    plt.figure(figsize=(12, 8))
    y_col = 'AUC_raw' if 'AUC_raw' in perf_df.columns else 'AUC'
    sns.barplot(x='Model', y=y_col, data=perf_df, hue='Model', palette='magma', legend=False)
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.3)
    plt.ylim(0.4, 1.0)
    plt.title(f'Performance of 15 Models ({target_col})', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig2c_comparison_{target_col}.png'), dpi=300)
    plt.close()

    out = (perf_df, trained_models)
    if return_search_objects and search_objects:
        out = out + (search_objects,)
    if return_xy_test:
        groups_train = df.iloc[train_idx]['ID'].to_numpy()
        groups_test = df.iloc[test_idx]['ID'].to_numpy()
        out = out + (X_test, y_test, X_train, y_train, groups_train, groups_test)
    return out
