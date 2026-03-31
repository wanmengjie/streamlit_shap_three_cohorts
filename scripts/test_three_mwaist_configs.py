# -*- coding: utf-8 -*-
"""
三方案快速对比：mwaist/systo/diasto 删除策略对 AUC 的影响。
方案 A：只删 mwaist
方案 B：删 mwaist + systo + diasto
方案 C：维持现状（不删 mwaist）

使用 LR 默认参数快速测试，约 1-2 分钟完成。
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CESD_CUTOFF, COGNITION_CUTOFF, TARGET_COL, RANDOM_SEED
from data.charls_complete_preprocessing import reapply_cohort_definition
from utils.charls_feature_lists import get_exclude_cols, CONTINUOUS_FOR_SCALING
from utils.charls_script_data_loader import load_supervised_prediction_df

# 当前基础删除列（不含 mwaist/systo/diasto）；sleep 保留供预测
BASE_DROP = ['rgrip', 'grip_strength_avg', 'lgrip', 'wspeed', 'psyche', 'puff']

CONFIGS = {
    'A_只删mwaist': BASE_DROP + ['mwaist'],
    'B_删mwaist+systo+diasto': BASE_DROP + ['mwaist', 'systo', 'diasto'],
    'C_维持现状': BASE_DROP,
}


def _load_and_prepare(drop_cols):
    """与主流程同一数据源；不应用 config.COLS_TO_DROP，再按方案删除列并重算队列。"""
    df = load_supervised_prediction_df(apply_config_drop=False)
    if df is None:
        return None
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    return reapply_cohort_definition(df, CESD_CUTOFF, COGNITION_CUTOFF)


def _quick_auc(df_sub):
    """LR 默认参数，单次划分，快速得 AUC"""
    Y = TARGET_COL
    exclude = get_exclude_cols(df_sub, target_col=Y)
    W_cols = [c for c in df_sub.columns if c not in exclude]
    X = df_sub[W_cols].select_dtypes(include=[np.number])
    y = df_sub[Y].astype(int)
    if X.shape[1] == 0 or len(X) < 50:
        return np.nan
    num_cols = X.columns.tolist()
    scale_cols = [c for c in num_cols if c in CONTINUOUS_FOR_SCALING and c in df_sub.columns]
    pass_cols = [c for c in num_cols if c not in scale_cols]
    if not scale_cols:
        scale_cols, pass_cols = num_cols, []
    transformers = [
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), scale_cols)
    ]
    if pass_cols:
        transformers.append(('pass', Pipeline([('imputer', SimpleImputer(strategy='median'))]), pass_cols))
    pipe = Pipeline([
        ('prep', ColumnTransformer(transformers=transformers)),
        ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight='balanced'))
    ])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=df_sub['ID']))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, proba)


def main():
    results = []
    for config_name, drop_cols in CONFIGS.items():
        print(f"\n>>> {config_name}: drop {drop_cols}")
        df = _load_and_prepare(drop_cols)
        if df is None:
            print("  数据加载失败")
            continue
        for cohort_key, g in [('A', 0), ('B', 1), ('C', 2)]:
            df_sub = df[df['baseline_group'] == g]
            if len(df_sub) < 50:
                continue
            auc = _quick_auc(df_sub)
            results.append({'config': config_name, 'cohort': f'Cohort_{cohort_key}', 'AUC': auc, 'n': len(df_sub)})
            print(f"  Cohort_{cohort_key}: AUC={auc:.4f} (n={len(df_sub)})")

    res_df = pd.DataFrame(results)
    out_csv = os.path.join('LIU_JUE_STRATEGIC_SUMMARY', 'mwaist_three_configs_auc.csv')
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    res_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n>>> 结果已保存: {out_csv}")

    # 打印对比表
    print("\n" + "="*60)
    print("AUC 对比汇总")
    print("="*60)
    for cohort in ['Cohort_A', 'Cohort_B', 'Cohort_C']:
        sub = res_df[res_df['cohort'] == cohort]
        if len(sub) == 0:
            continue
        vals = sub.set_index('config')['AUC']
        best = vals.idxmax()
        print(f"{cohort}: A={vals.get('A_只删mwaist', np.nan):.4f}  B={vals.get('B_删mwaist+systo+diasto', np.nan):.4f}  C={vals.get('C_维持现状', np.nan):.4f}  | 最佳={best}")


if __name__ == '__main__':
    main()
