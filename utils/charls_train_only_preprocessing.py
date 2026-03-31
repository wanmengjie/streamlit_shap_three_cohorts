# -*- coding: utf-8 -*-
"""
Numeric preprocessing aligned with modeling.charls_model_comparison:
SimpleImputer + StandardScaler (continuous) live in sklearn Pipeline / ColumnTransformer,
fitted only on the development train split, then transform the full matrix.

Bulk MICE / step1_imputed_full (if used) is a separate layer; do not replace it with
per-fold imputation here — this module fixes sklearn-side median imputation leakage only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.charls_feature_lists import CONTINUOUS_FOR_SCALING


def get_train_indices_for_preprocessor(df: pd.DataFrame, y_col: str = 'is_comorbidity_next', random_state=None) -> np.ndarray:
    """
    Same policy as compare_models: temporal split when USE_TEMPORAL_SPLIT and wave present;
    else GroupShuffleSplit(test_size=0.2) by ID.
    Returns positional indices into df (0 .. len-1).
    """
    if random_state is None:
        try:
            from config import RANDOM_SEED
            random_state = RANDOM_SEED
        except ImportError:
            random_state = 500
    if y_col not in df.columns:
        raise ValueError(f"get_train_indices_for_preprocessor: missing column {y_col!r}")
    if 'ID' not in df.columns:
        raise ValueError("get_train_indices_for_preprocessor: df must contain 'ID'")
    y = df[y_col].astype(int)
    n = len(df)
    dummy_X = np.zeros((n, 1))

    try:
        from config import USE_TEMPORAL_SPLIT
    except ImportError:
        USE_TEMPORAL_SPLIT = False
    use_temporal = bool(USE_TEMPORAL_SPLIT) and 'wave' in df.columns
    if use_temporal:
        max_wave = df['wave'].max()
        train_mask = (df['wave'] < max_wave).to_numpy()
        test_mask = (df['wave'] == max_wave).to_numpy()
        if train_mask.sum() >= 50 and test_mask.sum() >= 20:
            return np.where(train_mask)[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, _ = next(gss.split(dummy_X, y, groups=df['ID']))
    return train_idx


def fit_transform_numeric_train_only(df_aligned: pd.DataFrame, X_num: pd.DataFrame, y_col: str = 'is_comorbidity_next') -> pd.DataFrame:
    """
    Fit ColumnTransformer on train indices only; transform all rows of X_num.
    df_aligned must be row-aligned with X_num (same index and length).
    """
    if len(df_aligned) != len(X_num) or not df_aligned.index.equals(X_num.index):
        raise ValueError('fit_transform_numeric_train_only: df_aligned and X_num must be row-aligned')
    train_idx = get_train_indices_for_preprocessor(df_aligned, y_col=y_col)
    num_cols = X_num.columns.tolist()
    scale_cols = [c for c in num_cols if c in CONTINUOUS_FOR_SCALING]
    pass_cols = [c for c in num_cols if c not in scale_cols]
    if not scale_cols:
        scale_cols, pass_cols = num_cols, []
    transformers = [
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), scale_cols),
    ]
    if pass_cols:
        transformers.append(
            ('pass', Pipeline([('imputer', SimpleImputer(strategy='median'))]), pass_cols),
        )
    pre = ColumnTransformer(transformers=transformers)
    pre.fit(X_num.iloc[train_idx])
    Xt = pre.transform(X_num)
    out_cols = scale_cols + pass_cols
    return pd.DataFrame(Xt, columns=out_cols, index=X_num.index)
