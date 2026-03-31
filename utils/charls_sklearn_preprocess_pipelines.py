# -*- coding: utf-8 -*-
"""
Shared sklearn preprocessing for numeric features: IterativeImputer (MICE-style) inside Pipeline
branches, combined with StandardScaler for continuous columns — mirrors compare_models layout.

Import order: enable_iterative_imputer must be loaded before IterativeImputer.
"""
from __future__ import annotations

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.compose import ColumnTransformer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.charls_feature_lists import CONTINUOUS_FOR_SCALING


def make_iterative_imputer(random_state=None, max_iter=None) -> IterativeImputer:
    if random_state is None:
        try:
            from config import RANDOM_SEED
            random_state = RANDOM_SEED
        except ImportError:
            random_state = 500
    if max_iter is None:
        try:
            from config import ITERATIVE_IMPUTER_MAX_ITER
        except ImportError:
            ITERATIVE_IMPUTER_MAX_ITER = 10
        max_iter = int(ITERATIVE_IMPUTER_MAX_ITER)
    return IterativeImputer(
        random_state=random_state,
        max_iter=max_iter,
        initial_strategy='median',
        sample_posterior=False,
    )


def build_numeric_column_transformer(num_cols: list) -> ColumnTransformer:
    """
    Build ColumnTransformer for numeric columns only (same column grouping as charls_model_comparison).
    Each branch uses its own IterativeImputer instance (fit inside CV / train subset only).
    """
    scale_cols = [c for c in num_cols if c in CONTINUOUS_FOR_SCALING]
    pass_cols = [c for c in num_cols if c not in scale_cols]
    if not scale_cols:
        scale_cols, pass_cols = num_cols, []
    transformers = [
        ('num', Pipeline([('imputer', make_iterative_imputer()), ('scaler', StandardScaler())]), scale_cols),
    ]
    if pass_cols:
        transformers.append(
            ('pass', Pipeline([('imputer', make_iterative_imputer())]), pass_cols),
        )
    return ColumnTransformer(transformers=transformers)
