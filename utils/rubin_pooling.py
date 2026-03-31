# -*- coding: utf-8 -*-
"""
Rubin's rules for pooling estimates from multiple imputation.

Formula (Rubin 1987):
  Q_bar = (1/m) * sum(Q_m)           # pooled point estimate
  U_bar = (1/m) * sum(U_m)           # within-imputation variance (U_m = SE_m^2)
  B     = (1/(m-1)) * sum((Q_m - Q_bar)^2)  # between-imputation variance
  T     = U_bar + (1 + 1/m) * B      # total variance
  SE_pooled = sqrt(T)
  df = (m-1) * (1 + U_bar/((1+1/m)*B))^2  # degrees of freedom for t-test
"""
import numpy as np


def rubin_pool(estimates, variances=None, ses=None):
    """
    Pool scalar estimates using Rubin's rules.

    Parameters
    ----------
    estimates : array-like
        Point estimates from each imputation (Q_1, ..., Q_m).
    variances : array-like, optional
        Within-imputation variances (U_1, ..., U_m). If None, use ses^2.
    ses : array-like, optional
        Standard errors from each imputation. Used if variances is None.

    Returns
    -------
    dict
        'Q_bar': pooled point estimate
        'SE': pooled standard error
        'var_total': total variance T
        'df': approximate degrees of freedom
    """
    estimates = np.asarray(estimates, dtype=float)
    m = len(estimates)
    if m == 0:
        return {'Q_bar': np.nan, 'SE': np.nan, 'var_total': np.nan, 'df': np.nan}
    if m < 2:
        return {'Q_bar': float(estimates[0]), 'SE': np.nan, 'var_total': np.nan, 'df': np.nan}

    Q_bar = np.mean(estimates)
    if variances is not None:
        U_arr = np.asarray(variances, dtype=float)
    elif ses is not None:
        U_arr = np.asarray(ses, dtype=float) ** 2
    else:
        U_arr = np.full(m, np.nan)

    U_bar = np.nanmean(U_arr) if np.any(np.isfinite(U_arr)) else 0.0
    B = np.var(estimates, ddof=1) if m > 1 else 0.0
    T = U_bar + (1 + 1.0 / m) * B
    T = max(T, 0.0)
    SE = np.sqrt(T)

    # Barnard-Rubin df (small m correction)
    if B > 1e-12:
        lam = (1 + 1.0 / m) * B / T
        df = (m - 1) / (lam ** 2)
    else:
        df = np.inf

    return {'Q_bar': float(Q_bar), 'SE': float(SE), 'var_total': float(T), 'df': float(df)}


def rubin_pool_ci(Q_bar, SE, df, alpha=0.05):
    """
    Compute pooled confidence interval using t-distribution.

    Parameters
    ----------
    Q_bar : float
        Pooled point estimate.
    SE : float
        Pooled standard error.
    df : float
        Degrees of freedom (can be inf).
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    tuple
        (lower, upper) bounds.
    """
    if np.isinf(df) or df > 100:
        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)
    else:
        from scipy import stats
        z = stats.t.ppf(1 - alpha / 2, df)
    half = z * SE
    return (float(Q_bar - half), float(Q_bar + half))
