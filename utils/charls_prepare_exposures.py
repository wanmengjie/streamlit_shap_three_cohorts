# -*- coding: utf-8 -*-
"""
Derived exposure columns shared by main pipeline and `load_df_for_analysis`.
Kept in `utils/` so Streamlit / scripts can import without the `scripts` package.
"""
import numpy as np
import pandas as pd


def prepare_exposures(df: pd.DataFrame) -> pd.DataFrame:
    """Construct sleep_adequate (sleep ≥ 6h) before COLS_TO_DROP; smoking uses smokev elsewhere."""
    if "sleep" in df.columns:
        df["sleep_adequate"] = (df["sleep"].clip(0, 24) >= 6).astype(int)
        df.loc[df["sleep"].isna(), "sleep_adequate"] = np.nan
    return df
