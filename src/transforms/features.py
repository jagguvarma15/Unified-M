"""
Feature engineering utilities for MMM.

Creates model-ready datasets from raw marketing data.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def pivot_media_spend(
    df: pd.DataFrame,
    date_col: str = "date",
    channel_col: str = "channel",
    value_col: str = "spend",
    agg_func: str = "sum",
) -> pd.DataFrame:
    """
    Pivot long-format media spend to wide format for MMM.
    
    Transforms:
        date | channel | spend    →    date | google_spend | meta_spend | ...
    
    Args:
        df: Long-format DataFrame with channel-level spend
        date_col: Column name for date
        channel_col: Column name for channel
        value_col: Column name for spend values
        agg_func: Aggregation function if duplicates exist
    
    Returns:
        Wide-format DataFrame with one column per channel
    """
    # Pivot
    pivoted = df.pivot_table(
        index=date_col,
        columns=channel_col,
        values=value_col,
        aggfunc=agg_func,
        fill_value=0,
    ).reset_index()
    
    # Rename columns to include "_spend" suffix
    pivoted.columns = [
        f"{col}_spend" if col != date_col else col
        for col in pivoted.columns
    ]
    
    return pivoted


def add_time_features(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Add time-based features for modeling seasonality and trends.
    
    Args:
        df: DataFrame with date column
        date_col: Column name for date
    
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    
    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Basic time features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["week"] = df[date_col].dt.isocalendar().week
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["quarter"] = df[date_col].dt.quarter
    
    # Trend (days since start)
    df["trend"] = (df[date_col] - df[date_col].min()).dt.days
    
    # Weekend indicator
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    return df


def add_fourier_features(
    df: pd.DataFrame,
    date_col: str = "date",
    period: int = 365,
    n_order: int = 3,
    prefix: str = "fourier",
) -> pd.DataFrame:
    """
    Add Fourier terms for modeling seasonal patterns.
    
    Fourier terms capture periodic patterns without needing
    dummy variables for each period.
    
    Args:
        df: DataFrame with date column
        date_col: Column name for date
        period: Length of seasonal period (365 for yearly, 7 for weekly)
        n_order: Number of Fourier pairs (higher = more flexibility)
        prefix: Prefix for feature names
    
    Returns:
        DataFrame with Fourier features
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Day of year (or week) as fraction of period
    t = df[date_col].dt.dayofyear / period
    
    for k in range(1, n_order + 1):
        df[f"{prefix}_sin_{k}"] = np.sin(2 * np.pi * k * t)
        df[f"{prefix}_cos_{k}"] = np.cos(2 * np.pi * k * t)
    
    return df


def normalize_features(
    df: pd.DataFrame,
    columns: list[str],
    method: Literal["zscore", "minmax", "maxabs"] = "zscore",
) -> tuple[pd.DataFrame, dict]:
    """
    Normalize features for model stability.
    
    Args:
        df: DataFrame with features to normalize
        columns: Columns to normalize
        method: 'zscore', 'minmax', or 'maxabs'
    
    Returns:
        Tuple of (normalized DataFrame, normalization parameters)
    """
    df = df.copy()
    params = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / (std + 1e-8)
            params[col] = {"method": "zscore", "mean": mean, "std": std}
        
        elif method == "minmax":
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
            params[col] = {"method": "minmax", "min": min_val, "max": max_val}
        
        elif method == "maxabs":
            max_abs = df[col].abs().max()
            df[col] = df[col] / (max_abs + 1e-8)
            params[col] = {"method": "maxabs", "max_abs": max_abs}
    
    return df, params


def denormalize_features(
    df: pd.DataFrame,
    columns: list[str],
    params: dict,
) -> pd.DataFrame:
    """
    Reverse normalization using saved parameters.
    
    Args:
        df: Normalized DataFrame
        columns: Columns to denormalize
        params: Parameters from normalize_features()
    
    Returns:
        Denormalized DataFrame
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns or col not in params:
            continue
        
        p = params[col]
        
        if p["method"] == "zscore":
            df[col] = df[col] * p["std"] + p["mean"]
        elif p["method"] == "minmax":
            df[col] = df[col] * (p["max"] - p["min"]) + p["min"]
        elif p["method"] == "maxabs":
            df[col] = df[col] * p["max_abs"]
    
    return df


def create_mmm_features(
    media_spend: pd.DataFrame,
    outcomes: pd.DataFrame,
    controls: pd.DataFrame | None = None,
    target_col: str = "revenue",
    date_col: str = "date",
    channel_col: str = "channel",
    spend_col: str = "spend",
    add_time: bool = True,
    add_fourier: bool = True,
    fourier_order: int = 3,
    exposure_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Create a complete MMM-ready dataset from component DataFrames.
    
    This is the main transformation function that:
    1. Pivots media spend to wide format
    2. Joins with outcomes
    3. Joins with control variables
    4. Adds time/seasonality features
    5. Optionally pivots exposure columns (impressions, GRP, reach)
    
    Args:
        media_spend: Long-format media spend data
        outcomes: Outcome data (revenue, conversions, etc.)
        controls: Optional control variables
        target_col: Column to use as target (y)
        date_col: Date column name
        channel_col: Channel column name
        spend_col: Spend value column name
        add_time: Whether to add time features
        add_fourier: Whether to add Fourier seasonality features
        fourier_order: Number of Fourier terms
        exposure_cols: Optional list of exposure metric columns present in
            media_spend (e.g. ["impressions", "reach", "grp"]).  Each will be
            pivoted to wide format as {channel}_{metric} and included alongside
            spend columns.
    
    Returns:
        MMM-ready DataFrame with:
        - date: Date column
        - y: Target variable
        - {channel}_spend: One column per media channel
        - {channel}_{exposure}: One column per exposure metric per channel
        - Control variables
        - Time/seasonality features (if enabled)
    """
    # 1. Pivot media spend
    media_wide = pivot_media_spend(
        media_spend,
        date_col=date_col,
        channel_col=channel_col,
        value_col=spend_col,
    )
    
    # 1b. Pivot exposure columns if provided
    if exposure_cols:
        for exp_col in exposure_cols:
            if exp_col in media_spend.columns:
                exp_wide = pivot_media_spend(
                    media_spend,
                    date_col=date_col,
                    channel_col=channel_col,
                    value_col=exp_col,
                )
                # Rename columns: {channel}_spend -> {channel}_{exp_col}
                rename_map = {}
                for col in exp_wide.columns:
                    if col.endswith("_spend") and col != date_col:
                        rename_map[col] = col.replace("_spend", f"_{exp_col}")
                exp_wide = exp_wide.rename(columns=rename_map)
                media_wide = media_wide.merge(exp_wide, on=date_col, how="left")
    
    # 2. Prepare outcomes
    outcomes_clean = outcomes[[date_col, target_col]].copy()
    outcomes_clean = outcomes_clean.rename(columns={target_col: "y"})
    
    # 3. Merge
    df = media_wide.merge(outcomes_clean, on=date_col, how="inner")
    
    # 4. Add controls if provided
    if controls is not None:
        df = df.merge(controls, on=date_col, how="left")
    
    # 5. Add time features
    if add_time:
        df = add_time_features(df, date_col=date_col)
    
    if add_fourier:
        df = add_fourier_features(df, date_col=date_col, n_order=fourier_order)
    
    # 6. Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    return df


def get_media_columns(df: pd.DataFrame, suffix: str = "_spend") -> list[str]:
    """Get all media spend columns from a DataFrame."""
    return [col for col in df.columns if col.endswith(suffix)]


def create_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """
    Create lagged versions of specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to lag
        lags: List of lag periods
    
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    return df


def create_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
    agg_funcs: list[str] = ["mean", "sum"],
) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Args:
        df: Input DataFrame
        columns: Columns to compute rolling features for
        windows: Window sizes
        agg_funcs: Aggregation functions ('mean', 'sum', 'std', 'min', 'max')
    
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    for col in columns:
        for window in windows:
            for func in agg_funcs:
                df[f"{col}_roll{window}_{func}"] = (
                    df[col].rolling(window=window, min_periods=1).agg(func)
                )
    
    return df

