"""
Contribution decomposition and ROI calculation utilities.

Breaks down total response into channel-level contributions
and computes return on investment metrics.
"""

import numpy as np
import pandas as pd
from loguru import logger


def decompose_contributions(
    df: pd.DataFrame,
    coefficients: dict[str, float],
    baseline: float,
    media_cols: list[str],
    target_col: str = "y",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Decompose total response into channel contributions.
    
    Args:
        df: DataFrame with transformed media columns
        coefficients: Channel coefficients from model
        baseline: Baseline (intercept) contribution
        media_cols: List of media column names
        target_col: Target variable column
        date_col: Date column
    
    Returns:
        DataFrame with contribution columns for each channel
    
    Example:
        >>> contributions = decompose_contributions(
        ...     df, 
        ...     coefficients={"google_spend": 0.5, "meta_spend": 0.3},
        ...     baseline=1000,
        ...     media_cols=["google_spend", "meta_spend"],
        ... )
    """
    result = pd.DataFrame({date_col: df[date_col]})
    
    total_contrib = np.zeros(len(df))
    
    # Channel contributions
    for col in media_cols:
        if col in coefficients and col in df.columns:
            contrib = df[col].values * coefficients[col]
            result[f"{col}_contribution"] = contrib
            total_contrib += contrib
    
    # Baseline
    result["baseline_contribution"] = baseline
    total_contrib += baseline
    
    # Predicted vs actual
    result["predicted"] = total_contrib
    if target_col in df.columns:
        result["actual"] = df[target_col].values
        result["residual"] = result["actual"] - result["predicted"]
    
    return result


def compute_channel_roi(
    contributions: pd.DataFrame,
    spend_df: pd.DataFrame,
    media_cols: list[str],
    spend_suffix: str = "_spend",
    contrib_suffix: str = "_contribution",
) -> dict[str, dict[str, float]]:
    """
    Compute ROI metrics for each channel.
    
    ROI = (Contribution - Spend) / Spend
    
    Args:
        contributions: DataFrame with channel contributions
        spend_df: DataFrame with spend data
        media_cols: List of media column names (without suffix)
        spend_suffix: Suffix for spend columns
        contrib_suffix: Suffix for contribution columns
    
    Returns:
        Dictionary with ROI metrics per channel:
        - total_spend: Sum of spend
        - total_contribution: Sum of contribution
        - roi: Return on investment
        - roas: Return on ad spend (contribution / spend)
    """
    roi_metrics = {}
    
    for col in media_cols:
        spend_col = f"{col}" if col.endswith(spend_suffix) else f"{col}{spend_suffix}"
        contrib_col = f"{col}{contrib_suffix}"
        
        # Handle column name variations
        if spend_col not in spend_df.columns:
            spend_col = col
        if contrib_col not in contributions.columns:
            contrib_col = f"{col.replace(spend_suffix, '')}{contrib_suffix}"
        
        if spend_col in spend_df.columns and contrib_col in contributions.columns:
            total_spend = spend_df[spend_col].sum()
            total_contribution = contributions[contrib_col].sum()
            
            roi_metrics[col] = {
                "total_spend": float(total_spend),
                "total_contribution": float(total_contribution),
                "roi": float((total_contribution - total_spend) / (total_spend + 1e-8)),
                "roas": float(total_contribution / (total_spend + 1e-8)),
            }
    
    return roi_metrics


def compute_marginal_roi(
    response_curves: dict[str, pd.DataFrame],
    current_spend: dict[str, float],
) -> dict[str, float]:
    """
    Compute marginal ROI at current spend levels.
    
    Marginal ROI = derivative of response curve at current spend.
    This tells you the incremental return per additional dollar.
    
    Args:
        response_curves: Dict mapping channel -> DataFrame with spend/response
        current_spend: Current spend level per channel
    
    Returns:
        Dictionary with marginal ROI per channel
    """
    marginal_roi = {}
    
    for channel, curve_df in response_curves.items():
        spend = curve_df["spend"].values
        response = curve_df["response"].values
        
        current = current_spend.get(channel, 0)
        
        # Find closest spend level
        idx = np.abs(spend - current).argmin()
        
        # Compute numerical derivative
        if idx < len(spend) - 1:
            dx = spend[idx + 1] - spend[idx]
            dy = response[idx + 1] - response[idx]
            marginal_roi[channel] = float(dy / (dx + 1e-8))
        else:
            marginal_roi[channel] = 0.0
    
    return marginal_roi


def compute_contribution_share(
    contributions: pd.DataFrame,
    media_cols: list[str],
    contrib_suffix: str = "_contribution",
) -> dict[str, float]:
    """
    Compute share of total contribution per channel.
    
    Args:
        contributions: DataFrame with channel contributions
        media_cols: List of media column names
        contrib_suffix: Suffix for contribution columns
    
    Returns:
        Dictionary with contribution share (0-1) per channel
    """
    shares = {}
    
    # Sum contributions
    total_contrib = 0
    channel_totals = {}
    
    for col in media_cols:
        contrib_col = f"{col}{contrib_suffix}"
        if contrib_col not in contributions.columns:
            contrib_col = f"{col.replace('_spend', '')}{contrib_suffix}"
        
        if contrib_col in contributions.columns:
            total = contributions[contrib_col].sum()
            channel_totals[col] = total
            total_contrib += abs(total)
    
    # Compute shares
    for col, total in channel_totals.items():
        shares[col] = float(abs(total) / (total_contrib + 1e-8))
    
    return shares


def compute_weekly_contributions(
    contributions: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Aggregate contributions to weekly level.
    
    Args:
        contributions: Daily contribution DataFrame
        date_col: Date column name
    
    Returns:
        Weekly aggregated contributions
    """
    df = contributions.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["week"] = df[date_col].dt.to_period("W").dt.start_time
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Aggregate
    weekly = df.groupby("week")[numeric_cols].sum().reset_index()
    weekly = weekly.rename(columns={"week": date_col})
    
    return weekly


def create_contribution_summary(
    contributions: pd.DataFrame,
    spend_df: pd.DataFrame,
    media_cols: list[str],
) -> pd.DataFrame:
    """
    Create a summary table of channel performance.
    
    Args:
        contributions: Contribution DataFrame
        spend_df: Spend DataFrame
        media_cols: List of media columns
    
    Returns:
        Summary DataFrame with spend, contribution, ROI per channel
    """
    roi_metrics = compute_channel_roi(contributions, spend_df, media_cols)
    shares = compute_contribution_share(contributions, media_cols)
    
    summary_data = []
    
    for col in media_cols:
        if col in roi_metrics:
            metrics = roi_metrics[col]
            summary_data.append({
                "channel": col.replace("_spend", ""),
                "total_spend": metrics["total_spend"],
                "total_contribution": metrics["total_contribution"],
                "roi": metrics["roi"],
                "roas": metrics["roas"],
                "contribution_share": shares.get(col, 0),
            })
    
    summary = pd.DataFrame(summary_data)
    
    # Sort by contribution share
    summary = summary.sort_values("contribution_share", ascending=False)
    
    return summary

