"""
Adstock transformations for modeling carryover effects in media.

Adstock captures the lagged effect of advertising - today's ad spend
affects not just today's sales, but also future sales with decaying impact.
"""

import numpy as np
import pandas as pd
from typing import Literal


def geometric_adstock(
    x: np.ndarray | pd.Series,
    alpha: float,
    l_max: int = 8,
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply geometric (exponential decay) adstock transformation.
    
    The most common adstock model. Effect decays exponentially:
    adstock[t] = x[t] + alpha * adstock[t-1]
    
    Args:
        x: Input media spend/impressions array
        alpha: Decay rate (0-1). Higher = longer carryover.
               0 = no carryover, 1 = infinite carryover
        l_max: Maximum lag to consider
        normalize: If True, weights sum to 1
    
    Returns:
        Adstocked values
    
    Example:
        >>> spend = np.array([100, 0, 0, 0, 0])
        >>> geometric_adstock(spend, alpha=0.5, l_max=4)
        # Returns decaying effect over time
    """
    x = np.asarray(x)
    
    # Build decay weights
    weights = np.array([alpha ** i for i in range(l_max)])
    
    if normalize:
        weights = weights / weights.sum()
    
    # Apply convolution (filter)
    adstocked = np.convolve(x, weights)[:len(x)]
    
    return adstocked


def weibull_adstock(
    x: np.ndarray | pd.Series,
    shape: float,
    scale: float,
    l_max: int = 8,
    adstock_type: Literal["cdf", "pdf"] = "cdf",
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply Weibull adstock transformation.
    
    More flexible than geometric - can model delayed peaks and
    various decay patterns.
    
    - CDF type: Flexible decay, starts at 0
    - PDF type: Can have a peak in the middle (delayed effect)
    
    Args:
        x: Input media array
        shape: Weibull shape parameter (k).
               <1: decreasing hazard, >1: increasing hazard
        scale: Weibull scale parameter (lambda)
        l_max: Maximum lag
        adstock_type: 'cdf' or 'pdf'
        normalize: If True, weights sum to 1
    
    Returns:
        Adstocked values
    """
    from scipy.stats import weibull_min
    
    x = np.asarray(x)
    t = np.arange(l_max)
    
    if adstock_type == "cdf":
        # CDF: cumulative effect
        weights = weibull_min.cdf(t, c=shape, scale=scale)
        weights = np.diff(np.concatenate([[0], weights]))
    else:
        # PDF: can have delayed peak
        weights = weibull_min.pdf(t, c=shape, scale=scale)
    
    if normalize and weights.sum() > 0:
        weights = weights / weights.sum()
    
    adstocked = np.convolve(x, weights)[:len(x)]
    
    return adstocked


def delayed_adstock(
    x: np.ndarray | pd.Series,
    alpha: float,
    theta: int,
    l_max: int = 8,
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply delayed adstock transformation.
    
    Models situations where ad effect peaks after a delay,
    then decays geometrically.
    
    Args:
        x: Input media array
        alpha: Decay rate after peak
        theta: Delay before peak effect (in periods)
        l_max: Maximum lag
        normalize: If True, weights sum to 1
    
    Returns:
        Adstocked values
    """
    x = np.asarray(x)
    
    # Build weights with delay
    weights = np.zeros(l_max)
    for i in range(l_max):
        if i < theta:
            # Before peak: building up
            weights[i] = (i + 1) / (theta + 1)
        else:
            # After peak: geometric decay
            weights[i] = alpha ** (i - theta)
    
    if normalize and weights.sum() > 0:
        weights = weights / weights.sum()
    
    adstocked = np.convolve(x, weights)[:len(x)]
    
    return adstocked


def apply_adstock(
    df: pd.DataFrame,
    media_columns: list[str],
    adstock_params: dict[str, dict],
    adstock_type: str = "geometric",
) -> pd.DataFrame:
    """
    Apply adstock transformation to multiple media columns.
    
    Args:
        df: DataFrame with media columns
        media_columns: List of column names to transform
        adstock_params: Dict mapping column -> params
            e.g., {"google_spend": {"alpha": 0.7}, "tv_spend": {"alpha": 0.9}}
        adstock_type: 'geometric', 'weibull', or 'delayed'
    
    Returns:
        DataFrame with adstocked columns (suffixed with '_adstock')
    
    Example:
        >>> params = {
        ...     "google_spend": {"alpha": 0.5, "l_max": 4},
        ...     "meta_spend": {"alpha": 0.7, "l_max": 8},
        ... }
        >>> df_transformed = apply_adstock(df, ["google_spend", "meta_spend"], params)
    """
    df = df.copy()
    
    adstock_funcs = {
        "geometric": geometric_adstock,
        "weibull": weibull_adstock,
        "delayed": delayed_adstock,
    }
    
    if adstock_type not in adstock_funcs:
        raise ValueError(f"Unknown adstock type: {adstock_type}")
    
    func = adstock_funcs[adstock_type]
    
    for col in media_columns:
        params = adstock_params.get(col, {"alpha": 0.5})
        df[f"{col}_adstock"] = func(df[col].values, **params)
    
    return df


def estimate_adstock_halflife(alpha: float) -> float:
    """
    Calculate the half-life of geometric adstock.
    
    Half-life = number of periods for effect to decay to 50%.
    
    Args:
        alpha: Decay rate
    
    Returns:
        Half-life in periods
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1 (exclusive)")
    
    return np.log(0.5) / np.log(alpha)


def estimate_adstock_90life(alpha: float) -> float:
    """
    Calculate the 90%-life of geometric adstock.
    
    90%-life = number of periods for 90% of cumulative effect.
    
    Args:
        alpha: Decay rate
    
    Returns:
        90%-life in periods
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1 (exclusive)")
    
    return np.log(0.1) / np.log(alpha)

