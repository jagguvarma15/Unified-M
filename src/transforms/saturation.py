"""
Saturation (diminishing returns) transformations for media response curves.

Saturation models capture the fact that doubling spend doesn't double conversions.
There's a point of diminishing returns where additional spend yields less incremental response.
"""

import numpy as np
import pandas as pd


def hill_saturation(
    x: np.ndarray | pd.Series,
    K: float,
    S: float,
) -> np.ndarray:
    """
    Apply Hill function saturation transformation.
    
    Most common saturation function in MMM. Also known as:
    - Hill equation
    - Michaelis-Menten (when S=1)
    
    Formula: y = x^S / (K^S + x^S)
    
    Args:
        x: Input media spend (should be non-negative)
        K: Half-saturation point (EC50). Spend level at 50% max effect.
           Higher K = more spend needed to saturate
        S: Hill coefficient (steepness).
           S=1: standard hyperbolic
           S>1: steeper (more S-shaped)
           S<1: flatter
    
    Returns:
        Saturated values in [0, 1] range
    
    Example:
        >>> spend = np.array([0, 100, 500, 1000, 5000])
        >>> hill_saturation(spend, K=1000, S=1)
        # Returns: [0, 0.09, 0.33, 0.5, 0.83]
    """
    x = np.asarray(x, dtype=float)
    
    # Handle edge cases
    x = np.maximum(x, 0)  # Ensure non-negative
    
    # Hill function
    x_s = np.power(x, S)
    K_s = np.power(K, S)
    
    return x_s / (K_s + x_s)


def logistic_saturation(
    x: np.ndarray | pd.Series,
    L: float = 1.0,
    k: float = 1.0,
    x0: float = 0.0,
) -> np.ndarray:
    """
    Apply logistic (S-curve) saturation transformation.
    
    Useful when there's both a lower threshold (awareness) and
    upper saturation.
    
    Formula: y = L / (1 + exp(-k * (x - x0)))
    
    Args:
        x: Input media spend
        L: Maximum value (supremum)
        k: Steepness of the curve. Higher = sharper transition.
        x0: Midpoint (x value at L/2)
    
    Returns:
        Saturated values
    
    Note:
        Unlike Hill, this doesn't start at 0 when x=0.
        Use hill_saturation for 0-anchored responses.
    """
    x = np.asarray(x, dtype=float)
    
    return L / (1 + np.exp(-k * (x - x0)))


def michaelis_menten_saturation(
    x: np.ndarray | pd.Series,
    Vmax: float,
    Km: float,
) -> np.ndarray:
    """
    Apply Michaelis-Menten saturation (special case of Hill with S=1).
    
    Classic enzyme kinetics model, commonly used in pharma/bio modeling.
    
    Formula: y = Vmax * x / (Km + x)
    
    Args:
        x: Input media spend
        Vmax: Maximum response (asymptote)
        Km: Half-saturation constant (spend at Vmax/2)
    
    Returns:
        Saturated values
    """
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0)
    
    return Vmax * x / (Km + x)


def reach_saturation(
    x: np.ndarray | pd.Series,
    reach_ceiling: float,
    reach_rate: float,
) -> np.ndarray:
    """
    Apply reach-based saturation model.
    
    Models media saturation based on audience reach - you can't
    reach more than 100% of the target audience.
    
    Formula: y = ceiling * (1 - exp(-rate * x))
    
    Args:
        x: Input media spend/GRPs
        reach_ceiling: Maximum reachable audience (0-1 or percentage)
        reach_rate: Rate of reach accumulation
    
    Returns:
        Estimated reach/saturated values
    """
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0)
    
    return reach_ceiling * (1 - np.exp(-reach_rate * x))


def apply_saturation(
    df: pd.DataFrame,
    media_columns: list[str],
    saturation_params: dict[str, dict],
    saturation_type: str = "hill",
) -> pd.DataFrame:
    """
    Apply saturation transformation to multiple media columns.
    
    Args:
        df: DataFrame with media columns
        media_columns: List of column names to transform
        saturation_params: Dict mapping column -> params
            e.g., {"google_spend": {"K": 1000, "S": 1.5}}
        saturation_type: 'hill', 'logistic', 'michaelis_menten', or 'reach'
    
    Returns:
        DataFrame with saturated columns (suffixed with '_saturated')
    
    Example:
        >>> params = {
        ...     "google_spend_adstock": {"K": 1000, "S": 1.2},
        ...     "meta_spend_adstock": {"K": 500, "S": 0.8},
        ... }
        >>> df_sat = apply_saturation(df, list(params.keys()), params)
    """
    df = df.copy()
    
    saturation_funcs = {
        "hill": hill_saturation,
        "logistic": logistic_saturation,
        "michaelis_menten": michaelis_menten_saturation,
        "reach": reach_saturation,
    }
    
    if saturation_type not in saturation_funcs:
        raise ValueError(f"Unknown saturation type: {saturation_type}")
    
    func = saturation_funcs[saturation_type]
    
    for col in media_columns:
        params = saturation_params.get(col, {})
        
        # Set default params based on saturation type
        if saturation_type == "hill" and not params:
            params = {"K": df[col].mean(), "S": 1.0}
        elif saturation_type == "michaelis_menten" and not params:
            params = {"Vmax": 1.0, "Km": df[col].mean()}
        
        df[f"{col}_saturated"] = func(df[col].values, **params)
    
    return df


def estimate_marginal_response(
    x: np.ndarray | pd.Series,
    saturation_type: str = "hill",
    **params
) -> np.ndarray:
    """
    Estimate marginal (incremental) response at each spend level.
    
    This is the derivative of the saturation function - tells you
    how much additional response you get per additional dollar.
    
    Args:
        x: Spend levels to evaluate
        saturation_type: 'hill' or 'michaelis_menten'
        **params: Parameters for saturation function
    
    Returns:
        Marginal response at each x
    """
    x = np.asarray(x, dtype=float)
    
    if saturation_type == "hill":
        K = params.get("K", 1.0)
        S = params.get("S", 1.0)
        
        # Derivative of Hill function
        K_s = np.power(K, S)
        x_s = np.power(x, S)
        
        # d/dx [x^S / (K^S + x^S)]
        return S * K_s * np.power(x, S - 1) / np.power(K_s + x_s, 2)
    
    elif saturation_type == "michaelis_menten":
        Vmax = params.get("Vmax", 1.0)
        Km = params.get("Km", 1.0)
        
        # Derivative: Vmax * Km / (Km + x)^2
        return Vmax * Km / np.power(Km + x, 2)
    
    else:
        # Numerical approximation
        eps = 1e-6
        func = {
            "logistic": logistic_saturation,
            "reach": reach_saturation,
        }.get(saturation_type)
        
        if func is None:
            raise ValueError(f"Unknown saturation type: {saturation_type}")
        
        return (func(x + eps, **params) - func(x, **params)) / eps

