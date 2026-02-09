"""
Optional Rust acceleration for transformations.

If the Rust extension is installed, use it for 10-100x speedup.
Otherwise, falls back to Python implementations.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

# Try to import Rust extension
try:
    import unified_m_core

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    unified_m_core = None


def geometric_adstock_rust(
    x: np.ndarray,
    alpha: float,
    l_max: int = 8,
    normalize: bool = True,
) -> np.ndarray:
    """
    Fast Rust implementation of geometric adstock.
    
    Falls back to Python if Rust extension not available.
    """
    if RUST_AVAILABLE:
        return unified_m_core.geometric_adstock_rust(x, alpha, l_max, normalize)
    else:
        # Fallback to Python implementation
        from .adstock import geometric_adstock
        return geometric_adstock(x, alpha, l_max, normalize)


def weibull_adstock_rust(
    x: np.ndarray,
    shape: float,
    scale: float,
    l_max: int = 8,
) -> np.ndarray:
    """Fast Rust implementation of Weibull adstock."""
    if RUST_AVAILABLE:
        return unified_m_core.weibull_adstock_rust(x, shape, scale, l_max)
    else:
        from .adstock import weibull_adstock
        return weibull_adstock(x, shape, scale, l_max)


def hill_saturation_rust(
    x: np.ndarray,
    k: float,
    s: float,
    coef: float = 1.0,
) -> np.ndarray:
    """Fast Rust implementation of Hill saturation."""
    if RUST_AVAILABLE:
        return unified_m_core.hill_saturation_rust(x, k, s, coef)
    else:
        from .saturation import hill_saturation
        return hill_saturation(x, k, s, coef)


def optimize_budget_rust(
    response_params: list[tuple[str, tuple[float, float, float]]],
    total_budget: float,
    min_budget_pct: float = 0.0,
    max_budget_pct: float = 1.0,
    channel_constraints: list[tuple[str, tuple[float, float]]] | None = None,
) -> dict[str, float]:
    """
    Fast Rust implementation of budget optimization.
    
    Args:
        response_params: List of (channel, (K, S, coef)) tuples
        total_budget: Total budget to allocate
        min_budget_pct: Minimum % per channel
        max_budget_pct: Maximum % per channel
        channel_constraints: List of (channel, (min, max)) tuples
    
    Returns:
        Dict mapping channel -> optimal spend
    """
    if RUST_AVAILABLE:
        constraints = channel_constraints or []
        result = unified_m_core.optimize_budget_rust(
            response_params,
            total_budget,
            min_budget_pct,
            max_budget_pct,
            constraints,
        )
        # Convert PyDict to regular dict
        return {k: float(v) for k, v in result.items()}
    else:
        # Fallback to Python implementation
        from ..optimization.allocator import BudgetOptimizer
        
        # Build response curves from params
        curves = {}
        for channel, (k, s, coef) in response_params:
            def make_curve(k, s, coef):
                def curve(x):
                    x = max(0, x)
                    return coef * (x ** s) / (k ** s + x ** s)
                return curve
            curves[channel] = make_curve(k, s, coef)
        
        optimizer = BudgetOptimizer(
            response_curves=curves,
            total_budget=total_budget,
            min_budget_pct=min_budget_pct,
            max_budget_pct=max_budget_pct,
            channel_constraints=dict(channel_constraints) if channel_constraints else None,
        )
        
        result = optimizer.optimize()
        return result.optimal_allocation
