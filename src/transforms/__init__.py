"""
Transform layer for Unified-M.

Provides adstock, saturation, and feature engineering transformations
for preparing data for MMM modeling.
"""

from transforms.adstock import (
    geometric_adstock,
    weibull_adstock,
    delayed_adstock,
    apply_adstock,
)
from transforms.saturation import (
    hill_saturation,
    logistic_saturation,
    michaelis_menten_saturation,
    apply_saturation,
    estimate_marginal_response,
)
from transforms.features import (
    create_mmm_features,
    pivot_media_spend,
    add_time_features,
    add_fourier_features,
    normalize_features,
)

__all__ = [
    # Adstock
    "geometric_adstock",
    "weibull_adstock",
    "delayed_adstock",
    "apply_adstock",
    # Saturation
    "hill_saturation",
    "logistic_saturation",
    "michaelis_menten_saturation",
    "apply_saturation",
    "estimate_marginal_response",
    # Features
    "create_mmm_features",
    "pivot_media_spend",
    "add_time_features",
    "add_fourier_features",
    "normalize_features",
]

