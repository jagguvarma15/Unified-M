"""
Marketing Mix Model (MMM) implementation for Unified-M.

Wraps PyMC-Marketing with additional utilities for training,
evaluation, and contribution decomposition.
"""

from mmm.model import (
    UnifiedMMM,
    MMMResults,
)
from mmm.evaluation import (
    evaluate_model,
    cross_validate,
    compute_mape,
    compute_rmse,
)
from mmm.decomposition import (
    decompose_contributions,
    compute_channel_roi,
    compute_marginal_roi,
)

__all__ = [
    "UnifiedMMM",
    "MMMResults",
    "evaluate_model",
    "cross_validate",
    "compute_mape",
    "compute_rmse",
    "decompose_contributions",
    "compute_channel_roi",
    "compute_marginal_roi",
]

