"""
Marketing Mix Model (MMM) implementation for Unified-M.

Wraps PyMC-Marketing with additional utilities for training,
evaluation, and contribution decomposition.
"""

from unified_m.mmm.model import (
    UnifiedMMM,
    MMMResults,
)
from unified_m.mmm.evaluation import (
    evaluate_model,
    cross_validate,
    compute_mape,
    compute_rmse,
)
from unified_m.mmm.decomposition import (
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

