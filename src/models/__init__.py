"""
Pluggable model backends for Unified-M.

Ships with:
  - ``BuiltinMMM``   -- zero-dependency Ridge + BayesianRidge baseline
  - ``PyMCAdapter``   -- wraps PyMC-Marketing (optional install)

Planned / community:
  - ``MeridianAdapter`` -- Google Meridian
  - ``RobynAdapter``    -- Meta Robyn (via rpy2)

Use ``get_model(name)`` to instantiate the right backend by name.
"""

from models.registry import get_model, list_backends, register_backend
from models.evaluation import (
    compute_mape,
    compute_rmse,
    compute_mae,
    compute_r2,
    evaluate_model,
    cross_validate,
    compute_residual_diagnostics,
    compute_contribution_stability,
)

__all__ = [
    "get_model",
    "list_backends",
    "register_backend",
    "compute_mape",
    "compute_rmse",
    "compute_mae",
    "compute_r2",
    "evaluate_model",
    "cross_validate",
    "compute_residual_diagnostics",
    "compute_contribution_stability",
]
