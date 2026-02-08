"""
Reconciliation layer for Unified-M.

Fuses MMM estimates with incrementality tests and attribution signals
to produce unified channel-level lift estimates with calibrated uncertainty.
"""

from reconciliation.fusion import (
    ReconciliationEngine,
    ReconciliationResult,
    weighted_average_fusion,
    bayesian_fusion,
)
from reconciliation.calibration import (
    calibrate_mmm_with_tests,
    compute_calibration_factors,
)

__all__ = [
    "ReconciliationEngine",
    "ReconciliationResult",
    "weighted_average_fusion",
    "bayesian_fusion",
    "calibrate_mmm_with_tests",
    "compute_calibration_factors",
]

