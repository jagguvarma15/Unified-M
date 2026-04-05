"""
Reconciliation module -- fuses measurement signals into unified estimates.
"""

from .calibration import (
    calibrate_mmm_with_tests,
    compute_blended_estimates,
    compute_calibration_factors,
    create_calibration_report,
    estimate_test_coverage,
)
from .engine import (
    ChannelEstimate,
    ReconciliationEngine,
    ReconciliationResult,
)

__all__ = [
    "ReconciliationEngine",
    "ReconciliationResult",
    "ChannelEstimate",
    "compute_calibration_factors",
    "calibrate_mmm_with_tests",
    "compute_blended_estimates",
    "estimate_test_coverage",
    "create_calibration_report",
]
