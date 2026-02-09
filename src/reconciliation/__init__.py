"""
Reconciliation module -- fuses measurement signals into unified estimates.
"""

from .engine import (
    ReconciliationEngine,
    ReconciliationResult,
    ChannelEstimate,
)
from .calibration import (
    compute_calibration_factors,
    calibrate_mmm_with_tests,
    compute_blended_estimates,
    estimate_test_coverage,
    create_calibration_report,
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
