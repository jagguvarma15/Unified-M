"""
Calibration evaluation -- compares MMM predictions against experiment results.

Key metrics:
  - Predicted vs measured lift scatter data
  - Coverage: % of tests where MMM CI contains measured lift
  - Lift error: median |predicted - measured| / measured
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class CalibrationPoint:
    """One test comparison point."""

    test_id: str
    channel: str
    measured_lift: float
    predicted_lift: float
    predicted_ci_lower: float
    predicted_ci_upper: float
    within_ci: bool
    error_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "channel": self.channel,
            "measured_lift": self.measured_lift,
            "predicted_lift": self.predicted_lift,
            "predicted_ci_lower": self.predicted_ci_lower,
            "predicted_ci_upper": self.predicted_ci_upper,
            "within_ci": self.within_ci,
            "error_pct": self.error_pct,
        }


@dataclass
class CalibrationReport:
    """Full calibration evaluation report."""

    points: list[CalibrationPoint] = field(default_factory=list)
    coverage: float = 0.0          # % of tests within CI
    median_lift_error: float = 0.0  # median |pred - meas| / meas
    mean_lift_error: float = 0.0
    n_tests: int = 0
    calibration_quality: str = ""  # good / fair / poor

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "median_lift_error": self.median_lift_error,
            "mean_lift_error": self.mean_lift_error,
            "n_tests": self.n_tests,
            "calibration_quality": self.calibration_quality,
            "points": [p.to_dict() for p in self.points],
        }


def evaluate_calibration(
    experiments: pd.DataFrame,
    mmm_params: dict[str, Any],
    contributions: pd.DataFrame | None = None,
) -> CalibrationReport:
    """
    Compare MMM predictions against experiment results.

    Args:
        experiments:    DataFrame with test results (IncrementalityTestInput schema).
        mmm_params:     Model parameters dict (from get_parameters()).
        contributions:  Optional contributions DataFrame for time-period matching.

    Returns:
        CalibrationReport with per-test comparisons and aggregate metrics.
    """
    if experiments is None or len(experiments) == 0:
        return CalibrationReport(calibration_quality="no_tests")

    coefficients = mmm_params.get("coefficients", {})
    points: list[CalibrationPoint] = []

    for _, test in experiments.iterrows():
        channel = test["channel"]

        # Get MMM predicted coefficient for this channel
        # Match by channel name (try with/without _spend suffix)
        pred_coef = coefficients.get(channel, coefficients.get(f"{channel}_spend", None))
        if pred_coef is None:
            logger.debug(f"No MMM coefficient for channel {channel}, skipping")
            continue

        measured_lift = test["lift_estimate"]

        # Approximate MMM predicted lift (coefficient as proxy)
        # In a full implementation, we'd compute the MMM-predicted lift
        # for the specific test period by summing contributions
        predicted_lift = pred_coef

        # Approximate CI from coefficient (wider if no posterior)
        ci_half = abs(predicted_lift) * 0.2  # 20% uncertainty band
        pred_ci_lower = predicted_lift - ci_half * 1.96
        pred_ci_upper = predicted_lift + ci_half * 1.96

        # Check if measured lift is within predicted CI
        within_ci = pred_ci_lower <= measured_lift <= pred_ci_upper

        # Relative error
        if abs(measured_lift) > 1e-8:
            error_pct = abs(predicted_lift - measured_lift) / abs(measured_lift) * 100
        else:
            error_pct = 0.0

        points.append(CalibrationPoint(
            test_id=test.get("test_id", ""),
            channel=channel,
            measured_lift=measured_lift,
            predicted_lift=predicted_lift,
            predicted_ci_lower=pred_ci_lower,
            predicted_ci_upper=pred_ci_upper,
            within_ci=within_ci,
            error_pct=error_pct,
        ))

    if not points:
        return CalibrationReport(calibration_quality="no_matching_tests")

    # Aggregate metrics
    n_tests = len(points)
    coverage = sum(1 for p in points if p.within_ci) / n_tests
    errors = [p.error_pct for p in points]
    median_error = float(np.median(errors))
    mean_error = float(np.mean(errors))

    # Quality grading
    if coverage >= 0.8 and median_error < 30:
        quality = "good"
    elif coverage >= 0.6 and median_error < 50:
        quality = "fair"
    else:
        quality = "poor"

    logger.info(
        f"Calibration eval: coverage={coverage:.0%}, "
        f"median_error={median_error:.1f}%, quality={quality}"
    )

    return CalibrationReport(
        points=points,
        coverage=round(coverage, 4),
        median_lift_error=round(median_error, 4),
        mean_lift_error=round(mean_error, 4),
        n_tests=n_tests,
        calibration_quality=quality,
    )
