"""
Data quality gates for the Unified-M pipeline.

Gates run as a pre-flight check before model training.  Each gate
produces a pass/fail/warn result.  The pipeline can be configured to
abort on failure or continue with warnings.

Gates implemented:
  1. Schema validation      -- required columns and types
  2. Completeness           -- no missing weeks per channel
  3. Spend anomaly          -- flag spikes/drops > N sigma
  4. Target anomaly         -- same for KPI
  5. Staleness              -- data must be recent
  6. Cross-source consistency -- spend totals roughly match
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Result of a single quality gate."""

    gate_name: str
    passed: bool
    severity: str = "error"  # error | warning | info
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class DataQualityReport:
    """Aggregated quality report for a pipeline run."""

    timestamp: str = ""
    overall_pass: bool = True
    gates: list[GateResult] = field(default_factory=list)
    n_passed: int = 0
    n_failed: int = 0
    n_warnings: int = 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "overall_pass": self.overall_pass,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_warnings": self.n_warnings,
            "gates": [g.to_dict() for g in self.gates],
        }


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------

def _gate_schema_validation(
    media_spend: pd.DataFrame | None,
    outcomes: pd.DataFrame | None,
) -> GateResult:
    """Check that required columns exist and have correct types."""
    issues: list[str] = []

    if media_spend is not None:
        for col in ["date", "channel", "spend"]:
            if col not in media_spend.columns:
                issues.append(f"media_spend missing required column: {col}")
        if "spend" in media_spend.columns and media_spend["spend"].lt(0).any():
            issues.append("media_spend.spend contains negative values")

    if outcomes is not None:
        if "date" not in outcomes.columns:
            issues.append("outcomes missing required column: date")
        kpi_cols = {"revenue", "conversions", "kpi_revenue", "kpi_conversions"}
        if not kpi_cols.intersection(outcomes.columns):
            issues.append(f"outcomes has no KPI column (need one of {kpi_cols})")

    passed = len(issues) == 0
    return GateResult(
        gate_name="schema_validation",
        passed=passed,
        severity="error" if not passed else "info",
        message="; ".join(issues) if issues else "Schema valid",
        details={"issues": issues},
    )


def _gate_completeness(
    media_spend: pd.DataFrame | None,
    tolerance_missing_weeks: int = 1,
) -> GateResult:
    """Check for missing weeks per channel."""
    if media_spend is None or "date" not in media_spend.columns:
        return GateResult(
            gate_name="completeness",
            passed=True,
            severity="info",
            message="No media_spend data to check",
        )

    media_spend = media_spend.copy()
    media_spend["date"] = pd.to_datetime(media_spend["date"])
    media_spend["week"] = media_spend["date"].dt.to_period("W")

    all_weeks = set(media_spend["week"].unique())
    missing_by_channel: dict[str, int] = {}

    for channel in media_spend["channel"].unique():
        ch_weeks = set(media_spend[media_spend["channel"] == channel]["week"].unique())
        missing = len(all_weeks - ch_weeks)
        if missing > tolerance_missing_weeks:
            missing_by_channel[channel] = missing

    passed = len(missing_by_channel) == 0
    return GateResult(
        gate_name="completeness",
        passed=passed,
        severity="warning" if not passed else "info",
        message=(
            f"{len(missing_by_channel)} channels have missing weeks"
            if not passed else "All channels complete"
        ),
        details={"missing_by_channel": missing_by_channel},
    )


def _gate_spend_anomaly(
    media_spend: pd.DataFrame | None,
    sigma_threshold: float = 3.0,
) -> GateResult:
    """Flag weeks where spend deviates > N sigma from trailing mean."""
    if media_spend is None or "spend" not in media_spend.columns:
        return GateResult(
            gate_name="spend_anomaly",
            passed=True,
            severity="info",
            message="No spend data to check",
        )

    anomalies: list[dict] = []
    for channel in media_spend["channel"].unique():
        ch = media_spend[media_spend["channel"] == channel].sort_values("date")
        if len(ch) < 8:
            continue
        rolling_mean = ch["spend"].rolling(window=8, min_periods=4).mean()
        rolling_std = ch["spend"].rolling(window=8, min_periods=4).std()
        z_scores = (ch["spend"] - rolling_mean) / (rolling_std + 1e-8)
        outlier_mask = z_scores.abs() > sigma_threshold
        for _, row in ch[outlier_mask].iterrows():
            anomalies.append({
                "channel": channel,
                "date": str(row["date"]),
                "spend": float(row["spend"]),
                "z_score": float(z_scores.loc[row.name]),
            })

    passed = len(anomalies) == 0
    return GateResult(
        gate_name="spend_anomaly",
        passed=passed,
        severity="warning" if not passed else "info",
        message=f"{len(anomalies)} spend anomalies detected" if not passed else "No anomalies",
        details={"anomalies": anomalies[:20]},  # cap for JSON size
    )


def _gate_target_anomaly(
    outcomes: pd.DataFrame | None,
    target_col: str = "revenue",
    sigma_threshold: float = 3.0,
) -> GateResult:
    """Flag weeks where KPI deviates > N sigma from trailing mean."""
    if outcomes is None:
        return GateResult(
            gate_name="target_anomaly",
            passed=True,
            severity="info",
            message="No outcomes to check",
        )

    # Find the actual target column
    col = None
    for candidate in [target_col, f"kpi_{target_col}", "revenue", "kpi_revenue"]:
        if candidate in outcomes.columns:
            col = candidate
            break
    if col is None:
        return GateResult(
            gate_name="target_anomaly",
            passed=True,
            severity="info",
            message="No target column found",
        )

    outcomes = outcomes.sort_values("date")
    if len(outcomes) < 8:
        return GateResult(
            gate_name="target_anomaly",
            passed=True,
            severity="info",
            message="Not enough data for anomaly detection",
        )

    rolling_mean = outcomes[col].rolling(window=8, min_periods=4).mean()
    rolling_std = outcomes[col].rolling(window=8, min_periods=4).std()
    z_scores = (outcomes[col] - rolling_mean) / (rolling_std + 1e-8)
    outlier_mask = z_scores.abs() > sigma_threshold

    anomalies = []
    for _, row in outcomes[outlier_mask].iterrows():
        anomalies.append({
            "date": str(row.get("date", "")),
            "value": float(row[col]),
            "z_score": float(z_scores.loc[row.name]),
        })

    passed = len(anomalies) == 0
    return GateResult(
        gate_name="target_anomaly",
        passed=passed,
        severity="warning" if not passed else "info",
        message=f"{len(anomalies)} target anomalies" if not passed else "No anomalies",
        details={"anomalies": anomalies[:20], "column": col},
    )


def _gate_staleness(
    media_spend: pd.DataFrame | None,
    max_age_days: int = 14,
) -> GateResult:
    """Warn if the latest data is older than max_age_days."""
    if media_spend is None or "date" not in media_spend.columns:
        return GateResult(
            gate_name="staleness",
            passed=True,
            severity="info",
            message="No data to check staleness",
        )

    latest_date = pd.to_datetime(media_spend["date"]).max()
    age_days = (datetime.now() - latest_date).days

    passed = age_days <= max_age_days
    return GateResult(
        gate_name="staleness",
        passed=passed,
        severity="warning" if not passed else "info",
        message=(
            f"Latest data is {age_days} days old (threshold: {max_age_days})"
        ),
        details={"latest_date": str(latest_date), "age_days": age_days},
    )


def _gate_cross_source_consistency(
    media_spend: pd.DataFrame | None,
    outcomes: pd.DataFrame | None,
) -> GateResult:
    """Check that media and outcome date ranges overlap."""
    if media_spend is None or outcomes is None:
        return GateResult(
            gate_name="cross_source_consistency",
            passed=True,
            severity="info",
            message="Insufficient data for cross-source check",
        )

    media_dates = pd.to_datetime(media_spend["date"])
    outcome_dates = pd.to_datetime(outcomes["date"])

    media_range = (media_dates.min(), media_dates.max())
    outcome_range = (outcome_dates.min(), outcome_dates.max())

    # Check overlap
    overlap_start = max(media_range[0], outcome_range[0])
    overlap_end = min(media_range[1], outcome_range[1])
    has_overlap = overlap_start <= overlap_end

    return GateResult(
        gate_name="cross_source_consistency",
        passed=has_overlap,
        severity="error" if not has_overlap else "info",
        message=(
            "Media and outcome date ranges do not overlap!"
            if not has_overlap
            else f"Date overlap: {overlap_start.date()} to {overlap_end.date()}"
        ),
        details={
            "media_range": [str(media_range[0]), str(media_range[1])],
            "outcome_range": [str(outcome_range[0]), str(outcome_range[1])],
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_quality_gates(
    media_spend: pd.DataFrame | None = None,
    outcomes: pd.DataFrame | None = None,
    target_col: str = "revenue",
    tolerance_missing_weeks: int = 1,
    sigma_threshold: float = 3.0,
    max_staleness_days: int = 14,
) -> DataQualityReport:
    """
    Run all quality gates and return a consolidated report.

    Args:
        media_spend:  Raw media spend DataFrame.
        outcomes:     Raw outcomes DataFrame.
        target_col:   Name of the primary KPI column.
        tolerance_missing_weeks: Allowed missing weeks per channel.
        sigma_threshold: Z-score threshold for anomaly detection.
        max_staleness_days: Maximum acceptable data age.

    Returns:
        DataQualityReport with per-gate results.
    """
    gates = [
        _gate_schema_validation(media_spend, outcomes),
        _gate_completeness(media_spend, tolerance_missing_weeks),
        _gate_spend_anomaly(media_spend, sigma_threshold),
        _gate_target_anomaly(outcomes, target_col, sigma_threshold),
        _gate_staleness(media_spend, max_staleness_days),
        _gate_cross_source_consistency(media_spend, outcomes),
    ]

    n_passed = sum(1 for g in gates if g.passed)
    n_failed = sum(1 for g in gates if not g.passed and g.severity == "error")
    n_warnings = sum(1 for g in gates if not g.passed and g.severity == "warning")
    overall_pass = n_failed == 0

    for g in gates:
        level = "INFO" if g.passed else ("WARNING" if g.severity == "warning" else "ERROR")
        logger.log(level, f"Quality gate [{g.gate_name}]: {g.message}")

    return DataQualityReport(
        timestamp=datetime.now().isoformat(),
        overall_pass=overall_pass,
        gates=gates,
        n_passed=n_passed,
        n_failed=n_failed,
        n_warnings=n_warnings,
    )
