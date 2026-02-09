"""
Canonical data contracts for Unified-M.

These Pydantic models define the single source of truth for every data
boundary in the framework.  Any MMM engine -- built-in, Meridian, Robyn,
or custom -- must accept and produce data that conforms to these contracts.

The contracts are intentionally strict on required fields but permissive
on additional columns so teams can attach domain-specific metadata without
forking the schema.

Design principles:
  - All dates are ISO-8601 strings or datetime objects (coerced).
  - Money amounts are floats in the currency of the project.
  - Uncertainty is always represented as a (lower, upper) credible interval.
  - Every output includes a confidence_score in [0, 1].
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TestType(str, Enum):
    GEO_LIFT = "geo_lift"
    HOLDOUT = "holdout"
    SWITCHBACK = "switchback"
    SYNTHETIC_CONTROL = "synthetic_control"
    CONVERSION_LIFT = "conversion_lift"
    OTHER = "other"


class AttributionModel(str, Enum):
    LAST_CLICK = "last_click"
    FIRST_CLICK = "first_click"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    POSITION_BASED = "position_based"
    DATA_DRIVEN = "data_driven"
    PLATFORM_REPORTED = "platform_reported"
    OTHER = "other"


class FusionMethod(str, Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    BAYESIAN = "bayesian"
    HIERARCHICAL = "hierarchical"


# ---------------------------------------------------------------------------
# Input contracts  (what users feed into the framework)
# ---------------------------------------------------------------------------

class MediaSpendInput(BaseModel):
    """One row of media spend data (long format: one row per date x channel)."""

    date: datetime
    channel: str = Field(min_length=1, max_length=120)
    spend: float = Field(ge=0)
    impressions: float | None = Field(default=None, ge=0)
    clicks: float | None = Field(default=None, ge=0)

    class Config:
        extra = "allow"


class OutcomeInput(BaseModel):
    """One row of outcome / KPI data."""

    date: datetime
    revenue: float | None = Field(default=None, ge=0)
    conversions: float | None = Field(default=None, ge=0)

    class Config:
        extra = "allow"


class ControlInput(BaseModel):
    """One row of control-variable data (non-media factors)."""

    date: datetime

    class Config:
        extra = "allow"


class IncrementalityTestInput(BaseModel):
    """Result of a single incrementality / lift test."""

    test_id: str
    channel: str
    start_date: datetime
    end_date: datetime
    test_type: TestType
    lift_estimate: float
    lift_ci_lower: float
    lift_ci_upper: float
    confidence_level: float | None = Field(default=0.95, ge=0, le=1)
    spend_during_test: float | None = Field(default=None, ge=0)

    class Config:
        extra = "allow"


class AttributionInput(BaseModel):
    """One row of platform / MTA attribution data."""

    date: datetime
    channel: str
    model_type: AttributionModel
    attributed_conversions: float | None = Field(default=None, ge=0)
    attributed_revenue: float | None = Field(default=None, ge=0)

    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Internal / pipeline contracts
# ---------------------------------------------------------------------------

class ChannelConfig(BaseModel):
    """Per-channel configuration for transforms and priors."""

    name: str
    adstock_type: str = "geometric"
    adstock_params: dict[str, float] = Field(default_factory=lambda: {"alpha": 0.5, "l_max": 8})
    saturation_type: str = "hill"
    saturation_params: dict[str, float] = Field(default_factory=lambda: {"K": 5000, "S": 1.0})
    spend_column: str | None = None  # auto-derived if None


class MMMDataset(BaseModel):
    """
    Validated, model-ready dataset.

    This is the single object that every model adapter receives.  It
    wraps a pandas DataFrame together with metadata about which columns
    are media, controls, and the target.
    """

    date_column: str = "date"
    target_column: str = "y"
    media_columns: list[str] = Field(default_factory=list)
    control_columns: list[str] = Field(default_factory=list)
    channel_configs: dict[str, ChannelConfig] = Field(default_factory=dict)
    n_rows: int = 0
    date_range: tuple[str, str] = ("", "")
    data_hash: str = ""

    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Output contracts  (what the framework produces)
# ---------------------------------------------------------------------------

class ModelMetrics(BaseModel):
    """Standard model-fit metrics."""

    mape: float = Field(description="Mean Absolute Percentage Error (%)")
    rmse: float = Field(description="Root Mean Squared Error")
    mae: float = Field(description="Mean Absolute Error")
    r_squared: float = Field(description="R-squared (coefficient of determination)")
    nrmse: float | None = Field(default=None, description="Normalized RMSE")

    # Bayesian diagnostics (None when not applicable)
    rhat_max: float | None = Field(default=None, description="Max R-hat across parameters")
    ess_min: float | None = Field(default=None, description="Min effective sample size")
    divergences: int | None = Field(default=None, description="Number of divergent transitions")


class ChannelResult(BaseModel):
    """Unified result for a single channel after reconciliation."""

    channel: str
    lift_estimate: float
    lift_ci_lower: float
    lift_ci_upper: float
    confidence_score: float = Field(ge=0, le=1)

    total_spend: float = 0.0
    total_contribution: float = 0.0
    roi: float = 0.0
    roas: float = 0.0
    marginal_roi: float = 0.0

    # Source signal weights
    mmm_weight_used: float = 0.0
    incrementality_weight_used: float = 0.0
    attribution_weight_used: float = 0.0

    # Calibration
    calibration_factor: float | None = None
    last_test_date: str | None = None
    last_test_type: str | None = None


class ResponseCurvePoint(BaseModel):
    """Single point on a channel response curve."""
    spend: float
    response: float
    marginal_response: float = 0.0


class OptimizationOutput(BaseModel):
    """Result of a budget optimization run."""

    total_budget: float
    expected_response: float
    expected_roi: float
    improvement_pct: float = 0.0
    success: bool = True
    message: str = ""
    channel_allocations: dict[str, float] = Field(default_factory=dict)
    current_allocations: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Run metadata  (audit trail)
# ---------------------------------------------------------------------------

class RunManifest(BaseModel):
    """
    Immutable record of a single pipeline run.

    Written to ``runs/<run_id>/manifest.json`` so every result is
    traceable back to the exact config, data, and code that produced it.
    """

    run_id: str
    timestamp: str
    duration_seconds: float = 0.0
    status: str = "pending"  # pending | running | completed | failed

    # What was run
    model_backend: str = ""
    pipeline_steps: list[str] = Field(default_factory=list)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)

    # Data fingerprint
    data_hash: str = ""
    n_rows: int = 0
    n_channels: int = 0
    date_range: tuple[str, str] = ("", "")

    # Results summary
    metrics: ModelMetrics | None = None
    n_channel_results: int = 0
    total_incremental_value: float = 0.0

    # Error info (if failed)
    error_message: str | None = None
    error_step: str | None = None
