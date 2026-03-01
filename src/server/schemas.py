"""
API response schemas for FastAPI OpenAPI generation.

These models are used as response_model contracts so frontend types can be
generated from a stable backend schema.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.contracts import RunManifest


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    latest_run: str | None
    version: str
    cache: dict[str, Any] | None = None


class RootResponse(BaseModel):
    name: str
    version: str
    docs: str


class MessageResponse(BaseModel):
    message: str


class StatusResponse(BaseModel):
    status: str


class AdapterBackend(BaseModel):
    name: str
    available: bool
    install_hint: str | None = None


class AdaptersConnectors(BaseModel):
    database: list[str]
    cloud: list[str]
    ad_platforms: list[str]


class AdaptersResponse(BaseModel):
    model_backends: list[AdapterBackend]
    connectors: AdaptersConnectors
    cache: dict[str, Any]


class RunsResponse(BaseModel):
    runs: list[RunManifest]


class ContributionsResponse(BaseModel):
    data: list[dict[str, Any]]
    n_rows: int


class ReconciliationChannelEstimate(BaseModel):
    model_config = ConfigDict(extra="allow")
    channel: str | None = None
    lift_estimate: float
    roi_estimate: float | None = None
    lift_ci_lower: float | None = None
    lift_ci_upper: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    confidence_score: float


class ReconciliationResponse(BaseModel):
    channel_estimates: dict[str, ReconciliationChannelEstimate]
    total_incremental_value: float = 0.0
    reconciliation_method: str | None = None
    timestamp: str | None = None


class ResponseCurveChannel(BaseModel):
    spend: list[float] | None = None
    response: list[float] | None = None
    marginal_response: list[float] | None = None


class AdstockParams(BaseModel):
    decay: float | None = None
    max_lag: int | None = None
    alpha: float | None = None
    l_max: int | None = None


class SaturationParams(BaseModel):
    K: float | None = None
    S: float | None = None


class ParametersResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    coefficients: dict[str, float] = Field(default_factory=dict)
    intercept: float | None = None
    adstock: dict[str, AdstockParams] = Field(default_factory=dict)
    saturation: dict[str, SaturationParams] = Field(default_factory=dict)
    adstock_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    saturation_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    ridge_alpha: float | None = None


class OptimizationResponse(BaseModel):
    optimal_allocation: dict[str, float]
    expected_response: float
    expected_roi: float
    current_allocation: dict[str, float]
    current_response: float
    improvement_pct: float
    total_budget: float
    success: bool
    message: str
    iterations: int


class DiagnosticsChartPoint(BaseModel):
    date: str
    actual: float
    predicted: float
    residual: float | None = None


class DiagnosticsResponse(BaseModel):
    metrics: dict[str, float]
    chart: list[DiagnosticsChartPoint]
    residual_stats: dict[str, float] | None = None


class ROASChannel(BaseModel):
    channel: str
    total_contribution: float
    total_spend: float
    roas: float
    marginal_roi: float | None = None
    cpa: float | None = None


class ROASSummary(BaseModel):
    total_spend: float
    total_contribution: float
    blended_roas: float


class ROASResponse(BaseModel):
    channels: list[ROASChannel]
    summary: ROASSummary


class WaterfallChannel(BaseModel):
    name: str
    value: float


class WaterfallResponse(BaseModel):
    baseline: float
    channels: list[WaterfallChannel]
    total: float


class DataSourceStatus(BaseModel):
    exists: bool
    rows: int | None = None
    columns: list[str] | None = None
    size_bytes: int | None = None
    error: str | None = None


class UploadDataResponse(BaseModel):
    status: str
    data_type: str
    path: str
    rows: int
    columns: list[str]


class PipelineRunTriggerResponse(BaseModel):
    job_id: str
    status: str


class PipelineJobResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    current_step: str
    progress_pct: int
    logs: list[str]
    error: str | None = None
    run_id: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    finished_at: str | None = None


class PipelineJobsResponse(BaseModel):
    jobs: list[PipelineJobResponse]


class CalibrationResponse(BaseModel):
    n_tests: int
    points: list[dict[str, Any]]
    coverage: float | int | None = None
    median_lift_error: float | int | None = None
    mean_lift_error: float | int | None = None
    calibration_quality: str | None = None


class DataQualityGateResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str | None = None
    gate_name: str | None = None
    passed: bool
    severity: str
    message: str | None = None
    details: Any | None = None


class DataQualityResponse(BaseModel):
    timestamp: str
    overall_pass: bool
    n_passed: int
    n_failed: int
    n_warnings: int
    gates: list[DataQualityGateResult]


class ChannelInsight(BaseModel):
    channel: str
    current_spend: float
    optimal_spend: float
    marginal_roi: float
    saturation_point: float
    headroom_pct: float
    status: Literal["under-invested", "efficient", "over-saturated"]
    coefficient: float


class ChannelInsightsResponse(BaseModel):
    channels: list[ChannelInsight]


class SpendPacingChannel(BaseModel):
    channel: str
    planned: float
    actual: float
    diff: float
    pacing_pct: float
    status: Literal["on-track", "over", "under"]


class SpendPacingPoint(BaseModel):
    date: str
    actual: float


class SpendPacingResponse(BaseModel):
    total_planned: float
    total_actual: float
    pacing_pct: float
    channels: list[SpendPacingChannel]
    cumulative: list[SpendPacingPoint]


class ReportTopChannel(BaseModel):
    channel: str
    contribution: float
    share_pct: float


class ReportSummaryResponse(BaseModel):
    run_id: str | None
    generated_at: str
    metrics: dict[str, float]
    roas_summary: dict[str, float]
    top_channels: list[ReportTopChannel]
    recommendations: list[str]
    improvement_pct: float


class SavedConnector(BaseModel):
    id: str
    name: str
    type: str
    subtype: str
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    last_tested: str | None = None
    status: Literal["untested", "connected", "failed"]


class SavedConnectorListResponse(BaseModel):
    connectors: list[SavedConnector]


class ConnectorTestResponse(BaseModel):
    status: str
    connected: bool
    message: str


class ConnectorFetchResponse(BaseModel):
    status: str
    rows: int
    columns: list[str]
    data_type: str
    path: str | None = None
