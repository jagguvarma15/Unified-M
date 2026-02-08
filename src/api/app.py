"""
FastAPI application factory and routes.

Serves precomputed results from the data pipeline.
All heavy computation happens in the batch pipeline - 
the API only reads and serves artifacts.
"""

from datetime import datetime
from pathlib import Path
from typing import Any
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from loguru import logger

from config import get_config


# =============================================================================
# Pydantic Models for API
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class ChannelContribution(BaseModel):
    channel: str
    contribution: float
    contribution_share: float
    spend: float
    roi: float


class ContributionsResponse(BaseModel):
    date_range: tuple[str, str]
    total_contribution: float
    baseline_contribution: float
    channels: list[ChannelContribution]


class ChannelEstimate(BaseModel):
    channel: str
    lift_estimate: float
    lift_ci_lower: float
    lift_ci_upper: float
    confidence_score: float
    roi_estimate: float


class ReconciliationResponse(BaseModel):
    timestamp: str
    method: str
    total_incremental_value: float
    channels: list[ChannelEstimate]


class AllocationRecommendation(BaseModel):
    channel: str
    current_spend: float
    recommended_spend: float
    change_pct: float
    expected_response: float
    marginal_roi: float


class OptimizationResponse(BaseModel):
    total_budget: float
    expected_response: float
    improvement_pct: float
    recommendations: list[AllocationRecommendation]


class ResponseCurvePoint(BaseModel):
    spend: float
    response: float


class ResponseCurveResponse(BaseModel):
    channel: str
    curve: list[ResponseCurvePoint]
    current_spend: float
    optimal_spend: float


class ScenarioComparison(BaseModel):
    name: str
    total_budget: float
    expected_response: float
    roi: float
    allocation: dict[str, float]


class ScenariosResponse(BaseModel):
    scenarios: list[ScenarioComparison]


# =============================================================================
# Data Store (reads precomputed artifacts)
# =============================================================================

class ResultsStore:
    """
    Manages access to precomputed pipeline results.
    
    All results are loaded from disk (Parquet/JSON files)
    produced by the batch training pipeline.
    """
    
    def __init__(self, outputs_path: Path | None = None):
        config = get_config()
        self.outputs_path = outputs_path or config.storage.outputs_path
        self._cache: dict[str, Any] = {}
        self._cache_time: dict[str, datetime] = {}
        self._cache_ttl = 60  # seconds
    
    def _get_cached(self, key: str) -> Any | None:
        """Get cached value if not expired."""
        if key in self._cache:
            age = (datetime.now() - self._cache_time[key]).seconds
            if age < self._cache_ttl:
                return self._cache[key]
        return None
    
    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached value."""
        self._cache[key] = value
        self._cache_time[key] = datetime.now()
    
    def get_contributions(self) -> dict | None:
        """Load channel contributions."""
        cached = self._get_cached("contributions")
        if cached:
            return cached
        
        path = self.outputs_path / "contributions.parquet"
        if not path.exists():
            return None
        
        df = pd.read_parquet(path)
        result = {
            "data": df.to_dict(orient="records"),
            "summary": self._summarize_contributions(df),
        }
        
        self._set_cached("contributions", result)
        return result
    
    def _summarize_contributions(self, df: pd.DataFrame) -> dict:
        """Summarize contribution data."""
        # Identify contribution columns
        contrib_cols = [c for c in df.columns if c.endswith("_contribution")]
        
        summary = {
            "date_range": (
                str(df["date"].min()) if "date" in df.columns else None,
                str(df["date"].max()) if "date" in df.columns else None,
            ),
            "total_contribution": sum(df[c].sum() for c in contrib_cols),
            "baseline_contribution": df["baseline_contribution"].sum() if "baseline_contribution" in df.columns else 0,
            "channels": {},
        }
        
        for col in contrib_cols:
            channel = col.replace("_contribution", "")
            summary["channels"][channel] = {
                "total": df[col].sum(),
                "share": df[col].sum() / summary["total_contribution"] if summary["total_contribution"] > 0 else 0,
            }
        
        return summary
    
    def get_reconciliation(self) -> dict | None:
        """Load reconciliation results."""
        cached = self._get_cached("reconciliation")
        if cached:
            return cached
        
        path = self.outputs_path / "reconciliation.json"
        if not path.exists():
            return None
        
        with open(path) as f:
            result = json.load(f)
        
        self._set_cached("reconciliation", result)
        return result
    
    def get_optimization(self) -> dict | None:
        """Load optimization results."""
        cached = self._get_cached("optimization")
        if cached:
            return cached
        
        path = self.outputs_path / "optimization.json"
        if not path.exists():
            return None
        
        with open(path) as f:
            result = json.load(f)
        
        self._set_cached("optimization", result)
        return result
    
    def get_response_curves(self) -> dict | None:
        """Load response curves."""
        cached = self._get_cached("response_curves")
        if cached:
            return cached
        
        path = self.outputs_path / "response_curves.json"
        if not path.exists():
            return None
        
        with open(path) as f:
            result = json.load(f)
        
        self._set_cached("response_curves", result)
        return result
    
    def get_mmm_results(self) -> dict | None:
        """Load MMM model results."""
        cached = self._get_cached("mmm_results")
        if cached:
            return cached
        
        path = self.outputs_path / "mmm_results.json"
        if not path.exists():
            return None
        
        with open(path) as f:
            result = json.load(f)
        
        self._set_cached("mmm_results", result)
        return result
    
    def get_scenarios(self) -> dict | None:
        """Load scenario comparisons."""
        path = self.outputs_path / "scenarios.json"
        if not path.exists():
            return None
        
        with open(path) as f:
            return json.load(f)


# Global store instance
_store: ResultsStore | None = None


def get_store() -> ResultsStore:
    global _store
    if _store is None:
        _store = ResultsStore()
    return _store


# =============================================================================
# FastAPI Application
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Unified-M API",
        description="API for Unified Marketing Measurement results",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ==========================================================================
    # Health & Meta Endpoints
    # ==========================================================================
    
    @app.get("/health", response_model=HealthResponse)
    def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="0.1.0",
        )
    
    @app.get("/")
    def root():
        """Root endpoint with API info."""
        return {
            "name": "Unified-M API",
            "version": "0.1.0",
            "docs": "/docs",
            "endpoints": {
                "contributions": "/api/v1/contributions",
                "reconciliation": "/api/v1/reconciliation",
                "optimization": "/api/v1/optimization",
                "response_curves": "/api/v1/response-curves",
                "scenarios": "/api/v1/scenarios",
            },
        }
    
    # ==========================================================================
    # MMM Results Endpoints
    # ==========================================================================
    
    @app.get("/api/v1/contributions")
    def get_contributions(
        start_date: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
        end_date: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    ):
        """
        Get channel contribution decomposition.
        
        Returns the breakdown of total response by channel,
        showing how much each channel contributed to outcomes.
        """
        store = get_store()
        data = store.get_contributions()
        
        if data is None:
            raise HTTPException(
                status_code=404,
                detail="Contributions not available. Run the training pipeline first.",
            )
        
        return data
    
    @app.get("/api/v1/mmm/summary")
    def get_mmm_summary():
        """
        Get MMM model summary.
        
        Returns model fit metrics, coefficients, and parameter estimates.
        """
        store = get_store()
        data = store.get_mmm_results()
        
        if data is None:
            raise HTTPException(
                status_code=404,
                detail="MMM results not available. Run the training pipeline first.",
            )
        
        return data
    
    # ==========================================================================
    # Reconciliation Endpoints
    # ==========================================================================
    
    @app.get("/api/v1/reconciliation")
    def get_reconciliation():
        """
        Get reconciled channel estimates.
        
        Returns unified lift estimates combining MMM, incrementality tests,
        and attribution signals with calibrated uncertainty.
        """
        store = get_store()
        data = store.get_reconciliation()
        
        if data is None:
            raise HTTPException(
                status_code=404,
                detail="Reconciliation results not available. Run the pipeline first.",
            )
        
        return data
    
    @app.get("/api/v1/reconciliation/channel/{channel}")
    def get_channel_reconciliation(channel: str):
        """Get reconciliation details for a specific channel."""
        store = get_store()
        data = store.get_reconciliation()
        
        if data is None:
            raise HTTPException(status_code=404, detail="Results not available")
        
        estimates = data.get("channel_estimates", {})
        if channel not in estimates:
            raise HTTPException(
                status_code=404,
                detail=f"Channel '{channel}' not found",
            )
        
        return estimates[channel]
    
    # ==========================================================================
    # Optimization Endpoints
    # ==========================================================================
    
    @app.get("/api/v1/optimization")
    def get_optimization():
        """
        Get budget optimization recommendations.
        
        Returns optimal budget allocation across channels
        to maximize expected response.
        """
        store = get_store()
        data = store.get_optimization()
        
        if data is None:
            raise HTTPException(
                status_code=404,
                detail="Optimization results not available. Run the pipeline first.",
            )
        
        return data
    
    @app.get("/api/v1/response-curves")
    def get_response_curves(
        channel: str | None = Query(None, description="Filter by channel"),
    ):
        """
        Get response curves for channels.
        
        Response curves show the relationship between spend
        and expected response, capturing saturation effects.
        """
        store = get_store()
        data = store.get_response_curves()
        
        if data is None:
            raise HTTPException(
                status_code=404,
                detail="Response curves not available. Run the pipeline first.",
            )
        
        if channel:
            if channel not in data:
                raise HTTPException(
                    status_code=404,
                    detail=f"Channel '{channel}' not found",
                )
            return {channel: data[channel]}
        
        return data
    
    @app.get("/api/v1/scenarios")
    def get_scenarios():
        """
        Get budget scenario comparisons.
        
        Compare different budget allocation scenarios
        to support strategic planning.
        """
        store = get_store()
        data = store.get_scenarios()
        
        if data is None:
            raise HTTPException(
                status_code=404,
                detail="Scenarios not available. Run the pipeline first.",
            )
        
        return data
    
    return app


# Create default app instance
app = create_app()


# =============================================================================
# CLI Entry Point
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server."""
    import uvicorn
    
    logger.info(f"Starting Unified-M API on {host}:{port}")
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=reload,
    )

