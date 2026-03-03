"""
FastAPI application for Unified-M.

Design principles:
  - Read-only: all data comes from the artifact store.
  - Instant: no model inference on the hot path.
  - CORS-open by default for local dashboards.
  - Every endpoint returns JSON with a consistent envelope.
"""

from __future__ import annotations

from datetime import datetime
import tempfile
import os
import threading
from collections import Counter

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
import json
import re

from core.artifacts import ArtifactStore
from config import get_config
from core.contracts import RunManifest
from server.schemas import (
    AdaptersResponse,
    CalibrationResponse,
    ChannelInsightsResponse,
    ConnectorFetchResponse,
    ConnectorRevealResponse,
    ConnectorTestResponse,
    ContributionsResponse,
    DataQualityResponse,
    DataSourceStatus,
    DiagnosticsResponse,
    HealthResponse,
    MessageResponse,
    OptimizationResponse,
    ParametersResponse,
    PipelineJobResponse,
    PipelineJobsResponse,
    PipelineRunTriggerResponse,
    ReconciliationResponse,
    ReportSummaryResponse,
    RootResponse,
    ROASResponse,
    RunsResponse,
    SavedConnector,
    SavedConnectorListResponse,
    SpendPacingResponse,
    StatusResponse,
    TelemetryIngestRequest,
    TelemetrySummaryResponse,
    UploadDataResponse,
    WaterfallResponse,
    ResponseCurveChannel,
)


# Known pipeline data types; custom names allowed via _is_valid_custom_data_type
KNOWN_DATA_TYPES = frozenset({
    "media_spend", "outcomes", "controls", "incrementality_tests", "attribution",
})
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MB


def _is_valid_data_type(data_type: str) -> bool:
    """Accept known types or custom names: letter, then alphanumeric/underscore, 1–64 chars."""
    if data_type in KNOWN_DATA_TYPES:
        return True
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$", data_type))


def _validate_data_type(data_type: str) -> None:
    if not _is_valid_data_type(data_type):
        raise HTTPException(
            400,
            "data_type must be one of media_spend, outcomes, controls, incrementality_tests, attribution, "
            "or a custom name (letter followed by letters, numbers, underscores, max 64 chars)",
        )


def _max_upload_size_bytes() -> int:
    """
    Maximum accepted upload size in bytes.

    Controlled via MAX_UPLOAD_SIZE_MB (default: 200 MB).
    """
    raw = os.getenv("MAX_UPLOAD_SIZE_MB", "200")
    try:
        mb = int(raw)
    except (TypeError, ValueError):
        mb = 200
    mb = max(1, mb)
    return mb * 1024 * 1024


async def _stream_upload_to_tempfile(
    upload: UploadFile,
    suffix: str,
    max_bytes: int,
) -> tuple[Path, int]:
    """
    Stream an UploadFile to a temporary file in chunks with size enforcement.
    """
    written = 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(
                    413,
                    f"File too large. Max upload size is {max_bytes // (1024 * 1024)} MB.",
                )
            tmp.write(chunk)
    return tmp_path, written


def _normalize_parameters_payload(data: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure parameter payload always exposes canonical keys used by the UI/API:
      - adstock[channel] = {decay, max_lag, alpha, l_max}
      - saturation[channel] = {K, S}

    Legacy keys (adstock_params/saturation_params) are preserved for backward
    compatibility.
    """
    normalized = dict(data)
    adstock_params = normalized.get("adstock_params", {}) or {}
    saturation_params = normalized.get("saturation_params", {}) or {}

    if "adstock" not in normalized:
        normalized["adstock"] = {
            ch: {
                "decay": p.get("alpha"),
                "max_lag": p.get("l_max"),
                "alpha": p.get("alpha"),
                "l_max": p.get("l_max"),
            }
            for ch, p in adstock_params.items()
            if isinstance(p, dict)
        }
    if "saturation" not in normalized:
        normalized["saturation"] = {
            ch: {
                "K": p.get("K"),
                "S": p.get("S"),
            }
            for ch, p in saturation_params.items()
            if isinstance(p, dict)
        }
    return normalized


def _normalize_optimization_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure optimization payload exposes canonical keys used by the UI."""
    normalized = dict(data)

    optimal = normalized.get("optimal_allocation") or normalized.get("channel_allocations") or {}
    current = normalized.get("current_allocation") or normalized.get("current_allocations") or {}
    normalized["optimal_allocation"] = {str(k): float(v) for k, v in optimal.items()}
    normalized["current_allocation"] = {str(k): float(v) for k, v in current.items()}

    if "expected_response" not in normalized:
        normalized["expected_response"] = float(normalized.get("optimized_response", 0.0))
    if "current_response" not in normalized:
        normalized["current_response"] = float(normalized.get("baseline_response", 0.0))
    if "total_budget" not in normalized:
        normalized["total_budget"] = float(sum(normalized["optimal_allocation"].values()))
    normalized.setdefault("expected_roi", 0.0)
    normalized.setdefault("improvement_pct", 0.0)
    normalized.setdefault("success", True)
    normalized.setdefault("message", "")
    normalized.setdefault("iterations", 0)

    return normalized


def _normalize_reconciliation_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure reconciliation channel estimates include ci_lower/ci_upper aliases."""
    normalized = dict(data)
    estimates = normalized.get("channel_estimates", {}) or {}
    out: dict[str, Any] = {}

    for ch, raw in estimates.items():
        est = dict(raw) if isinstance(raw, dict) else {}
        if "ci_lower" not in est:
            est["ci_lower"] = est.get("lift_ci_lower")
        if "ci_upper" not in est:
            est["ci_upper"] = est.get("lift_ci_upper")
        if "lift_ci_lower" not in est:
            est["lift_ci_lower"] = est.get("ci_lower")
        if "lift_ci_upper" not in est:
            est["lift_ci_upper"] = est.get("ci_upper")
        out[ch] = est

    normalized["channel_estimates"] = out
    normalized.setdefault("total_incremental_value", 0.0)
    return normalized


def _normalize_channel_spend_key(channel: str) -> str:
    return channel if channel.endswith("_spend") else f"{channel}_spend"


# ---------------------------------------------------------------------------
# Artifact reader (thin cache over the store)
# ---------------------------------------------------------------------------

class ArtifactReader:
    """
    Reads the latest run's artifacts and caches them in memory.
    """

    def __init__(self, store: ArtifactStore):
        self._store = store
        self._cache: dict[str, Any] = {}
        self._cache_run: str | None = None

    def _ensure_cache(self) -> str | None:
        run_id = self._store.get_latest_run_id()
        if run_id is None:
            return None
        if run_id != self._cache_run:
            self._cache.clear()
            self._cache_run = run_id
        return run_id

    def get(self, name: str) -> Any | None:
        run_id = self._ensure_cache()
        if run_id is None:
            return None
        if name in self._cache:
            return self._cache[name]
        try:
            data = self._store.load_json(run_id, name)
            self._cache[name] = data
            return data
        except Exception:
            return None

    def get_dataframe_as_dict(self, name: str) -> Any | None:
        run_id = self._ensure_cache()
        if run_id is None:
            return None
        cache_key = f"df_{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            df = self._store.load_dataframe(run_id, name)
            data = {"data": df.to_dict(orient="records"), "n_rows": len(df)}
            self._cache[cache_key] = data
            return data
        except Exception:
            return None

    def invalidate(self) -> None:
        self._cache.clear()
        self._cache_run = None


class TelemetryBuffer:
    """Small in-memory rolling buffer for product telemetry events."""

    def __init__(self, max_events: int = 2000):
        self._max_events = max_events
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def ingest(self, events: list[dict[str, Any]]) -> None:
        with self._lock:
            self._events.extend(events)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            counts = Counter(str(e.get("event", "unknown")) for e in self._events)
            return {
                "total_events": len(self._events),
                "by_event": dict(counts),
                "window_seconds": 3600,
            }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(runs_dir: str | Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    config = get_config()
    _runs_dir = (Path(runs_dir) if runs_dir else config.storage.runs_path).resolve()
    store = ArtifactStore(_runs_dir)
    reader = ArtifactReader(store)
    telemetry = TelemetryBuffer()

    # Import cache layer (Redis with in-memory fallback)
    from server.cache import get_cache, cache_key as make_cache_key
    cache = get_cache()

    application = FastAPI(
        title="Unified-M API",
        description=(
            "Local API for Unified Marketing Measurement. "
            "Serves precomputed results from the pipeline."
        ),
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Optional bearer-token auth (reads API_AUTH_TOKEN from env)
    from server.auth import BearerAuthMiddleware
    application.add_middleware(BearerAuthMiddleware)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @application.get("/health", response_model=HealthResponse)
    def health():
        run_id = store.get_latest_run_id()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "latest_run": run_id,
            "version": "0.2.0",
            "cache": cache.stats(),
        }

    @application.get("/api/cache/stats")
    def cache_stats():
        return cache.stats()

    @application.post("/api/cache/clear", response_model=MessageResponse)
    def cache_clear():
        cache.clear()
        reader.invalidate()
        return {"message": "Cache cleared"}

    @application.get("/", response_model=RootResponse)
    def root():
        return {
            "name": "Unified-M API",
            "version": "0.2.0",
            "docs": "/docs",
        }

    @application.post("/api/v1/telemetry", response_model=StatusResponse)
    def ingest_telemetry(payload: TelemetryIngestRequest):
        events = [event.model_dump() for event in payload.events]
        if events:
            telemetry.ingest(events)
        return {"status": "ok"}

    @application.get("/api/v1/telemetry/summary", response_model=TelemetrySummaryResponse)
    def telemetry_summary():
        return telemetry.summary()

    # ------------------------------------------------------------------
    # Adapter Discovery
    # ------------------------------------------------------------------

    @application.get("/api/v1/adapters", response_model=AdaptersResponse)
    def list_adapters():
        """Discover available model backends, connectors, and cache status."""
        from models.registry import list_backends, _REGISTRY, _auto_discover
        _auto_discover()

        backend_info = []
        known_backends = {
            "builtin": None,
            "pymc": "pip install pymc-marketing",
            "meridian": "pip install google-meridian",
            "numpyro": "pip install numpyro",
        }
        available = set(list_backends())
        for name, hint in known_backends.items():
            backend_info.append({
                "name": name,
                "available": name in available,
                "install_hint": hint if name not in available else None,
            })

        return {
            "model_backends": backend_info,
            "connectors": {
                "database": ["postgresql", "mysql", "sqlite", "sqlserver"],
                "cloud": ["s3", "azure"],
                "ad_platforms": ["google_ads", "meta_ads", "tiktok_ads", "amazon_ads"],
            },
            "cache": cache.stats(),
        }

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    @application.get("/api/v1/runs", response_model=RunsResponse)
    def list_runs(limit: int = Query(default=20, ge=1, le=100)):
        """List recent pipeline runs with their manifests."""
        runs = store.list_runs(limit=limit)
        return {"runs": [r.model_dump() for r in runs]}

    @application.get("/api/v1/runs/{run_id}", response_model=RunManifest)
    def get_run(run_id: str):
        """Get the manifest for a specific run."""
        try:
            manifest = store.load_manifest(run_id)
            return manifest.model_dump()
        except Exception:
            raise HTTPException(404, f"Run '{run_id}' not found")

    @application.get("/api/v1/compare-runs", response_model=dict[str, Any])
    def compare_runs(
        run_a: str = Query(..., description="First run ID"),
        run_b: str = Query(..., description="Second run ID"),
    ):
        """Compare two runs (config diff, metrics, coefficients, allocations)."""
        if run_a == run_b:
            raise HTTPException(400, "Cannot compare a run to itself. Select two different runs.")
        try:
            from core.exceptions import ArtifactError
            return store.compare_runs(run_a, run_b)
        except ArtifactError as e:
            detail = str(e)
            if getattr(e, "run_id", None):
                detail += f" (run: {e.run_id})"
            logger.warning("Compare runs failed: %s", detail)
            raise HTTPException(404, detail)
        except Exception as e:
            logger.exception("Run compare failed: %s", e)
            raise HTTPException(400, str(e))

    # ------------------------------------------------------------------
    # Results (from latest run)
    # ------------------------------------------------------------------

    @application.get("/api/v1/contributions", response_model=ContributionsResponse)
    def get_contributions():
        """Channel contribution decomposition from the latest run."""
        data = reader.get_dataframe_as_dict("contributions")
        if data is None:
            raise HTTPException(404, "No contributions available. Run the pipeline first.")
        return data

    @application.get("/api/v1/reconciliation", response_model=ReconciliationResponse)
    def get_reconciliation():
        """Reconciled channel estimates with uncertainty."""
        data = reader.get("reconciliation")
        if data is None:
            raise HTTPException(404, "No reconciliation results. Run the pipeline first.")
        return _normalize_reconciliation_payload(data)

    @application.get("/api/v1/optimization", response_model=OptimizationResponse)
    def get_optimization():
        """Budget optimization recommendations."""
        data = reader.get("optimization")
        if data is None:
            raise HTTPException(404, "No optimization results. Run the pipeline first.")
        return _normalize_optimization_payload(data)

    @application.get("/api/v1/response-curves", response_model=dict[str, ResponseCurveChannel])
    def get_response_curves(channel: str | None = Query(default=None)):
        """Response (saturation) curves per channel."""
        data = reader.get("response_curves")
        if data is None:
            raise HTTPException(404, "No response curves. Run the pipeline first.")
        if channel and channel not in data:
            raise HTTPException(404, f"Channel '{channel}' not found")
        return {channel: data[channel]} if channel else data

    @application.get("/api/v1/parameters", response_model=ParametersResponse)
    def get_parameters():
        """Model parameters (coefficients, adstock, saturation)."""
        data = reader.get("parameters")
        if data is None:
            raise HTTPException(404, "No parameters. Run the pipeline first.")
        return _normalize_parameters_payload(data)

    @application.get("/api/v1/diagnostics", response_model=DiagnosticsResponse)
    def get_diagnostics():
        """Model diagnostics: actual vs predicted, residuals, fit stats."""
        contrib_data = reader.get_dataframe_as_dict("contributions")
        if contrib_data is None or not contrib_data.get("data"):
            raise HTTPException(404, "No contribution data for diagnostics.")

        rows = contrib_data["data"]
        actuals = [r.get("actual", 0) or 0 for r in rows]
        predicted = [r.get("predicted", 0) or 0 for r in rows]
        dates = [r.get("date", "") for r in rows]

        y_true = np.array(actuals, dtype=float)
        y_pred = np.array(predicted, dtype=float)
        residuals = y_true - y_pred

        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else 0
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        mae = float(np.mean(np.abs(residuals)))

        # Durbin-Watson
        diff = np.diff(residuals)
        dw = float(np.sum(diff ** 2) / (np.sum(residuals ** 2) + 1e-12))

        # Downsample for chart
        step = max(1, len(dates) // 200)
        chart_rows = [
            {
                "date": dates[i],
                "actual": actuals[i],
                "predicted": predicted[i],
                "residual": float(residuals[i]),
            }
            for i in range(0, len(dates), step)
        ]

        return {
            "metrics": {
                "r_squared": round(r2, 4),
                "mape": round(mape, 2),
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "durbin_watson": round(dw, 4),
                "n_observations": len(dates),
            },
            "chart": chart_rows,
            "residual_stats": {
                "mean": round(float(np.mean(residuals)), 4),
                "std": round(float(np.std(residuals)), 4),
                "min": round(float(np.min(residuals)), 4),
                "max": round(float(np.max(residuals)), 4),
            },
        }

    @application.get("/api/v1/roas", response_model=ROASResponse)
    def get_roas():
        """Channel-level ROAS / ROI analysis."""
        contrib_data = reader.get_dataframe_as_dict("contributions")
        optim_raw = reader.get("optimization")
        params = reader.get("parameters")
        optim_data = _normalize_optimization_payload(optim_raw) if optim_raw else None

        if contrib_data is None or not contrib_data.get("data"):
            raise HTTPException(404, "No data for ROAS analysis.")

        rows = contrib_data["data"]
        reserved = {"date", "actual", "predicted", "baseline"}
        channels = [k for k in rows[0].keys() if k not in reserved]
        current_alloc: dict[str, float] = {}
        if optim_data:
            current_alloc = {
                _normalize_channel_spend_key(str(ch)): float(v)
                for ch, v in optim_data.get("current_allocation", {}).items()
            }

        channel_roas = []
        for ch in channels:
            total_contribution = sum(float(r.get(ch, 0) or 0) for r in rows)
            spend = 0.0
            spend = current_alloc.get(_normalize_channel_spend_key(ch), 0.0)
            roas = total_contribution / spend if spend > 0 else 0
            mroi = 0.0
            if params and "coefficients" in params:
                mroi = params["coefficients"].get(ch, 0)

            channel_roas.append({
                "channel": ch,
                "total_contribution": round(total_contribution, 2),
                "total_spend": round(spend, 2),
                "roas": round(roas, 4),
                "marginal_roi": round(mroi, 4),
                "cpa": round(spend / total_contribution, 2) if total_contribution > 0 else 0,
            })

        channel_roas.sort(key=lambda x: x["roas"], reverse=True)

        return {
            "channels": channel_roas,
            "summary": {
                "total_spend": round(sum(c["total_spend"] for c in channel_roas), 2),
                "total_contribution": round(sum(c["total_contribution"] for c in channel_roas), 2),
                "blended_roas": round(
                    sum(c["total_contribution"] for c in channel_roas)
                    / (sum(c["total_spend"] for c in channel_roas) + 1e-8),
                    4,
                ),
            },
        }

    @application.get("/api/v1/waterfall", response_model=WaterfallResponse)
    def get_waterfall():
        """Waterfall decomposition of total response."""
        contrib_data = reader.get_dataframe_as_dict("contributions")
        if contrib_data is None or not contrib_data.get("data"):
            raise HTTPException(404, "No contribution data for waterfall.")

        rows = contrib_data["data"]
        reserved = {"date", "actual", "predicted", "baseline"}
        channels = [k for k in rows[0].keys() if k not in reserved]

        baseline = sum(float(r.get("baseline", 0) or 0) for r in rows)
        channel_totals = []
        for ch in channels:
            total = sum(float(r.get(ch, 0) or 0) for r in rows)
            channel_totals.append({"name": ch, "value": round(total, 2)})

        channel_totals.sort(key=lambda x: abs(x["value"]), reverse=True)
        total_response = sum(float(r.get("actual", 0) or 0) for r in rows)

        return {
            "baseline": round(baseline, 2),
            "channels": channel_totals,
            "total": round(total_response, 2),
        }

    # ------------------------------------------------------------------
    # Data Management
    # ------------------------------------------------------------------

    @application.get("/api/v1/data/status", response_model=dict[str, DataSourceStatus])
    def get_data_status():
        """Check which data sources are available."""
        config = get_config()
        processed = config.storage.processed_path

        def check_file(name: str) -> dict:
            path = processed / name
            exists = path.exists()
            info = {}
            if exists:
                try:
                    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                    info = {
                        "exists": True,
                        "rows": len(df),
                        "columns": list(df.columns),
                        "size_bytes": path.stat().st_size,
                    }
                except Exception as e:
                    info = {"exists": True, "error": str(e)}
            else:
                info = {"exists": False}
            return info

        # Start with the 5 known types
        known = ["media_spend", "outcomes", "controls", "incrementality_tests", "attribution"]
        result: dict[str, Any] = {}
        for name in known:
            result[name] = check_file(f"{name}.parquet")

        # Discover any custom parquet files
        if processed.exists():
            for p in sorted(processed.glob("*.parquet")):
                key = p.stem
                if key not in result:
                    result[key] = check_file(p.name)

        return result

    @application.post("/api/v1/data/upload", response_model=UploadDataResponse)
    async def upload_data(
        data_type: str = Form(...),
        file: UploadFile = File(...),
    ):
        """
        Upload a data file (CSV or Parquet).

        data_type: one of the known types or a custom name (e.g. promo_flags, weather).
        """
        _validate_data_type(data_type)

        config = get_config()
        processed = config.storage.processed_path
        processed.mkdir(parents=True, exist_ok=True)

        # Determine file extension
        ext = Path(file.filename).suffix.lower()
        if ext not in [".csv", ".parquet"]:
            ext = ".parquet"  # default

        output_path = processed / f"{data_type}{ext}"
        max_upload_bytes = _max_upload_size_bytes()
        tmp_path: Path | None = None

        try:
            tmp_path, _ = await _stream_upload_to_tempfile(
                file,
                suffix=ext,
                max_bytes=max_upload_bytes,
            )

            # Save to disk
            if ext == ".parquet":
                df = pd.read_parquet(tmp_path)
                df.to_parquet(output_path, index=False)
            else:
                df = pd.read_csv(tmp_path)
                # Convert to parquet for consistency
                output_path = processed / f"{data_type}.parquet"
                df.to_parquet(output_path, index=False)

            return {
                "status": "success",
                "data_type": data_type,
                "path": str(output_path),
                "rows": len(df),
                "columns": list(df.columns),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to upload {data_type}")
            raise HTTPException(500, f"Upload failed: {str(e)}")
        finally:
            try:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            await file.close()

    from server.jobs import JobManager
    job_manager = JobManager()

    def _run_pipeline_sync(
        model: str,
        target: str,
        budget: float | None,
        use_sample_data: bool = False,
        on_progress: Any = None,
    ) -> dict:
        if use_sample_data:
            if on_progress is not None:
                try:
                    on_progress("connect", "Generating sample data")
                except Exception:
                    pass
            from pipeline.runner import Pipeline
            from config import load_config as _load_cfg

            cfg = _load_cfg()
            cfg.ensure_directories()

            with tempfile.TemporaryDirectory(prefix="unified_m_sample_") as tmp:
                tmp_dir = Path(tmp)
                dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=365, freq="D")
                np.random.seed(42)

                channels = ["google_search", "meta_facebook", "meta_instagram", "tiktok", "tv_linear"]
                media_records: list[dict[str, Any]] = []
                for date in dates:
                    for channel in channels:
                        base = {
                            "google_search": 1200, "meta_facebook": 900,
                            "meta_instagram": 600, "tiktok": 400, "tv_linear": 1500,
                        }[channel]
                        seasonal = 1 + 0.3 * np.sin(date.dayofyear / 365 * 2 * np.pi)
                        noise = 1 + np.random.normal(0, 0.15)
                        spend = max(0, base * seasonal * noise)
                        media_records.append({
                            "date": date,
                            "geo": "national",
                            "channel": channel,
                            "spend": round(spend, 2),
                            "impressions": round(spend * np.random.uniform(40, 60), 0),
                            "clicks": round(spend * np.random.uniform(0.3, 0.5), 0),
                        })
                media_df = pd.DataFrame(media_records)
                media_path = tmp_dir / "media_spend.parquet"
                media_df.to_parquet(media_path, index=False)

                outcomes_records: list[dict[str, Any]] = []
                for date in dates:
                    base_revenue = 50000
                    seasonal = 1 + 0.2 * np.sin(date.dayofyear / 365 * 2 * np.pi)
                    trend = 1 + (date - dates[0]).days / 365 * 0.1
                    noise = np.random.normal(1, 0.08)
                    revenue = base_revenue * seasonal * trend * noise
                    conversions = round(revenue / 100 * np.random.uniform(0.8, 1.2), 0)
                    outcomes_records.append({
                        "date": date,
                        "geo": "national",
                        "revenue": round(revenue, 2),
                        "conversions": conversions,
                        "new_customers": round(conversions * 0.3, 0),
                    })
                outcomes_df = pd.DataFrame(outcomes_records)
                outcomes_path = tmp_dir / "outcomes.parquet"
                outcomes_df.to_parquet(outcomes_path, index=False)

                controls_df = pd.DataFrame({
                    "date": dates,
                    "geo": "national",
                    "is_holiday": [1 if d.dayofweek >= 5 else 0 for d in dates],
                    "promo": np.random.binomial(1, 0.1, len(dates)),
                    "price_index": np.random.normal(1.0, 0.03, len(dates)).clip(0.8, 1.2).round(3),
                })
                controls_path = tmp_dir / "controls.parquet"
                controls_df.to_parquet(controls_path, index=False)

                tests_records: list[dict[str, Any]] = []
                for i, channel in enumerate(channels[:3]):
                    start = dates[60 + i * 30]
                    end = dates[90 + i * 30]
                    lift = np.random.uniform(0.05, 0.25)
                    tests_records.append({
                        "test_id": f"test_{channel}_sample",
                        "channel": channel,
                        "start_date": start,
                        "end_date": end,
                        "test_type": "geo_lift",
                        "lift_estimate": round(lift, 4),
                        "lift_ci_lower": round(lift * 0.6, 4),
                        "lift_ci_upper": round(lift * 1.4, 4),
                        "confidence_level": 0.95,
                        "spend_during_test": round(np.random.uniform(20000, 80000), 2),
                    })
                tests_df = pd.DataFrame(tests_records)
                tests_path = tmp_dir / "incrementality_tests.parquet"
                tests_df.to_parquet(tests_path, index=False)

                cfg_dict = cfg.to_flat_dict()
                cfg_dict["sample_data"] = True
                pipe = Pipeline(config=cfg_dict, runs_dir=cfg.storage.runs_path)
                pipe.connect(
                    media_spend=media_path,
                    outcomes=outcomes_path,
                    controls=controls_path,
                    incrementality_tests=tests_path,
                )
                results = pipe.run(model=model, target_col=target, total_budget=budget, on_progress=on_progress)

            run_id = pipe.run_id
            if run_id:
                run_dir = cfg.storage.runs_path / run_id
                (run_dir / ".sample_run").write_text("sample")
            return {"run_id": run_id, "metrics": results.get("metrics", {})}

        from pipeline.runner import Pipeline
        from config import load_config as _load_cfg

        cfg = _load_cfg()
        cfg.ensure_directories()
        processed = cfg.storage.processed_path

        pipe = Pipeline(config=cfg.to_flat_dict(), runs_dir=cfg.storage.runs_path)
        pipe.connect(
            media_spend=processed / "media_spend.parquet"
            if (processed / "media_spend.parquet").exists() else None,
            outcomes=processed / "outcomes.parquet"
            if (processed / "outcomes.parquet").exists() else None,
            controls=processed / "controls.parquet"
            if (processed / "controls.parquet").exists() else None,
            incrementality_tests=processed / "incrementality_tests.parquet"
            if (processed / "incrementality_tests.parquet").exists() else None,
            attribution=processed / "attribution.parquet"
            if (processed / "attribution.parquet").exists() else None,
        )
        results = pipe.run(
            model=model,
            target_col=target,
            total_budget=budget,
            on_progress=on_progress,
        )
        return {"run_id": pipe.run_id, "metrics": results.get("metrics", {})}

    @application.post("/api/v1/pipeline/run", response_model=PipelineRunTriggerResponse)
    async def trigger_pipeline(
        model: str = Form(default="builtin"),
        target: str = Form(default="revenue"),
        budget: float | None = Form(default=None),
        use_sample_data: bool = Form(default=False),
    ):
        """
        Trigger a pipeline run asynchronously. Returns a job_id that can
        be polled via GET /api/v1/pipeline/jobs/{job_id}.
        """
        job = job_manager.create_job()
        job_manager.start_pipeline(
            job,
            _run_pipeline_sync,
            on_complete=lambda: (reader.invalidate(), cache.clear()),
            model=model,
            target=target,
            budget=budget,
            use_sample_data=use_sample_data,
        )
        return {"job_id": job.job_id, "status": "pending"}

    @application.get("/api/v1/pipeline/jobs", response_model=PipelineJobsResponse)
    def list_jobs(limit: int = Query(default=20, ge=1, le=100)):
        """List recent pipeline jobs."""
        return {"jobs": job_manager.list_jobs(limit=limit)}

    @application.get("/api/v1/pipeline/jobs/{job_id}", response_model=PipelineJobResponse)
    def get_job(job_id: str):
        """Get status of a pipeline job."""
        job = job_manager.get_job(job_id)
        if job is None:
            raise HTTPException(404, f"Job '{job_id}' not found")
        return job.to_dict()

    # ------------------------------------------------------------------
    # Calibration, Stability, Data Quality
    # ------------------------------------------------------------------

    _empty_calibration = {"n_tests": 0, "points": [], "coverage": 0, "median_lift_error": 0, "mean_lift_error": 0, "calibration_quality": "no_tests"}

    @application.get("/api/v1/calibration", response_model=CalibrationResponse)
    def calibration():
        """Calibration: MMM predicted vs. experiment-measured lift. Returns empty payload when no data."""
        try:
            run_id = store.get_latest_run_id()
            ck = make_cache_key("calibration", str(run_id))
            cached = cache.get(ck)
            if cached is not None:
                return cached

            data = reader.get("calibration_eval")
            if data is None and run_id:
                params = reader.get("parameters")
                if params:
                    tests_path = _runs_dir / run_id / "incrementality_tests.parquet"
                    alt_tests = config.storage.processed_path / "incrementality_tests.parquet"
                    tests_df = None
                    if tests_path.exists():
                        tests_df = pd.read_parquet(tests_path)
                    elif alt_tests.exists():
                        tests_df = pd.read_parquet(alt_tests)
                    if tests_df is not None and len(tests_df) > 0:
                        from models.calibration_eval import evaluate_calibration
                        report = evaluate_calibration(tests_df, params)
                        data = report.to_dict()
            if data is None:
                data = _empty_calibration
            cache.set(ck, data, ttl=600)
            return data
        except Exception as e:
            logger.exception("Calibration endpoint error: %s", e)
            return _empty_calibration

    @application.get("/api/v1/stability", response_model=dict[str, Any])
    def stability():
        """Recommendation stability metrics across runs. Returns empty payload when fewer than 2 runs."""
        try:
            run_id = store.get_latest_run_id()
            ck = make_cache_key("stability", str(run_id))
            cached = cache.get(ck)
            if cached is not None:
                return cached

            data = reader.get("stability_metrics")
            if data is None:
                all_runs = store.list_runs(limit=2)
                if len(all_runs) < 2:
                    data = {}
                else:
                    curr_id = all_runs[0].run_id
                    prev_id = all_runs[1].run_id
                    curr_opt = store.load_json(curr_id, "optimization")
                    prev_opt = store.load_json(prev_id, "optimization")
                    curr_params = store.load_json(curr_id, "parameters")
                    prev_params = store.load_json(prev_id, "parameters")
                    from models.evaluation import compute_stability_report
                    curr_alloc = curr_opt.get("optimal_allocation", {}) if curr_opt else None
                    prev_alloc = prev_opt.get("optimal_allocation", {}) if prev_opt else None
                    contributions_data = reader.get_dataframe_as_dict("contributions")
                    contrib_df = None
                    if contributions_data and isinstance(contributions_data.get("data"), list):
                        contrib_df = pd.DataFrame(contributions_data["data"])
                    data = compute_stability_report(
                        current_allocation=curr_alloc,
                        previous_allocation=prev_alloc,
                        current_params=curr_params,
                        previous_params=prev_params,
                        contributions=contrib_df,
                    )
            cache.set(ck, data, ttl=600)
            return data
        except Exception as e:
            logger.exception("Stability endpoint error: %s", e)
            return {}

    _empty_data_quality = {
        "timestamp": "",
        "overall_pass": True,
        "n_passed": 0,
        "n_failed": 0,
        "n_warnings": 0,
        "gates": [],
    }

    @application.get("/api/v1/data-quality", response_model=DataQualityResponse)
    def data_quality():
        """Data quality gate results from the latest run. Returns empty payload when no data."""
        try:
            run_id = store.get_latest_run_id()
            ck = make_cache_key("data_quality", str(run_id))
            cached = cache.get(ck)
            if cached is not None:
                return cached

            data = reader.get("data_quality_report")
            if data is None:
                processed = config.storage.processed_path
                ms_path = processed / "media_spend.parquet"
                oc_path = processed / "outcomes.parquet"
                ms_df = pd.read_parquet(ms_path) if ms_path.exists() else None
                oc_df = pd.read_parquet(oc_path) if oc_path.exists() else None

                if ms_df is None and oc_df is None:
                    data = {**_empty_data_quality, "timestamp": datetime.now().isoformat()}
                else:
                    from quality.gates import run_quality_gates
                    report = run_quality_gates(media_spend=ms_df, outcomes=oc_df)
                    data = report.to_dict()
            cache.set(ck, data, ttl=300)
            return data
        except Exception as e:
            logger.exception("Data-quality endpoint error: %s", e)
            return {**_empty_data_quality, "timestamp": datetime.now().isoformat()}

    # ------------------------------------------------------------------
    # Channel Insights (saturation alerts + marginal ROI)
    # ------------------------------------------------------------------

    @application.get("/api/v1/channel-insights", response_model=ChannelInsightsResponse)
    def channel_insights():
        """Per-channel saturation status, marginal ROI, and headroom."""
        try:
            optim_data = reader.get("optimization")
            params_raw = reader.get("parameters")
            curves_raw = reader.get("response_curves")

            if not optim_data or not params_raw:
                raise HTTPException(404, "No optimization or parameter data.")
            params = _normalize_parameters_payload(params_raw)

            current_alloc = optim_data.get("current_allocation", {})
            optimal_alloc = optim_data.get("optimal_allocation", {})
            channels = list(current_alloc.keys())
            coefficients = params.get("coefficients", {})
            sat_params = params.get("saturation", {}) or params.get("saturation_params", {})

            insights = []
            for ch in channels:
                cur_spend = current_alloc.get(ch, 0)
                opt_spend = optimal_alloc.get(ch, 0)
                coeff = coefficients.get(ch, 0)

                # Compute marginal ROI at current spend using saturation derivative
                sp = sat_params.get(ch, {})
                K = sp.get("K", cur_spend + 1) if isinstance(sp, dict) else (cur_spend + 1)
                S = sp.get("S", 1.0) if isinstance(sp, dict) else 1.0

                delta = max(cur_spend * 0.01, 1.0)
                from transforms.saturation import hill_saturation as _hs  # noqa: E402
                resp_at = float(_hs(np.array([cur_spend]), K=K, S=S)[0]) * coeff
                resp_at_plus = float(_hs(np.array([cur_spend + delta]), K=K, S=S)[0]) * coeff
                marginal_roi = (resp_at_plus - resp_at) / delta if delta > 0 else 0

                # Saturation point: where marginal ROI drops below 10% of average ROI
                avg_roi = resp_at / cur_spend if cur_spend > 0 else 1
                threshold = avg_roi * 0.1
                sat_point = cur_spend
                for test_spend in np.linspace(cur_spend, cur_spend * 5 + 1, 200):
                    r1 = float(_hs(np.array([test_spend]), K=K, S=S)[0]) * coeff
                    r2 = float(_hs(np.array([test_spend + delta]), K=K, S=S)[0]) * coeff
                    if (r2 - r1) / delta < threshold:
                        sat_point = float(test_spend)
                        break
                else:
                    sat_point = float(cur_spend * 5)

                headroom_pct = round(((sat_point - cur_spend) / cur_spend * 100) if cur_spend > 0 else 100, 1)
                if headroom_pct < 10:
                    status = "over-saturated"
                elif headroom_pct > 80:
                    status = "under-invested"
                else:
                    status = "efficient"

                insights.append({
                    "channel": ch,
                    "current_spend": round(cur_spend, 2),
                    "optimal_spend": round(opt_spend, 2),
                    "marginal_roi": round(marginal_roi, 6),
                    "saturation_point": round(sat_point, 2),
                    "headroom_pct": headroom_pct,
                    "status": status,
                    "coefficient": round(coeff, 6),
                })

            insights.sort(key=lambda x: x["marginal_roi"], reverse=True)
            return {"channels": insights}
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Channel insights error: %s", e)
            return {"channels": []}

    # ------------------------------------------------------------------
    # Spend Pacing (plan vs actual tracker)
    # ------------------------------------------------------------------

    @application.get("/api/v1/spend-pacing", response_model=SpendPacingResponse)
    def spend_pacing():
        """Compare planned (optimal) allocation vs actual spend from media data."""
        try:
            optim_data = reader.get("optimization")
            config = get_config()
            processed = config.storage.processed_path
            ms_path = processed / "media_spend.parquet"

            if not optim_data:
                raise HTTPException(404, "No optimization data for pacing.")

            def _normalize_channel_key(channel: str) -> str:
                return channel if channel.endswith("_spend") else f"{channel}_spend"

            optimal_raw = optim_data.get("optimal_allocation", {})
            optimal = {
                _normalize_channel_key(str(ch)): float(v)
                for ch, v in optimal_raw.items()
            }
            channels = list(optimal.keys())

            # Load actual spend from media_spend
            actual_by_channel: dict[str, float] = {}
            cumulative: list[dict] = []
            if ms_path.exists():
                ms_df = pd.read_parquet(ms_path)
                if "channel" in ms_df.columns and "spend" in ms_df.columns:
                    grouped = ms_df.groupby("channel")["spend"].sum().to_dict()
                    actual_by_channel = {
                        _normalize_channel_key(str(ch)): float(v)
                        for ch, v in grouped.items()
                    }
                    if "date" in ms_df.columns:
                        ms_df["date"] = pd.to_datetime(ms_df["date"])
                        cum = ms_df.groupby("date")["spend"].sum().sort_index().cumsum().reset_index()
                        step = max(1, len(cum) // 60)
                        cumulative = [
                            {"date": str(row["date"])[:10], "actual": round(float(row["spend"]), 2)}
                            for _, row in cum.iloc[::step].iterrows()
                        ]

            total_planned = sum(optimal.values())
            total_actual = sum(actual_by_channel.get(ch, 0) for ch in channels)
            pacing_pct = round(total_actual / total_planned * 100, 1) if total_planned > 0 else 0

            pacing_channels = []
            for ch in channels:
                planned = optimal.get(ch, 0)
                actual = actual_by_channel.get(ch, 0)
                ch_pacing = round(actual / planned * 100, 1) if planned > 0 else 0
                diff = actual - planned
                if ch_pacing > 115:
                    ch_status = "over"
                elif ch_pacing < 85:
                    ch_status = "under"
                else:
                    ch_status = "on-track"
                pacing_channels.append({
                    "channel": ch,
                    "planned": round(planned, 2),
                    "actual": round(actual, 2),
                    "diff": round(diff, 2),
                    "pacing_pct": ch_pacing,
                    "status": ch_status,
                })
            pacing_channels.sort(key=lambda x: abs(x["diff"]), reverse=True)

            return {
                "total_planned": round(total_planned, 2),
                "total_actual": round(total_actual, 2),
                "pacing_pct": pacing_pct,
                "channels": pacing_channels,
                "cumulative": cumulative,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Spend pacing error: %s", e)
            return {"total_planned": 0, "total_actual": 0, "pacing_pct": 0, "channels": [], "cumulative": []}

    # ------------------------------------------------------------------
    # Executive Summary / Report
    # ------------------------------------------------------------------

    @application.get("/api/v1/report/summary", response_model=ReportSummaryResponse)
    def report_summary():
        """One-click executive summary payload."""
        try:
            run_id = store.get_latest_run_id()
            contrib_data = reader.get_dataframe_as_dict("contributions")
            optim_data = reader.get("optimization")
            params = reader.get("parameters")
            roas_data = None
            try:
                roas_data = get_roas()
            except Exception:
                pass
            diag_data = None
            try:
                diag_data = get_diagnostics()
            except Exception:
                pass

            metrics = {}
            if diag_data and "metrics" in diag_data:
                metrics = diag_data["metrics"]

            # Top channels by contribution
            top_channels = []
            if contrib_data and contrib_data.get("data"):
                rows = contrib_data["data"]
                reserved = {"date", "actual", "predicted", "baseline"}
                chs = [k for k in rows[0].keys() if k not in reserved]
                ch_totals = [(ch, sum(float(r.get(ch, 0) or 0) for r in rows)) for ch in chs]
                ch_totals.sort(key=lambda x: abs(x[1]), reverse=True)
                total_contrib = sum(abs(v) for _, v in ch_totals)
                for ch, val in ch_totals[:5]:
                    pct = round(abs(val) / total_contrib * 100, 1) if total_contrib > 0 else 0
                    top_channels.append({"channel": ch, "contribution": round(val, 2), "share_pct": pct})

            # Key recommendations
            recommendations = []
            if optim_data:
                current = optim_data.get("current_allocation", {})
                optimal = optim_data.get("optimal_allocation", {})
                changes = []
                for ch in optimal:
                    cur = current.get(ch, 0)
                    opt = optimal.get(ch, 0)
                    diff = opt - cur
                    pct = round(diff / cur * 100, 1) if cur > 0 else 0
                    changes.append((ch, diff, pct))
                changes.sort(key=lambda x: abs(x[1]), reverse=True)
                for ch, diff, pct in changes[:3]:
                    action = "Increase" if diff > 0 else "Decrease"
                    recommendations.append(f"{action} {ch} by ${abs(diff):,.0f} ({abs(pct):.0f}%)")

            roas_summary = {}
            if roas_data and "summary" in roas_data:
                roas_summary = roas_data["summary"]

            return {
                "run_id": run_id,
                "generated_at": datetime.now().isoformat(),
                "metrics": metrics,
                "roas_summary": roas_summary,
                "top_channels": top_channels,
                "recommendations": recommendations,
                "improvement_pct": optim_data.get("improvement_pct", 0) if optim_data else 0,
            }
        except Exception as e:
            logger.exception("Report summary error: %s", e)
            return {"run_id": None, "generated_at": datetime.now().isoformat(), "metrics": {}, "roas_summary": {}, "top_channels": [], "recommendations": [], "improvement_pct": 0}

    # ------------------------------------------------------------------
    # Cache control
    # ------------------------------------------------------------------

    @application.post("/api/v1/refresh", response_model=StatusResponse)
    def refresh_cache():
        """Force the server to re-read artifacts from disk."""
        reader.invalidate()
        cache.clear()
        return {"status": "cache_invalidated"}

    # ------------------------------------------------------------------
    # Datapoint Connectors
    # ------------------------------------------------------------------

    @application.post("/api/v1/datapoint/test", response_model=ConnectorTestResponse)
    async def test_datapoint_connection(
        connection_type: str = Form(...),
        connection_config: str = Form(...),
    ):
        """
        Test connection to a datapoint (database or cloud storage).
        
        connection_type: 'database' or 'cloud'
        connection_config: JSON string with connection parameters
        """
        import json
        
        try:
            config = json.loads(connection_config)
            
            if connection_type == "database":
                from connectors.database import create_database_connector
                db_type = config.pop("db_type")
                connector = create_database_connector(db_type, **config)
                success = connector.test_connection()
                connector.close()
                
                return {
                    "status": "success" if success else "failed",
                    "connected": success,
                    "message": "Connection successful" if success else "Connection failed",
                }
            
            elif connection_type == "cloud":
                from connectors.cloud import create_cloud_connector
                cloud_type = config.pop("cloud_type")
                connector = create_cloud_connector(cloud_type, **config)
                success = connector.test_connection()
                
                return {
                    "status": "success" if success else "failed",
                    "connected": success,
                    "message": "Connection successful" if success else "Connection failed",
                }
            
            else:
                raise HTTPException(400, f"Invalid connection_type: {connection_type}")
        
        except Exception as e:
            logger.exception("Datapoint connection test failed")
            return {
                "status": "error",
                "connected": False,
                "message": str(e),
            }

    @application.post("/api/v1/datapoint/fetch", response_model=ConnectorFetchResponse)
    async def fetch_datapoint_data(
        connection_type: str = Form(...),
        connection_config: str = Form(...),
        query_or_path: str = Form(...),
        data_type: str = Form(...),
    ):
        """
        Fetch data from a datapoint connection.
        
        For databases: query_or_path is SQL query
        For cloud: query_or_path is file path
        data_type: known type or custom name (e.g. promo_flags).
        """
        import json
        import tempfile
        from pathlib import Path

        _validate_data_type(data_type)

        try:
            config = json.loads(connection_config)
            config_obj = get_config()
            processed = config_obj.storage.processed_path
            
            if connection_type == "database":
                from connectors.database import create_database_connector
                db_type = config.pop("db_type")
                connector = create_database_connector(db_type, **config)
                df = connector.load(query_or_path)
                connector.close()
            
            elif connection_type == "cloud":
                from connectors.cloud import create_cloud_connector
                cloud_type = config.pop("cloud_type")
                connector = create_cloud_connector(cloud_type, **config)
                df = connector.load(query_or_path)
            
            else:
                raise HTTPException(400, f"Invalid connection_type: {connection_type}")
            
            # Save to processed directory
            output_path = processed / f"{data_type}.parquet"
            df.to_parquet(output_path, index=False)
            
            # Invalidate cache to refresh data status
            reader.invalidate()
            
            return {
                "status": "success",
                "rows": len(df),
                "columns": list(df.columns),
                "path": str(output_path),
                "data_type": data_type,
            }
        
        except Exception as e:
            logger.exception("Datapoint data fetch failed")
            raise HTTPException(500, f"Failed to fetch data: {str(e)}")

    @application.post("/api/v1/datapoint/upload", response_model=UploadDataResponse)
    async def upload_datapoint_file(
        file: UploadFile = File(...),
        data_type: str = Form(...),
    ):
        """
        Upload a file (CSV, Parquet, Excel) as a datapoint.

        data_type: known type or custom name (e.g. promo_flags).
        """
        _validate_data_type(data_type)

        from connectors.local import auto_connect

        # Validate file extension
        ext = Path(file.filename).suffix.lower()
        if ext not in [".csv", ".parquet", ".xlsx", ".xls"]:
            raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: .csv, .parquet, .xlsx, .xls")

        config = get_config()
        processed = config.storage.processed_path
        processed.mkdir(parents=True, exist_ok=True)
        max_upload_bytes = _max_upload_size_bytes()
        tmp_path: Path | None = None

        try:
            tmp_path, _ = await _stream_upload_to_tempfile(
                file,
                suffix=ext,
                max_bytes=max_upload_bytes,
            )

            # Load using connector
            connector = auto_connect(tmp_path)
            df = connector.load(tmp_path)

            # Save to processed directory as parquet
            output_path = processed / f"{data_type}.parquet"
            df.to_parquet(output_path, index=False)

            # Invalidate cache
            reader.invalidate()

            return {
                "status": "success",
                "data_type": data_type,
                "path": str(output_path),
                "rows": len(df),
                "columns": list(df.columns),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to upload datapoint file: {data_type}")
            raise HTTPException(500, f"Upload failed: {str(e)}")
        finally:
            try:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            await file.close()

    # ------------------------------------------------------------------
    # Connector Registry (saved connections CRUD)
    # ------------------------------------------------------------------

    from connectors.registry import ConnectorStore
    connector_store = ConnectorStore(config.storage.raw_path.parent / "connectors")

    @application.get("/api/v1/connectors", response_model=SavedConnectorListResponse)
    def list_connectors():
        """List all saved connections (configs omitted for security)."""
        return {"connectors": connector_store.list()}

    @application.post("/api/v1/connectors", response_model=SavedConnector)
    async def create_connector(
        name: str = Form(...),
        connector_type: str = Form(...),
        subtype: str = Form(...),
        connector_config: str = Form(...),
    ):
        """Create a saved connection."""
        try:
            cfg = json.loads(connector_config)
        except json.JSONDecodeError:
            raise HTTPException(400, "connector_config must be valid JSON")
        record = connector_store.create(name, connector_type, subtype, cfg)
        return record

    @application.get("/api/v1/connectors/{connector_id}", response_model=SavedConnector)
    def get_connector(connector_id: str):
        """Get a saved connection (config masked)."""
        record = connector_store.get(connector_id, include_secrets=False)
        if record is None:
            raise HTTPException(404, "Connector not found")
        return record

    @application.post("/api/v1/connectors/{connector_id}/reveal-secrets", response_model=ConnectorRevealResponse)
    def reveal_connector_secrets(
        connector_id: str,
        x_reveal_token: str | None = Header(default=None, alias="X-Reveal-Token"),
    ):
        """
        Reveal decrypted connector config when explicitly authorized.

        Requires a second factor token via X-Reveal-Token that matches
        CONNECTOR_REVEAL_TOKEN.
        """
        required = os.getenv("CONNECTOR_REVEAL_TOKEN", "")
        if not required:
            raise HTTPException(
                403,
                "Secret reveal is disabled. Set CONNECTOR_REVEAL_TOKEN to enable.",
            )
        if not x_reveal_token or x_reveal_token != required:
            raise HTTPException(403, "Invalid or missing X-Reveal-Token")

        record = connector_store.get(connector_id, include_secrets=True)
        if record is None:
            raise HTTPException(404, "Connector not found")
        return {"id": record["id"], "config": record.get("config", {})}

    @application.put("/api/v1/connectors/{connector_id}", response_model=SavedConnector)
    async def update_connector(
        connector_id: str,
        name: str | None = Form(default=None),
        connector_config: str | None = Form(default=None),
    ):
        """Update a saved connection."""
        cfg = None
        if connector_config is not None:
            try:
                cfg = json.loads(connector_config)
            except json.JSONDecodeError:
                raise HTTPException(400, "connector_config must be valid JSON")
        record = connector_store.update(connector_id, name=name, config=cfg)
        if record is None:
            raise HTTPException(404, "Connector not found")
        return record

    @application.delete("/api/v1/connectors/{connector_id}", response_model=StatusResponse)
    def delete_connector(connector_id: str):
        """Delete a saved connection."""
        if not connector_store.delete(connector_id):
            raise HTTPException(404, "Connector not found")
        return {"status": "deleted"}

    @application.post("/api/v1/connectors/{connector_id}/test", response_model=ConnectorTestResponse)
    def test_connector(connector_id: str):
        """Test a saved connection."""
        record = connector_store.get(connector_id, include_secrets=True)
        if record is None:
            raise HTTPException(404, "Connector not found")

        cfg = dict(record["config"])
        conn_type = record["type"]
        subtype = record["subtype"]
        success = False
        message = ""

        try:
            if conn_type == "database":
                from connectors.database import create_database_connector
                cfg["db_type"] = subtype
                db_type = cfg.pop("db_type")
                connector = create_database_connector(db_type, **cfg)
                success = connector.test_connection()
                connector.close()
            elif conn_type == "cloud":
                from connectors.cloud import create_cloud_connector
                cfg["cloud_type"] = subtype
                cloud_type = cfg.pop("cloud_type")
                connector = create_cloud_connector(cloud_type, **cfg)
                success = connector.test_connection()
            else:
                message = f"Unknown connector type: {conn_type}"
        except Exception as e:
            message = str(e)

        connector_store.set_test_result(connector_id, success)
        return {
            "status": "success" if success else "failed",
            "connected": success,
            "message": message or ("Connection successful" if success else "Connection failed"),
        }

    @application.post("/api/v1/connectors/{connector_id}/fetch", response_model=ConnectorFetchResponse)
    async def fetch_from_connector(
        connector_id: str,
        query_or_path: str = Form(...),
        data_type: str = Form(...),
    ):
        """Fetch data from a saved connection and store as a data source."""
        _validate_data_type(data_type)
        record = connector_store.get(connector_id, include_secrets=True)
        if record is None:
            raise HTTPException(404, "Connector not found")

        cfg = dict(record["config"])
        conn_type = record["type"]
        subtype = record["subtype"]
        config_obj = get_config()
        processed = config_obj.storage.processed_path

        try:
            if conn_type == "database":
                from connectors.database import create_database_connector
                connector = create_database_connector(subtype, **cfg)
                df = connector.load(query_or_path)
                connector.close()
            elif conn_type == "cloud":
                from connectors.cloud import create_cloud_connector
                connector = create_cloud_connector(subtype, **cfg)
                df = connector.load(query_or_path)
            else:
                raise HTTPException(400, f"Unknown connector type: {conn_type}")

            output_path = processed / f"{data_type}.parquet"
            df.to_parquet(output_path, index=False)
            reader.invalidate()

            return {
                "status": "success",
                "rows": len(df),
                "columns": list(df.columns),
                "data_type": data_type,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Connector fetch failed")
            raise HTTPException(500, f"Fetch failed: {str(e)}")

    return application


# Default instance for ``uvicorn server.app:app``
app = create_app()


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Start the API server (also serves the built UI if ui/dist/ exists)."""
    import uvicorn
    _mount_ui(app)
    logger.info(f"Starting Unified-M API on {host}:{port}")
    uvicorn.run("server.app:app", host=host, port=port, reload=reload)


def _mount_ui(application: FastAPI) -> None:
    """
    If the React UI has been built (``cd ui && bun run build``), serve the
    static assets from FastAPI so a single process serves both API and UI.

    Routes defined with ``@app.get()`` take priority over the mount, so
    ``/api/*``, ``/health``, ``/docs`` all keep working.
    """
    ui_dist = Path(__file__).resolve().parent.parent.parent / "ui" / "dist"
    if not ui_dist.is_dir():
        return

    from fastapi.staticfiles import StaticFiles

    # ``html=True`` makes StaticFiles serve index.html for directory
    # requests and 404s — exactly what a single-page app needs.
    application.mount("/", StaticFiles(directory=ui_dist, html=True), name="ui")
    logger.info(f"Serving built UI from {ui_dist}")
