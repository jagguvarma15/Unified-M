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

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import io
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
import json
import re

from core.artifacts import ArtifactStore
from config import get_config


# Known pipeline data types; custom names allowed via _is_valid_custom_data_type
KNOWN_DATA_TYPES = frozenset({
    "media_spend", "outcomes", "controls", "incrementality_tests", "attribution",
})


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


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(runs_dir: str | Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    config = get_config()
    _runs_dir = Path(runs_dir) if runs_dir else config.storage.runs_path
    store = ArtifactStore(_runs_dir)
    reader = ArtifactReader(store)

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

    @application.get("/health")
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

    @application.post("/api/cache/clear")
    def cache_clear():
        cache.clear()
        reader.invalidate()
        return {"message": "Cache cleared"}

    @application.get("/")
    def root():
        return {
            "name": "Unified-M API",
            "version": "0.2.0",
            "docs": "/docs",
        }

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    @application.get("/api/v1/runs")
    def list_runs(limit: int = Query(default=20, ge=1, le=100)):
        """List recent pipeline runs with their manifests."""
        runs = store.list_runs(limit=limit)
        return {"runs": [r.model_dump() for r in runs]}

    @application.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str):
        """Get the manifest for a specific run."""
        try:
            manifest = store.load_manifest(run_id)
            return manifest.model_dump()
        except Exception:
            raise HTTPException(404, f"Run '{run_id}' not found")

    @application.get("/api/v1/runs/compare")
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
            raise HTTPException(404, str(e))
        except Exception as e:
            logger.exception("Run compare failed: %s", e)
            raise HTTPException(400, str(e))

    # ------------------------------------------------------------------
    # Results (from latest run)
    # ------------------------------------------------------------------

    @application.get("/api/v1/contributions")
    def get_contributions():
        """Channel contribution decomposition from the latest run."""
        data = reader.get_dataframe_as_dict("contributions")
        if data is None:
            raise HTTPException(404, "No contributions available. Run the pipeline first.")
        return data

    @application.get("/api/v1/reconciliation")
    def get_reconciliation():
        """Reconciled channel estimates with uncertainty."""
        data = reader.get("reconciliation")
        if data is None:
            raise HTTPException(404, "No reconciliation results. Run the pipeline first.")
        return data

    @application.get("/api/v1/optimization")
    def get_optimization():
        """Budget optimization recommendations."""
        data = reader.get("optimization")
        if data is None:
            raise HTTPException(404, "No optimization results. Run the pipeline first.")
        return data

    @application.get("/api/v1/response-curves")
    def get_response_curves(channel: str | None = Query(default=None)):
        """Response (saturation) curves per channel."""
        data = reader.get("response_curves")
        if data is None:
            raise HTTPException(404, "No response curves. Run the pipeline first.")
        if channel and channel not in data:
            raise HTTPException(404, f"Channel '{channel}' not found")
        return {channel: data[channel]} if channel else data

    @application.get("/api/v1/parameters")
    def get_parameters():
        """Model parameters (coefficients, adstock, saturation)."""
        data = reader.get("parameters")
        if data is None:
            raise HTTPException(404, "No parameters. Run the pipeline first.")
        return data

    @application.get("/api/v1/diagnostics")
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

    @application.get("/api/v1/roas")
    def get_roas():
        """Channel-level ROAS / ROI analysis."""
        contrib_data = reader.get_dataframe_as_dict("contributions")
        optim_data = reader.get("optimization")
        params = reader.get("parameters")

        if contrib_data is None or not contrib_data.get("data"):
            raise HTTPException(404, "No data for ROAS analysis.")

        rows = contrib_data["data"]
        reserved = {"date", "actual", "predicted", "baseline"}
        channels = [k for k in rows[0].keys() if k not in reserved]

        channel_roas = []
        for ch in channels:
            total_contribution = sum(float(r.get(ch, 0) or 0) for r in rows)
            spend = 0.0
            if optim_data:
                spend = optim_data.get("current_allocation", {}).get(ch, 0)
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

    @application.get("/api/v1/waterfall")
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

    @application.get("/api/v1/data/status")
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

    @application.post("/api/v1/data/upload")
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

        try:
            # Read uploaded file
            contents = await file.read()

            # Save to disk
            if ext == ".parquet":
                df = pd.read_parquet(io.BytesIO(contents))
                df.to_parquet(output_path, index=False)
            else:
                df = pd.read_csv(io.BytesIO(contents))
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
        except Exception as e:
            logger.exception(f"Failed to upload {data_type}")
            raise HTTPException(500, f"Upload failed: {str(e)}")

    @application.post("/api/v1/pipeline/run")
    async def trigger_pipeline(
        model: str = Form(default="builtin"),
        target: str = Form(default="revenue"),
        budget: float | None = Form(default=None),
    ):
        """
        Trigger a pipeline run with the current data sources.

        This is a long-running operation. In production, you'd want to
        use a task queue (Celery, RQ, etc.) and return a job ID.
        """
        try:
            from pipeline.runner import Pipeline
            from config import load_config

            config = load_config()
            config.ensure_directories()

            processed = config.storage.processed_path

            pipe = Pipeline(
                config=config.to_flat_dict(),
                runs_dir=config.storage.runs_path,
            )

            pipe.connect(
                media_spend=processed / "media_spend.parquet"
                if (processed / "media_spend.parquet").exists()
                else None,
                outcomes=processed / "outcomes.parquet"
                if (processed / "outcomes.parquet").exists()
                else None,
                controls=processed / "controls.parquet"
                if (processed / "controls.parquet").exists()
                else None,
                incrementality_tests=processed / "incrementality_tests.parquet"
                if (processed / "incrementality_tests.parquet").exists()
                else None,
                attribution=processed / "attribution.parquet"
                if (processed / "attribution.parquet").exists()
                else None,
            )

            results = pipe.run(model=model, target_col=target, total_budget=budget)

            # Invalidate cache so new results are visible
            reader.invalidate()

            return {
                "status": "success",
                "run_id": pipe.run_id,
                "metrics": results.get("metrics", {}),
            }
        except Exception as e:
            logger.exception("Pipeline run failed")
            raise HTTPException(500, f"Pipeline failed: {str(e)}")

    # ------------------------------------------------------------------
    # Calibration, Stability, Data Quality
    # ------------------------------------------------------------------

    _empty_calibration = {"n_tests": 0, "points": [], "coverage": 0, "median_lift_error": 0, "mean_lift_error": 0, "calibration_quality": "no_tests"}

    @application.get("/api/v1/calibration")
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

    @application.get("/api/v1/stability")
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

    @application.get("/api/v1/data-quality")
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

    @application.get("/api/v1/channel-insights")
    def channel_insights():
        """Per-channel saturation status, marginal ROI, and headroom."""
        try:
            optim_data = reader.get("optimization")
            params = reader.get("parameters")
            curves_raw = reader.get("response_curves")

            if not optim_data or not params:
                raise HTTPException(404, "No optimization or parameter data.")

            current_alloc = optim_data.get("current_allocation", {})
            optimal_alloc = optim_data.get("optimal_allocation", {})
            channels = list(current_alloc.keys())
            coefficients = params.get("coefficients", {})
            sat_params = params.get("saturation", {})

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

    @application.get("/api/v1/spend-pacing")
    def spend_pacing():
        """Compare planned (optimal) allocation vs actual spend from media data."""
        try:
            optim_data = reader.get("optimization")
            contrib_data = reader.get_dataframe_as_dict("contributions")
            config = get_config()
            processed = config.storage.processed_path
            ms_path = processed / "media_spend.parquet"

            if not optim_data:
                raise HTTPException(404, "No optimization data for pacing.")

            optimal = optim_data.get("optimal_allocation", {})
            channels = list(optimal.keys())

            # Load actual spend from media_spend
            actual_by_channel: dict[str, float] = {}
            cumulative: list[dict] = []
            if ms_path.exists():
                ms_df = pd.read_parquet(ms_path)
                if "channel" in ms_df.columns and "spend" in ms_df.columns:
                    actual_by_channel = ms_df.groupby("channel")["spend"].sum().to_dict()
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

    @application.get("/api/v1/report/summary")
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

    @application.post("/api/v1/refresh")
    def refresh_cache():
        """Force the server to re-read artifacts from disk."""
        reader.invalidate()
        cache.clear()
        return {"status": "cache_invalidated"}

    # ------------------------------------------------------------------
    # Datapoint Connectors
    # ------------------------------------------------------------------

    @application.post("/api/v1/datapoint/test")
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

    @application.post("/api/v1/datapoint/fetch")
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

    @application.post("/api/v1/datapoint/upload")
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
        import tempfile
        import io
        
        # Validate file extension
        ext = Path(file.filename).suffix.lower()
        if ext not in [".csv", ".parquet", ".xlsx", ".xls"]:
            raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: .csv, .parquet, .xlsx, .xls")
        
        config = get_config()
        processed = config.storage.processed_path
        processed.mkdir(parents=True, exist_ok=True)
        
        try:
            # Read uploaded file
            contents = await file.read()
            
            # Save to temp file for connector
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            
            # Load using connector
            connector = auto_connect(tmp_path)
            df = connector.load(tmp_path)
            
            # Save to processed directory as parquet
            output_path = processed / f"{data_type}.parquet"
            df.to_parquet(output_path, index=False)
            
            # Cleanup temp file
            Path(tmp_path).unlink()
            
            # Invalidate cache
            reader.invalidate()
            
            return {
                "status": "success",
                "data_type": data_type,
                "path": str(output_path),
                "rows": len(df),
                "columns": list(df.columns),
            }
        
        except Exception as e:
            logger.exception(f"Failed to upload datapoint file: {data_type}")
            raise HTTPException(500, f"Upload failed: {str(e)}")

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
