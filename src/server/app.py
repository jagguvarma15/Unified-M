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
from pathlib import Path
from typing import Any
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from core.artifacts import ArtifactStore
from config import get_config


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
        }

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
        run_a: str = Query(...),
        run_b: str = Query(...),
    ):
        """Compare two runs (config diff, metric diff)."""
        try:
            return store.compare_runs(run_a, run_b)
        except Exception as e:
            raise HTTPException(404, str(e))

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

    # ------------------------------------------------------------------
    # Cache control
    # ------------------------------------------------------------------

    @application.post("/api/v1/refresh")
    def refresh_cache():
        """Force the server to re-read artifacts from disk."""
        reader.invalidate()
        return {"status": "cache_invalidated"}

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
