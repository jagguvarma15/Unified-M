"""
Versioned artifact store for Unified-M.

Every pipeline run creates an immutable ``runs/<run_id>/`` directory
containing:

    manifest.json        -- RunManifest (config snapshot, data hash, metrics)
    contributions.parquet
    response_curves.json
    reconciliation.json
    optimization.json
    model/               -- backend-specific model state

The store also maintains a ``runs/latest`` symlink so the API and UI
always know where to read from without tracking run ids.

This gives full audit-trail capabilities:
  - "Why did this change?"  -- diff two manifests
  - "What was the model?"   -- reload from model/
  - "What data was used?"   -- check data_hash in manifest
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from loguru import logger

from core.contracts import RunManifest
from core.exceptions import ArtifactError


class ArtifactStore:
    """
    Manages versioned run artifacts on the local filesystem.

    Usage::

        store = ArtifactStore(Path("runs"))
        run_id = store.create_run(config_dict)
        store.save_dataframe(run_id, "contributions", df)
        store.save_json(run_id, "reconciliation", data)
        store.finalise_run(run_id, manifest)
    """

    def __init__(self, base_path: Path | str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def create_run(self, config_snapshot: dict | None = None) -> str:
        """
        Create a new run directory and return its id.

        The run id is a UTC timestamp + short uuid fragment so runs sort
        chronologically in the filesystem.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        short_id = uuid4().hex[:8]
        run_id = f"{ts}_{short_id}"

        run_dir = self.base_path / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "model").mkdir()

        # Write initial manifest
        manifest = RunManifest(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="running",
            config_snapshot=config_snapshot or {},
        )
        self._write_manifest(run_id, manifest)

        logger.info(f"Created run {run_id}")
        return run_id

    def finalise_run(
        self,
        run_id: str,
        manifest: RunManifest,
    ) -> None:
        """
        Write the final manifest and update the ``latest`` symlink.
        """
        manifest.status = "completed"
        self._write_manifest(run_id, manifest)
        self._update_latest(run_id)
        logger.info(f"Finalised run {run_id}")

    def fail_run(self, run_id: str, error: str, step: str = "") -> None:
        """Mark a run as failed."""
        manifest = self.load_manifest(run_id)
        manifest.status = "failed"
        manifest.error_message = error
        manifest.error_step = step
        self._write_manifest(run_id, manifest)
        logger.error(f"Run {run_id} failed at step '{step}': {error}")

    # ------------------------------------------------------------------
    # Read / write helpers
    # ------------------------------------------------------------------

    def save_dataframe(
        self,
        run_id: str,
        name: str,
        df: pd.DataFrame,
    ) -> Path:
        """Save a DataFrame as Parquet inside the run directory."""
        path = self._run_dir(run_id) / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.debug(f"Saved {name}.parquet ({len(df)} rows) to run {run_id}")
        return path

    def save_json(
        self,
        run_id: str,
        name: str,
        data: Any,
    ) -> Path:
        """Save a JSON-serialisable object inside the run directory."""
        path = self._run_dir(run_id) / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
        logger.debug(f"Saved {name}.json to run {run_id}")
        return path

    def load_dataframe(self, run_id: str, name: str) -> pd.DataFrame:
        """Load a Parquet artifact."""
        path = self._run_dir(run_id) / f"{name}.parquet"
        if not path.exists():
            raise ArtifactError(f"Artifact '{name}.parquet' not found", run_id=run_id)
        return pd.read_parquet(path)

    def load_json(self, run_id: str, name: str) -> Any:
        """Load a JSON artifact."""
        path = self._run_dir(run_id) / f"{name}.json"
        if not path.exists():
            raise ArtifactError(f"Artifact '{name}.json' not found", run_id=run_id)
        with open(path) as f:
            return json.load(f)

    def load_json_if_exists(self, run_id: str, name: str) -> Any | None:
        """Load a JSON artifact if present; return None if missing (for optional artifacts in compare)."""
        path = self._run_dir(run_id) / f"{name}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def load_dataframe_if_exists(self, run_id: str, name: str) -> pd.DataFrame | None:
        """Load a Parquet artifact if present; return None if missing."""
        path = self._run_dir(run_id) / f"{name}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def load_manifest(self, run_id: str) -> RunManifest:
        """Load the manifest for a run."""
        data = self.load_json(run_id, "manifest")
        return RunManifest(**data)

    def get_model_dir(self, run_id: str) -> Path:
        """Return the model/ subdirectory for a run."""
        return self._run_dir(run_id) / "model"

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_runs(self, limit: int = 50) -> list[RunManifest]:
        """Return manifests for the most recent *limit* runs."""
        runs = []
        for d in sorted(self.base_path.iterdir(), reverse=True):
            if d.is_dir() and (d / "manifest.json").exists():
                try:
                    runs.append(self.load_manifest(d.name))
                except Exception:
                    continue
            if len(runs) >= limit:
                break
        return runs

    def get_latest_run_id(self) -> str | None:
        """Return the run_id pointed to by the ``latest`` symlink."""
        latest = self.base_path / "latest"
        if latest.exists():
            # Resolve symlink or read the file
            if latest.is_symlink():
                return latest.resolve().name
            # Fallback: read as text
            return latest.read_text().strip()
        # No symlink -- find newest directory
        runs = self.list_runs(limit=1)
        return runs[0].run_id if runs else None

    def compare_runs(self, run_a: str, run_b: str) -> dict:
        """
        Return an advanced, verifiable comparison of two runs.

        Loads manifests as raw JSON (no Pydantic validation) so older or
        partial manifests still work. Loads parameters, optimization, and
        optionally contributions from each run's artifacts.
        """
        # Load manifests as raw JSON so we don't fail on schema/validation (e.g. older manifests)
        ma = self._load_manifest_dict(run_a)
        mb = self._load_manifest_dict(run_b)

        # Optional artifacts (per-run; may be missing for failed or partial runs)
        params_a = self.load_json_if_exists(run_a, "parameters")
        params_b = self.load_json_if_exists(run_b, "parameters")
        opt_a = self.load_json_if_exists(run_a, "optimization")
        opt_b = self.load_json_if_exists(run_b, "optimization")
        contrib_df_a = self.load_dataframe_if_exists(run_a, "contributions")
        contrib_df_b = self.load_dataframe_if_exists(run_b, "contributions")

        def _metrics_dict(m: Any) -> dict[str, Any]:
            if m is None:
                return {}
            if isinstance(m, dict):
                return m
            if hasattr(m, "model_dump"):
                return m.model_dump()
            return {}

        metrics_a_raw = _metrics_dict(ma.get("metrics"))
        metrics_b_raw = _metrics_dict(mb.get("metrics"))

        # Verification: run ids and data fingerprints so clients can verify they're comparing the right runs
        data_hash_a = ma.get("data_hash") or ""
        data_hash_b = mb.get("data_hash") or ""
        model_backend_a = ma.get("model_backend") or ""
        model_backend_b = mb.get("model_backend") or ""
        verification = {
            "run_a": run_a,
            "run_b": run_b,
            "timestamp_a": ma.get("timestamp") or "",
            "timestamp_b": mb.get("timestamp") or "",
            "data_hash_a": data_hash_a,
            "data_hash_b": data_hash_b,
            "data_hash_changed": data_hash_a != data_hash_b,
            "model_backend_a": model_backend_a,
            "model_backend_b": model_backend_b,
            "model_backend_changed": model_backend_a != model_backend_b,
        }

        # Metrics: raw and deltas (delta = B − A)
        metrics_a: dict[str, Any] = metrics_a_raw
        metrics_b: dict[str, Any] = metrics_b_raw
        metrics_delta: dict[str, float] = {}
        for key in set(metrics_a.keys()) | set(metrics_b.keys()):
            va = metrics_a.get(key)
            vb = metrics_b.get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                metrics_delta[key] = round(float(vb) - float(va), 6)

        # Coefficients: from parameters.json (coefficients key)
        coefficients_a: dict[str, float] = {}
        coefficients_b: dict[str, float] = {}
        if params_a and isinstance(params_a.get("coefficients"), dict):
            coefficients_a = {k: float(v) for k, v in params_a["coefficients"].items()}
        if params_b and isinstance(params_b.get("coefficients"), dict):
            coefficients_b = {k: float(v) for k, v in params_b["coefficients"].items()}
        all_coef_channels = sorted(set(coefficients_a.keys()) | set(coefficients_b.keys()))
        coefficient_diff = {ch: round(coefficients_b.get(ch, 0) - coefficients_a.get(ch, 0), 6) for ch in all_coef_channels}

        # Allocations: from optimization.json (optimal_allocation or channel_allocations)
        def _get_allocation(opt: dict | None) -> dict[str, float]:
            if not opt:
                return {}
            alloc = opt.get("optimal_allocation") or opt.get("channel_allocations") or {}
            return {k: float(v) for k, v in alloc.items()}

        allocation_a = _get_allocation(opt_a)
        allocation_b = _get_allocation(opt_b)
        all_alloc_channels = sorted(set(allocation_a.keys()) | set(allocation_b.keys()))
        allocation_diff = {ch: round(allocation_b.get(ch, 0) - allocation_a.get(ch, 0), 2) for ch in all_alloc_channels}

        # Current allocation (baseline) if present
        def _get_current_allocation(opt: dict | None) -> dict[str, float]:
            if not opt:
                return {}
            curr = opt.get("current_allocation") or opt.get("current_allocations") or {}
            return {k: float(v) for k, v in curr.items()}

        current_allocation_a = _get_current_allocation(opt_a)
        current_allocation_b = _get_current_allocation(opt_b)

        # Contribution totals per channel (from contributions.parquet) for verifiable contribution shift
        def _contribution_totals(df: pd.DataFrame | None) -> dict[str, float]:
            if df is None or df.empty:
                return {}
            reserved = {"date", "actual", "predicted", "baseline"}
            cols = [c for c in df.columns if c not in reserved]
            totals = {}
            for c in cols:
                try:
                    totals[c] = round(float(df[c].abs().sum()), 2)
                except (TypeError, ValueError):
                    pass
            return totals

        contribution_totals_a = _contribution_totals(contrib_df_a)
        contribution_totals_b = _contribution_totals(contrib_df_b)
        contrib_channels = sorted(set(contribution_totals_a.keys()) | set(contribution_totals_b.keys()))
        contribution_diff = {
            ch: round(contribution_totals_b.get(ch, 0) - contribution_totals_a.get(ch, 0), 2)
            for ch in contrib_channels
        }

        # Config diff (shallow)
        config_changes = _dict_diff(ma.get("config_snapshot") or {}, mb.get("config_snapshot") or {})

        n_rows_a = ma.get("n_rows") or 0
        n_rows_b = mb.get("n_rows") or 0
        n_channels_a = ma.get("n_channels") or 0
        n_channels_b = mb.get("n_channels") or 0

        return {
            "verification": verification,
            "run_a": run_a,
            "run_b": run_b,
            "config_changes": config_changes,
            "n_rows_a": n_rows_a,
            "n_rows_b": n_rows_b,
            "n_rows_change": n_rows_b - n_rows_a,
            "n_channels_a": n_channels_a,
            "n_channels_b": n_channels_b,
            "n_channels_change": n_channels_b - n_channels_a,
            "metrics_a": metrics_a,
            "metrics_b": metrics_b,
            "metrics_delta": metrics_delta,
            "coefficients_a": coefficients_a,
            "coefficients_b": coefficients_b,
            "coefficient_diff": coefficient_diff,
            "allocation_a": allocation_a,
            "allocation_b": allocation_b,
            "current_allocation_a": current_allocation_a,
            "current_allocation_b": current_allocation_b,
            "allocation_diff": allocation_diff,
            "contribution_totals_a": contribution_totals_a,
            "contribution_totals_b": contribution_totals_b,
            "contribution_diff": contribution_diff,
            "data_hash_changed": verification["data_hash_changed"],
            "model_backend_changed": verification["model_backend_changed"],
        }

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_data_hash(df: pd.DataFrame) -> str:
        """Deterministic SHA-256 of a DataFrame for reproducibility checks."""
        import numpy as np
        h = hashlib.sha256()
        for col in sorted(df.columns):
            h.update(col.encode())
            # Convert to numpy array to handle both regular arrays and nullable IntegerArray
            values = df[col].to_numpy(dtype=None, na_value=0)
            # Ensure we have a contiguous array for tobytes()
            if not values.flags['C_CONTIGUOUS']:
                values = np.ascontiguousarray(values)
            h.update(values.tobytes())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        d = self.base_path / run_id
        if not d.exists():
            raise ArtifactError(f"Run directory not found: {run_id}", run_id=run_id)
        return d

    def _load_manifest_dict(self, run_id: str) -> dict[str, Any]:
        """Load manifest.json as a raw dict (no Pydantic validation). Raises if run or manifest missing."""
        return self.load_json(run_id, "manifest")

    def _write_manifest(self, run_id: str, manifest: RunManifest) -> None:
        path = self._run_dir(run_id) / "manifest.json"
        with open(path, "w") as f:
            json.dump(manifest.model_dump(), f, indent=2, default=_json_default)

    def _update_latest(self, run_id: str) -> None:
        latest = self.base_path / "latest"
        # Remove old symlink / file
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        # Write run_id as text (symlinks can be fragile across platforms)
        latest.write_text(run_id)


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """JSON fallback serialiser for numpy and pandas types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _dict_diff(a: dict, b: dict) -> dict:
    """Shallow diff of two flat-ish dicts."""
    changes: dict[str, Any] = {}
    all_keys = set(a.keys()) | set(b.keys())
    for key in sorted(all_keys):
        va = a.get(key)
        vb = b.get(key)
        if va != vb:
            changes[key] = {"before": va, "after": vb}
    return changes
