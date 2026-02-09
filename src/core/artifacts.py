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
        Return a diff-style comparison of two runs.

        Useful for answering "why did results change between runs?"
        """
        ma = self.load_manifest(run_a)
        mb = self.load_manifest(run_b)

        diff: dict[str, Any] = {
            "run_a": run_a,
            "run_b": run_b,
        }

        # Config diff
        diff["config_changes"] = _dict_diff(ma.config_snapshot, mb.config_snapshot)

        # Data diff
        diff["data_hash_changed"] = ma.data_hash != mb.data_hash
        diff["n_rows_change"] = mb.n_rows - ma.n_rows
        diff["n_channels_change"] = mb.n_channels - ma.n_channels

        # Metrics diff
        if ma.metrics and mb.metrics:
            diff["metrics_a"] = ma.metrics.model_dump()
            diff["metrics_b"] = mb.metrics.model_dump()

        diff["model_backend_changed"] = ma.model_backend != mb.model_backend

        return diff

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_data_hash(df: pd.DataFrame) -> str:
        """Deterministic SHA-256 of a DataFrame for reproducibility checks."""
        h = hashlib.sha256()
        for col in sorted(df.columns):
            h.update(col.encode())
            h.update(df[col].values.tobytes())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        d = self.base_path / run_id
        if not d.exists():
            raise ArtifactError(f"Run directory not found: {run_id}", run_id=run_id)
        return d

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
