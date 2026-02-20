"""
Background job manager for long-running pipeline executions.

Runs each pipeline in a background thread and exposes state that the
API can poll.  Job state is held in-memory -- restarting the server
clears the history (by design: runs/ artifacts are the durable record).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from loguru import logger


PIPELINE_STEPS = [
    "connect",
    "quality_gates",
    "transform",
    "train",
    "reconcile",
    "optimise",
    "finalise",
]


@dataclass
class Job:
    job_id: str
    status: str = "pending"  # pending | running | completed | failed
    current_step: str = ""
    progress_pct: int = 0
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    run_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "current_step": self.current_step,
            "progress_pct": self.progress_pct,
            "logs": self.logs[-50:],
            "error": self.error,
            "run_id": self.run_id,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    """Manages pipeline jobs in background threads."""

    def __init__(self, max_history: int = 50):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_history = max_history

    def create_job(self) -> Job:
        job = Job(
            job_id=uuid4().hex[:12],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._trim_history()
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda j: j.created_at,
                reverse=True,
            )
        return [j.to_dict() for j in jobs[:limit]]

    def start_pipeline(
        self,
        job: Job,
        run_fn: Callable[..., dict[str, Any]],
        on_complete: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run *run_fn* in a background thread, feeding progress into *job*."""

        def _progress_callback(step: str, message: str = "") -> None:
            with self._lock:
                job.status = "running"
                job.current_step = step
                try:
                    idx = PIPELINE_STEPS.index(step)
                    job.progress_pct = int((idx + 1) / len(PIPELINE_STEPS) * 100)
                except ValueError:
                    pass
                if message:
                    job.logs.append(message)

        def _worker() -> None:
            try:
                job.status = "running"
                job.logs.append("Pipeline started")
                results = run_fn(on_progress=_progress_callback, **kwargs)
                with self._lock:
                    job.status = "completed"
                    job.progress_pct = 100
                    job.run_id = results.get("run_id")
                    job.metrics = results.get("metrics", {})
                    job.finished_at = datetime.now(timezone.utc).isoformat()
                    job.logs.append("Pipeline completed successfully")
                if on_complete:
                    on_complete()
            except Exception as exc:
                with self._lock:
                    job.status = "failed"
                    job.error = str(exc)
                    job.finished_at = datetime.now(timezone.utc).isoformat()
                    job.logs.append(f"Pipeline failed: {exc}")
                logger.exception("Background pipeline job failed")

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _trim_history(self) -> None:
        if len(self._jobs) > self._max_history:
            sorted_jobs = sorted(
                self._jobs.items(),
                key=lambda kv: kv[1].created_at,
            )
            for jid, _ in sorted_jobs[: len(self._jobs) - self._max_history]:
                del self._jobs[jid]
