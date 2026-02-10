"""
Prefect orchestration adapter for Unified-M.

Wraps the pipeline steps as Prefect tasks/flows for scheduling,
retry logic, and observability.

Install:
    pip install prefect

Run:
    python -m orchestration.prefect_adapter
    # or schedule via: prefect deployment run weekly-pipeline/weekly
"""

from __future__ import annotations

from typing import Any

from loguru import logger

try:
    from prefect import flow, task, get_run_logger
    from prefect.tasks import task_input_hash
    from datetime import timedelta

    _PREFECT_AVAILABLE = True
except ImportError:
    _PREFECT_AVAILABLE = False


if _PREFECT_AVAILABLE:

    @task(
        name="ingest_data",
        retries=2,
        retry_delay_seconds=60,
        description="Ingest raw data into bronze zone",
    )
    def ingest_data() -> dict[str, Any]:
        """Ingest raw data sources."""
        plogger = get_run_logger()
        from config import get_config

        config = get_config()
        config.ensure_directories()
        plogger.info(f"Ingesting from {config.storage.raw_path}")
        return {"status": "ingested", "path": str(config.storage.bronze_path)}

    @task(
        name="transform_data",
        description="Transform bronze -> silver -> gold",
    )
    def transform_data(ingest_result: dict) -> dict[str, Any]:
        """Transform through lakehouse zones."""
        plogger = get_run_logger()
        from config import get_config

        config = get_config()
        plogger.info("Building measurement mart")
        return {"status": "transformed", "path": str(config.storage.gold_path)}

    @task(
        name="run_quality_gates",
        description="Validate data quality",
    )
    def run_quality_gates(transform_result: dict) -> dict[str, Any]:
        """Run data quality gates."""
        plogger = get_run_logger()
        from quality.gates import run_quality_gates as _run_gates

        plogger.info("Running quality gates")
        # In a real run, pass actual DataFrames
        report = _run_gates()
        return report.to_dict()

    @task(
        name="train_model",
        retries=1,
        retry_delay_seconds=120,
        description="Train MMM model",
    )
    def train_model(
        quality_result: dict,
        model_backend: str = "builtin",
        target_col: str = "revenue",
    ) -> dict[str, Any]:
        """Train the MMM model."""
        plogger = get_run_logger()
        plogger.info(f"Training model (backend={model_backend})")

        from pipeline.runner import PipelineRunner
        from config import get_config

        config = get_config()
        runner = PipelineRunner(config)
        results = runner.train_model()
        return {"status": "trained", "metrics": results.get("metrics", {})}

    @task(
        name="reconcile",
        description="Reconcile MMM with experiments and attribution",
    )
    def reconcile(train_result: dict) -> dict[str, Any]:
        """Reconcile channel estimates."""
        plogger = get_run_logger()
        plogger.info("Reconciling contributions")
        return {"status": "reconciled"}

    @task(
        name="optimize",
        description="Optimize budget allocation",
    )
    def optimize(reconcile_result: dict) -> dict[str, Any]:
        """Run budget optimization."""
        plogger = get_run_logger()
        plogger.info("Optimizing budget allocation")
        return {"status": "optimized"}

    @task(
        name="export_artifacts",
        description="Save all outputs to artifact store",
    )
    def export_artifacts(optimize_result: dict) -> dict[str, Any]:
        """Export pipeline artifacts."""
        plogger = get_run_logger()
        plogger.info("Exporting artifacts")
        return {"status": "exported"}

    # ------------------------------------------------------------------
    # Flows
    # ------------------------------------------------------------------

    @flow(
        name="weekly-pipeline",
        description="Full Unified-M weekly pipeline: ingest → transform → validate → train → reconcile → optimize → export",
        retries=0,
    )
    def weekly_pipeline(
        model_backend: str = "builtin",
        target_col: str = "revenue",
    ) -> dict[str, Any]:
        """Full weekly pipeline flow."""
        plogger = get_run_logger()
        plogger.info(f"Starting weekly pipeline (model={model_backend})")

        ingest_result = ingest_data()
        transform_result = transform_data(ingest_result)
        quality_result = run_quality_gates(transform_result)
        train_result = train_model(quality_result, model_backend, target_col)
        reconcile_result = reconcile(train_result)
        optimize_result = optimize(reconcile_result)
        export_result = export_artifacts(optimize_result)

        plogger.info("Pipeline complete!")
        return export_result

    @flow(
        name="scenario-analysis",
        description="Run what-if scenario without retraining",
    )
    def scenario_analysis(
        budget: float = 100_000,
        channel_shifts: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """What-if scenario analysis flow."""
        plogger = get_run_logger()
        plogger.info(f"Running scenario: budget=${budget:,.0f}")
        return {"status": "scenario_complete", "budget": budget}

    @flow(
        name="experiment-calibration",
        description="Calibrate MMM priors from new experiment results",
    )
    def experiment_calibration(
        test_file: str = "",
    ) -> dict[str, Any]:
        """Experiment calibration flow."""
        plogger = get_run_logger()
        plogger.info(f"Calibrating from: {test_file}")
        return {"status": "calibration_complete"}

    # ------------------------------------------------------------------
    # Run directly
    # ------------------------------------------------------------------

    if __name__ == "__main__":
        weekly_pipeline(model_backend="builtin", target_col="revenue")

else:
    logger.debug("Prefect not installed -- orchestration adapter not available")
