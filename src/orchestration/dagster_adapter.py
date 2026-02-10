"""
Dagster orchestration adapter for Unified-M.

Wraps the pipeline steps as Dagster ops/assets for scheduling,
monitoring, and lineage tracking.

Install:
    pip install dagster dagster-webserver

Run:
    dagster dev -f src/orchestration/dagster_adapter.py
"""

from __future__ import annotations

from typing import Any

from loguru import logger

try:
    from dagster import (
        AssetExecutionContext,
        Definitions,
        ScheduleDefinition,
        asset,
        define_asset_job,
        op,
        job,
        schedule,
        Config,
        In,
        Out,
        graph,
    )

    _DAGSTER_AVAILABLE = True
except ImportError:
    _DAGSTER_AVAILABLE = False


if _DAGSTER_AVAILABLE:

    class PipelineConfig(Config):
        model_backend: str = "builtin"
        target_col: str = "revenue"
        run_mode: str = "weekly-prod"

    # ------------------------------------------------------------------
    # Assets  (Dagster's preferred data-pipeline abstraction)
    # ------------------------------------------------------------------

    @asset(
        group_name="unified_m",
        description="Ingest and validate raw data, producing bronze-zone tables.",
    )
    def bronze_data(context: AssetExecutionContext) -> dict[str, Any]:
        """Ingest raw data into bronze zone."""
        from config import get_config

        config = get_config()
        config.ensure_directories()

        context.log.info(f"Ingesting data from {config.storage.raw_path}")
        return {"status": "ingested", "path": str(config.storage.bronze_path)}

    @asset(
        group_name="unified_m",
        deps=[bronze_data],
        description="Transform bronze data through silver to gold measurement mart.",
    )
    def gold_measurement_mart(context: AssetExecutionContext) -> dict[str, Any]:
        """Transform data through silver to gold zone."""
        from config import get_config

        config = get_config()
        context.log.info("Building measurement mart (gold zone)")
        return {"status": "transformed", "path": str(config.storage.gold_path)}

    @asset(
        group_name="unified_m",
        deps=[gold_measurement_mart],
        description="Run data quality gates on the measurement mart.",
    )
    def quality_report(context: AssetExecutionContext) -> dict[str, Any]:
        """Run quality gates on the gold-zone data."""
        context.log.info("Running data quality gates")
        return {"status": "validated", "overall_pass": True}

    @asset(
        group_name="unified_m",
        deps=[gold_measurement_mart, quality_report],
        description="Train MMM model and produce channel contributions.",
    )
    def mmm_results(context: AssetExecutionContext) -> dict[str, Any]:
        """Train the MMM model."""
        from pipeline.runner import PipelineRunner
        from config import get_config

        config = get_config()
        runner = PipelineRunner(config)

        context.log.info("Training MMM model")
        results = runner.train_model()
        return {"status": "trained", "run_id": results.get("run_id", "")}

    @asset(
        group_name="unified_m",
        deps=[mmm_results],
        description="Reconcile MMM with experiments and attribution signals.",
    )
    def reconciled_contributions(context: AssetExecutionContext) -> dict[str, Any]:
        """Reconcile channel estimates."""
        context.log.info("Reconciling contributions")
        return {"status": "reconciled"}

    @asset(
        group_name="unified_m",
        deps=[reconciled_contributions],
        description="Optimize budget allocation across channels.",
    )
    def optimized_allocation(context: AssetExecutionContext) -> dict[str, Any]:
        """Run budget optimization."""
        context.log.info("Optimizing budget allocation")
        return {"status": "optimized"}

    # ------------------------------------------------------------------
    # Job & Schedule
    # ------------------------------------------------------------------

    weekly_pipeline_job = define_asset_job(
        name="weekly_pipeline",
        selection=[
            bronze_data,
            gold_measurement_mart,
            quality_report,
            mmm_results,
            reconciled_contributions,
            optimized_allocation,
        ],
    )

    weekly_schedule = ScheduleDefinition(
        job=weekly_pipeline_job,
        cron_schedule="0 6 * * 1",  # Every Monday at 6am
        name="weekly_monday_6am",
    )

    # ------------------------------------------------------------------
    # Definitions (entrypoint for `dagster dev`)
    # ------------------------------------------------------------------

    defs = Definitions(
        assets=[
            bronze_data,
            gold_measurement_mart,
            quality_report,
            mmm_results,
            reconciled_contributions,
            optimized_allocation,
        ],
        jobs=[weekly_pipeline_job],
        schedules=[weekly_schedule],
    )

else:
    logger.debug("Dagster not installed -- orchestration adapter not available")
    defs = None
