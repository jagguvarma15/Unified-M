"""
Run-mode logic for the Unified-M pipeline.

Each mode configures which pipeline steps execute, which model backend
to use, and how data is sourced.

Modes:
  local-dev              -- fast Ridge, sample/demo data
  weekly-prod            -- full Bayesian, latest ingested data
  backfill               -- retrain on a historical date window
  what-if                -- scenario planning, no retraining
  experiment-calibration -- update priors from new test results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class RunModeConfig:
    """Configuration derived from a run mode selection."""

    mode: str
    model_backend: str = "builtin"
    retrain: bool = True
    ingest_new_data: bool = True
    run_quality_gates: bool = True
    run_reconciliation: bool = True
    run_optimization: bool = True
    run_scenarios: bool = False
    update_priors_only: bool = False

    # Backfill-specific
    backfill_start: str | None = None
    backfill_end: str | None = None

    # What-if specific
    scenario_budget: float | None = None
    scenario_shifts: dict[str, float] = field(default_factory=dict)

    # Experiment-calibration specific
    test_file: str | None = None

    pipeline_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "model_backend": self.model_backend,
            "retrain": self.retrain,
            "ingest_new_data": self.ingest_new_data,
            "run_quality_gates": self.run_quality_gates,
            "run_reconciliation": self.run_reconciliation,
            "run_optimization": self.run_optimization,
            "run_scenarios": self.run_scenarios,
            "update_priors_only": self.update_priors_only,
            "pipeline_steps": self.pipeline_steps,
        }


def resolve_run_mode(
    mode: str = "local-dev",
    *,
    model: str | None = None,
    backfill_start: str | None = None,
    backfill_end: str | None = None,
    scenario_budget: float | None = None,
    scenario_shifts: dict[str, float] | None = None,
    test_file: str | None = None,
) -> RunModeConfig:
    """
    Translate a mode name into a concrete ``RunModeConfig``.

    Args:
        mode:             One of the five supported modes.
        model:            Override model backend (e.g. "pymc").
        backfill_start:   ISO date for backfill window start.
        backfill_end:     ISO date for backfill window end.
        scenario_budget:  Total budget for what-if scenarios.
        scenario_shifts:  Channel -> % shift for what-if.
        test_file:        Path to new experiment results.

    Returns:
        Fully resolved RunModeConfig.
    """
    mode = mode.lower().strip()

    if mode == "local-dev":
        cfg = RunModeConfig(
            mode=mode,
            model_backend=model or "builtin",
            retrain=True,
            ingest_new_data=True,
            run_quality_gates=True,
            run_reconciliation=True,
            run_optimization=True,
            pipeline_steps=[
                "ingest", "validate", "transform", "train",
                "reconcile", "optimize", "export",
            ],
        )

    elif mode == "weekly-prod":
        cfg = RunModeConfig(
            mode=mode,
            model_backend=model or "pymc",
            retrain=True,
            ingest_new_data=True,
            run_quality_gates=True,
            run_reconciliation=True,
            run_optimization=True,
            pipeline_steps=[
                "ingest", "validate", "transform", "train",
                "reconcile", "optimize", "evaluate", "export",
            ],
        )

    elif mode == "backfill":
        if not backfill_start or not backfill_end:
            raise ValueError("backfill mode requires --start and --end dates")
        cfg = RunModeConfig(
            mode=mode,
            model_backend=model or "builtin",
            retrain=True,
            ingest_new_data=True,
            run_quality_gates=True,
            run_reconciliation=True,
            run_optimization=False,
            backfill_start=backfill_start,
            backfill_end=backfill_end,
            pipeline_steps=[
                "ingest", "validate", "transform", "train",
                "reconcile", "export",
            ],
        )

    elif mode == "what-if":
        cfg = RunModeConfig(
            mode=mode,
            model_backend=model or "builtin",
            retrain=False,
            ingest_new_data=False,
            run_quality_gates=False,
            run_reconciliation=False,
            run_optimization=True,
            run_scenarios=True,
            scenario_budget=scenario_budget,
            scenario_shifts=scenario_shifts or {},
            pipeline_steps=["load_artifacts", "scenario", "export"],
        )

    elif mode == "experiment-calibration":
        cfg = RunModeConfig(
            mode=mode,
            model_backend=model or "builtin",
            retrain=False,
            ingest_new_data=True,
            run_quality_gates=True,
            run_reconciliation=True,
            run_optimization=True,
            update_priors_only=True,
            test_file=test_file,
            pipeline_steps=[
                "ingest_tests", "validate", "reconcile",
                "update_priors", "optimize", "export",
            ],
        )

    else:
        raise ValueError(
            f"Unknown run mode: '{mode}'. "
            f"Valid modes: local-dev, weekly-prod, backfill, what-if, experiment-calibration"
        )

    logger.info(
        f"Resolved run mode '{mode}' -> backend={cfg.model_backend}, "
        f"retrain={cfg.retrain}, steps={cfg.pipeline_steps}"
    )
    return cfg
