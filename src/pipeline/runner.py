"""
Pipeline runner -- the single entry point for an end-to-end Unified-M run.

Orchestrates:
  1. Connect   -- load data from configured sources
  2. Validate  -- check schemas
  3. Transform -- create MMM-ready features (adstock, saturation, time)
  4. Train     -- fit the selected model backend
  5. Reconcile -- fuse MMM + tests + attribution into unified estimates
  6. Optimise  -- allocate budget using fitted response curves
  7. Export    -- write versioned artifacts

Every step writes its output into the run artifact directory so the
full provenance is captured.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from core.artifacts import ArtifactStore
from core.contracts import (
    ChannelResult,
    ModelMetrics,
    RunManifest,
)
from core.exceptions import PipelineError
from connectors.local import load_file, ParquetConnector
from models.registry import get_model
from transforms.features import (
    create_mmm_features,
    pivot_media_spend,
    add_time_features,
    add_fourier_features,
)
from reconciliation.engine import ReconciliationEngine
from optimization.allocator import BudgetOptimizer


class Pipeline:
    """
    End-to-end Unified-M pipeline.

    Example::

        pipe = Pipeline(config)
        pipe.connect(
            media_spend="data/media_spend.csv",
            outcomes="data/outcomes.csv",
        )
        results = pipe.run(model="builtin")
        print(results["metrics"])
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        runs_dir: str | Path = "runs",
    ):
        self._config = config or {}
        self._store = ArtifactStore(Path(runs_dir))
        self._run_id: str | None = None

        # Raw data
        self._media_spend: pd.DataFrame | None = None
        self._outcomes: pd.DataFrame | None = None
        self._controls: pd.DataFrame | None = None
        self._incrementality_tests: pd.DataFrame | None = None
        self._attribution: pd.DataFrame | None = None

        # Processed
        self._mmm_input: pd.DataFrame | None = None
        self._media_cols: list[str] = []
        self._control_cols: list[str] = []
        self._target_col: str = "y"

    # ------------------------------------------------------------------
    # Step 1: Connect
    # ------------------------------------------------------------------

    def connect(
        self,
        media_spend: str | Path | pd.DataFrame | None = None,
        outcomes: str | Path | pd.DataFrame | None = None,
        controls: str | Path | pd.DataFrame | None = None,
        incrementality_tests: str | Path | pd.DataFrame | None = None,
        attribution: str | Path | pd.DataFrame | None = None,
    ) -> "Pipeline":
        """
        Load data from files or accept pre-loaded DataFrames.

        At minimum ``media_spend`` and ``outcomes`` are required.
        """
        logger.info("Pipeline: connecting to data sources...")

        self._media_spend = self._load_or_use(media_spend, "media_spend")
        self._outcomes = self._load_or_use(outcomes, "outcomes")
        self._controls = self._load_or_use(controls, "controls")
        self._incrementality_tests = self._load_or_use(incrementality_tests, "incrementality_tests")
        self._attribution = self._load_or_use(attribution, "attribution")

        if self._media_spend is None:
            raise PipelineError("media_spend data is required", step="connect")
        if self._outcomes is None:
            raise PipelineError("outcomes data is required", step="connect")

        # Coerce date columns
        for df_name in ["_media_spend", "_outcomes", "_controls", "_incrementality_tests", "_attribution"]:
            df = getattr(self, df_name)
            if df is not None and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

        n_channels = self._media_spend["channel"].nunique() if "channel" in self._media_spend.columns else 0
        logger.info(
            f"Connected: media_spend={len(self._media_spend)} rows, "
            f"outcomes={len(self._outcomes)} rows, "
            f"channels={n_channels}, "
            f"has_tests={self._incrementality_tests is not None}, "
            f"has_attribution={self._attribution is not None}"
        )
        return self

    # ------------------------------------------------------------------
    # Step 2-7: Run
    # ------------------------------------------------------------------

    def run(
        self,
        model: str = "builtin",
        target_col: str = "revenue",
        total_budget: float | None = None,
        model_kwargs: dict[str, Any] | None = None,
        reconciliation_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full pipeline and return results.

        Args:
            model:         Model backend name ("builtin", "pymc", ...).
            target_col:    Outcome column to use as the target.
            total_budget:  Budget for optimization (auto-computed if None).
            model_kwargs:  Extra keyword arguments for the model's fit().
            reconciliation_weights: Override weights for reconciliation.

        Returns:
            Dictionary with keys: metrics, contributions, reconciliation,
            optimization, response_curves, run_id.
        """
        if self._media_spend is None:
            raise PipelineError("Call connect() before run()", step="run")

        t0 = time.time()

        # Create a new artifact run
        self._run_id = self._store.create_run(config_snapshot={
            "model": model,
            "target_col": target_col,
            "total_budget": total_budget,
            **self._config,
        })

        results: dict[str, Any] = {"run_id": self._run_id}

        try:
            # -- Transform ------------------------------------------------
            logger.info("Pipeline step: transform")
            self._transform(target_col)
            self._store.save_dataframe(self._run_id, "mmm_input", self._mmm_input)

            # -- Train ----------------------------------------------------
            logger.info(f"Pipeline step: train (backend={model})")
            mmm = get_model(model, **(model_kwargs or {}))
            fit_meta = mmm.fit(
                df=self._mmm_input,
                target_col="y",
                media_cols=self._media_cols,
                control_cols=self._control_cols,
            )

            # Save model state
            mmm.save_state(self._store.get_model_dir(self._run_id))

            metrics_dict = fit_meta.get("metrics", {})
            results["metrics"] = metrics_dict

            # Contributions
            contributions = mmm.get_channel_contributions()
            contrib_df = pd.DataFrame(contributions)
            contrib_df["date"] = self._mmm_input["date"].values
            contrib_df["actual"] = self._mmm_input["y"].values
            contrib_df["predicted"] = mmm.predict(self._mmm_input)
            self._store.save_dataframe(self._run_id, "contributions", contrib_df)
            results["contributions"] = contrib_df

            # Response curves
            curves = mmm.get_response_curves()
            curves_json = {
                ch: df.to_dict(orient="list") for ch, df in curves.items()
            }
            self._store.save_json(self._run_id, "response_curves", curves_json)
            results["response_curves"] = curves

            # Parameters
            params = mmm.get_parameters()
            self._store.save_json(self._run_id, "parameters", params)
            results["parameters"] = params

            # -- Reconcile ------------------------------------------------
            logger.info("Pipeline step: reconcile")
            recon_result = self._reconcile(params, reconciliation_weights)
            self._store.save_json(self._run_id, "reconciliation", recon_result)
            results["reconciliation"] = recon_result

            # -- Optimise -------------------------------------------------
            logger.info("Pipeline step: optimise")
            budget = total_budget or float(self._media_spend["spend"].sum())
            opt_result = self._optimise(curves, params, budget)
            self._store.save_json(self._run_id, "optimization", opt_result)
            results["optimization"] = opt_result

            # -- Finalise -------------------------------------------------
            duration = time.time() - t0
            data_hash = ArtifactStore.compute_data_hash(self._mmm_input)

            manifest = RunManifest(
                run_id=self._run_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(duration, 2),
                status="completed",
                model_backend=model,
                pipeline_steps=["connect", "transform", "train", "reconcile", "optimise"],
                config_snapshot=self._config,
                data_hash=data_hash,
                n_rows=len(self._mmm_input),
                n_channels=len(self._media_cols),
                date_range=(
                    str(self._mmm_input["date"].min()),
                    str(self._mmm_input["date"].max()),
                ),
                metrics=ModelMetrics(**metrics_dict) if metrics_dict else None,
                n_channel_results=len(recon_result.get("channel_estimates", {})),
                total_incremental_value=recon_result.get("total_incremental_value", 0),
            )
            self._store.finalise_run(self._run_id, manifest)

            logger.info(
                f"Pipeline complete in {duration:.1f}s.  "
                f"Run: {self._run_id}  "
                f"MAPE={metrics_dict.get('mape', 'N/A')}%  "
                f"R2={metrics_dict.get('r_squared', 'N/A')}"
            )

        except Exception as exc:
            self._store.fail_run(self._run_id, str(exc), step="unknown")
            raise PipelineError(f"Pipeline failed: {exc}", step="run") from exc

        return results

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _transform(self, target_col: str) -> None:
        """Create MMM-ready dataset from raw inputs."""
        # Determine target
        if target_col in self._outcomes.columns:
            t_col = target_col
        elif "revenue" in self._outcomes.columns:
            t_col = "revenue"
        elif "conversions" in self._outcomes.columns:
            t_col = "conversions"
        else:
            raise PipelineError(
                f"Target column '{target_col}' not found in outcomes. "
                f"Available: {list(self._outcomes.columns)}",
                step="transform",
            )

        self._mmm_input = create_mmm_features(
            media_spend=self._media_spend,
            outcomes=self._outcomes,
            controls=self._controls,
            target_col=t_col,
        )

        # Identify column types
        self._media_cols = [c for c in self._mmm_input.columns if c.endswith("_spend")]
        control_candidates = [
            "is_holiday", "promo", "promotion", "price_index",
            "trend", "is_weekend",
        ]
        self._control_cols = [
            c for c in self._mmm_input.columns
            if c in control_candidates or c.startswith("fourier_")
        ]

        logger.info(
            f"Transform complete: {len(self._mmm_input)} rows, "
            f"{len(self._media_cols)} media cols, "
            f"{len(self._control_cols)} control cols"
        )

    def _reconcile(
        self,
        params: dict[str, Any],
        weights: dict[str, float] | None,
    ) -> dict:
        """Run reconciliation engine."""
        w = weights or {}
        engine = ReconciliationEngine(
            mmm_weight=w.get("mmm", 0.5),
            incrementality_weight=w.get("incrementality", 0.3),
            attribution_weight=w.get("attribution", 0.2),
        )

        result = engine.reconcile(
            mmm_results=params,
            incrementality_tests=self._incrementality_tests,
            attribution_data=self._attribution,
            channels=self._media_cols,
        )
        return result.to_dict()

    def _optimise(
        self,
        curves: dict[str, pd.DataFrame],
        params: dict[str, Any],
        total_budget: float,
    ) -> dict:
        """Run budget optimisation."""
        # Build callable response curves from DataFrame data
        response_fns = {}
        for ch, curve_df in curves.items():
            spend_arr = curve_df["spend"].values
            response_arr = curve_df["response"].values

            def _make_fn(s, r):
                def fn(x):
                    return float(np.interp(x, s, r))
                return fn

            response_fns[ch] = _make_fn(spend_arr, response_arr)

        optimizer = BudgetOptimizer(
            response_curves=response_fns,
            total_budget=total_budget,
        )

        # Set current allocation from data
        if self._media_spend is not None and "channel" in self._media_spend.columns:
            current = (
                self._media_spend.groupby("channel")["spend"].sum()
                .to_dict()
            )
            # Map to _spend column names
            current_mapped = {f"{k}_spend": v for k, v in current.items()}
            optimizer.set_current_allocation(current_mapped)

        result = optimizer.optimize()
        return result.to_dict()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_or_use(
        source: str | Path | pd.DataFrame | None,
        name: str,
    ) -> pd.DataFrame | None:
        """Load from file if string/Path, use directly if DataFrame."""
        if source is None:
            return None
        if isinstance(source, pd.DataFrame):
            return source.copy()
        path = Path(source)
        if not path.exists():
            logger.warning(f"Data source not found for {name}: {path}")
            return None
        return load_file(path)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def store(self) -> ArtifactStore:
        return self._store

    @property
    def run_id(self) -> str | None:
        return self._run_id
