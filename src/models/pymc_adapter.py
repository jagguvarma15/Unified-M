"""
PyMC-Marketing adapter for Unified-M.

Wraps pymc-marketing's ``DelayedSaturatedMMM`` behind the canonical
``BaseMMM`` interface.  This adapter is loaded only when
``pymc-marketing`` is installed; otherwise it silently stays
unregistered and the framework falls back to the built-in model.

Install:
    pip install pymc-marketing
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from core.base_model import BaseMMM

# Guard import -- if pymc-marketing is missing, this module will
# fail to import and the registry will skip it automatically.
from pymc_marketing.mmm import (
    DelayedSaturatedMMM,
    GeometricAdstock,
    LogisticSaturation,
)

from models.registry import register_backend


class PyMCAdapter(BaseMMM):
    """
    Bayesian MMM powered by PyMC-Marketing.

    Provides:
      - Full posterior distributions (not just point estimates).
      - Learned adstock and saturation parameters.
      - Proper uncertainty quantification.

    Requires ``pymc-marketing >= 0.9.0``.
    """

    def __init__(
        self,
        adstock_l_max: int = 8,
        yearly_seasonality: int = 2,
        n_samples: int = 1000,
        n_chains: int = 4,
        target_accept: float = 0.9,
    ):
        self._adstock_l_max = adstock_l_max
        self._yearly_seasonality = yearly_seasonality
        self._n_samples = n_samples
        self._n_chains = n_chains
        self._target_accept = target_accept

        self._model: DelayedSaturatedMMM | None = None
        self._trace = None
        self._data: pd.DataFrame | None = None
        self._media_cols: list[str] = []
        self._control_cols: list[str] = []
        self._target_col: str = "y"
        self._date_col: str = "date"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "pymc"

    @property
    def description(self) -> str:
        return "Bayesian MMM via PyMC-Marketing (MCMC inference)"

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "y",
        media_cols: list[str] | None = None,
        control_cols: list[str] | None = None,
        date_col: str = "date",
        **kwargs: Any,
    ) -> dict[str, Any]:
        logger.info("Fitting PyMC-Marketing MMM...")

        self._data = df.copy()
        self._target_col = target_col
        self._date_col = date_col
        self._media_cols = media_cols or [c for c in df.columns if c.endswith("_spend")]
        self._control_cols = control_cols or []

        n_samples = kwargs.pop("n_samples", self._n_samples)
        n_chains = kwargs.pop("n_chains", self._n_chains)
        target_accept = kwargs.pop("target_accept", self._target_accept)
        random_seed = kwargs.pop("random_seed", 42)

        self._model = DelayedSaturatedMMM(
            date_column=date_col,
            channel_columns=self._media_cols,
            control_columns=self._control_cols if self._control_cols else None,
            adstock=GeometricAdstock(l_max=self._adstock_l_max),
            saturation=LogisticSaturation(),
            yearly_seasonality=self._yearly_seasonality,
        )

        y = df[target_col].values
        self._model.fit(
            X=df,
            y=y,
            target_col=target_col,
            draws=n_samples,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=random_seed,
        )

        self._trace = self._model.fit_result

        # Compute metrics
        y_pred = (
            self._model.posterior_predictive
            .mean(dim=["chain", "draw"])["y"]
            .values
        )
        metrics = self.get_metrics(y, y_pred)

        # Bayesian diagnostics
        import arviz as az
        summary = az.summary(self._trace)
        metrics["rhat_max"] = float(summary["r_hat"].max()) if "r_hat" in summary.columns else None
        metrics["ess_min"] = float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns else None
        if hasattr(self._trace, "sample_stats"):
            div = self._trace.sample_stats.get("diverging")
            metrics["divergences"] = int(div.sum()) if div is not None else 0

        logger.info(
            f"PyMC MMM fit complete.  "
            f"R2={metrics['r_squared']:.3f}  MAPE={metrics['mape']:.2f}%"
        )

        return {"metrics": metrics}

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        pred = self._model.sample_posterior_predictive(df)
        return pred.posterior_predictive["y"].mean(dim=["chain", "draw"]).values

    # ------------------------------------------------------------------
    # Contributions
    # ------------------------------------------------------------------

    def get_channel_contributions(self) -> dict[str, np.ndarray]:
        self._check_fitted()
        raw = self._model.compute_channel_contribution_original_scale()
        contributions: dict[str, np.ndarray] = {}
        for i, ch in enumerate(self._media_cols):
            contributions[ch] = raw[:, :, i].mean(axis=(0, 1)) if raw.ndim > 2 else raw[:, i]
        return contributions

    # ------------------------------------------------------------------
    # Response curves
    # ------------------------------------------------------------------

    def get_response_curves(
        self,
        spend_grid: np.ndarray | None = None,
        n_points: int = 100,
    ) -> dict[str, pd.DataFrame]:
        self._check_fitted()
        posterior = self._trace.posterior

        curves: dict[str, pd.DataFrame] = {}
        for i, ch in enumerate(self._media_cols):
            max_spend = self._data[ch].max() * 1.5
            grid = spend_grid if spend_grid is not None else np.linspace(0, max_spend, n_points)

            coef = float(posterior["beta_channel"][:, :, i].mean()) if "beta_channel" in posterior else 1.0
            lam = float(posterior["saturation_lam"][:, :, i].mean()) if "saturation_lam" in posterior else 1.0

            response = coef * (1 - np.exp(-lam * grid))
            marginal = coef * lam * np.exp(-lam * grid)

            curves[ch] = pd.DataFrame({
                "spend": grid,
                "response": response,
                "marginal_response": marginal,
            })

        return curves

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_parameters(self) -> dict[str, Any]:
        self._check_fitted()
        posterior = self._trace.posterior

        coefficients = {}
        adstock_params = {}
        saturation_params = {}

        for i, ch in enumerate(self._media_cols):
            if "beta_channel" in posterior:
                coefficients[ch] = float(posterior["beta_channel"][:, :, i].mean())
            if "adstock_alpha" in posterior:
                adstock_params[ch] = {
                    "alpha": float(posterior["adstock_alpha"][:, :, i].mean()),
                    "l_max": self._adstock_l_max,
                }
            if "saturation_lam" in posterior:
                saturation_params[ch] = {
                    "lam": float(posterior["saturation_lam"][:, :, i].mean()),
                }

        intercept = float(posterior["intercept"].mean()) if "intercept" in posterior else 0.0

        return {
            "coefficients": coefficients,
            "intercept": intercept,
            "adstock_params": adstock_params,
            "saturation_params": saturation_params,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, directory: Path) -> None:
        self._check_fitted()
        with open(directory / "pymc_model.pkl", "wb") as f:
            pickle.dump({
                "model": self._model,
                "trace": self._trace,
                "media_cols": self._media_cols,
                "control_cols": self._control_cols,
                "target_col": self._target_col,
                "date_col": self._date_col,
            }, f)
        params = self.get_parameters()
        with open(directory / "parameters.json", "w") as f:
            json.dump(params, f, indent=2)
        logger.info(f"Saved PyMC model state to {directory}")

    def load_state(self, directory: Path) -> None:
        pkl = directory / "pymc_model.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"PyMC model state not found: {pkl}")
        with open(pkl, "rb") as f:
            state = pickle.load(f)
        self._model = state["model"]
        self._trace = state["trace"]
        self._media_cols = state["media_cols"]
        self._control_cols = state["control_cols"]
        self._target_col = state["target_col"]
        self._date_col = state["date_col"]
        logger.info(f"Loaded PyMC model state from {directory}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._model is None:
            from core.exceptions import ModelNotFittedError
            raise ModelNotFittedError("PyMCAdapter")


# ---------------------------------------------------------------------------
# Self-register
# ---------------------------------------------------------------------------

register_backend("pymc", PyMCAdapter)
