"""
Google Meridian adapter for Unified-M.

Wraps Google's Meridian MMM behind the canonical ``BaseMMM`` interface.
Meridian is Google's open-source Bayesian MMM built on JAX/NumPyro.

Install:
    pip install google-meridian

This adapter is loaded only when ``google-meridian`` is installed;
otherwise it silently stays unregistered.
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

try:
    from meridian import Meridian
    from meridian import spec as meridian_spec

    _MERIDIAN_AVAILABLE = True
except ImportError:
    _MERIDIAN_AVAILABLE = False


class MeridianAdapter(BaseMMM):
    """
    Bayesian MMM powered by Google Meridian (JAX/NumPyro backend).

    Provides:
      - Geo-level hierarchical modeling
      - Bayesian inference via NumPyro (NUTS)
      - Prior calibration from experiments
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_chains: int = 4,
        n_warmup: int = 500,
        **kwargs: Any,
    ):
        if not _MERIDIAN_AVAILABLE:
            raise ImportError(
                "google-meridian is not installed. "
                "Run: pip install google-meridian"
            )

        self._n_samples = n_samples
        self._n_chains = n_chains
        self._n_warmup = n_warmup

        self._model: Any = None
        self._data: pd.DataFrame | None = None
        self._media_cols: list[str] = []
        self._control_cols: list[str] = []
        self._target_col: str = "y"
        self._date_col: str = "date"
        self._fitted_params: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "meridian"

    @property
    def description(self) -> str:
        return "Google Meridian: Bayesian geo-level MMM (JAX/NumPyro)"

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "y",
        media_cols: list[str] | None = None,
        control_cols: list[str] | None = None,
        date_col: str = "date",
        **kwargs: Any,
    ) -> dict[str, Any]:
        logger.info("Fitting Meridian MMM (JAX/NumPyro)...")

        self._data = df.copy()
        self._target_col = target_col
        self._date_col = date_col
        self._media_cols = media_cols or [c for c in df.columns if c.endswith("_spend")]
        self._control_cols = control_cols or []

        # Build Meridian input data structure
        n_times = len(df)
        n_media = len(self._media_cols)

        media_data = df[self._media_cols].values.reshape(1, n_times, n_media)
        target_data = df[target_col].values.reshape(1, n_times)

        control_data = None
        if self._control_cols:
            n_ctrl = len(self._control_cols)
            control_data = df[self._control_cols].values.reshape(1, n_times, n_ctrl)

        input_data = meridian_spec.InputData(
            media=media_data,
            media_names=self._media_cols,
            target=target_data,
            controls=control_data,
            control_names=self._control_cols if self._control_cols else None,
        )

        self._model = Meridian(input_data=input_data)
        self._model.fit(
            n_samples=self._n_samples,
            n_chains=self._n_chains,
            n_warmup=self._n_warmup,
        )

        # Extract metrics
        y_pred = self._model.predict().mean(axis=0).flatten()[:n_times]
        y_true = df[target_col].values
        metrics = self.get_metrics(y_true, y_pred)

        logger.info(
            f"Meridian fit complete. R2={metrics['r_squared']:.3f} "
            f"MAPE={metrics['mape']:.2f}%"
        )

        return {"metrics": metrics}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        # For Meridian, prediction uses the fitted model's posterior
        pred = self._model.predict()
        return pred.mean(axis=0).flatten()[:len(df)]

    def get_channel_contributions(self) -> dict[str, np.ndarray]:
        self._check_fitted()
        contributions: dict[str, np.ndarray] = {}
        media_contrib = self._model.get_media_contribution()
        mean_contrib = media_contrib.mean(axis=0)  # average over posterior

        for i, ch in enumerate(self._media_cols):
            if mean_contrib.ndim == 3:
                contributions[ch] = mean_contrib[0, :, i]
            else:
                contributions[ch] = mean_contrib[:, i]

        return contributions

    def get_response_curves(
        self,
        spend_grid: np.ndarray | None = None,
        n_points: int = 100,
    ) -> dict[str, pd.DataFrame]:
        self._check_fitted()
        curves: dict[str, pd.DataFrame] = {}

        for i, ch in enumerate(self._media_cols):
            max_spend = self._data[ch].max() * 1.5
            grid = spend_grid if spend_grid is not None else np.linspace(0, max_spend, n_points)

            # Use model's response function if available
            response = self._model.get_response_curve(channel_idx=i, spend_values=grid)
            marginal = np.gradient(response, grid)

            curves[ch] = pd.DataFrame({
                "spend": grid,
                "response": response,
                "marginal_response": marginal,
            })

        return curves

    def get_parameters(self) -> dict[str, Any]:
        self._check_fitted()
        posterior = self._model.get_posterior_samples()

        coefficients = {}
        for i, ch in enumerate(self._media_cols):
            if "beta_media" in posterior:
                coefficients[ch] = float(np.mean(posterior["beta_media"][:, i]))

        return {
            "coefficients": coefficients,
            "intercept": float(np.mean(posterior.get("intercept", [0]))),
            "adstock_params": {},
            "saturation_params": {},
        }

    def save_state(self, directory: Path) -> None:
        self._check_fitted()
        with open(directory / "meridian_model.pkl", "wb") as f:
            pickle.dump({
                "model": self._model,
                "media_cols": self._media_cols,
                "control_cols": self._control_cols,
                "target_col": self._target_col,
                "date_col": self._date_col,
            }, f)
        params = self.get_parameters()
        with open(directory / "parameters.json", "w") as f:
            json.dump(params, f, indent=2)
        logger.info(f"Saved Meridian model state to {directory}")

    def load_state(self, directory: Path) -> None:
        pkl = directory / "meridian_model.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"Meridian model not found: {pkl}")
        with open(pkl, "rb") as f:
            state = pickle.load(f)
        self._model = state["model"]
        self._media_cols = state["media_cols"]
        self._control_cols = state["control_cols"]
        self._target_col = state["target_col"]
        self._date_col = state["date_col"]

    def _check_fitted(self) -> None:
        if self._model is None:
            from core.exceptions import ModelNotFittedError
            raise ModelNotFittedError("MeridianAdapter")


# Self-register only if Meridian is available
if _MERIDIAN_AVAILABLE:
    from models.registry import register_backend
    register_backend("meridian", MeridianAdapter)
