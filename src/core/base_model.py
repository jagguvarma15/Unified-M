"""
Abstract base class for pluggable MMM engines.

Any model backend -- built-in ridge, PyMC-Marketing, Google Meridian,
Meta Robyn, or a custom implementation -- must subclass ``BaseMMM`` and
implement the methods below.  The pipeline, API, and UI only talk to
this interface, making the model layer completely swappable.

Design goals:
  - Accept a pandas DataFrame (not a framework-specific object).
  - Return plain numpy arrays and Python dicts (no framework lock-in).
  - Serialise / deserialise model state to a directory so any run is
    reproducible without the original in-memory model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseMMM(ABC):
    """
    Abstract interface that every MMM backend must implement.

    Subclasses must implement:
      - fit()
      - predict()
      - get_channel_contributions()
      - get_response_curves()
      - get_parameters()
      - save_state() / load_state()

    Optional overrides:
      - name() -- human-readable name shown in logs / UI
      - description() -- longer text
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier, e.g. 'builtin', 'pymc', 'meridian'."""
        return self.__class__.__name__

    @property
    def description(self) -> str:
        """One-line description for the UI / logs."""
        return "Base MMM model"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        media_cols: list[str],
        control_cols: list[str] | None = None,
        date_col: str = "date",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Fit the model to data.

        Args:
            df:           Training DataFrame (already transformed if needed).
            target_col:   Name of the target (y) column.
            media_cols:   Names of media spend columns.
            control_cols: Names of control-variable columns (optional).
            date_col:     Name of date column.
            **kwargs:     Backend-specific options (samples, chains, ...).

        Returns:
            Dictionary of fit metadata (metrics, diagnostics, timing, ...).
        """
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate point predictions for new (or training) data.

        Returns:
            1-D array of predicted target values, same length as *df*.
        """
        ...

    @abstractmethod
    def get_channel_contributions(self) -> dict[str, np.ndarray]:
        """
        Decompose fitted values into per-channel contribution arrays.

        Returns:
            Mapping of channel_name -> 1-D array of daily contributions.
        """
        ...

    @abstractmethod
    def get_response_curves(
        self,
        spend_grid: np.ndarray | None = None,
        n_points: int = 100,
    ) -> dict[str, pd.DataFrame]:
        """
        Return the response (saturation) curve for each channel.

        Each DataFrame must have at least columns ``spend`` and
        ``response``.  Optionally include ``marginal_response``.

        Args:
            spend_grid: Custom spend values; auto-generated when None.
            n_points:   Number of points on each curve.

        Returns:
            Mapping of channel_name -> DataFrame(spend, response, ...).
        """
        ...

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """
        Return fitted parameters as a JSON-serialisable dictionary.

        Must include at least:
          - ``coefficients``   : dict[channel, float]
          - ``adstock_params`` : dict[channel, dict]
          - ``saturation_params`` : dict[channel, dict]
          - ``intercept``      : float
        """
        ...

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @abstractmethod
    def save_state(self, directory: Path) -> None:
        """
        Persist everything needed to reload the model later.

        Implementations should write to *directory* (already exists).
        """
        ...

    @abstractmethod
    def load_state(self, directory: Path) -> None:
        """
        Restore model state from a previous ``save_state()`` call.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience helpers (can be overridden)
    # ------------------------------------------------------------------

    def get_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Compute standard regression metrics."""
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-12))
        nrmse = float(rmse / (y_true.mean() + 1e-12))
        return {
            "mape": round(mape, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r_squared": round(r2, 4),
            "nrmse": round(nrmse, 4),
        }
