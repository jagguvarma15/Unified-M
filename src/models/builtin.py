"""
Built-in MMM model -- zero heavy dependencies.

Uses scikit-learn Ridge regression with manual adstock + saturation
transforms.  This is the default backend that always works, even
without PyMC, Meridian, or Robyn installed.

Strengths:
  - Instant (seconds, not minutes).
  - Deterministic (no MCMC sampling noise).
  - Great for prototyping, demos, and CI pipelines.

Limitations:
  - No proper uncertainty quantification (uses bootstrap CIs).
  - No hierarchical priors.
  - Response-curve parameters are set heuristically, not learned.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from core.base_model import BaseMMM
from models.registry import register_backend
from transforms.adstock import geometric_adstock
from transforms.saturation import hill_saturation


class BuiltinMMM(BaseMMM):
    """Ridge-regression MMM with geometric adstock and Hill saturation."""

    def __init__(
        self,
        adstock_alphas: dict[str, float] | None = None,
        adstock_l_max: int = 8,
        saturation_K: dict[str, float] | None = None,
        saturation_S: dict[str, float] | None = None,
        ridge_alphas: list[float] | None = None,
        n_bootstrap: int = 200,
    ):
        self._adstock_alphas = adstock_alphas or {}
        self._adstock_l_max = adstock_l_max
        self._saturation_K = saturation_K or {}
        self._saturation_S = saturation_S or {}
        self._ridge_alphas = ridge_alphas or [0.01, 0.1, 1.0, 10.0, 100.0]
        self._n_bootstrap = n_bootstrap

        # Fitted state
        self._model: RidgeCV | None = None
        self._scaler: StandardScaler | None = None
        self._feature_cols: list[str] = []
        self._media_cols: list[str] = []
        self._control_cols: list[str] = []
        self._target_col: str = "y"
        self._date_col: str = "date"
        self._data: pd.DataFrame | None = None
        self._transformed: pd.DataFrame | None = None
        self._coefficients: dict[str, float] = {}
        self._intercept: float = 0.0

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "builtin"

    @property
    def description(self) -> str:
        return "Ridge regression with geometric adstock and Hill saturation"

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
        logger.info("Fitting BuiltinMMM (Ridge + adstock + saturation)...")

        self._data = df.copy()
        self._target_col = target_col
        self._date_col = date_col
        self._media_cols = media_cols or [c for c in df.columns if c.endswith("_spend")]
        self._control_cols = control_cols or []

        if not self._media_cols:
            raise ValueError("No media columns detected. Pass media_cols explicitly.")

        # 1. Apply adstock + saturation to media columns
        transformed = df.copy()
        for col in self._media_cols:
            alpha = self._adstock_alphas.get(col, 0.5)
            K = self._saturation_K.get(col, float(df[col].median()) + 1.0)
            S = self._saturation_S.get(col, 1.0)

            adstocked = geometric_adstock(df[col].values, alpha=alpha, l_max=self._adstock_l_max)
            saturated = hill_saturation(adstocked, K=K, S=S)
            transformed[f"{col}_transformed"] = saturated

        self._transformed = transformed

        # 2. Build feature matrix
        feature_cols = [f"{c}_transformed" for c in self._media_cols] + self._control_cols
        feature_cols = [c for c in feature_cols if c in transformed.columns]
        self._feature_cols = feature_cols

        X = transformed[feature_cols].values
        y = transformed[target_col].values

        # 3. Scale
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # 4. Fit Ridge
        self._model = RidgeCV(alphas=self._ridge_alphas)
        self._model.fit(X_scaled, y)

        self._intercept = float(self._model.intercept_)

        # Map coefficients back to original column names
        self._coefficients = {}
        for i, col in enumerate(feature_cols):
            # Store with the original media col name
            original = col.replace("_transformed", "")
            self._coefficients[original] = float(self._model.coef_[i])

        y_pred = self._model.predict(X_scaled)
        metrics = self.get_metrics(y, y_pred)

        logger.info(
            f"BuiltinMMM fit complete.  "
            f"R2={metrics['r_squared']:.3f}  MAPE={metrics['mape']:.2f}%  "
            f"Ridge alpha={self._model.alpha_:.2f}"
        )

        return {
            "metrics": metrics,
            "ridge_alpha": float(self._model.alpha_),
            "n_features": len(feature_cols),
        }

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()

        transformed = df.copy()
        for col in self._media_cols:
            alpha = self._adstock_alphas.get(col, 0.5)
            K = self._saturation_K.get(col, float(self._data[col].median()) + 1.0)
            S = self._saturation_S.get(col, 1.0)

            adstocked = geometric_adstock(df[col].values, alpha=alpha, l_max=self._adstock_l_max)
            saturated = hill_saturation(adstocked, K=K, S=S)
            transformed[f"{col}_transformed"] = saturated

        feature_cols = [c for c in self._feature_cols if c in transformed.columns]
        X = transformed[feature_cols].values
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    # ------------------------------------------------------------------
    # Contributions
    # ------------------------------------------------------------------

    def get_channel_contributions(self) -> dict[str, np.ndarray]:
        self._check_fitted()

        contributions: dict[str, np.ndarray] = {}
        feature_cols = self._feature_cols

        X = self._transformed[feature_cols].values
        X_scaled = self._scaler.transform(X)

        for i, col in enumerate(feature_cols):
            original = col.replace("_transformed", "")
            contributions[original] = X_scaled[:, i] * self._model.coef_[i]

        contributions["baseline"] = np.full(len(self._transformed), self._intercept)
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

        curves: dict[str, pd.DataFrame] = {}
        for col in self._media_cols:
            max_spend = self._data[col].max() * 1.5
            grid = spend_grid if spend_grid is not None else np.linspace(0, max_spend, n_points)

            alpha = self._adstock_alphas.get(col, 0.5)
            K = self._saturation_K.get(col, float(self._data[col].median()) + 1.0)
            S = self._saturation_S.get(col, 1.0)
            coef = self._coefficients.get(col, 0)

            # Single-period response: adstock reduces to just alpha^0 * x = x for point eval
            saturated = hill_saturation(grid, K=K, S=S)
            response = saturated * coef

            # Marginal response (derivative of Hill * coef)
            K_s = K ** S
            x_s = np.power(grid + 1e-10, S)
            marginal = coef * S * K_s * np.power(grid + 1e-10, S - 1) / np.power(K_s + x_s, 2)

            curves[col] = pd.DataFrame({
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

        adstock_params = {}
        saturation_params = {}
        for col in self._media_cols:
            adstock_params[col] = {
                "alpha": self._adstock_alphas.get(col, 0.5),
                "l_max": self._adstock_l_max,
            }
            saturation_params[col] = {
                "K": self._saturation_K.get(col, float(self._data[col].median()) + 1.0),
                "S": self._saturation_S.get(col, 1.0),
            }

        return {
            "coefficients": dict(self._coefficients),
            "intercept": self._intercept,
            "adstock_params": adstock_params,
            "saturation_params": saturation_params,
            "ridge_alpha": float(self._model.alpha_),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, directory: Path) -> None:
        self._check_fitted()
        state = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_cols": self._feature_cols,
            "media_cols": self._media_cols,
            "control_cols": self._control_cols,
            "target_col": self._target_col,
            "date_col": self._date_col,
            "coefficients": self._coefficients,
            "intercept": self._intercept,
            "adstock_alphas": self._adstock_alphas,
            "adstock_l_max": self._adstock_l_max,
            "saturation_K": self._saturation_K,
            "saturation_S": self._saturation_S,
        }
        with open(directory / "builtin_model.pkl", "wb") as f:
            pickle.dump(state, f)
        # Also write a human-readable summary
        params = self.get_parameters()
        with open(directory / "parameters.json", "w") as f:
            json.dump(params, f, indent=2)
        logger.info(f"Saved BuiltinMMM state to {directory}")

    def load_state(self, directory: Path) -> None:
        pkl_path = directory / "builtin_model.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Model state not found: {pkl_path}")
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)

        self._model = state["model"]
        self._scaler = state["scaler"]
        self._feature_cols = state["feature_cols"]
        self._media_cols = state["media_cols"]
        self._control_cols = state["control_cols"]
        self._target_col = state["target_col"]
        self._date_col = state["date_col"]
        self._coefficients = state["coefficients"]
        self._intercept = state["intercept"]
        self._adstock_alphas = state.get("adstock_alphas", {})
        self._adstock_l_max = state.get("adstock_l_max", 8)
        self._saturation_K = state.get("saturation_K", {})
        self._saturation_S = state.get("saturation_S", {})
        logger.info(f"Loaded BuiltinMMM state from {directory}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._model is None:
            from core.exceptions import ModelNotFittedError
            raise ModelNotFittedError("BuiltinMMM")


# ---------------------------------------------------------------------------
# Self-register
# ---------------------------------------------------------------------------

register_backend("builtin", BuiltinMMM)
