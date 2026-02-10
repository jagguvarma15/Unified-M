"""
NumPyro adapter for Unified-M.

A lightweight Bayesian MMM using NumPyro (JAX backend) directly,
without the PyMC-Marketing or Meridian wrapper. This gives maximum
flexibility in model specification while staying fast (JAX JIT).

Install:
    pip install numpyro jax jaxlib

This adapter is loaded only when NumPyro is installed.
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
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    _NUMPYRO_AVAILABLE = True
except ImportError:
    _NUMPYRO_AVAILABLE = False


def _mmm_model(
    media: Any,
    controls: Any,
    y: Any = None,
    n_media: int = 0,
    n_controls: int = 0,
) -> None:
    """NumPyro generative model for MMM."""
    # Priors
    intercept = numpyro.sample("intercept", dist.Normal(0, 10))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5))

    # Media coefficients (positive)
    beta_media = numpyro.sample(
        "beta_media",
        dist.HalfNormal(jnp.ones(n_media) * 2),
    )

    # Adstock decay
    adstock_alpha = numpyro.sample(
        "adstock_alpha",
        dist.Beta(jnp.ones(n_media) * 2, jnp.ones(n_media) * 2),
    )

    # Saturation (Hill) parameters
    saturation_lam = numpyro.sample(
        "saturation_lam",
        dist.HalfNormal(jnp.ones(n_media) * 1),
    )

    # Control coefficients
    if n_controls > 0:
        beta_controls = numpyro.sample(
            "beta_controls",
            dist.Normal(jnp.zeros(n_controls), 2),
        )
    else:
        beta_controls = jnp.zeros(0)

    # Transform media: simple saturation (1 - exp(-lam * x))
    media_saturated = 1.0 - jnp.exp(-saturation_lam * media)
    media_effect = jnp.sum(beta_media * media_saturated, axis=-1)

    # Control effect
    ctrl_effect = jnp.sum(beta_controls * controls, axis=-1) if n_controls > 0 else 0.0

    mu = intercept + media_effect + ctrl_effect

    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


class NumPyroAdapter(BaseMMM):
    """
    Lightweight Bayesian MMM using NumPyro/JAX directly.

    Faster than PyMC for medium-sized datasets thanks to JAX JIT.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_chains: int = 4,
        n_warmup: int = 500,
        **kwargs: Any,
    ):
        if not _NUMPYRO_AVAILABLE:
            raise ImportError(
                "numpyro is not installed. "
                "Run: pip install numpyro jax jaxlib"
            )

        self._n_samples = n_samples
        self._n_chains = n_chains
        self._n_warmup = n_warmup

        self._mcmc: Any = None
        self._samples: dict[str, Any] = {}
        self._data: pd.DataFrame | None = None
        self._media_cols: list[str] = []
        self._control_cols: list[str] = []
        self._target_col: str = "y"
        self._date_col: str = "date"

    @property
    def name(self) -> str:
        return "numpyro"

    @property
    def description(self) -> str:
        return "NumPyro/JAX Bayesian MMM (NUTS sampler)"

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "y",
        media_cols: list[str] | None = None,
        control_cols: list[str] | None = None,
        date_col: str = "date",
        **kwargs: Any,
    ) -> dict[str, Any]:
        logger.info("Fitting NumPyro MMM (JAX/NUTS)...")

        self._data = df.copy()
        self._target_col = target_col
        self._date_col = date_col
        self._media_cols = media_cols or [c for c in df.columns if c.endswith("_spend")]
        self._control_cols = control_cols or []

        # Prepare data as JAX arrays
        media = jnp.array(df[self._media_cols].values, dtype=jnp.float32)
        y = jnp.array(df[target_col].values, dtype=jnp.float32)

        if self._control_cols:
            controls = jnp.array(df[self._control_cols].values, dtype=jnp.float32)
        else:
            controls = jnp.zeros((len(df), 0), dtype=jnp.float32)

        # Normalize
        media_scale = jnp.maximum(media.max(axis=0), 1e-8)
        media_norm = media / media_scale
        y_scale = jnp.maximum(y.max(), 1e-8)
        y_norm = y / y_scale

        # Run MCMC
        kernel = NUTS(_mmm_model)
        self._mcmc = MCMC(
            kernel,
            num_warmup=self._n_warmup,
            num_samples=self._n_samples,
            num_chains=self._n_chains,
        )
        rng_key = jax.random.PRNGKey(42)
        self._mcmc.run(
            rng_key,
            media=media_norm,
            controls=controls,
            y=y_norm,
            n_media=len(self._media_cols),
            n_controls=len(self._control_cols),
        )

        self._samples = self._mcmc.get_samples()

        # Compute predictions for metrics
        y_pred = self._predict_internal(media_norm, controls) * float(y_scale)
        metrics = self.get_metrics(np.array(y), np.array(y_pred))

        logger.info(
            f"NumPyro fit complete. R2={metrics['r_squared']:.3f} "
            f"MAPE={metrics['mape']:.2f}%"
        )

        return {"metrics": metrics}

    def _predict_internal(self, media: Any, controls: Any) -> np.ndarray:
        """Predict using posterior mean parameters."""
        intercept = float(jnp.mean(self._samples["intercept"]))
        beta_media = jnp.mean(self._samples["beta_media"], axis=0)
        sat_lam = jnp.mean(self._samples["saturation_lam"], axis=0)

        media_sat = 1.0 - jnp.exp(-sat_lam * media)
        media_effect = jnp.sum(beta_media * media_sat, axis=-1)

        ctrl_effect = 0.0
        if "beta_controls" in self._samples and controls.shape[-1] > 0:
            beta_ctrl = jnp.mean(self._samples["beta_controls"], axis=0)
            ctrl_effect = jnp.sum(beta_ctrl * controls, axis=-1)

        return np.array(intercept + media_effect + ctrl_effect)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        media = jnp.array(df[self._media_cols].values, dtype=jnp.float32)
        media_scale = jnp.maximum(jnp.array(self._data[self._media_cols].values).max(axis=0), 1e-8)
        media_norm = media / media_scale

        controls = jnp.zeros((len(df), 0), dtype=jnp.float32)
        if self._control_cols:
            controls = jnp.array(df[self._control_cols].values, dtype=jnp.float32)

        y_scale = float(jnp.array(self._data[self._target_col].values).max())
        return self._predict_internal(media_norm, controls) * y_scale

    def get_channel_contributions(self) -> dict[str, np.ndarray]:
        self._check_fitted()
        media = jnp.array(self._data[self._media_cols].values, dtype=jnp.float32)
        media_scale = jnp.maximum(media.max(axis=0), 1e-8)
        media_norm = media / media_scale
        y_scale = float(self._data[self._target_col].max())

        beta = jnp.mean(self._samples["beta_media"], axis=0)
        lam = jnp.mean(self._samples["saturation_lam"], axis=0)

        contributions: dict[str, np.ndarray] = {}
        for i, ch in enumerate(self._media_cols):
            sat = 1.0 - jnp.exp(-lam[i] * media_norm[:, i])
            contributions[ch] = np.array(beta[i] * sat * y_scale)

        intercept = float(jnp.mean(self._samples["intercept"])) * y_scale
        contributions["baseline"] = np.full(len(self._data), intercept)
        return contributions

    def get_response_curves(
        self,
        spend_grid: np.ndarray | None = None,
        n_points: int = 100,
    ) -> dict[str, pd.DataFrame]:
        self._check_fitted()
        curves: dict[str, pd.DataFrame] = {}
        beta = np.array(jnp.mean(self._samples["beta_media"], axis=0))
        lam = np.array(jnp.mean(self._samples["saturation_lam"], axis=0))

        for i, ch in enumerate(self._media_cols):
            max_spend = float(self._data[ch].max()) * 1.5
            grid = spend_grid if spend_grid is not None else np.linspace(0, max_spend, n_points)
            media_scale = float(self._data[ch].max()) or 1.0
            grid_norm = grid / media_scale

            response = beta[i] * (1.0 - np.exp(-lam[i] * grid_norm))
            marginal = beta[i] * lam[i] / media_scale * np.exp(-lam[i] * grid_norm)

            curves[ch] = pd.DataFrame({
                "spend": grid,
                "response": response,
                "marginal_response": marginal,
            })

        return curves

    def get_parameters(self) -> dict[str, Any]:
        self._check_fitted()
        coefficients = {}
        adstock_params = {}
        saturation_params = {}

        beta = np.array(jnp.mean(self._samples["beta_media"], axis=0))
        lam = np.array(jnp.mean(self._samples["saturation_lam"], axis=0))

        for i, ch in enumerate(self._media_cols):
            coefficients[ch] = float(beta[i])
            saturation_params[ch] = {"lam": float(lam[i])}

            if "adstock_alpha" in self._samples:
                alpha = float(jnp.mean(self._samples["adstock_alpha"][:, i]))
                adstock_params[ch] = {"alpha": alpha}

        return {
            "coefficients": coefficients,
            "intercept": float(jnp.mean(self._samples["intercept"])),
            "adstock_params": adstock_params,
            "saturation_params": saturation_params,
        }

    def save_state(self, directory: Path) -> None:
        self._check_fitted()
        # Convert JAX arrays to numpy for pickling
        samples_np = {k: np.array(v) for k, v in self._samples.items()}
        with open(directory / "numpyro_model.pkl", "wb") as f:
            pickle.dump({
                "samples": samples_np,
                "media_cols": self._media_cols,
                "control_cols": self._control_cols,
                "target_col": self._target_col,
                "date_col": self._date_col,
            }, f)
        params = self.get_parameters()
        with open(directory / "parameters.json", "w") as f:
            json.dump(params, f, indent=2)

    def load_state(self, directory: Path) -> None:
        pkl = directory / "numpyro_model.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"NumPyro model not found: {pkl}")
        with open(pkl, "rb") as f:
            state = pickle.load(f)
        self._samples = {k: jnp.array(v) for k, v in state["samples"].items()}
        self._media_cols = state["media_cols"]
        self._control_cols = state["control_cols"]
        self._target_col = state["target_col"]
        self._date_col = state["date_col"]

    def _check_fitted(self) -> None:
        if not self._samples:
            from core.exceptions import ModelNotFittedError
            raise ModelNotFittedError("NumPyroAdapter")


# Self-register only if NumPyro is available
if _NUMPYRO_AVAILABLE:
    from models.registry import register_backend
    register_backend("numpyro", NumPyroAdapter)
