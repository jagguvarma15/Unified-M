"""
Core MMM model implementation wrapping PyMC-Marketing.

Provides a high-level interface for training, predicting, and analyzing
Marketing Mix Models with Bayesian inference.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import pickle

import arviz as az
import numpy as np
import pandas as pd
from loguru import logger

# PyMC-Marketing imports (with fallback for when not installed)
try:
    from pymc_marketing.mmm import (
        DelayedSaturatedMMM,
        GeometricAdstock,
        LogisticSaturation,
    )
    PYMC_MARKETING_AVAILABLE = True
except ImportError:
    PYMC_MARKETING_AVAILABLE = False
    logger.warning("pymc-marketing not installed. Using baseline model.")


@dataclass
class MMMResults:
    """
    Container for MMM training results.
    
    Stores model outputs, posterior samples, and derived metrics
    for downstream use (reconciliation, optimization, API).
    """
    
    # Model fit
    trace: Any = None  # ArviZ InferenceData
    model: Any = None  # PyMC model object
    
    # Posterior summaries
    channel_contributions: pd.DataFrame | None = None
    baseline_contribution: float = 0.0
    
    # Response curves (for optimization)
    response_curves: dict[str, dict] = field(default_factory=dict)
    
    # Metrics
    metrics: dict[str, float] = field(default_factory=dict)
    
    # Parameters (posterior means)
    adstock_params: dict[str, float] = field(default_factory=dict)
    saturation_params: dict[str, dict] = field(default_factory=dict)
    coefficients: dict[str, float] = field(default_factory=dict)
    
    # Metadata
    training_date_range: tuple[str, str] | None = None
    channels: list[str] = field(default_factory=list)
    target_variable: str = "y"
    
    def to_dict(self) -> dict:
        """Convert results to dictionary for serialization."""
        return {
            "channel_contributions": (
                self.channel_contributions.to_dict() 
                if self.channel_contributions is not None else None
            ),
            "baseline_contribution": self.baseline_contribution,
            "response_curves": self.response_curves,
            "metrics": self.metrics,
            "adstock_params": self.adstock_params,
            "saturation_params": self.saturation_params,
            "coefficients": self.coefficients,
            "training_date_range": self.training_date_range,
            "channels": self.channels,
            "target_variable": self.target_variable,
        }
    
    def save(self, path: Path | str) -> None:
        """Save results to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save summary as JSON
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        # Save full results as pickle (includes trace)
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self, f)
        
        logger.info(f"Saved MMM results to {path}")
    
    @classmethod
    def load(cls, path: Path | str) -> "MMMResults":
        """Load results from disk."""
        path = Path(path)
        
        with open(path.with_suffix(".pkl"), "rb") as f:
            return pickle.load(f)


class UnifiedMMM:
    """
    High-level MMM interface for Unified-M.
    
    Wraps PyMC-Marketing's DelayedSaturatedMMM with:
    - Simplified configuration
    - Automatic prior setting
    - Result extraction utilities
    - Fallback baseline model
    
    Example:
        >>> mmm = UnifiedMMM(
        ...     date_col="date",
        ...     target_col="y",
        ...     media_cols=["google_spend", "meta_spend", "tv_spend"],
        ...     control_cols=["seasonality", "promo"],
        ... )
        >>> results = mmm.fit(df)
        >>> contributions = mmm.get_contributions()
    """
    
    def __init__(
        self,
        date_col: str = "date",
        target_col: str = "y",
        media_cols: list[str] | None = None,
        control_cols: list[str] | None = None,
        adstock_max_lag: int = 8,
        yearly_seasonality: int = 2,
        **kwargs,
    ):
        """
        Initialize MMM configuration.
        
        Args:
            date_col: Name of date column
            target_col: Name of target (y) column
            media_cols: List of media spend columns (auto-detected if None)
            control_cols: List of control variable columns
            adstock_max_lag: Maximum lag for adstock transformation
            yearly_seasonality: Fourier order for yearly seasonality
            **kwargs: Additional arguments passed to PyMC-Marketing model
        """
        self.date_col = date_col
        self.target_col = target_col
        self.media_cols = media_cols or []
        self.control_cols = control_cols or []
        self.adstock_max_lag = adstock_max_lag
        self.yearly_seasonality = yearly_seasonality
        self.extra_kwargs = kwargs
        
        self._model = None
        self._trace = None
        self._data: pd.DataFrame | None = None
        self._results: MMMResults | None = None
    
    def fit(
        self,
        df: pd.DataFrame,
        n_samples: int = 1000,
        n_chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> MMMResults:
        """
        Fit the MMM model.
        
        Args:
            df: Training data with date, target, media, and control columns
            n_samples: Number of posterior samples per chain
            n_chains: Number of MCMC chains
            target_accept: Target acceptance probability
            random_seed: Random seed for reproducibility
        
        Returns:
            MMMResults with posterior summaries and metrics
        """
        logger.info("Fitting MMM model...")
        
        self._data = df.copy()
        
        # Auto-detect media columns if not specified
        if not self.media_cols:
            self.media_cols = [c for c in df.columns if c.endswith("_spend")]
            logger.info(f"Auto-detected media columns: {self.media_cols}")
        
        if not PYMC_MARKETING_AVAILABLE:
            logger.warning("Using baseline model (pymc-marketing not available)")
            return self._fit_baseline(df)
        
        return self._fit_pymc(df, n_samples, n_chains, target_accept, random_seed)
    
    def _fit_pymc(
        self,
        df: pd.DataFrame,
        n_samples: int,
        n_chains: int,
        target_accept: float,
        random_seed: int,
    ) -> MMMResults:
        """Fit using PyMC-Marketing."""
        
        # Prepare data in PyMC-Marketing format
        X = df[self.media_cols].values
        y = df[self.target_col].values
        
        # Build model
        self._model = DelayedSaturatedMMM(
            date_column=self.date_col,
            channel_columns=self.media_cols,
            control_columns=self.control_cols if self.control_cols else None,
            adstock=GeometricAdstock(l_max=self.adstock_max_lag),
            saturation=LogisticSaturation(),
            yearly_seasonality=self.yearly_seasonality,
            **self.extra_kwargs,
        )
        
        # Fit
        self._model.fit(
            X=df,
            y=y,
            target_col=self.target_col,
            draws=n_samples,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=random_seed,
        )
        
        self._trace = self._model.fit_result
        
        # Extract results
        self._results = self._extract_results(df)
        
        return self._results
    
    def _fit_baseline(self, df: pd.DataFrame) -> MMMResults:
        """
        Fit baseline model when PyMC-Marketing is not available.
        
        Uses simple ridge regression as a fallback.
        """
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        
        # Prepare features
        feature_cols = self.media_cols + self.control_cols
        X = df[feature_cols].values
        y = df[self.target_col].values
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit ridge
        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        model.fit(X_scaled, y)
        
        # Predictions
        y_pred = model.predict(X_scaled)
        
        # Create results
        coefficients = dict(zip(feature_cols, model.coef_ * scaler.scale_))
        
        # Compute contributions
        contributions = pd.DataFrame({
            self.date_col: df[self.date_col],
        })
        
        for i, col in enumerate(feature_cols):
            contributions[col] = X_scaled[:, i] * model.coef_[i]
        
        contributions["baseline"] = model.intercept_
        contributions["predicted"] = y_pred
        contributions["actual"] = y
        
        # Metrics
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
        
        self._results = MMMResults(
            channel_contributions=contributions,
            baseline_contribution=model.intercept_,
            metrics={"mape": mape, "rmse": rmse, "r2": r2},
            coefficients=coefficients,
            training_date_range=(
                str(df[self.date_col].min()),
                str(df[self.date_col].max()),
            ),
            channels=self.media_cols,
            target_variable=self.target_col,
        )
        
        logger.info(f"Baseline model fit complete. MAPE: {mape:.2f}%, R²: {r2:.3f}")
        
        return self._results
    
    def _extract_results(self, df: pd.DataFrame) -> MMMResults:
        """Extract results from fitted PyMC-Marketing model."""
        
        # Get posterior means for parameters
        posterior = self._trace.posterior
        
        # Adstock parameters
        adstock_params = {}
        if "adstock_alpha" in posterior:
            for i, channel in enumerate(self.media_cols):
                adstock_params[channel] = float(
                    posterior["adstock_alpha"][:, :, i].mean()
                )
        
        # Saturation parameters
        saturation_params = {}
        if "saturation_lam" in posterior:
            for i, channel in enumerate(self.media_cols):
                saturation_params[channel] = {
                    "lam": float(posterior["saturation_lam"][:, :, i].mean())
                }
        
        # Coefficients
        coefficients = {}
        if "beta_channel" in posterior:
            for i, channel in enumerate(self.media_cols):
                coefficients[channel] = float(
                    posterior["beta_channel"][:, :, i].mean()
                )
        
        # Get contributions
        contributions = self._model.compute_channel_contribution_original_scale()
        
        # Metrics
        y_pred = self._model.posterior_predictive.mean(dim=["chain", "draw"])["y"].values
        y_actual = df[self.target_col].values
        
        mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-8))) * 100
        rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
        r2 = 1 - np.sum((y_actual - y_pred) ** 2) / np.sum((y_actual - y_actual.mean()) ** 2)
        
        return MMMResults(
            trace=self._trace,
            model=self._model,
            channel_contributions=contributions,
            baseline_contribution=float(posterior.get("intercept", [0]).mean()),
            metrics={"mape": mape, "rmse": rmse, "r2": r2},
            adstock_params=adstock_params,
            saturation_params=saturation_params,
            coefficients=coefficients,
            training_date_range=(
                str(df[self.date_col].min()),
                str(df[self.date_col].max()),
            ),
            channels=self.media_cols,
            target_variable=self.target_col,
        )
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for new data.
        
        Args:
            df: Data with same columns as training data
        
        Returns:
            Array of predictions
        """
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if PYMC_MARKETING_AVAILABLE and hasattr(self._model, "sample_posterior_predictive"):
            pred = self._model.sample_posterior_predictive(df)
            return pred.posterior_predictive["y"].mean(dim=["chain", "draw"]).values
        else:
            # Baseline model prediction
            from sklearn.preprocessing import StandardScaler
            feature_cols = self.media_cols + self.control_cols
            X = df[feature_cols].values
            # This is simplified - in practice you'd store the scaler
            return np.zeros(len(df))  # Placeholder
    
    def get_contributions(self) -> pd.DataFrame:
        """Get channel contribution decomposition."""
        if self._results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._results.channel_contributions
    
    def get_response_curves(
        self,
        spend_range: np.ndarray | None = None,
        n_points: int = 100,
    ) -> dict[str, pd.DataFrame]:
        """
        Get response curves for each channel.
        
        Args:
            spend_range: Array of spend values to evaluate
            n_points: Number of points if spend_range not provided
        
        Returns:
            Dict mapping channel -> DataFrame with spend and response columns
        """
        if self._data is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        curves = {}
        
        for channel in self.media_cols:
            if spend_range is None:
                max_spend = self._data[channel].max() * 1.5
                spend = np.linspace(0, max_spend, n_points)
            else:
                spend = spend_range
            
            # Get response using saturation parameters
            if channel in self._results.saturation_params:
                params = self._results.saturation_params[channel]
                coef = self._results.coefficients.get(channel, 1.0)
                
                # Logistic saturation
                lam = params.get("lam", 1.0)
                response = coef * (1 - np.exp(-lam * spend))
            else:
                # Linear fallback
                coef = self._results.coefficients.get(channel, 0)
                response = coef * spend
            
            curves[channel] = pd.DataFrame({
                "spend": spend,
                "response": response,
            })
        
        return curves
    
    @property
    def results(self) -> MMMResults | None:
        """Get the latest results."""
        return self._results

