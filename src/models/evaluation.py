"""
Model evaluation utilities for MMM.

Provides metrics, cross-validation, and diagnostic functions.
"""

import numpy as np
import pandas as pd
from typing import Callable
from loguru import logger


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAPE as percentage (0-100)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Avoid division by zero
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        RMSE in same units as y
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R-squared (coefficient of determination).

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        R-squared value (higher is better, max 1.0)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    return float(1 - (ss_res / ss_tot))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAE in same units as y
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute all standard evaluation metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with mape, rmse, mae, r2 metrics
    """
    return {
        "mape": compute_mape(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
    }


def cross_validate(
    model_fn: Callable,
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "y",
    n_splits: int = 5,
    test_size: float = 0.2,
    gap: int = 0,
) -> dict[str, list[float]]:
    """
    Time-series cross-validation for MMM.

    Uses expanding window (train on past, test on future) to
    respect temporal ordering.

    Args:
        model_fn: Function that takes (train_df) and returns fitted model
        df: Full dataset sorted by date
        date_col: Date column name
        target_col: Target column name
        n_splits: Number of CV folds
        test_size: Fraction of data for each test set
        gap: Number of periods to skip between train and test

    Returns:
        Dictionary with lists of metrics for each fold
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    test_n = int(n * test_size)

    results: dict[str, list[float]] = {"mape": [], "rmse": [], "mae": [], "r2": []}

    for fold in range(n_splits):
        # Calculate split points
        test_end = n - fold * test_n
        test_start = test_end - test_n
        train_end = test_start - gap

        if train_end < test_n:
            logger.warning(f"Skipping fold {fold}: not enough training data")
            continue

        # Split data
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        logger.info(
            f"Fold {fold + 1}/{n_splits}: "
            f"Train {len(train_df)} rows, Test {len(test_df)} rows"
        )

        # Fit model
        model = model_fn(train_df)

        # Predict
        y_pred = model.predict(test_df)
        y_true = test_df[target_col].values

        # Evaluate
        fold_metrics = evaluate_model(y_true, y_pred)

        for metric, value in fold_metrics.items():
            results[metric].append(value)

        logger.info(f"  MAPE: {fold_metrics['mape']:.2f}%, R2: {fold_metrics['r2']:.3f}")

    return results


def compute_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute residual diagnostics to check model assumptions.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with diagnostic statistics
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    return {
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "skewness": float(_skewness(residuals)),
        "kurtosis": float(_kurtosis(residuals)),
        "durbin_watson": float(_durbin_watson(residuals)),
    }


def _skewness(x: np.ndarray) -> float:
    """Compute skewness of distribution."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    return float(np.sum(((x - mean) / std) ** 3) * n / ((n - 1) * (n - 2)))


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis of distribution."""
    mean = np.mean(x)
    std = np.std(x)
    m4 = np.mean((x - mean) ** 4)
    return float(m4 / (std ** 4) - 3)


def _durbin_watson(residuals: np.ndarray) -> float:
    """
    Compute Durbin-Watson statistic for autocorrelation.

    Values close to 2 indicate no autocorrelation.
    Values < 2 indicate positive autocorrelation.
    Values > 2 indicate negative autocorrelation.
    """
    diff = np.diff(residuals)
    return float(np.sum(diff ** 2) / np.sum(residuals ** 2))


def compute_contribution_stability(
    contributions: pd.DataFrame,
    window: int = 4,
) -> dict[str, float]:
    """
    Assess stability of channel contributions over time.

    Unstable contributions (high variance) may indicate model issues.

    Args:
        contributions: DataFrame with channel contribution columns
        window: Rolling window size for stability calculation

    Returns:
        Dictionary with stability metrics per channel
    """
    stability = {}

    # Get contribution columns (exclude date, actual, predicted)
    contrib_cols = [
        c for c in contributions.columns
        if c not in ["date", "actual", "predicted", "baseline"]
    ]

    for col in contrib_cols:
        rolling_std = contributions[col].rolling(window=window).std()
        rolling_mean = contributions[col].rolling(window=window).mean()

        # Coefficient of variation
        cv = (rolling_std / (rolling_mean.abs() + 1e-8)).mean()

        stability[col] = float(cv)

    return stability
