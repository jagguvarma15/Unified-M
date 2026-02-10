"""
Backtesting framework for MMM evaluation.

Extends the basic cross-validation in evaluation.py with:
  - WMAPE (Weighted MAPE)
  - Out-of-time holdout
  - One-channel-out sensitivity test
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class BacktestResult:
    """Result of a backtest evaluation."""

    holdout_weeks: int = 0
    in_sample_metrics: dict[str, float] = field(default_factory=dict)
    out_of_sample_metrics: dict[str, float] = field(default_factory=dict)
    channel_sensitivity: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "holdout_weeks": self.holdout_weeks,
            "in_sample_metrics": self.in_sample_metrics,
            "out_of_sample_metrics": self.out_of_sample_metrics,
            "channel_sensitivity": self.channel_sensitivity,
        }


def compute_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted MAPE: sum(|errors|) / sum(|actuals|).

    Better than MAPE for sparse weeks since it weights by magnitude.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total_actual = np.sum(np.abs(y_true))
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAPE, WMAPE, RMSE, MAE, R2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else 0.0
    wmape = compute_wmape(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))

    return {
        "mape": round(mape, 4),
        "wmape": round(wmape, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r_squared": round(r2, 4),
    }


def holdout_backtest(
    model_fn: Callable,
    df: pd.DataFrame,
    holdout_weeks: int = 8,
    target_col: str = "y",
    date_col: str = "date",
) -> BacktestResult:
    """
    Out-of-time holdout backtest.

    Trains on all data except the last ``holdout_weeks`` weeks,
    then evaluates on the held-out period.

    Args:
        model_fn:        Callable that takes (train_df) and returns a fitted model
                         with a predict(df) method.
        df:              Full dataset sorted by date.
        holdout_weeks:   Number of weeks to hold out.
        target_col:      Target column name.
        date_col:        Date column name.

    Returns:
        BacktestResult with in-sample and out-of-sample metrics.
    """
    df = df.sort_values(date_col).reset_index(drop=True)

    # Split: train on all except last N weeks
    split_idx = len(df) - holdout_weeks
    if split_idx < holdout_weeks:
        logger.warning("Not enough data for holdout backtest")
        return BacktestResult(holdout_weeks=holdout_weeks)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info(f"Holdout backtest: {len(train_df)} train, {len(test_df)} test rows")

    # Fit model
    model = model_fn(train_df)

    # In-sample metrics
    y_train_pred = model.predict(train_df)
    in_sample = compute_all_metrics(train_df[target_col].values, y_train_pred)

    # Out-of-sample metrics
    y_test_pred = model.predict(test_df)
    out_of_sample = compute_all_metrics(test_df[target_col].values, y_test_pred)

    logger.info(
        f"Holdout results: in-sample MAPE={in_sample['mape']:.2f}%, "
        f"out-of-sample MAPE={out_of_sample['mape']:.2f}%"
    )

    return BacktestResult(
        holdout_weeks=holdout_weeks,
        in_sample_metrics=in_sample,
        out_of_sample_metrics=out_of_sample,
    )


def channel_sensitivity_test(
    model_fn: Callable,
    df: pd.DataFrame,
    media_cols: list[str],
    target_col: str = "y",
) -> dict[str, float]:
    """
    One-channel-out sensitivity test.

    For each media channel, zero out its spend and retrain.
    The expected lift should decrease -- if it doesn't, the channel
    coefficient may not be trustworthy.

    Args:
        model_fn:    Callable taking (df) returning fitted model.
        df:          Full dataset.
        media_cols:  List of media spend columns.
        target_col:  Target column.

    Returns:
        Dict of channel -> predicted total lift decrease when zeroed out.
    """
    # Baseline: full model predictions
    baseline_model = model_fn(df)
    baseline_pred = baseline_model.predict(df)
    baseline_total = float(baseline_pred.sum())

    sensitivity: dict[str, float] = {}

    for col in media_cols:
        # Zero out this channel
        df_zeroed = df.copy()
        df_zeroed[col] = 0.0

        zeroed_pred = baseline_model.predict(df_zeroed)
        zeroed_total = float(zeroed_pred.sum())

        # Lift decrease
        decrease = baseline_total - zeroed_total
        decrease_pct = decrease / baseline_total if baseline_total > 0 else 0.0

        sensitivity[col] = round(decrease_pct, 4)
        logger.debug(f"Channel {col}: zeroing decreases prediction by {decrease_pct:.2%}")

    return sensitivity
