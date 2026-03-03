"""Regression tests for BaseMMM metric helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.base_model import BaseMMM


class _DummyMMM(BaseMMM):
    def fit(self, df: pd.DataFrame, target_col: str, media_cols: list[str], control_cols: list[str] | None = None, date_col: str = "date", **kwargs: Any) -> dict[str, Any]:
        return {}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df))

    def get_channel_contributions(self) -> dict[str, np.ndarray]:
        return {}

    def get_response_curves(self, spend_grid: np.ndarray | None = None, n_points: int = 100) -> dict[str, pd.DataFrame]:
        return {}

    def get_parameters(self) -> dict[str, Any]:
        return {}

    def save_state(self, directory: Path) -> None:
        return None

    def load_state(self, directory: Path) -> None:
        return None


def test_get_metrics_all_zero_targets_sets_mape_zero():
    model = _DummyMMM()
    y_true = np.array([0.0, 0.0, 0.0], dtype=float)
    y_pred = np.array([1.0, -1.0, 0.5], dtype=float)

    metrics = model.get_metrics(y_true=y_true, y_pred=y_pred)

    assert metrics["mape"] == 0.0
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["r_squared"])
    assert np.isfinite(metrics["nrmse"])
