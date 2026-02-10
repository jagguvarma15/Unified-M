"""
Switchback experiment analysis.

Switchback experiments alternate treatment ON/OFF across time periods
(or geo x time cells).  Analysis uses difference-in-means with
cluster-robust standard errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SwitchbackResult:
    """Result of a switchback experiment analysis."""

    test_id: str
    channel: str
    lift_estimate: float
    lift_ci_lower: float
    lift_ci_upper: float
    lift_pct: float
    p_value: float
    is_significant: bool
    n_on_periods: int = 0
    n_off_periods: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "channel": self.channel,
            "lift_estimate": self.lift_estimate,
            "lift_ci_lower": self.lift_ci_lower,
            "lift_ci_upper": self.lift_ci_upper,
            "lift_pct": self.lift_pct,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "n_on_periods": self.n_on_periods,
            "n_off_periods": self.n_off_periods,
        }


class SwitchbackAnalyzer:
    """
    Analyze switchback experiments.

    Expects data with a binary ``is_treatment`` column indicating
    whether the channel was ON (1) or OFF (0) in each period.

    Usage::

        analyzer = SwitchbackAnalyzer()
        result = analyzer.analyze(
            data=df,
            treatment_col="is_treatment",
            kpi_col="kpi_revenue",
        )
    """

    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level

    def analyze(
        self,
        data: pd.DataFrame,
        treatment_col: str = "is_treatment",
        kpi_col: str = "kpi_revenue",
        cluster_col: str | None = None,
        test_id: str = "switchback_test",
        channel: str = "unknown",
    ) -> SwitchbackResult:
        """
        Run switchback analysis (difference-in-means).

        Args:
            data:           DataFrame with rows per period (or geo x period).
            treatment_col:  Binary column (1=on, 0=off).
            kpi_col:        KPI to measure.
            cluster_col:    Optional column for cluster-robust SEs.
            test_id:        Identifier for the test.
            channel:        Channel being tested.

        Returns:
            SwitchbackResult with lift and confidence intervals.
        """
        on = data[data[treatment_col] == 1][kpi_col].values
        off = data[data[treatment_col] == 0][kpi_col].values

        if len(on) == 0 or len(off) == 0:
            logger.warning("Switchback analysis: insufficient data in on/off groups")
            return SwitchbackResult(
                test_id=test_id,
                channel=channel,
                lift_estimate=0.0,
                lift_ci_lower=0.0,
                lift_ci_upper=0.0,
                lift_pct=0.0,
                p_value=1.0,
                is_significant=False,
            )

        mean_on = np.mean(on)
        mean_off = np.mean(off)
        lift = mean_on - mean_off
        lift_pct = lift / mean_off if mean_off != 0 else 0.0

        # Standard error (cluster-robust if cluster_col provided)
        if cluster_col and cluster_col in data.columns:
            se = self._cluster_robust_se(data, treatment_col, kpi_col, cluster_col)
        else:
            var_on = np.var(on, ddof=1) / len(on)
            var_off = np.var(off, ddof=1) / len(off)
            se = np.sqrt(var_on + var_off)

        # t-test
        from scipy import stats
        alpha = 1 - self.confidence_level
        z = stats.norm.ppf(1 - alpha / 2)

        ci_lower = lift - z * se
        ci_upper = lift + z * se

        t_stat = lift / (se + 1e-12)
        p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

        is_significant = p_value < alpha

        logger.info(
            f"Switchback result: lift={lift:.2f} ({lift_pct:.1%}), "
            f"p={p_value:.4f}, significant={is_significant}"
        )

        return SwitchbackResult(
            test_id=test_id,
            channel=channel,
            lift_estimate=lift,
            lift_ci_lower=ci_lower,
            lift_ci_upper=ci_upper,
            lift_pct=lift_pct,
            p_value=p_value,
            is_significant=is_significant,
            n_on_periods=len(on),
            n_off_periods=len(off),
        )

    @staticmethod
    def _cluster_robust_se(
        data: pd.DataFrame,
        treatment_col: str,
        kpi_col: str,
        cluster_col: str,
    ) -> float:
        """Compute cluster-robust standard error."""
        clusters = data[cluster_col].unique()
        cluster_lifts = []

        for cluster in clusters:
            c_data = data[data[cluster_col] == cluster]
            on = c_data[c_data[treatment_col] == 1][kpi_col]
            off = c_data[c_data[treatment_col] == 0][kpi_col]
            if len(on) > 0 and len(off) > 0:
                cluster_lifts.append(on.mean() - off.mean())

        if len(cluster_lifts) < 2:
            return 1e8  # effectively infinite SE

        cluster_lifts = np.array(cluster_lifts)
        n = len(cluster_lifts)
        return float(np.std(cluster_lifts, ddof=1) / np.sqrt(n))
