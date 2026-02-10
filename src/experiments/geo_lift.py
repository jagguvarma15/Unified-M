"""
Geo-lift experiment analysis.

Implements a simplified CausalImpact-style analysis using
synthetic control / difference-in-differences for geo holdout tests.

The analyzer:
  1. Takes pre/post period data for treatment and control geos
  2. Builds a counterfactual from control geos
  3. Estimates lift as treatment_actual - counterfactual
  4. Computes confidence intervals via bootstrap or Bayesian posterior
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class GeoLiftResult:
    """Result of a geo-lift experiment analysis."""

    test_id: str
    channel: str
    lift_estimate: float
    lift_ci_lower: float
    lift_ci_upper: float
    lift_pct: float
    p_value: float
    is_significant: bool

    # Details
    treatment_actual: float = 0.0
    counterfactual: float = 0.0
    pre_period_fit_r2: float = 0.0
    n_treatment_geos: int = 0
    n_control_geos: int = 0

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
            "treatment_actual": self.treatment_actual,
            "counterfactual": self.counterfactual,
            "pre_period_fit_r2": self.pre_period_fit_r2,
            "n_treatment_geos": self.n_treatment_geos,
            "n_control_geos": self.n_control_geos,
        }


class GeoLiftAnalyzer:
    """
    Analyze geo-lift experiments using synthetic control.

    Usage::

        analyzer = GeoLiftAnalyzer()
        result = analyzer.analyze(
            data=df,
            treatment_geos=["CA", "OR", "WA"],
            control_geos=["TX", "AZ", "NV"],
            pre_start="2024-01-01",
            pre_end="2024-03-31",
            post_start="2024-04-01",
            post_end="2024-05-31",
        )
    """

    def __init__(
        self,
        confidence_level: float = 0.90,
        n_bootstrap: int = 1000,
    ):
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap

    def analyze(
        self,
        data: pd.DataFrame,
        treatment_geos: list[str],
        control_geos: list[str],
        pre_start: str,
        pre_end: str,
        post_start: str,
        post_end: str,
        kpi_col: str = "kpi_revenue",
        geo_col: str = "geo",
        date_col: str = "week_start",
        test_id: str = "geo_lift_test",
        channel: str = "unknown",
    ) -> GeoLiftResult:
        """
        Run geo-lift analysis.

        Args:
            data:            DataFrame with geo x week granularity.
            treatment_geos:  List of geo codes in the treatment group.
            control_geos:    List of geo codes in the control group.
            pre_start/end:   Pre-test period boundaries.
            post_start/end:  Post-test (treatment) period boundaries.
            kpi_col:         KPI column to measure.
            geo_col:         Geo identifier column.
            date_col:        Date column.

        Returns:
            GeoLiftResult with lift estimate and confidence intervals.
        """
        data = data.copy()
        data[date_col] = pd.to_datetime(data[date_col])

        # Split into pre and post periods
        pre_mask = (data[date_col] >= pre_start) & (data[date_col] <= pre_end)
        post_mask = (data[date_col] >= post_start) & (data[date_col] <= post_end)

        treat_mask = data[geo_col].isin(treatment_geos)
        ctrl_mask = data[geo_col].isin(control_geos)

        # Aggregate to weekly totals per group
        treat_pre = data[pre_mask & treat_mask].groupby(date_col)[kpi_col].sum()
        ctrl_pre = data[pre_mask & ctrl_mask].groupby(date_col)[kpi_col].sum()
        treat_post = data[post_mask & treat_mask].groupby(date_col)[kpi_col].sum()
        ctrl_post = data[post_mask & ctrl_mask].groupby(date_col)[kpi_col].sum()

        # Fit scaling factor from pre-period (simple ratio method)
        if ctrl_pre.sum() > 0:
            scale_factor = treat_pre.sum() / ctrl_pre.sum()
        else:
            scale_factor = 1.0

        # Pre-period fit quality
        if len(ctrl_pre) > 0 and len(treat_pre) > 0:
            aligned = pd.DataFrame({"treat": treat_pre, "ctrl": ctrl_pre * scale_factor}).dropna()
            if len(aligned) > 1:
                ss_res = ((aligned["treat"] - aligned["ctrl"]) ** 2).sum()
                ss_tot = ((aligned["treat"] - aligned["treat"].mean()) ** 2).sum()
                r2 = 1 - ss_res / (ss_tot + 1e-8)
            else:
                r2 = 0.0
        else:
            r2 = 0.0

        # Counterfactual = control_post * scale_factor
        counterfactual = ctrl_post.sum() * scale_factor
        treatment_actual = treat_post.sum()
        lift = treatment_actual - counterfactual

        # Lift as percentage
        lift_pct = lift / counterfactual if counterfactual > 0 else 0.0

        # Bootstrap CI
        ci_lower, ci_upper, p_value = self._bootstrap_ci(
            treat_post.values, ctrl_post.values, scale_factor
        )

        alpha = 1 - self.confidence_level
        is_significant = p_value < alpha

        logger.info(
            f"Geo-lift result: lift={lift:.2f} ({lift_pct:.1%}), "
            f"p={p_value:.4f}, significant={is_significant}"
        )

        return GeoLiftResult(
            test_id=test_id,
            channel=channel,
            lift_estimate=lift,
            lift_ci_lower=ci_lower,
            lift_ci_upper=ci_upper,
            lift_pct=lift_pct,
            p_value=p_value,
            is_significant=is_significant,
            treatment_actual=treatment_actual,
            counterfactual=counterfactual,
            pre_period_fit_r2=r2,
            n_treatment_geos=len(treatment_geos),
            n_control_geos=len(control_geos),
        )

    def _bootstrap_ci(
        self,
        treat_vals: np.ndarray,
        ctrl_vals: np.ndarray,
        scale_factor: float,
    ) -> tuple[float, float, float]:
        """Compute bootstrap confidence interval for lift."""
        rng = np.random.default_rng(42)
        n_treat = len(treat_vals)
        n_ctrl = len(ctrl_vals)

        if n_treat == 0 or n_ctrl == 0:
            return 0.0, 0.0, 1.0

        lifts = []
        for _ in range(self.n_bootstrap):
            t_sample = rng.choice(treat_vals, size=n_treat, replace=True)
            c_sample = rng.choice(ctrl_vals, size=n_ctrl, replace=True)
            boot_lift = t_sample.sum() - c_sample.sum() * scale_factor
            lifts.append(boot_lift)

        lifts = np.array(lifts)
        alpha = 1 - self.confidence_level
        ci_lower = float(np.percentile(lifts, alpha / 2 * 100))
        ci_upper = float(np.percentile(lifts, (1 - alpha / 2) * 100))

        # Two-sided p-value: proportion of bootstrap samples with opposite sign
        observed_lift = treat_vals.sum() - ctrl_vals.sum() * scale_factor
        if observed_lift >= 0:
            p_value = float(np.mean(lifts <= 0)) * 2
        else:
            p_value = float(np.mean(lifts >= 0)) * 2
        p_value = min(p_value, 1.0)

        return ci_lower, ci_upper, p_value
