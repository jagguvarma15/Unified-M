"""
Reconciliation engine -- fuses MMM, incrementality tests, and attribution
into a single consistent set of channel lift estimates with uncertainty.

This is the core of "unified measurement": rather than reporting three
separate numbers, we produce one calibrated estimate per channel, with
a confidence score that reflects how much evidence backs it.

Fusion strategies:
  - weighted_average : simple weighted combination (default)
  - bayesian         : MMM posterior as prior, test as likelihood

The engine also computes calibration factors so future MMM runs can
be anchored to real-world test results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChannelEstimate:
    """Unified estimate for a single channel."""

    channel: str
    lift_estimate: float
    roi_estimate: float
    lift_ci_lower: float
    lift_ci_upper: float
    confidence_score: float  # 0-1

    mmm_contribution: float = 0.0
    incrementality_contribution: float = 0.0
    attribution_contribution: float = 0.0

    calibration_factor: float | None = None
    last_test_date: str | None = None
    data_quality_score: float = 1.0

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "lift_estimate": self.lift_estimate,
            "roi_estimate": self.roi_estimate,
            "lift_ci_lower": self.lift_ci_lower,
            "lift_ci_upper": self.lift_ci_upper,
            "confidence_score": self.confidence_score,
            "mmm_contribution": self.mmm_contribution,
            "incrementality_contribution": self.incrementality_contribution,
            "attribution_contribution": self.attribution_contribution,
            "calibration_factor": self.calibration_factor,
            "last_test_date": self.last_test_date,
            "data_quality_score": self.data_quality_score,
        }


@dataclass
class ReconciliationResult:
    """Complete reconciliation output for all channels."""

    channel_estimates: dict[str, ChannelEstimate] = field(default_factory=dict)
    total_incremental_value: float = 0.0
    reconciliation_method: str = "weighted_average"
    timestamp: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        records = [est.to_dict() for est in self.channel_estimates.values()]
        return pd.DataFrame(records)

    def to_dict(self) -> dict:
        return {
            "channel_estimates": {
                k: v.to_dict() for k, v in self.channel_estimates.items()
            },
            "total_incremental_value": self.total_incremental_value,
            "reconciliation_method": self.reconciliation_method,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved reconciliation results to {path}")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ReconciliationEngine:
    """
    Fuses multiple measurement signals into unified channel estimates.

    Usage::

        engine = ReconciliationEngine(mmm_weight=0.5)
        result = engine.reconcile(
            mmm_results=params_dict,
            incrementality_tests=test_df,
        )
    """

    def __init__(
        self,
        mmm_weight: float = 0.5,
        incrementality_weight: float = 0.3,
        attribution_weight: float = 0.2,
        fusion_method: Literal["weighted_average", "bayesian"] = "weighted_average",
        confidence_threshold: float = 0.8,
    ):
        total = mmm_weight + incrementality_weight + attribution_weight
        self.mmm_weight = mmm_weight / total
        self.incrementality_weight = incrementality_weight / total
        self.attribution_weight = attribution_weight / total
        self.fusion_method = fusion_method
        self.confidence_threshold = confidence_threshold

    def reconcile(
        self,
        mmm_results: dict | None = None,
        incrementality_tests: pd.DataFrame | None = None,
        attribution_data: pd.DataFrame | None = None,
        channels: list[str] | None = None,
    ) -> ReconciliationResult:
        """
        Reconcile all measurement signals into unified estimates.

        Args:
            mmm_results:          Model parameters dict (must have 'coefficients').
            incrementality_tests: DataFrame with test results.
            attribution_data:     DataFrame with attribution data.
            channels:             Explicit channel list; auto-detected if None.

        Returns:
            ReconciliationResult with per-channel estimates and metadata.
        """
        if channels is None:
            channels = self._detect_channels(mmm_results, incrementality_tests, attribution_data)

        logger.info(f"Reconciling {len(channels)} channels via {self.fusion_method}")

        if self.fusion_method == "bayesian":
            return self._bayesian_fusion(
                channels, mmm_results, incrementality_tests, attribution_data,
            )

        return self._weighted_average_fusion(
            channels, mmm_results, incrementality_tests, attribution_data,
        )

    # ------------------------------------------------------------------
    # Weighted average
    # ------------------------------------------------------------------

    def _weighted_average_fusion(
        self,
        channels: list[str],
        mmm_results: dict | None,
        incrementality_tests: pd.DataFrame | None,
        attribution_data: pd.DataFrame | None,
    ) -> ReconciliationResult:
        channel_estimates: dict[str, ChannelEstimate] = {}
        total_value = 0.0

        for channel in channels:
            mmm_lift, mmm_ci, mmm_ok = self._extract_mmm(channel, mmm_results)
            incr_lift, incr_ci, incr_ok, last_test = self._extract_incrementality(
                channel, incrementality_tests,
            )
            attr_lift, attr_ok = self._extract_attribution(channel, attribution_data)

            # Weighted combination of available signals
            numerator = 0.0
            denominator = 0.0

            if mmm_ok:
                numerator += self.mmm_weight * mmm_lift
                denominator += self.mmm_weight
            if incr_ok:
                numerator += self.incrementality_weight * incr_lift
                denominator += self.incrementality_weight
            if attr_ok:
                numerator += self.attribution_weight * attr_lift
                denominator += self.attribution_weight

            final_lift = numerator / denominator if denominator > 0 else 0.0
            confidence = denominator  # 0-1 based on available evidence

            # Best-available CI
            if incr_ok:
                ci = incr_ci
            elif mmm_ok:
                ci = mmm_ci
            else:
                ci = (final_lift * 0.5, final_lift * 1.5)

            # Calibration factor = test / MMM
            cal_factor = None
            if incr_ok and mmm_ok and mmm_lift != 0:
                cal_factor = incr_lift / mmm_lift

            estimate = ChannelEstimate(
                channel=channel,
                lift_estimate=final_lift,
                roi_estimate=final_lift,
                lift_ci_lower=ci[0],
                lift_ci_upper=ci[1],
                confidence_score=confidence,
                mmm_contribution=mmm_lift * self.mmm_weight if mmm_ok else 0,
                incrementality_contribution=incr_lift * self.incrementality_weight if incr_ok else 0,
                attribution_contribution=attr_lift * self.attribution_weight if attr_ok else 0,
                calibration_factor=cal_factor,
                last_test_date=last_test,
            )
            channel_estimates[channel] = estimate
            total_value += final_lift

        return ReconciliationResult(
            channel_estimates=channel_estimates,
            total_incremental_value=total_value,
            reconciliation_method="weighted_average",
            timestamp=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # Bayesian fusion
    # ------------------------------------------------------------------

    def _bayesian_fusion(
        self,
        channels: list[str],
        mmm_results: dict | None,
        incrementality_tests: pd.DataFrame | None,
        attribution_data: pd.DataFrame | None,
    ) -> ReconciliationResult:
        """MMM posterior as prior, test results as likelihood. Normal-Normal conjugate."""
        channel_estimates: dict[str, ChannelEstimate] = {}
        total_value = 0.0

        for channel in channels:
            mmm_lift, mmm_ci, mmm_ok = self._extract_mmm(channel, mmm_results)
            incr_lift, incr_ci, incr_ok, last_test = self._extract_incrementality(
                channel, incrementality_tests,
            )

            # Prior from MMM
            prior_mean = mmm_lift if mmm_ok else 0.0
            prior_std = abs(mmm_lift) * 0.3 + 0.01

            # Likelihood from test
            if incr_ok:
                likelihood_mean = incr_lift
                likelihood_std = (incr_ci[1] - incr_ci[0]) / 3.92

                prior_prec = 1 / (prior_std ** 2)
                like_prec = 1 / (likelihood_std ** 2 + 1e-12)
                post_prec = prior_prec + like_prec

                posterior_mean = (prior_prec * prior_mean + like_prec * likelihood_mean) / post_prec
                posterior_std = np.sqrt(1 / post_prec)
                confidence = min(1.0, like_prec / (prior_prec + 0.01))
            else:
                posterior_mean = prior_mean
                posterior_std = prior_std
                confidence = 0.3

            ci_lower = posterior_mean - 1.96 * posterior_std
            ci_upper = posterior_mean + 1.96 * posterior_std

            cal_factor = None
            if incr_ok and mmm_ok and mmm_lift != 0:
                cal_factor = incr_lift / mmm_lift

            estimate = ChannelEstimate(
                channel=channel,
                lift_estimate=posterior_mean,
                roi_estimate=posterior_mean,
                lift_ci_lower=ci_lower,
                lift_ci_upper=ci_upper,
                confidence_score=confidence,
                mmm_contribution=prior_mean,
                incrementality_contribution=incr_lift if incr_ok else 0,
                calibration_factor=cal_factor,
                last_test_date=last_test,
            )
            channel_estimates[channel] = estimate
            total_value += posterior_mean

        return ReconciliationResult(
            channel_estimates=channel_estimates,
            total_incremental_value=total_value,
            reconciliation_method="bayesian",
            timestamp=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # Signal extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_mmm(
        channel: str,
        mmm_results: dict | None,
    ) -> tuple[float, tuple[float, float], bool]:
        if not mmm_results:
            return 0.0, (0.0, 0.0), False
        coef = mmm_results.get("coefficients", {}).get(channel, 0)
        if coef == 0:
            return 0.0, (0.0, 0.0), False
        return coef, (coef * 0.8, coef * 1.2), True

    @staticmethod
    def _extract_incrementality(
        channel: str,
        tests: pd.DataFrame | None,
    ) -> tuple[float, tuple[float, float], bool, str | None]:
        if tests is None or "channel" not in tests.columns:
            return 0.0, (0.0, 0.0), False, None
        ch_tests = tests[tests["channel"] == channel]
        if len(ch_tests) == 0:
            return 0.0, (0.0, 0.0), False, None
        latest = ch_tests.sort_values("end_date").iloc[-1]
        lift = latest["lift_estimate"]
        ci = (latest["lift_ci_lower"], latest["lift_ci_upper"])
        return lift, ci, True, str(latest["end_date"])

    @staticmethod
    def _extract_attribution(
        channel: str,
        attr: pd.DataFrame | None,
    ) -> tuple[float, bool]:
        if attr is None or "channel" not in attr.columns:
            return 0.0, False
        ch_attr = attr[attr["channel"] == channel]
        if len(ch_attr) == 0:
            return 0.0, False
        if "attributed_conversions" in ch_attr.columns:
            return float(ch_attr["attributed_conversions"].sum()), True
        if "attributed_revenue" in ch_attr.columns:
            return float(ch_attr["attributed_revenue"].sum()), True
        return 0.0, False

    @staticmethod
    def _detect_channels(
        mmm_results: dict | None,
        tests: pd.DataFrame | None,
        attr: pd.DataFrame | None,
    ) -> list[str]:
        channels: set[str] = set()
        if mmm_results:
            channels.update(mmm_results.get("coefficients", {}).keys())
            channels.update(mmm_results.get("channels", []))
        if tests is not None and "channel" in tests.columns:
            channels.update(tests["channel"].unique())
        if attr is not None and "channel" in attr.columns:
            channels.update(attr["channel"].unique())
        return sorted(channels)
