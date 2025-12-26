"""
Fusion strategies for combining measurement signals.

Implements various approaches to reconcile MMM, incrementality tests,
and attribution data into unified estimates.
"""

from dataclasses import dataclass, field
from typing import Literal
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ChannelEstimate:
    """
    Unified estimate for a single channel.
    
    Combines information from multiple measurement sources.
    """
    
    channel: str
    
    # Point estimates
    lift_estimate: float  # Final reconciled lift
    roi_estimate: float  # Return on investment
    
    # Uncertainty
    lift_ci_lower: float
    lift_ci_upper: float
    confidence_score: float  # 0-1, higher = more confident
    
    # Source contributions
    mmm_contribution: float = 0.0
    incrementality_contribution: float = 0.0
    attribution_contribution: float = 0.0
    
    # Metadata
    data_quality_score: float = 1.0
    last_test_date: str | None = None
    
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
            "data_quality_score": self.data_quality_score,
            "last_test_date": self.last_test_date,
        }


@dataclass 
class ReconciliationResult:
    """
    Complete reconciliation output for all channels.
    """
    
    channel_estimates: dict[str, ChannelEstimate] = field(default_factory=dict)
    total_incremental_value: float = 0.0
    reconciliation_method: str = "weighted_average"
    timestamp: str = ""
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy analysis."""
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
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved reconciliation results to {path}")


class ReconciliationEngine:
    """
    Engine for reconciling multiple measurement signals.
    
    Supports pluggable fusion strategies:
    - weighted_average: Simple weighted combination
    - bayesian: Bayesian updating with priors from MMM
    - hierarchical: Multi-level calibration
    
    Example:
        >>> engine = ReconciliationEngine(
        ...     mmm_weight=0.5,
        ...     incrementality_weight=0.3,
        ...     attribution_weight=0.2,
        ... )
        >>> result = engine.reconcile(
        ...     mmm_results=mmm_results,
        ...     incrementality_tests=test_df,
        ...     attribution_data=attr_df,
        ... )
    """
    
    def __init__(
        self,
        mmm_weight: float = 0.5,
        incrementality_weight: float = 0.3,
        attribution_weight: float = 0.2,
        fusion_method: Literal["weighted_average", "bayesian", "hierarchical"] = "weighted_average",
        confidence_threshold: float = 0.8,
    ):
        """
        Initialize reconciliation engine.
        
        Args:
            mmm_weight: Weight for MMM estimates (0-1)
            incrementality_weight: Weight for incrementality tests (0-1)
            attribution_weight: Weight for attribution data (0-1)
            fusion_method: Strategy for combining estimates
            confidence_threshold: Minimum confidence for estimates
        """
        # Normalize weights
        total = mmm_weight + incrementality_weight + attribution_weight
        self.mmm_weight = mmm_weight / total
        self.incrementality_weight = incrementality_weight / total
        self.attribution_weight = attribution_weight / total
        
        self.fusion_method = fusion_method
        self.confidence_threshold = confidence_threshold
        
        self._mmm_results = None
        self._incrementality_tests = None
        self._attribution_data = None
    
    def reconcile(
        self,
        mmm_results: dict | None = None,
        incrementality_tests: pd.DataFrame | None = None,
        attribution_data: pd.DataFrame | None = None,
        channels: list[str] | None = None,
    ) -> ReconciliationResult:
        """
        Reconcile all measurement signals.
        
        Args:
            mmm_results: Dictionary with MMM outputs (coefficients, contributions)
            incrementality_tests: DataFrame with test results
            attribution_data: DataFrame with attribution data
            channels: List of channels to reconcile (auto-detected if None)
        
        Returns:
            ReconciliationResult with unified estimates
        """
        self._mmm_results = mmm_results
        self._incrementality_tests = incrementality_tests
        self._attribution_data = attribution_data
        
        # Auto-detect channels
        if channels is None:
            channels = self._detect_channels()
        
        logger.info(f"Reconciling {len(channels)} channels using {self.fusion_method}")
        
        # Select fusion method
        if self.fusion_method == "weighted_average":
            return weighted_average_fusion(
                channels=channels,
                mmm_results=mmm_results,
                incrementality_tests=incrementality_tests,
                attribution_data=attribution_data,
                mmm_weight=self.mmm_weight,
                incrementality_weight=self.incrementality_weight,
                attribution_weight=self.attribution_weight,
            )
        elif self.fusion_method == "bayesian":
            return bayesian_fusion(
                channels=channels,
                mmm_results=mmm_results,
                incrementality_tests=incrementality_tests,
                attribution_data=attribution_data,
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _detect_channels(self) -> list[str]:
        """Auto-detect channels from available data sources."""
        channels = set()
        
        if self._mmm_results:
            if "channels" in self._mmm_results:
                channels.update(self._mmm_results["channels"])
            elif "coefficients" in self._mmm_results:
                channels.update(self._mmm_results["coefficients"].keys())
        
        if self._incrementality_tests is not None and "channel" in self._incrementality_tests.columns:
            channels.update(self._incrementality_tests["channel"].unique())
        
        if self._attribution_data is not None and "channel" in self._attribution_data.columns:
            channels.update(self._attribution_data["channel"].unique())
        
        return list(channels)


def weighted_average_fusion(
    channels: list[str],
    mmm_results: dict | None,
    incrementality_tests: pd.DataFrame | None,
    attribution_data: pd.DataFrame | None,
    mmm_weight: float = 0.5,
    incrementality_weight: float = 0.3,
    attribution_weight: float = 0.2,
) -> ReconciliationResult:
    """
    Simple weighted average fusion of measurement signals.
    
    This is the most straightforward reconciliation approach:
    1. Extract lift estimates from each source
    2. Compute weighted average
    3. Propagate uncertainty
    """
    from datetime import datetime
    
    channel_estimates = {}
    total_value = 0.0
    
    for channel in channels:
        # Extract MMM estimate
        mmm_lift = 0.0
        mmm_ci = (0.0, 0.0)
        mmm_available = False
        
        if mmm_results:
            coef = mmm_results.get("coefficients", {}).get(channel, 0)
            if coef != 0:
                mmm_lift = coef
                # Approximate CI from coefficient (would be better from posterior)
                mmm_ci = (coef * 0.8, coef * 1.2)
                mmm_available = True
        
        # Extract incrementality test estimate
        incr_lift = 0.0
        incr_ci = (0.0, 0.0)
        incr_available = False
        last_test = None
        
        if incrementality_tests is not None:
            channel_tests = incrementality_tests[
                incrementality_tests["channel"] == channel
            ]
            if len(channel_tests) > 0:
                # Use most recent test
                latest = channel_tests.sort_values("end_date").iloc[-1]
                incr_lift = latest["lift_estimate"]
                incr_ci = (latest["lift_ci_lower"], latest["lift_ci_upper"])
                incr_available = True
                last_test = str(latest["end_date"])
        
        # Extract attribution estimate
        attr_lift = 0.0
        attr_available = False
        
        if attribution_data is not None:
            channel_attr = attribution_data[
                attribution_data["channel"] == channel
            ]
            if len(channel_attr) > 0:
                # Aggregate attribution
                if "attributed_conversions" in channel_attr.columns:
                    attr_lift = channel_attr["attributed_conversions"].sum()
                elif "attributed_revenue" in channel_attr.columns:
                    attr_lift = channel_attr["attributed_revenue"].sum()
                attr_available = True
        
        # Compute weighted average
        numerator = 0.0
        denominator = 0.0
        
        if mmm_available:
            numerator += mmm_weight * mmm_lift
            denominator += mmm_weight
        
        if incr_available:
            numerator += incrementality_weight * incr_lift
            denominator += incrementality_weight
        
        if attr_available:
            numerator += attribution_weight * attr_lift
            denominator += attribution_weight
        
        if denominator > 0:
            final_lift = numerator / denominator
        else:
            final_lift = 0.0
        
        # Compute uncertainty (wider if fewer sources)
        confidence = denominator  # 0-1 based on available sources
        
        # CI from weighted combination
        if incr_available:
            ci_lower = incr_ci[0]
            ci_upper = incr_ci[1]
        elif mmm_available:
            ci_lower = mmm_ci[0]
            ci_upper = mmm_ci[1]
        else:
            ci_lower = final_lift * 0.5
            ci_upper = final_lift * 1.5
        
        # Create channel estimate
        estimate = ChannelEstimate(
            channel=channel,
            lift_estimate=final_lift,
            roi_estimate=final_lift,  # Simplified; would need spend data
            lift_ci_lower=ci_lower,
            lift_ci_upper=ci_upper,
            confidence_score=confidence,
            mmm_contribution=mmm_lift * mmm_weight if mmm_available else 0,
            incrementality_contribution=incr_lift * incrementality_weight if incr_available else 0,
            attribution_contribution=attr_lift * attribution_weight if attr_available else 0,
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


def bayesian_fusion(
    channels: list[str],
    mmm_results: dict | None,
    incrementality_tests: pd.DataFrame | None,
    attribution_data: pd.DataFrame | None,
) -> ReconciliationResult:
    """
    Bayesian fusion treating MMM as prior and tests as likelihood.
    
    More sophisticated than weighted average:
    1. Use MMM posteriors as priors
    2. Update with incrementality test results (likelihood)
    3. Produce calibrated posteriors with proper uncertainty
    """
    from datetime import datetime
    
    channel_estimates = {}
    total_value = 0.0
    
    for channel in channels:
        # Get MMM prior
        prior_mean = 0.0
        prior_std = 1.0  # Vague prior if no MMM
        
        if mmm_results and "coefficients" in mmm_results:
            coef = mmm_results["coefficients"].get(channel, 0)
            prior_mean = coef
            prior_std = abs(coef) * 0.3 + 0.01  # Assume 30% uncertainty
        
        # Get test likelihood
        likelihood_mean = None
        likelihood_std = None
        last_test = None
        
        if incrementality_tests is not None:
            channel_tests = incrementality_tests[
                incrementality_tests["channel"] == channel
            ]
            if len(channel_tests) > 0:
                latest = channel_tests.sort_values("end_date").iloc[-1]
                likelihood_mean = latest["lift_estimate"]
                # Estimate std from CI (assuming 95% CI)
                likelihood_std = (latest["lift_ci_upper"] - latest["lift_ci_lower"]) / 3.92
                last_test = str(latest["end_date"])
        
        # Bayesian update (conjugate normal-normal)
        if likelihood_mean is not None and likelihood_std is not None:
            # Posterior precision = prior precision + likelihood precision
            prior_precision = 1 / (prior_std ** 2)
            likelihood_precision = 1 / (likelihood_std ** 2)
            posterior_precision = prior_precision + likelihood_precision
            
            # Posterior mean = weighted average by precision
            posterior_mean = (
                prior_precision * prior_mean + likelihood_precision * likelihood_mean
            ) / posterior_precision
            
            posterior_std = np.sqrt(1 / posterior_precision)
            
            confidence = min(1.0, likelihood_precision / (prior_precision + 0.01))
        else:
            # No test data, use prior
            posterior_mean = prior_mean
            posterior_std = prior_std
            confidence = 0.3  # Low confidence without test
        
        # 95% credible interval
        ci_lower = posterior_mean - 1.96 * posterior_std
        ci_upper = posterior_mean + 1.96 * posterior_std
        
        estimate = ChannelEstimate(
            channel=channel,
            lift_estimate=posterior_mean,
            roi_estimate=posterior_mean,
            lift_ci_lower=ci_lower,
            lift_ci_upper=ci_upper,
            confidence_score=confidence,
            mmm_contribution=prior_mean,
            incrementality_contribution=likelihood_mean or 0,
            attribution_contribution=0,
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

