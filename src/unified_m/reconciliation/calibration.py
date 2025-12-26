"""
Calibration utilities for adjusting MMM estimates with incrementality tests.

When incrementality tests are available, they can be used to calibrate
(adjust) MMM estimates to be more accurate.
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_calibration_factors(
    mmm_estimates: dict[str, float],
    test_results: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute calibration factors from incrementality tests.
    
    Calibration factor = Test estimate / MMM estimate
    
    A factor > 1 means MMM underestimated the channel.
    A factor < 1 means MMM overestimated the channel.
    
    Args:
        mmm_estimates: Dictionary of channel -> MMM lift estimate
        test_results: DataFrame with incrementality test results
    
    Returns:
        Dictionary of channel -> calibration factor
    """
    calibration_factors = {}
    
    for channel, mmm_estimate in mmm_estimates.items():
        # Find matching test
        channel_tests = test_results[test_results["channel"] == channel]
        
        if len(channel_tests) == 0:
            # No test for this channel, use factor of 1
            calibration_factors[channel] = 1.0
            continue
        
        # Use most recent test
        latest = channel_tests.sort_values("end_date").iloc[-1]
        test_estimate = latest["lift_estimate"]
        
        if mmm_estimate != 0:
            factor = test_estimate / mmm_estimate
        else:
            factor = 1.0
        
        calibration_factors[channel] = factor
        
        logger.info(
            f"Channel {channel}: MMM={mmm_estimate:.2f}, "
            f"Test={test_estimate:.2f}, Factor={factor:.2f}"
        )
    
    return calibration_factors


def calibrate_mmm_with_tests(
    mmm_contributions: pd.DataFrame,
    calibration_factors: dict[str, float],
    contribution_suffix: str = "_contribution",
) -> pd.DataFrame:
    """
    Apply calibration factors to MMM contributions.
    
    Args:
        mmm_contributions: DataFrame with channel contribution columns
        calibration_factors: Dictionary of channel -> calibration factor
        contribution_suffix: Suffix for contribution columns
    
    Returns:
        Calibrated contributions DataFrame
    """
    calibrated = mmm_contributions.copy()
    
    for channel, factor in calibration_factors.items():
        col = f"{channel}{contribution_suffix}"
        
        if col not in calibrated.columns:
            # Try without suffix
            col = channel
        
        if col in calibrated.columns:
            calibrated[col] = calibrated[col] * factor
            logger.info(f"Calibrated {col} with factor {factor:.2f}")
    
    # Recalculate predicted if it exists
    contrib_cols = [
        c for c in calibrated.columns 
        if c.endswith(contribution_suffix) or c == "baseline_contribution"
    ]
    
    if contrib_cols:
        calibrated["predicted_calibrated"] = calibrated[contrib_cols].sum(axis=1)
    
    return calibrated


def estimate_test_coverage(
    channels: list[str],
    test_results: pd.DataFrame,
    lookback_days: int = 365,
) -> dict[str, dict]:
    """
    Estimate test coverage for each channel.
    
    Returns metrics about how well each channel is covered by tests:
    - has_test: Whether any test exists
    - recency_days: Days since most recent test
    - test_count: Number of tests
    - avg_confidence: Average CI width
    
    Args:
        channels: List of channels to evaluate
        test_results: DataFrame with test results
        lookback_days: Consider tests within this many days
    
    Returns:
        Dictionary of channel -> coverage metrics
    """
    from datetime import datetime, timedelta
    
    coverage = {}
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    
    for channel in channels:
        channel_tests = test_results[test_results["channel"] == channel]
        
        if len(channel_tests) == 0:
            coverage[channel] = {
                "has_test": False,
                "recency_days": None,
                "test_count": 0,
                "avg_confidence": 0.0,
            }
            continue
        
        # Filter to recent tests
        channel_tests = channel_tests.copy()
        channel_tests["end_date"] = pd.to_datetime(channel_tests["end_date"])
        recent_tests = channel_tests[channel_tests["end_date"] >= cutoff_date]
        
        if len(recent_tests) == 0:
            coverage[channel] = {
                "has_test": True,
                "recency_days": (datetime.now() - channel_tests["end_date"].max()).days,
                "test_count": 0,
                "avg_confidence": 0.0,
            }
            continue
        
        # Compute metrics
        latest = recent_tests["end_date"].max()
        recency = (datetime.now() - latest).days
        
        # Confidence from CI width (narrower = more confident)
        ci_width = (
            recent_tests["lift_ci_upper"] - recent_tests["lift_ci_lower"]
        ).mean()
        lift_mean = recent_tests["lift_estimate"].abs().mean()
        relative_ci = ci_width / (lift_mean + 1e-8)
        confidence = max(0, 1 - relative_ci)  # 0-1, higher = narrower CI
        
        coverage[channel] = {
            "has_test": True,
            "recency_days": recency,
            "test_count": len(recent_tests),
            "avg_confidence": confidence,
        }
    
    return coverage


def compute_blended_estimates(
    mmm_estimates: dict[str, float],
    test_estimates: dict[str, float],
    test_confidence: dict[str, float],
    min_test_confidence: float = 0.5,
) -> dict[str, float]:
    """
    Compute blended estimates based on test confidence.
    
    Higher test confidence = more weight on test result.
    Lower test confidence = more weight on MMM.
    
    Args:
        mmm_estimates: Channel -> MMM estimate
        test_estimates: Channel -> test estimate
        test_confidence: Channel -> confidence (0-1)
        min_test_confidence: Minimum confidence to use any test weight
    
    Returns:
        Blended estimates per channel
    """
    blended = {}
    
    for channel in mmm_estimates:
        mmm = mmm_estimates[channel]
        
        if channel in test_estimates and channel in test_confidence:
            test = test_estimates[channel]
            conf = test_confidence[channel]
            
            if conf >= min_test_confidence:
                # Blend based on confidence
                weight = conf
                blended[channel] = weight * test + (1 - weight) * mmm
            else:
                blended[channel] = mmm
        else:
            blended[channel] = mmm
    
    return blended


def create_calibration_report(
    mmm_estimates: dict[str, float],
    test_results: pd.DataFrame,
    calibration_factors: dict[str, float],
) -> pd.DataFrame:
    """
    Create a detailed calibration report.
    
    Shows MMM estimates, test results, calibration factors,
    and recommendations for each channel.
    """
    records = []
    
    for channel, mmm_est in mmm_estimates.items():
        channel_tests = test_results[test_results["channel"] == channel]
        
        record = {
            "channel": channel,
            "mmm_estimate": mmm_est,
            "has_test": len(channel_tests) > 0,
            "calibration_factor": calibration_factors.get(channel, 1.0),
        }
        
        if len(channel_tests) > 0:
            latest = channel_tests.sort_values("end_date").iloc[-1]
            record.update({
                "test_estimate": latest["lift_estimate"],
                "test_ci_lower": latest["lift_ci_lower"],
                "test_ci_upper": latest["lift_ci_upper"],
                "test_date": latest["end_date"],
                "test_type": latest["test_type"],
            })
            
            # Recommendation
            factor = calibration_factors[channel]
            if factor > 1.2:
                record["recommendation"] = "MMM underestimating - increase priors"
            elif factor < 0.8:
                record["recommendation"] = "MMM overestimating - decrease priors"
            else:
                record["recommendation"] = "MMM aligned with test"
        else:
            record["recommendation"] = "Schedule incrementality test"
        
        records.append(record)
    
    return pd.DataFrame(records)

