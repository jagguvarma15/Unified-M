"""
Power analysis for experiment planning.

Helps answer:
  - "How many weeks/geos do I need to detect a 10% lift?"
  - "What's the minimum detectable effect given my sample size?"
"""

from __future__ import annotations

import numpy as np
from loguru import logger


def compute_required_sample_size(
    baseline_mean: float,
    baseline_std: float,
    mde_pct: float = 0.10,
    alpha: float = 0.10,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """
    Compute required sample size (per group) for a t-test.

    Args:
        baseline_mean: Expected mean of the KPI in the control group.
        baseline_std:  Expected standard deviation of the KPI.
        mde_pct:       Minimum detectable effect as fraction (0.10 = 10%).
        alpha:         Significance level.
        power:         Statistical power (1 - Type II error rate).
        two_sided:     Whether the test is two-sided.

    Returns:
        Required number of observations per group.
    """
    from scipy import stats

    mde = baseline_mean * mde_pct

    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    n = ((z_alpha + z_beta) ** 2 * 2 * baseline_std ** 2) / (mde ** 2)
    n = int(np.ceil(n))

    logger.info(
        f"Power analysis: need {n} obs/group to detect {mde_pct:.0%} lift "
        f"(alpha={alpha}, power={power})"
    )
    return n


def compute_mde(
    baseline_mean: float,
    baseline_std: float,
    n_per_group: int,
    alpha: float = 0.10,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """
    Compute minimum detectable effect given a sample size.

    Args:
        baseline_mean: Expected KPI mean.
        baseline_std:  Expected KPI standard deviation.
        n_per_group:   Number of observations per group.
        alpha:         Significance level.
        power:         Statistical power.
        two_sided:     Two-sided test.

    Returns:
        MDE as a fraction of the baseline mean (e.g. 0.08 = 8%).
    """
    from scipy import stats

    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    mde_absolute = (z_alpha + z_beta) * np.sqrt(2 * baseline_std ** 2 / n_per_group)
    mde_pct = mde_absolute / baseline_mean if baseline_mean > 0 else 0.0

    logger.info(
        f"MDE analysis: with n={n_per_group}/group, MDE = {mde_pct:.1%} "
        f"(alpha={alpha}, power={power})"
    )
    return float(mde_pct)


def compute_experiment_duration(
    baseline_weekly_mean: float,
    baseline_weekly_std: float,
    n_geos_per_group: int,
    mde_pct: float = 0.10,
    alpha: float = 0.10,
    power: float = 0.80,
) -> int:
    """
    Compute how many weeks an experiment needs to run.

    For geo experiments with ``n_geos_per_group`` geos, each observed
    weekly. Uses the geo x week as the unit of observation.

    Args:
        baseline_weekly_mean:  Mean weekly KPI per geo.
        baseline_weekly_std:   Std of weekly KPI per geo.
        n_geos_per_group:      Number of geos in treatment/control.
        mde_pct:               Minimum detectable effect.
        alpha:                 Significance level.
        power:                 Statistical power.

    Returns:
        Required number of weeks.
    """
    total_n = compute_required_sample_size(
        baseline_mean=baseline_weekly_mean,
        baseline_std=baseline_weekly_std,
        mde_pct=mde_pct,
        alpha=alpha,
        power=power,
    )

    weeks = int(np.ceil(total_n / n_geos_per_group))
    weeks = max(weeks, 2)  # minimum 2 weeks

    logger.info(
        f"Experiment duration: {weeks} weeks with {n_geos_per_group} geos/group "
        f"to detect {mde_pct:.0%} lift"
    )
    return weeks
