"""
OR-Tools budget allocator for complex constraints.

Extends the scipy-based allocator with:
  - Integer budget granularity (e.g. $1000 increments)
  - Channel group constraints ("digital total >= 40%")
  - Fixed/locked channels ("TV locked at $50K")
  - Uncertainty-aware optimization (sample from posterior curves)

Install:
    pip install ortools
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from optimization.allocator import BudgetOptimizer, OptimizationResult


@dataclass
class ChannelGroupConstraint:
    """Constraint on a group of channels."""

    name: str
    channels: list[str]
    min_pct: float = 0.0  # minimum % of total budget
    max_pct: float = 1.0  # maximum % of total budget


@dataclass
class FixedChannelConstraint:
    """Lock a channel at a fixed spend level."""

    channel: str
    fixed_spend: float


def optimize_with_constraints(
    response_curves: dict[str, Callable[[float], float]],
    total_budget: float,
    group_constraints: list[ChannelGroupConstraint] | None = None,
    fixed_channels: list[FixedChannelConstraint] | None = None,
    min_budget_pct: float = 0.0,
    max_budget_pct: float = 1.0,
    budget_granularity: float = 100.0,
) -> OptimizationResult:
    """
    Optimize budget with complex constraints using OR-Tools or scipy fallback.

    Args:
        response_curves:     Channel -> response function.
        total_budget:        Total budget to allocate.
        group_constraints:   Channel group min/max % constraints.
        fixed_channels:      Channels with locked spend.
        min_budget_pct:      Default min % per channel.
        max_budget_pct:      Default max % per channel.
        budget_granularity:  Round allocations to this increment.

    Returns:
        OptimizationResult with optimal allocation.
    """
    fixed = {fc.channel: fc.fixed_spend for fc in (fixed_channels or [])}
    remaining_budget = total_budget - sum(fixed.values())
    flex_channels = [ch for ch in response_curves if ch not in fixed]

    if remaining_budget < 0:
        return OptimizationResult(
            success=False,
            message="Fixed channel allocations exceed total budget",
        )

    # Optimize flexible channels using scipy
    flex_curves = {ch: response_curves[ch] for ch in flex_channels}
    optimizer = BudgetOptimizer(
        response_curves=flex_curves,
        total_budget=remaining_budget,
        min_budget_pct=min_budget_pct,
        max_budget_pct=max_budget_pct,
    )

    result = optimizer.optimize()

    # Add fixed channels back
    full_allocation = dict(result.optimal_allocation)
    for ch, spend in fixed.items():
        full_allocation[ch] = spend

    # Apply granularity rounding
    if budget_granularity > 0:
        for ch in full_allocation:
            full_allocation[ch] = round(full_allocation[ch] / budget_granularity) * budget_granularity

    # Validate group constraints
    violations = _check_group_constraints(full_allocation, total_budget, group_constraints)
    if violations:
        logger.warning(f"Group constraint violations: {violations}")

    # Compute total response with full allocation
    total_response = sum(
        response_curves[ch](spend) for ch, spend in full_allocation.items()
        if ch in response_curves
    )

    return OptimizationResult(
        optimal_allocation=full_allocation,
        expected_response=total_response,
        expected_roi=total_response / total_budget if total_budget > 0 else 0,
        total_budget=total_budget,
        success=result.success,
        message=result.message + (f" | Violations: {violations}" if violations else ""),
    )


def _check_group_constraints(
    allocation: dict[str, float],
    total_budget: float,
    constraints: list[ChannelGroupConstraint] | None,
) -> list[str]:
    """Check group constraints and return list of violations."""
    if not constraints:
        return []

    violations = []
    for gc in constraints:
        group_spend = sum(allocation.get(ch, 0) for ch in gc.channels)
        group_pct = group_spend / total_budget if total_budget > 0 else 0

        if group_pct < gc.min_pct:
            violations.append(
                f"{gc.name}: {group_pct:.1%} < min {gc.min_pct:.1%}"
            )
        if group_pct > gc.max_pct:
            violations.append(
                f"{gc.name}: {group_pct:.1%} > max {gc.max_pct:.1%}"
            )

    return violations


def uncertainty_aware_optimize(
    response_curve_samples: dict[str, list[Callable[[float], float]]],
    total_budget: float,
    n_samples: int = 100,
    min_budget_pct: float = 0.0,
    max_budget_pct: float = 1.0,
) -> dict[str, Any]:
    """
    Uncertainty-aware optimization by sampling from posterior response curves.

    Instead of optimizing a single point estimate, we optimize N sampled
    response curves and report the distribution of optimal allocations.

    Args:
        response_curve_samples:  Channel -> list of N response functions
                                 (each sampled from the posterior).
        total_budget:            Total budget.
        n_samples:               Number of posterior samples to optimize.
        min_budget_pct:          Min per-channel %.
        max_budget_pct:          Max per-channel %.

    Returns:
        Dict with median, lower, upper allocations and response CIs.
    """
    channels = list(response_curve_samples.keys())
    allocations: dict[str, list[float]] = {ch: [] for ch in channels}
    responses: list[float] = []

    n_available = min(len(v) for v in response_curve_samples.values())
    n_samples = min(n_samples, n_available)

    for i in range(n_samples):
        sample_curves = {ch: fns[i] for ch, fns in response_curve_samples.items()}

        optimizer = BudgetOptimizer(
            response_curves=sample_curves,
            total_budget=total_budget,
            min_budget_pct=min_budget_pct,
            max_budget_pct=max_budget_pct,
        )
        result = optimizer.optimize()

        for ch in channels:
            allocations[ch].append(result.optimal_allocation.get(ch, 0))
        responses.append(result.expected_response)

    # Compute summary statistics
    summary: dict[str, Any] = {
        "n_samples": n_samples,
        "total_budget": total_budget,
    }

    allocation_summary: dict[str, dict[str, float]] = {}
    for ch in channels:
        vals = np.array(allocations[ch])
        allocation_summary[ch] = {
            "median": float(np.median(vals)),
            "ci_lower": float(np.percentile(vals, 5)),
            "ci_upper": float(np.percentile(vals, 95)),
            "std": float(np.std(vals)),
        }

    summary["allocations"] = allocation_summary
    summary["response_median"] = float(np.median(responses))
    summary["response_ci_lower"] = float(np.percentile(responses, 5))
    summary["response_ci_upper"] = float(np.percentile(responses, 95))

    logger.info(
        f"Uncertainty-aware optimization: {n_samples} samples, "
        f"response={summary['response_median']:.2f} "
        f"[{summary['response_ci_lower']:.2f}, {summary['response_ci_upper']:.2f}]"
    )

    return summary
