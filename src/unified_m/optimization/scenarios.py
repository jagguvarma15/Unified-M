"""
Scenario planning utilities for budget optimization.

Create and compare different budget scenarios to support
strategic planning and what-if analysis.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from unified_m.optimization.allocator import BudgetOptimizer, OptimizationResult


@dataclass
class BudgetScenario:
    """
    A budget scenario for comparison.
    """
    
    name: str
    description: str
    total_budget: float
    allocation: dict[str, float]
    expected_response: float
    expected_roi: float
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "total_budget": self.total_budget,
            "allocation": self.allocation,
            "expected_response": self.expected_response,
            "expected_roi": self.expected_roi,
        }


def create_budget_scenarios(
    response_curves: dict[str, Callable[[float], float]],
    base_budget: float,
    current_allocation: dict[str, float] | None = None,
    budget_multipliers: list[float] | None = None,
) -> list[BudgetScenario]:
    """
    Create multiple budget scenarios for comparison.
    
    Args:
        response_curves: Channel -> response function mapping
        base_budget: Base budget level
        current_allocation: Current allocation (used for "current" scenario)
        budget_multipliers: List of multipliers (e.g., [0.8, 1.0, 1.2])
    
    Returns:
        List of BudgetScenario objects
    """
    if budget_multipliers is None:
        budget_multipliers = [0.7, 0.85, 1.0, 1.15, 1.3]
    
    scenarios = []
    
    # Current scenario (if provided)
    if current_allocation:
        current_response = sum(
            response_curves[ch](spend) 
            for ch, spend in current_allocation.items()
            if ch in response_curves
        )
        current_budget = sum(current_allocation.values())
        
        scenarios.append(BudgetScenario(
            name="Current",
            description="Current allocation",
            total_budget=current_budget,
            allocation=current_allocation,
            expected_response=current_response,
            expected_roi=current_response / current_budget if current_budget > 0 else 0,
        ))
    
    # Optimized scenarios at different budget levels
    for mult in budget_multipliers:
        budget = base_budget * mult
        
        optimizer = BudgetOptimizer(
            response_curves=response_curves,
            total_budget=budget,
        )
        
        result = optimizer.optimize()
        
        scenarios.append(BudgetScenario(
            name=f"Optimized ({mult:.0%})",
            description=f"Optimal allocation at {mult:.0%} of base budget",
            total_budget=budget,
            allocation=result.optimal_allocation,
            expected_response=result.expected_response,
            expected_roi=result.expected_roi,
        ))
    
    return scenarios


def compare_scenarios(
    scenarios: list[BudgetScenario],
) -> pd.DataFrame:
    """
    Create a comparison table of scenarios.
    
    Args:
        scenarios: List of BudgetScenario objects
    
    Returns:
        DataFrame comparing all scenarios
    """
    records = []
    
    # Get all channels
    all_channels = set()
    for s in scenarios:
        all_channels.update(s.allocation.keys())
    
    for scenario in scenarios:
        record = {
            "scenario": scenario.name,
            "description": scenario.description,
            "total_budget": scenario.total_budget,
            "expected_response": scenario.expected_response,
            "expected_roi": scenario.expected_roi,
        }
        
        # Add per-channel allocations
        for channel in sorted(all_channels):
            record[f"{channel}_spend"] = scenario.allocation.get(channel, 0)
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Add comparison columns
    if len(df) > 0:
        base_response = df["expected_response"].iloc[0]
        df["response_vs_base"] = (
            (df["expected_response"] - base_response) / base_response * 100
        )
    
    return df


def create_channel_shift_scenarios(
    response_curves: dict[str, Callable[[float], float]],
    current_allocation: dict[str, float],
    shift_channel: str,
    shift_amounts: list[float] | None = None,
) -> list[BudgetScenario]:
    """
    Create scenarios shifting budget to/from a specific channel.
    
    Useful for answering "what if we increase Google spend by 20%?"
    
    Args:
        response_curves: Channel -> response function
        current_allocation: Current allocation
        shift_channel: Channel to shift budget to/from
        shift_amounts: List of shift percentages (e.g., [-20, -10, 10, 20])
    
    Returns:
        List of scenarios with shifted budgets
    """
    if shift_amounts is None:
        shift_amounts = [-30, -20, -10, 10, 20, 30]
    
    scenarios = []
    total_budget = sum(current_allocation.values())
    other_channels = [ch for ch in current_allocation if ch != shift_channel]
    
    # Current scenario
    current_response = sum(
        response_curves[ch](spend)
        for ch, spend in current_allocation.items()
        if ch in response_curves
    )
    
    scenarios.append(BudgetScenario(
        name="Current",
        description="Current allocation",
        total_budget=total_budget,
        allocation=current_allocation,
        expected_response=current_response,
        expected_roi=current_response / total_budget,
    ))
    
    # Shifted scenarios
    for shift_pct in shift_amounts:
        shift_amount = current_allocation.get(shift_channel, 0) * (shift_pct / 100)
        
        new_allocation = current_allocation.copy()
        new_allocation[shift_channel] = max(
            0, 
            current_allocation.get(shift_channel, 0) + shift_amount
        )
        
        # Redistribute to other channels (proportionally)
        other_total = sum(current_allocation.get(ch, 0) for ch in other_channels)
        if other_total > 0:
            for ch in other_channels:
                proportion = current_allocation.get(ch, 0) / other_total
                new_allocation[ch] = max(
                    0,
                    current_allocation.get(ch, 0) - shift_amount * proportion
                )
        
        response = sum(
            response_curves[ch](spend)
            for ch, spend in new_allocation.items()
            if ch in response_curves
        )
        
        scenarios.append(BudgetScenario(
            name=f"{shift_channel} {shift_pct:+}%",
            description=f"Shift {shift_pct:+}% to {shift_channel}",
            total_budget=total_budget,
            allocation=new_allocation,
            expected_response=response,
            expected_roi=response / total_budget,
        ))
    
    return scenarios


def compute_efficiency_frontier(
    response_curves: dict[str, Callable[[float], float]],
    budget_range: tuple[float, float],
    n_points: int = 20,
) -> pd.DataFrame:
    """
    Compute the efficiency frontier (optimal response at each budget level).
    
    Args:
        response_curves: Channel -> response function
        budget_range: (min_budget, max_budget)
        n_points: Number of points on the frontier
    
    Returns:
        DataFrame with budget, optimal response, and optimal allocation
    """
    budgets = np.linspace(budget_range[0], budget_range[1], n_points)
    
    records = []
    
    for budget in budgets:
        optimizer = BudgetOptimizer(
            response_curves=response_curves,
            total_budget=budget,
        )
        
        result = optimizer.optimize()
        
        record = {
            "budget": budget,
            "optimal_response": result.expected_response,
            "roi": result.expected_roi,
        }
        
        # Add allocation
        for channel, spend in result.optimal_allocation.items():
            record[f"{channel}_spend"] = spend
            record[f"{channel}_pct"] = spend / budget * 100
        
        records.append(record)
    
    return pd.DataFrame(records)


def find_diminishing_returns_point(
    response_curves: dict[str, Callable[[float], float]],
    channel: str,
    threshold: float = 0.1,
    max_spend: float = 1000000,
) -> float:
    """
    Find the spend level where marginal ROI drops below threshold.
    
    Useful for identifying saturation points.
    
    Args:
        response_curves: Channel -> response function
        channel: Channel to analyze
        threshold: Marginal ROI threshold
        max_spend: Maximum spend to consider
    
    Returns:
        Spend level at diminishing returns point
    """
    if channel not in response_curves:
        return 0
    
    curve = response_curves[channel]
    
    # Search for point where marginal ROI < threshold
    delta = 100
    
    for spend in np.linspace(0, max_spend, 1000):
        marginal = (curve(spend + delta) - curve(spend)) / delta
        if marginal < threshold:
            return spend
    
    return max_spend

