"""
Budget optimization module.
"""

from .allocator import BudgetOptimizer, OptimizationResult, optimize_budget
from .scenarios import (
    BudgetScenario,
    create_budget_scenarios,
    compare_scenarios,
    create_channel_shift_scenarios,
    compute_efficiency_frontier,
    find_diminishing_returns_point,
)

__all__ = [
    "BudgetOptimizer",
    "OptimizationResult",
    "optimize_budget",
    "BudgetScenario",
    "create_budget_scenarios",
    "compare_scenarios",
    "create_channel_shift_scenarios",
    "compute_efficiency_frontier",
    "find_diminishing_returns_point",
]
