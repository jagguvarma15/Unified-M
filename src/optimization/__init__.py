"""
Budget optimization layer for Unified-M.

Provides optimal budget allocation based on response curves
and constraints using scipy.optimize.
"""

from optimization.allocator import (
    BudgetOptimizer,
    OptimizationResult,
    optimize_budget,
)
from optimization.scenarios import (
    create_budget_scenarios,
    compare_scenarios,
)

__all__ = [
    "BudgetOptimizer",
    "OptimizationResult",
    "optimize_budget",
    "create_budget_scenarios",
    "compare_scenarios",
]

