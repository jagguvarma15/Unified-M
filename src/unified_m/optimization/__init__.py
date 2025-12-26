"""
Budget optimization layer for Unified-M.

Provides optimal budget allocation based on response curves
and constraints using scipy.optimize.
"""

from unified_m.optimization.allocator import (
    BudgetOptimizer,
    OptimizationResult,
    optimize_budget,
)
from unified_m.optimization.scenarios import (
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

