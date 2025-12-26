"""
Budget allocation optimizer using response curves.

Finds optimal spend allocation across channels to maximize
total response subject to budget and channel constraints.
"""

from dataclasses import dataclass, field
from typing import Callable
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from loguru import logger


@dataclass
class OptimizationResult:
    """
    Results from budget optimization.
    """
    
    # Optimal allocation
    optimal_allocation: dict[str, float] = field(default_factory=dict)
    
    # Expected outcomes
    expected_response: float = 0.0
    expected_roi: float = 0.0
    
    # Comparison to current
    current_allocation: dict[str, float] = field(default_factory=dict)
    current_response: float = 0.0
    improvement_pct: float = 0.0
    
    # Optimization details
    total_budget: float = 0.0
    success: bool = True
    message: str = ""
    iterations: int = 0
    
    # Per-channel metrics
    channel_metrics: pd.DataFrame | None = None
    
    def to_dict(self) -> dict:
        return {
            "optimal_allocation": self.optimal_allocation,
            "expected_response": self.expected_response,
            "expected_roi": self.expected_roi,
            "current_allocation": self.current_allocation,
            "current_response": self.current_response,
            "improvement_pct": self.improvement_pct,
            "total_budget": self.total_budget,
            "success": self.success,
            "message": self.message,
            "iterations": self.iterations,
        }
    
    def save(self, path: Path | str) -> None:
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        if self.channel_metrics is not None:
            self.channel_metrics.to_csv(
                path.with_suffix(".csv"), index=False
            )


class BudgetOptimizer:
    """
    Optimize budget allocation across channels.
    
    Uses response curves to find the allocation that maximizes
    total response subject to budget constraints.
    
    Example:
        >>> optimizer = BudgetOptimizer(
        ...     response_curves={
        ...         "google": lambda x: 1000 * (1 - np.exp(-x/5000)),
        ...         "meta": lambda x: 800 * (1 - np.exp(-x/4000)),
        ...     },
        ...     total_budget=50000,
        ... )
        >>> result = optimizer.optimize()
    """
    
    def __init__(
        self,
        response_curves: dict[str, Callable[[float], float]] | None = None,
        response_params: dict[str, dict] | None = None,
        total_budget: float = 100000,
        min_budget_pct: float = 0.0,
        max_budget_pct: float = 1.0,
        channel_constraints: dict[str, tuple[float, float]] | None = None,
    ):
        """
        Initialize optimizer.
        
        Args:
            response_curves: Dictionary of channel -> response function
                Each function takes spend and returns response
            response_params: Alternative to response_curves - dict of
                channel -> {"K": ..., "S": ...} for Hill saturation
            total_budget: Total budget to allocate
            min_budget_pct: Minimum % of budget per channel (0-1)
            max_budget_pct: Maximum % of budget per channel (0-1)
            channel_constraints: Per-channel (min, max) spend overrides
        """
        self.total_budget = total_budget
        self.min_budget_pct = min_budget_pct
        self.max_budget_pct = max_budget_pct
        self.channel_constraints = channel_constraints or {}
        
        # Build response functions
        if response_curves:
            self.response_curves = response_curves
            self.channels = list(response_curves.keys())
        elif response_params:
            self.response_curves = self._build_response_curves(response_params)
            self.channels = list(response_params.keys())
        else:
            self.response_curves = {}
            self.channels = []
        
        self._current_allocation: dict[str, float] | None = None
    
    def _build_response_curves(
        self,
        params: dict[str, dict],
    ) -> dict[str, Callable[[float], float]]:
        """Build response curves from Hill saturation parameters."""
        curves = {}
        
        for channel, p in params.items():
            K = p.get("K", 1000)
            S = p.get("S", 1.0)
            coef = p.get("coefficient", 1.0)
            
            # Hill saturation: coef * x^S / (K^S + x^S)
            def make_curve(K, S, coef):
                def curve(x):
                    x = max(0, x)
                    return coef * (x ** S) / (K ** S + x ** S)
                return curve
            
            curves[channel] = make_curve(K, S, coef)
        
        return curves
    
    def set_current_allocation(self, allocation: dict[str, float]) -> None:
        """Set current allocation for comparison."""
        self._current_allocation = allocation
    
    def compute_response(self, allocation: dict[str, float]) -> float:
        """Compute total response for an allocation."""
        total = 0.0
        
        for channel, spend in allocation.items():
            if channel in self.response_curves:
                total += self.response_curves[channel](spend)
        
        return total
    
    def optimize(
        self,
        method: str = "SLSQP",
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
    ) -> OptimizationResult:
        """
        Find optimal budget allocation.
        
        Args:
            method: Optimization method (SLSQP, trust-constr, etc.)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
        
        Returns:
            OptimizationResult with optimal allocation
        """
        if not self.channels:
            return OptimizationResult(
                success=False,
                message="No channels configured",
            )
        
        n_channels = len(self.channels)
        
        # Objective: maximize response (minimize negative response)
        def objective(x):
            allocation = dict(zip(self.channels, x))
            return -self.compute_response(allocation)
        
        # Constraint: total spend = budget
        def budget_constraint(x):
            return np.sum(x) - self.total_budget
        
        # Bounds per channel
        bounds = []
        for channel in self.channels:
            if channel in self.channel_constraints:
                min_spend, max_spend = self.channel_constraints[channel]
            else:
                min_spend = self.min_budget_pct * self.total_budget
                max_spend = self.max_budget_pct * self.total_budget
            bounds.append((min_spend, max_spend))
        
        # Initial guess: equal allocation
        x0 = np.full(n_channels, self.total_budget / n_channels)
        
        # Optimize
        logger.info(f"Optimizing budget allocation for {n_channels} channels...")
        
        result = optimize.minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            constraints={"type": "eq", "fun": budget_constraint},
            options={"maxiter": max_iterations, "ftol": tolerance},
        )
        
        # Extract results
        optimal_allocation = dict(zip(self.channels, result.x))
        expected_response = -result.fun
        
        # Compare to current allocation
        current_response = 0.0
        if self._current_allocation:
            current_response = self.compute_response(self._current_allocation)
        
        improvement = (
            (expected_response - current_response) / (current_response + 1e-8) * 100
            if current_response > 0 else 0
        )
        
        # Per-channel metrics
        channel_data = []
        for channel in self.channels:
            optimal_spend = optimal_allocation[channel]
            current_spend = self._current_allocation.get(channel, 0) if self._current_allocation else 0
            optimal_resp = self.response_curves[channel](optimal_spend)
            current_resp = self.response_curves[channel](current_spend) if current_spend > 0 else 0
            
            channel_data.append({
                "channel": channel,
                "current_spend": current_spend,
                "optimal_spend": optimal_spend,
                "spend_change": optimal_spend - current_spend,
                "spend_change_pct": (optimal_spend - current_spend) / (current_spend + 1e-8) * 100,
                "current_response": current_resp,
                "optimal_response": optimal_resp,
                "marginal_roi": self._compute_marginal_roi(channel, optimal_spend),
            })
        
        channel_metrics = pd.DataFrame(channel_data)
        
        opt_result = OptimizationResult(
            optimal_allocation=optimal_allocation,
            expected_response=expected_response,
            expected_roi=expected_response / self.total_budget,
            current_allocation=self._current_allocation or {},
            current_response=current_response,
            improvement_pct=improvement,
            total_budget=self.total_budget,
            success=result.success,
            message=result.message,
            iterations=result.nit,
            channel_metrics=channel_metrics,
        )
        
        logger.info(
            f"Optimization complete. Expected response: {expected_response:.2f}, "
            f"Improvement: {improvement:.1f}%"
        )
        
        return opt_result
    
    def _compute_marginal_roi(self, channel: str, spend: float, delta: float = 100) -> float:
        """Compute marginal ROI at given spend level."""
        if channel not in self.response_curves:
            return 0.0
        
        curve = self.response_curves[channel]
        return (curve(spend + delta) - curve(spend)) / delta
    
    def optimize_for_target(
        self,
        target_response: float,
        method: str = "SLSQP",
    ) -> OptimizationResult:
        """
        Find minimum budget to achieve target response.
        
        Instead of maximizing response for fixed budget,
        minimizes budget to achieve a target response level.
        
        Args:
            target_response: Target response to achieve
            method: Optimization method
        
        Returns:
            OptimizationResult with minimum budget allocation
        """
        n_channels = len(self.channels)
        
        # Objective: minimize total spend
        def objective(x):
            return np.sum(x)
        
        # Constraint: response >= target
        def response_constraint(x):
            allocation = dict(zip(self.channels, x))
            return self.compute_response(allocation) - target_response
        
        # Bounds
        bounds = [(0, self.total_budget * 2) for _ in self.channels]
        
        # Initial guess
        x0 = np.full(n_channels, self.total_budget / n_channels)
        
        result = optimize.minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            constraints={"type": "ineq", "fun": response_constraint},
        )
        
        optimal_allocation = dict(zip(self.channels, result.x))
        total_budget = np.sum(result.x)
        expected_response = self.compute_response(optimal_allocation)
        
        return OptimizationResult(
            optimal_allocation=optimal_allocation,
            expected_response=expected_response,
            expected_roi=expected_response / total_budget if total_budget > 0 else 0,
            total_budget=total_budget,
            success=result.success,
            message=f"Minimum budget for {target_response:.0f} response: {total_budget:.0f}",
        )


def optimize_budget(
    response_curves: dict[str, Callable[[float], float]],
    total_budget: float,
    current_allocation: dict[str, float] | None = None,
    min_budget_pct: float = 0.0,
    max_budget_pct: float = 1.0,
) -> OptimizationResult:
    """
    Convenience function for simple budget optimization.
    
    Args:
        response_curves: Channel -> response function mapping
        total_budget: Total budget to allocate
        current_allocation: Current spend for comparison
        min_budget_pct: Minimum % per channel
        max_budget_pct: Maximum % per channel
    
    Returns:
        OptimizationResult with optimal allocation
    """
    optimizer = BudgetOptimizer(
        response_curves=response_curves,
        total_budget=total_budget,
        min_budget_pct=min_budget_pct,
        max_budget_pct=max_budget_pct,
    )
    
    if current_allocation:
        optimizer.set_current_allocation(current_allocation)
    
    return optimizer.optimize()

