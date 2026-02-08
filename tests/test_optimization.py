"""Tests for budget optimization."""

import numpy as np
import pytest

from optimization import BudgetOptimizer, optimize_budget


class TestBudgetOptimizer:
    """Test budget optimization."""
    
    def test_basic_optimization(self):
        """Test basic budget optimization."""
        # Simple response curves
        response_curves = {
            "google": lambda x: 1000 * (1 - np.exp(-x / 5000)),
            "meta": lambda x: 800 * (1 - np.exp(-x / 4000)),
        }
        
        optimizer = BudgetOptimizer(
            response_curves=response_curves,
            total_budget=10000,
        )
        
        result = optimizer.optimize()
        
        assert result.success
        assert len(result.optimal_allocation) == 2
        assert sum(result.optimal_allocation.values()) == pytest.approx(10000, rel=0.01)
    
    def test_budget_constraint(self):
        """Test that total budget is respected."""
        response_curves = {
            "ch1": lambda x: x ** 0.5,
            "ch2": lambda x: x ** 0.5,
            "ch3": lambda x: x ** 0.5,
        }
        
        total_budget = 50000
        
        optimizer = BudgetOptimizer(
            response_curves=response_curves,
            total_budget=total_budget,
        )
        
        result = optimizer.optimize()
        
        actual_total = sum(result.optimal_allocation.values())
        assert actual_total == pytest.approx(total_budget, rel=0.01)
    
    def test_channel_constraints(self):
        """Test per-channel min/max constraints."""
        response_curves = {
            "google": lambda x: x ** 0.5,
            "meta": lambda x: x ** 0.5,
        }
        
        optimizer = BudgetOptimizer(
            response_curves=response_curves,
            total_budget=10000,
            channel_constraints={
                "google": (3000, 7000),  # Google must be between 3k-7k
            },
        )
        
        result = optimizer.optimize()
        
        assert result.optimal_allocation["google"] >= 3000
        assert result.optimal_allocation["google"] <= 7000
    
    def test_response_params_input(self):
        """Test initialization with response params instead of curves."""
        response_params = {
            "google": {"K": 10000, "S": 1.0, "coefficient": 1000},
            "meta": {"K": 8000, "S": 1.2, "coefficient": 800},
        }
        
        optimizer = BudgetOptimizer(
            response_params=response_params,
            total_budget=20000,
        )
        
        result = optimizer.optimize()
        
        assert result.success
        assert "google" in result.optimal_allocation
        assert "meta" in result.optimal_allocation
    
    def test_compute_response(self):
        """Test response computation for given allocation."""
        response_curves = {
            "ch1": lambda x: x * 0.1,
            "ch2": lambda x: x * 0.2,
        }
        
        optimizer = BudgetOptimizer(
            response_curves=response_curves,
            total_budget=10000,
        )
        
        allocation = {"ch1": 1000, "ch2": 2000}
        response = optimizer.compute_response(allocation)
        
        expected = 1000 * 0.1 + 2000 * 0.2
        assert response == pytest.approx(expected)
    
    def test_improvement_calculation(self):
        """Test improvement calculation vs current allocation."""
        # Inefficient curves to create clear optimization potential
        response_curves = {
            "high_roi": lambda x: x * 2,  # 2x ROI
            "low_roi": lambda x: x * 0.5,  # 0.5x ROI
        }
        
        optimizer = BudgetOptimizer(
            response_curves=response_curves,
            total_budget=10000,
            min_budget_pct=0.0,
            max_budget_pct=1.0,
        )
        
        # Set suboptimal current allocation
        optimizer.set_current_allocation({
            "high_roi": 2000,  # Under-invested
            "low_roi": 8000,  # Over-invested
        })
        
        result = optimizer.optimize()
        
        # Optimizer should shift budget to high_roi
        assert result.optimal_allocation["high_roi"] > 2000
        assert result.improvement_pct > 0


class TestOptimizeBudgetFunction:
    """Test convenience function."""
    
    def test_optimize_budget_function(self):
        """Test the optimize_budget convenience function."""
        response_curves = {
            "google": lambda x: 500 * (1 - np.exp(-x / 5000)),
            "meta": lambda x: 400 * (1 - np.exp(-x / 4000)),
        }
        
        result = optimize_budget(
            response_curves=response_curves,
            total_budget=15000,
        )
        
        assert result.success
        assert result.total_budget == 15000
        assert result.expected_response > 0

