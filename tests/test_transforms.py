"""Tests for transform functions."""

import numpy as np
import pandas as pd
import pytest

from transforms import (
    geometric_adstock,
    weibull_adstock,
    hill_saturation,
    logistic_saturation,
    apply_adstock,
    apply_saturation,
    create_mmm_features,
    pivot_media_spend,
)


class TestAdstock:
    """Test adstock transformations."""
    
    def test_geometric_adstock_basic(self):
        """Test basic geometric adstock."""
        x = np.array([100, 0, 0, 0, 0])
        result = geometric_adstock(x, alpha=0.5, l_max=5, normalize=False)
        
        # First value should be 100 (immediate effect)
        assert result[0] == pytest.approx(100, rel=0.01)
        
        # Effect should decay
        assert result[1] < result[0]
        assert result[2] < result[1]
    
    def test_geometric_adstock_no_carryover(self):
        """Test with alpha=0 (no carryover)."""
        x = np.array([100, 50, 0, 0, 0])
        result = geometric_adstock(x, alpha=0.0, l_max=5, normalize=True)
        
        # Should be same as input (no carryover)
        np.testing.assert_array_almost_equal(result, x)
    
    def test_geometric_adstock_shape(self):
        """Test output shape matches input."""
        x = np.random.rand(100)
        result = geometric_adstock(x, alpha=0.7, l_max=8)
        
        assert len(result) == len(x)
    
    def test_weibull_adstock_shape(self):
        """Test Weibull adstock shape."""
        x = np.random.rand(100)
        result = weibull_adstock(x, shape=2.0, scale=2.0, l_max=8)
        
        assert len(result) == len(x)


class TestSaturation:
    """Test saturation transformations."""
    
    def test_hill_saturation_bounds(self):
        """Test Hill saturation stays in [0, 1]."""
        x = np.linspace(0, 10000, 100)
        result = hill_saturation(x, K=1000, S=1.0)
        
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_hill_saturation_at_zero(self):
        """Test Hill saturation is 0 at x=0."""
        result = hill_saturation(np.array([0]), K=1000, S=1.0)
        
        assert result[0] == 0
    
    def test_hill_saturation_at_k(self):
        """Test Hill saturation is 0.5 at x=K."""
        K = 1000
        result = hill_saturation(np.array([K]), K=K, S=1.0)
        
        assert result[0] == pytest.approx(0.5, rel=0.01)
    
    def test_hill_saturation_monotonic(self):
        """Test Hill saturation is monotonically increasing."""
        x = np.linspace(0, 10000, 100)
        result = hill_saturation(x, K=1000, S=1.0)
        
        assert np.all(np.diff(result) >= 0)
    
    def test_logistic_saturation_shape(self):
        """Test logistic saturation shape."""
        x = np.linspace(0, 10000, 100)
        result = logistic_saturation(x, L=1.0, k=0.001, x0=5000)
        
        assert len(result) == len(x)


class TestApplyTransforms:
    """Test batch transform applications."""
    
    def test_apply_adstock(self):
        """Test applying adstock to multiple columns."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "google_spend": np.random.rand(10) * 1000,
            "meta_spend": np.random.rand(10) * 800,
        })
        
        params = {
            "google_spend": {"alpha": 0.5},
            "meta_spend": {"alpha": 0.7},
        }
        
        result = apply_adstock(df, ["google_spend", "meta_spend"], params)
        
        assert "google_spend_adstock" in result.columns
        assert "meta_spend_adstock" in result.columns
    
    def test_apply_saturation(self):
        """Test applying saturation to multiple columns."""
        df = pd.DataFrame({
            "google_spend": np.linspace(0, 10000, 10),
            "meta_spend": np.linspace(0, 8000, 10),
        })
        
        params = {
            "google_spend": {"K": 5000, "S": 1.0},
            "meta_spend": {"K": 4000, "S": 1.2},
        }
        
        result = apply_saturation(df, ["google_spend", "meta_spend"], params)
        
        assert "google_spend_saturated" in result.columns
        assert "meta_spend_saturated" in result.columns


class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    def test_pivot_media_spend(self):
        """Test pivoting long-format media data."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "channel": ["google", "meta", "google", "meta"],
            "spend": [100, 200, 150, 250],
        })
        
        result = pivot_media_spend(df)
        
        assert "date" in result.columns
        assert "google_spend" in result.columns
        assert "meta_spend" in result.columns
        assert len(result) == 2  # 2 unique dates
    
    def test_create_mmm_features(self):
        """Test full feature creation pipeline."""
        media_spend = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "channel": ["google", "meta", "google", "meta"],
            "spend": [100, 200, 150, 250],
        })
        
        outcomes = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "revenue": [5000, 6000],
        })
        
        result = create_mmm_features(
            media_spend=media_spend,
            outcomes=outcomes,
            target_col="revenue",
        )
        
        assert "date" in result.columns
        assert "y" in result.columns
        assert "google_spend" in result.columns
        assert "meta_spend" in result.columns
        assert len(result) == 2

