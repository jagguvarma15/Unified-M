"""
Experiment analysis module for Unified-M.

Provides tools for analyzing incrementality experiments:
  - Geo-lift (CausalImpact-style synthetic control)
  - Switchback experiments
  - Power analysis for experiment planning
"""

from experiments.geo_lift import GeoLiftAnalyzer
from experiments.switchback import SwitchbackAnalyzer
from experiments.power_analysis import compute_required_sample_size, compute_mde

__all__ = [
    "GeoLiftAnalyzer",
    "SwitchbackAnalyzer",
    "compute_required_sample_size",
    "compute_mde",
]
