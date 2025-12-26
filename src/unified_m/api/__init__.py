"""
FastAPI application for serving Unified-M results.

Provides low-latency access to precomputed MMM outputs,
reconciliation results, and optimization recommendations.
"""

from unified_m.api.app import create_app, app

__all__ = ["create_app", "app"]

