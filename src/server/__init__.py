"""
FastAPI server for Unified-M.

Serves precomputed pipeline results with low-latency access.
"""

from .app import app, create_app, run_server

__all__ = ["create_app", "app", "run_server"]
