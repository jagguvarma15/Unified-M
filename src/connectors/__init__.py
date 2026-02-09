"""
Data connectors for Unified-M.

Supports CSV, Parquet, and DuckDB for local-first data access.
"""

from .local import (
    BaseConnector,
    CSVConnector,
    ParquetConnector,
    DuckDBConnector,
    auto_connect,
    load_file,
)

__all__ = [
    "BaseConnector",
    "CSVConnector",
    "ParquetConnector",
    "DuckDBConnector",
    "auto_connect",
    "load_file",
]
