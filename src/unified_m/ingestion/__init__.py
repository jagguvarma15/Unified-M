"""
Data ingestion layer for Unified-M.

Pluggable loaders for various data sources (CSV, Parquet, databases, APIs).
"""

from unified_m.ingestion.loaders import (
    BaseLoader,
    ParquetLoader,
    CSVLoader,
    DuckDBLoader,
    load_data,
)

__all__ = [
    "BaseLoader",
    "ParquetLoader",
    "CSVLoader",
    "DuckDBLoader",
    "load_data",
]

