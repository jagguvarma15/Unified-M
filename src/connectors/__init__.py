"""
Data connectors for Unified-M.

Supports CSV, Parquet, Excel, DuckDB, databases, and cloud storage.
"""

from .cloud import (
    AzureBlobConnector,
    CloudStorageConnector,
    S3Connector,
    create_cloud_connector,
)
from .database import (
    DatabaseConnector,
    MySQLConnector,
    PostgreSQLConnector,
    SQLiteConnector,
    SQLServerConnector,
    create_database_connector,
)
from .local import (
    BaseConnector,
    CSVConnector,
    DuckDBConnector,
    ExcelConnector,
    ParquetConnector,
    auto_connect,
    load_file,
)

__all__ = [
    "BaseConnector",
    "CSVConnector",
    "ParquetConnector",
    "ExcelConnector",
    "DuckDBConnector",
    "auto_connect",
    "load_file",
    "DatabaseConnector",
    "PostgreSQLConnector",
    "MySQLConnector",
    "SQLServerConnector",
    "SQLiteConnector",
    "create_database_connector",
    "CloudStorageConnector",
    "S3Connector",
    "AzureBlobConnector",
    "create_cloud_connector",
]
