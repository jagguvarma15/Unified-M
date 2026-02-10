"""
Data connectors for Unified-M.

Supports CSV, Parquet, Excel, DuckDB, databases, and cloud storage.
"""

from .local import (
    BaseConnector,
    CSVConnector,
    ParquetConnector,
    ExcelConnector,
    DuckDBConnector,
    auto_connect,
    load_file,
)
from .database import (
    DatabaseConnector,
    PostgreSQLConnector,
    MySQLConnector,
    SQLServerConnector,
    SQLiteConnector,
    create_database_connector,
)
from .cloud import (
    CloudStorageConnector,
    S3Connector,
    AzureBlobConnector,
    create_cloud_connector,
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
