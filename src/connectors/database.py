"""
Database connectors for Unified-M.

Supports PostgreSQL, MySQL, SQL Server, SQLite, and cloud databases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from loguru import logger

from core.exceptions import ConnectorError


class DatabaseConnector(ABC):
    """Base class for database connectors."""

    def __init__(self, connection_string: str | None = None, **kwargs: Any):
        self.connection_string = connection_string
        self.kwargs = kwargs
        self._conn = None

    @abstractmethod
    def connect(self) -> Any:
        """Establish database connection."""
        ...

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if connection is valid."""
        ...

    def load(self, query: str, **kwargs: Any) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        if self._conn is None:
            self._conn = self.connect()
        logger.info(f"Executing query: {query[:100]}...")
        return pd.read_sql(query, self._conn, **kwargs)

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


class PostgreSQLConnector(DatabaseConnector):
    """PostgreSQL database connector."""

    def __init__(self, host: str, port: int = 5432, database: str = "", 
                 user: str = "", password: str = "", **kwargs: Any):
        try:
            import psycopg2
        except ImportError:
            raise ConnectorError(
                "psycopg2 is not installed. Run: pip install psycopg2-binary"
            )
        
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        super().__init__(**kwargs)

    def connect(self) -> Any:
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                **self.kwargs
            )
            logger.info(f"Connected to PostgreSQL: {self.host}:{self.port}/{self.database}")
            return conn
        except Exception as e:
            raise ConnectorError(f"Failed to connect to PostgreSQL: {e}")

    def test_connection(self) -> bool:
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.close()
            return True
        except Exception as e:
            logger.error(f"PostgreSQL connection test failed: {e}")
            return False


class MySQLConnector(DatabaseConnector):
    """MySQL/MariaDB database connector."""

    def __init__(self, host: str, port: int = 3306, database: str = "",
                 user: str = "", password: str = "", **kwargs: Any):
        try:
            import pymysql
        except ImportError:
            raise ConnectorError(
                "pymysql is not installed. Run: pip install pymysql"
            )
        
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        super().__init__(**kwargs)

    def connect(self) -> Any:
        try:
            import pymysql
            conn = pymysql.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                **self.kwargs
            )
            logger.info(f"Connected to MySQL: {self.host}:{self.port}/{self.database}")
            return conn
        except Exception as e:
            raise ConnectorError(f"Failed to connect to MySQL: {e}")

    def test_connection(self) -> bool:
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.close()
            return True
        except Exception as e:
            logger.error(f"MySQL connection test failed: {e}")
            return False


class SQLServerConnector(DatabaseConnector):
    """SQL Server database connector."""

    def __init__(self, server: str, database: str = "", user: str = "",
                 password: str = "", driver: str = "ODBC Driver 17 for SQL Server", **kwargs: Any):
        try:
            import pyodbc
        except ImportError:
            raise ConnectorError(
                "pyodbc is not installed. Run: pip install pyodbc"
            )
        
        self.server = server
        self.database = database
        self.user = user
        self.password = password
        self.driver = driver
        super().__init__(**kwargs)

    def connect(self) -> Any:
        try:
            import pyodbc
            conn_str = (
                f"DRIVER={{{self.driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.user};"
                f"PWD={self.password}"
            )
            conn = pyodbc.connect(conn_str, **self.kwargs)
            logger.info(f"Connected to SQL Server: {self.server}/{self.database}")
            return conn
        except Exception as e:
            raise ConnectorError(f"Failed to connect to SQL Server: {e}")

    def test_connection(self) -> bool:
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.close()
            return True
        except Exception as e:
            logger.error(f"SQL Server connection test failed: {e}")
            return False


class SQLiteConnector(DatabaseConnector):
    """SQLite database connector."""

    def __init__(self, database: str, **kwargs: Any):
        try:
            import sqlite3
        except ImportError:
            raise ConnectorError("sqlite3 should be available in standard library")
        
        self.database = database
        super().__init__(**kwargs)

    def connect(self) -> Any:
        try:
            import sqlite3
            conn = sqlite3.connect(self.database, **self.kwargs)
            logger.info(f"Connected to SQLite: {self.database}")
            return conn
        except Exception as e:
            raise ConnectorError(f"Failed to connect to SQLite: {e}")

    def test_connection(self) -> bool:
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.close()
            return True
        except Exception as e:
            logger.error(f"SQLite connection test failed: {e}")
            return False


def create_database_connector(
    db_type: str,
    **kwargs: Any
) -> DatabaseConnector:
    """Factory function to create appropriate database connector."""
    db_type_lower = db_type.lower()
    
    if db_type_lower in ["postgresql", "postgres"]:
        return PostgreSQLConnector(**kwargs)
    elif db_type_lower in ["mysql", "mariadb"]:
        return MySQLConnector(**kwargs)
    elif db_type_lower in ["sqlserver", "mssql", "sql server"]:
        return SQLServerConnector(**kwargs)
    elif db_type_lower == "sqlite":
        return SQLiteConnector(**kwargs)
    else:
        raise ConnectorError(f"Unsupported database type: {db_type}")
