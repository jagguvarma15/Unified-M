"""
Local file connectors for CSV, Parquet, DuckDB, and SQLite.

All connectors follow the same contract:
  - ``load(source, **kw) -> pd.DataFrame``
  - ``save(df, dest, **kw) -> Path``

``auto_connect(path)`` picks the right connector based on file extension.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from core.exceptions import ConnectorError


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseConnector(ABC):
    """Interface that every local connector implements."""

    @abstractmethod
    def load(self, source: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Read data from *source* into a DataFrame."""
        ...

    @abstractmethod
    def save(self, df: pd.DataFrame, dest: str | Path, **kwargs: Any) -> Path:
        """Write *df* to *dest* and return the resolved path."""
        ...

    @staticmethod
    def _ensure_path(source: str | Path) -> Path:
        p = Path(source)
        if not p.exists():
            raise ConnectorError(f"Source not found: {source}", source=str(source))
        return p

    @staticmethod
    def _ensure_parent(dest: str | Path) -> Path:
        p = Path(dest)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

class CSVConnector(BaseConnector):
    """Read / write CSV files or directories of CSVs."""

    def __init__(self, delimiter: str = ",", encoding: str = "utf-8"):
        self.delimiter = delimiter
        self.encoding = encoding

    def load(self, source: str | Path, **kwargs: Any) -> pd.DataFrame:
        path = self._ensure_path(source)
        logger.info(f"Loading CSV from {path}")

        if path.is_dir():
            frames = [
                pd.read_csv(f, delimiter=self.delimiter, encoding=self.encoding, **kwargs)
                for f in sorted(path.glob("*.csv"))
            ]
            if not frames:
                raise ConnectorError(f"No CSV files in {path}", source=str(path))
            return pd.concat(frames, ignore_index=True)

        return pd.read_csv(path, delimiter=self.delimiter, encoding=self.encoding, **kwargs)

    def save(self, df: pd.DataFrame, dest: str | Path, **kwargs: Any) -> Path:
        path = self._ensure_parent(dest)
        df.to_csv(path, index=False, **kwargs)
        logger.info(f"Saved {len(df)} rows to {path}")
        return path


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

class ParquetConnector(BaseConnector):
    """Read / write Parquet files (single file or Hive-partitioned dir)."""

    def load(self, source: str | Path, **kwargs: Any) -> pd.DataFrame:
        path = self._ensure_path(source)
        logger.info(f"Loading Parquet from {path}")
        return pd.read_parquet(path, **kwargs)

    def save(self, df: pd.DataFrame, dest: str | Path, **kwargs: Any) -> Path:
        path = self._ensure_parent(dest)
        df.to_parquet(path, index=False, **kwargs)
        logger.info(f"Saved {len(df)} rows to {path}")
        return path


# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------

class DuckDBConnector(BaseConnector):
    """
    Use DuckDB for SQL-based file access or as a local database.

    Can query CSV / Parquet files directly *or* connect to a .duckdb file.
    """

    def __init__(self, database: str | Path | None = None):
        self._database = database
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            try:
                import duckdb
            except ImportError:
                raise ConnectorError("duckdb is not installed. Run: pip install duckdb")
            self._conn = duckdb.connect(str(self._database) if self._database else ":memory:")
        return self._conn

    def load(self, source: str | Path, **kwargs: Any) -> pd.DataFrame:
        src = str(source)

        # If it looks like SQL, execute directly
        if src.strip().upper().startswith("SELECT"):
            logger.info("Executing DuckDB SQL query")
            return self.conn.execute(src).df()

        path = self._ensure_path(source)
        logger.info(f"Loading via DuckDB from {path}")

        if path.suffix == ".parquet":
            return self.conn.execute(f"SELECT * FROM '{path}'").df()
        elif path.suffix == ".csv":
            return self.conn.execute(f"SELECT * FROM read_csv_auto('{path}')").df()
        else:
            raise ConnectorError(f"Unsupported file type for DuckDB: {path.suffix}", source=src)

    def save(self, df: pd.DataFrame, dest: str | Path, **kwargs: Any) -> Path:
        path = self._ensure_parent(dest)
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")
        return path

    def query(self, sql: str) -> pd.DataFrame:
        """Execute an arbitrary SQL statement and return a DataFrame."""
        return self.conn.execute(sql).df()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

_EXTENSION_MAP: dict[str, type[BaseConnector]] = {
    ".csv": CSVConnector,
    ".tsv": CSVConnector,
    ".parquet": ParquetConnector,
    ".pq": ParquetConnector,
    ".duckdb": DuckDBConnector,
    ".db": DuckDBConnector,
}


def auto_connect(source: str | Path) -> BaseConnector:
    """
    Return the correct connector based on file extension.

    Falls back to ``ParquetConnector`` for directories that contain
    ``.parquet`` files, ``CSVConnector`` for directories with ``.csv``.
    """
    path = Path(source)

    if path.is_dir():
        if any(path.glob("*.parquet")):
            return ParquetConnector()
        if any(path.glob("*.csv")):
            return CSVConnector()
        raise ConnectorError(f"Cannot auto-detect format for directory: {path}", source=str(path))

    ext = path.suffix.lower()
    cls = _EXTENSION_MAP.get(ext)
    if cls is None:
        raise ConnectorError(f"Unsupported file extension: {ext}", source=str(path))

    if cls is CSVConnector and ext == ".tsv":
        return CSVConnector(delimiter="\t")

    return cls()


def load_file(source: str | Path, **kwargs: Any) -> pd.DataFrame:
    """One-liner: auto-detect format and return a DataFrame."""
    connector = auto_connect(source)
    return connector.load(source, **kwargs)
