"""
Pluggable data loaders for various sources.

Supports Parquet, CSV, DuckDB, and extensible to APIs/databases.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import polars as pl
from loguru import logger


class BaseLoader(ABC):
    """
    Abstract base class for data loaders.
    
    Extend this class to create custom loaders for specific data sources.
    """
    
    @abstractmethod
    def load(self, source: str | Path, **kwargs) -> pd.DataFrame:
        """Load data from source and return as pandas DataFrame."""
        pass
    
    @abstractmethod
    def load_polars(self, source: str | Path, **kwargs) -> pl.DataFrame:
        """Load data from source and return as polars DataFrame."""
        pass
    
    def validate_source(self, source: str | Path) -> Path:
        """Validate that the source exists."""
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        return path


class ParquetLoader(BaseLoader):
    """
    Loader for Parquet files.
    
    Supports single files and directories of partitioned Parquet.
    """
    
    def load(self, source: str | Path, **kwargs) -> pd.DataFrame:
        """Load Parquet file(s) into pandas DataFrame."""
        path = self.validate_source(source)
        logger.info(f"Loading Parquet from {path}")
        
        if path.is_dir():
            # Load all parquet files in directory
            return pd.read_parquet(path, **kwargs)
        return pd.read_parquet(path, **kwargs)
    
    def load_polars(self, source: str | Path, **kwargs) -> pl.DataFrame:
        """Load Parquet file(s) into polars DataFrame."""
        path = self.validate_source(source)
        logger.info(f"Loading Parquet (polars) from {path}")
        
        if path.is_dir():
            return pl.read_parquet(f"{path}/*.parquet", **kwargs)
        return pl.read_parquet(path, **kwargs)


class CSVLoader(BaseLoader):
    """
    Loader for CSV files.
    
    Supports single files and directories of CSVs.
    """
    
    def __init__(self, delimiter: str = ",", encoding: str = "utf-8"):
        self.delimiter = delimiter
        self.encoding = encoding
    
    def load(self, source: str | Path, **kwargs) -> pd.DataFrame:
        """Load CSV file(s) into pandas DataFrame."""
        path = self.validate_source(source)
        logger.info(f"Loading CSV from {path}")
        
        if path.is_dir():
            # Load and concatenate all CSV files
            dfs = []
            for csv_file in path.glob("*.csv"):
                dfs.append(pd.read_csv(
                    csv_file,
                    delimiter=self.delimiter,
                    encoding=self.encoding,
                    **kwargs
                ))
            return pd.concat(dfs, ignore_index=True)
        
        return pd.read_csv(
            path,
            delimiter=self.delimiter,
            encoding=self.encoding,
            **kwargs
        )
    
    def load_polars(self, source: str | Path, **kwargs) -> pl.DataFrame:
        """Load CSV file(s) into polars DataFrame."""
        path = self.validate_source(source)
        logger.info(f"Loading CSV (polars) from {path}")
        
        if path.is_dir():
            return pl.read_csv(f"{path}/*.csv", **kwargs)
        return pl.read_csv(path, **kwargs)


class DuckDBLoader(BaseLoader):
    """
    Loader using DuckDB for SQL-based data access.
    
    Can query Parquet files directly or connect to DuckDB databases.
    """
    
    def __init__(self, database: str | Path | None = None):
        """
        Initialize DuckDB loader.
        
        Args:
            database: Path to DuckDB database file, or None for in-memory.
        """
        self.database = database
        self._conn: duckdb.DuckDBPyConnection | None = None
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create DuckDB connection."""
        if self._conn is None:
            if self.database:
                self._conn = duckdb.connect(str(self.database))
            else:
                self._conn = duckdb.connect()
        return self._conn
    
    def load(self, source: str | Path, **kwargs) -> pd.DataFrame:
        """
        Load data using DuckDB query.
        
        If source is a path, queries the file directly.
        If source is a SQL string, executes the query.
        """
        source_str = str(source)
        
        # Check if source is a SQL query or file path
        if source_str.strip().upper().startswith("SELECT"):
            logger.info("Executing DuckDB query")
            return self.conn.execute(source_str).df()
        
        # Source is a file path
        path = self.validate_source(source)
        logger.info(f"Loading via DuckDB from {path}")
        
        if path.suffix == ".parquet":
            return self.conn.execute(f"SELECT * FROM '{path}'").df()
        elif path.suffix == ".csv":
            return self.conn.execute(f"SELECT * FROM read_csv_auto('{path}')").df()
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
    def load_polars(self, source: str | Path, **kwargs) -> pl.DataFrame:
        """Load data via DuckDB and convert to polars."""
        return pl.from_pandas(self.load(source, **kwargs))
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary SQL query."""
        logger.info(f"Executing query: {sql[:100]}...")
        return self.conn.execute(sql).df()
    
    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class DataIngestion:
    """
    High-level data ingestion orchestrator.
    
    Manages loading from multiple sources and combines into unified datasets.
    """
    
    def __init__(self):
        self.parquet = ParquetLoader()
        self.csv = CSVLoader()
        self._duckdb: DuckDBLoader | None = None
    
    @property
    def duckdb(self) -> DuckDBLoader:
        """Lazy-initialize DuckDB loader."""
        if self._duckdb is None:
            self._duckdb = DuckDBLoader()
        return self._duckdb
    
    def load_media_spend(
        self,
        source: str | Path,
        loader_type: str = "auto",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load media spend data from source.
        
        Args:
            source: Path to data file or directory
            loader_type: One of 'auto', 'parquet', 'csv', 'duckdb'
        """
        loader = self._get_loader(source, loader_type)
        df = loader.load(source, **kwargs)
        
        # Ensure date column is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        logger.info(f"Loaded media spend: {len(df)} rows, {df['channel'].nunique() if 'channel' in df.columns else 'N/A'} channels")
        return df
    
    def load_outcomes(
        self,
        source: str | Path,
        loader_type: str = "auto",
        **kwargs
    ) -> pd.DataFrame:
        """Load outcome/response data."""
        loader = self._get_loader(source, loader_type)
        df = loader.load(source, **kwargs)
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        logger.info(f"Loaded outcomes: {len(df)} rows")
        return df
    
    def load_controls(
        self,
        source: str | Path,
        loader_type: str = "auto",
        **kwargs
    ) -> pd.DataFrame:
        """Load control variable data."""
        loader = self._get_loader(source, loader_type)
        df = loader.load(source, **kwargs)
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        logger.info(f"Loaded controls: {len(df)} rows, {len(df.columns) - 1} variables")
        return df
    
    def load_incrementality(
        self,
        source: str | Path,
        loader_type: str = "auto",
        **kwargs
    ) -> pd.DataFrame:
        """Load incrementality test results."""
        loader = self._get_loader(source, loader_type)
        df = loader.load(source, **kwargs)
        
        for col in ["start_date", "end_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        logger.info(f"Loaded incrementality tests: {len(df)} tests")
        return df
    
    def load_attribution(
        self,
        source: str | Path,
        loader_type: str = "auto",
        **kwargs
    ) -> pd.DataFrame:
        """Load attribution data."""
        loader = self._get_loader(source, loader_type)
        df = loader.load(source, **kwargs)
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        logger.info(f"Loaded attribution: {len(df)} rows")
        return df
    
    def _get_loader(self, source: str | Path, loader_type: str) -> BaseLoader:
        """Get appropriate loader based on type or auto-detect."""
        if loader_type == "auto":
            path = Path(source)
            if path.suffix == ".parquet" or (path.is_dir() and any(path.glob("*.parquet"))):
                return self.parquet
            elif path.suffix == ".csv" or (path.is_dir() and any(path.glob("*.csv"))):
                return self.csv
            else:
                return self.duckdb
        
        loaders = {
            "parquet": self.parquet,
            "csv": self.csv,
            "duckdb": self.duckdb,
        }
        
        if loader_type not in loaders:
            raise ValueError(f"Unknown loader type: {loader_type}")
        
        return loaders[loader_type]


# =============================================================================
# Convenience Functions
# =============================================================================

def load_data(
    source: str | Path,
    loader_type: str = "auto",
    **kwargs
) -> pd.DataFrame:
    """
    Simple function to load data from any supported source.
    
    Args:
        source: Path to file or directory
        loader_type: 'auto', 'parquet', 'csv', or 'duckdb'
        **kwargs: Additional arguments passed to the loader
    
    Returns:
        pandas DataFrame with loaded data
    """
    ingestion = DataIngestion()
    loader = ingestion._get_loader(source, loader_type)
    return loader.load(source, **kwargs)


def save_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Save DataFrame to Parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, **kwargs)
    logger.info(f"Saved {len(df)} rows to {path}")


def save_parquet_polars(df: pl.DataFrame, path: str | Path, **kwargs) -> None:
    """Save polars DataFrame to Parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, **kwargs)
    logger.info(f"Saved {len(df)} rows to {path}")

