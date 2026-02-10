"""
Base class for all ad-platform connectors.

Every ad-platform connector must:
  1. Implement ``fetch(start_date, end_date)`` to pull data from the API.
  2. Return a DataFrame conforming to the MediaSpendInput schema.
  3. Support a local file-drop fallback via ``load_from_file()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


class AdPlatformConnector(ABC):
    """Abstract base for ad-platform connectors."""

    platform_name: str = "unknown"

    @abstractmethod
    def fetch(
        self,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch data from the platform API.

        Returns:
            DataFrame with at least: date, channel, spend
        """
        ...

    @abstractmethod
    def test_connection(self) -> bool:
        """Test whether the API credentials are valid."""
        ...

    def load_from_file(self, path: str | Path) -> pd.DataFrame:
        """
        File-drop fallback: load from CSV or Parquet.

        Expected location: ``data/raw/{platform_name}/``
        """
        path = Path(path)
        logger.info(f"Loading {self.platform_name} data from file: {path}")

        if path.is_dir():
            frames = []
            for ext in ["*.parquet", "*.csv"]:
                for f in sorted(path.glob(ext)):
                    if f.suffix == ".parquet":
                        frames.append(pd.read_parquet(f))
                    else:
                        frames.append(pd.read_csv(f))
            if not frames:
                raise FileNotFoundError(f"No data files in {path}")
            return self.normalize(pd.concat(frames, ignore_index=True))

        if path.suffix == ".parquet":
            return self.normalize(pd.read_parquet(path))
        return self.normalize(pd.read_csv(path))

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names and add platform metadata.

        Ensures output has: date, channel, spend, impressions (optional),
        clicks (optional).
        """
        # Standardize column names
        col_map = {
            "Date": "date",
            "DATE": "date",
            "Channel": "channel",
            "CHANNEL": "channel",
            "Spend": "spend",
            "SPEND": "spend",
            "Cost": "spend",
            "cost": "spend",
            "cost_micros": "spend",
            "Impressions": "impressions",
            "IMPRESSIONS": "impressions",
            "Clicks": "clicks",
            "CLICKS": "clicks",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Ensure required columns
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if "channel" not in df.columns:
            df["channel"] = self.platform_name

        df["_source"] = self.platform_name
        return df
