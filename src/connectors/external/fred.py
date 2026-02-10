"""
FRED (Federal Reserve Economic Data) connector.

Fetches macro economic indicators like consumer confidence,
unemployment rate, CPI, etc.

Requires: FRED_API_KEY env var (free at https://fred.stlouisfed.org/docs/api/)

Fallback: download CSV from FRED website -> data/raw/fred/
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from loguru import logger


# Common FRED series for marketing measurement
DEFAULT_SERIES = {
    "UMCSENT": "ctrl_consumer_confidence",      # Univ of Michigan Consumer Sentiment
    "UNRATE": "ctrl_unemployment_rate",          # Unemployment Rate
    "CPIAUCSL": "ctrl_cpi",                      # Consumer Price Index
    "RSXFS": "ctrl_retail_sales",                # Advance Retail Sales
}


class FREDConnector:
    """Fetch macro indicators from FRED API."""

    def __init__(
        self,
        api_key: str | None = None,
        series_map: dict[str, str] | None = None,
    ):
        self.api_key = api_key or os.getenv("FRED_API_KEY", "")
        self.series_map = series_map or DEFAULT_SERIES

    def fetch(
        self,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch all configured series and merge into one DataFrame.

        Returns DataFrame with: date + one column per series.
        """
        import requests

        if not self.api_key:
            logger.warning("FRED_API_KEY not set. Use file-drop fallback (data/raw/fred/).")
            return pd.DataFrame(columns=["date"])

        frames = []
        for series_id, col_name in self.series_map.items():
            logger.info(f"Fetching FRED series: {series_id}")
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date,
            }
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            obs = data.get("observations", [])
            df = pd.DataFrame(obs)
            if len(df) > 0:
                df = df[["date", "value"]].copy()
                df["date"] = pd.to_datetime(df["date"])
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.rename(columns={"value": col_name})
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["date"])

        result = frames[0]
        for f in frames[1:]:
            result = result.merge(f, on="date", how="outer")

        result = result.sort_values("date").reset_index(drop=True)
        logger.info(f"Fetched {len(result)} observations from FRED")
        return result

    def load_from_file(self, path: str) -> pd.DataFrame:
        """File-drop fallback."""
        logger.info(f"Loading FRED data from file: {path}")
        return pd.read_csv(path, parse_dates=["date"])
