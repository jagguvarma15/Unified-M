"""
Weather data connector using Open-Meteo API (free, no key needed).

Fetches daily temperature and precipitation for a given location,
then aggregates to weekly for the measurement mart.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger


class WeatherConnector:
    """Fetch historical weather data from Open-Meteo."""

    def __init__(
        self,
        latitude: float = 39.8283,   # US centroid default
        longitude: float = -98.5795,
    ):
        self.latitude = latitude
        self.longitude = longitude

    def fetch(
        self,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch daily weather from Open-Meteo archive API.

        Returns DataFrame with: date, temperature_f, precipitation_in
        """
        import requests

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean,precipitation_sum",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "timezone": "America/New_York",
        }

        logger.info(f"Fetching weather data: {start_date} to {end_date}")
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        df = pd.DataFrame({
            "date": pd.to_datetime(daily.get("time", [])),
            "ctrl_temperature_f": daily.get("temperature_2m_mean", []),
            "ctrl_precipitation_in": daily.get("precipitation_sum", []),
        })

        logger.info(f"Fetched {len(df)} days of weather data")
        return df

    def fetch_weekly(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch and aggregate to ISO weeks."""
        df = self.fetch(start_date, end_date)
        df["week_start"] = df["date"].dt.to_period("W").dt.start_time
        weekly = df.groupby("week_start").agg(
            ctrl_temperature_f=("ctrl_temperature_f", "mean"),
            ctrl_precipitation_in=("ctrl_precipitation_in", "sum"),
        ).reset_index()
        return weekly
