"""
Holiday calendar connector.

Uses the ``holidays`` Python package (fully offline, no API calls).
Generates a binary holiday indicator per date/geo.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger


class HolidayConnector:
    """Generate holiday indicators using the ``holidays`` package."""

    def __init__(self, country: str = "US", state: str | None = None):
        self.country = country
        self.state = state

    def generate(
        self,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Generate daily holiday flags for a date range.

        Returns DataFrame with: date, ctrl_is_holiday, holiday_name
        """
        try:
            import holidays as holidays_lib
        except ImportError:
            raise ImportError(
                "holidays package not installed. Run: pip install holidays"
            )

        logger.info(f"Generating holiday calendar: {start_date} to {end_date} ({self.country})")

        if self.state:
            cal = holidays_lib.country_holidays(self.country, state=self.state)
        else:
            cal = holidays_lib.country_holidays(self.country)

        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        records = []
        for d in dates:
            is_holiday = d in cal
            records.append({
                "date": d,
                "ctrl_is_holiday": int(is_holiday),
                "holiday_name": cal.get(d, ""),
            })

        df = pd.DataFrame(records)
        n_holidays = df["ctrl_is_holiday"].sum()
        logger.info(f"Generated calendar: {len(df)} days, {n_holidays} holidays")
        return df

    def generate_weekly(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate and aggregate to ISO weeks (1 if any day is a holiday)."""
        df = self.generate(start_date, end_date)
        df["week_start"] = df["date"].dt.to_period("W").dt.start_time
        weekly = df.groupby("week_start").agg(
            ctrl_is_holiday=("ctrl_is_holiday", "max"),
        ).reset_index()
        return weekly
