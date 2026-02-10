"""
External data connectors for control variables.

  - Weather (Open-Meteo API -- free, no key required)
  - Macro indicators (FRED API -- free key)
  - Holiday calendars (``holidays`` Python package -- offline)
"""

from connectors.external.weather import WeatherConnector
from connectors.external.fred import FREDConnector
from connectors.external.holidays import HolidayConnector

__all__ = ["WeatherConnector", "FREDConnector", "HolidayConnector"]
