"""
Amazon Ads connector.

Fetches Sponsored Products / Brands / Display spend via the
Amazon Advertising API.

Local fallback:
    Download Sponsored Ads reports as CSV -> ``data/raw/amazon_ads/``

Required env vars:
    AMAZON_ADS_CLIENT_ID
    AMAZON_ADS_CLIENT_SECRET
    AMAZON_ADS_REFRESH_TOKEN
    AMAZON_ADS_PROFILE_ID
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from loguru import logger

from connectors.ad_platforms._base import AdPlatformConnector


class AmazonAdsConnector(AdPlatformConnector):
    """Fetch campaign spend from Amazon Advertising API."""

    platform_name = "amazon_ads"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        refresh_token: str | None = None,
        profile_id: str | None = None,
        **kwargs: Any,
    ):
        self.client_id = client_id or os.getenv("AMAZON_ADS_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("AMAZON_ADS_CLIENT_SECRET", "")
        self.refresh_token = refresh_token or os.getenv("AMAZON_ADS_REFRESH_TOKEN", "")
        self.profile_id = profile_id or os.getenv("AMAZON_ADS_PROFILE_ID", "")

    def fetch(
        self,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch campaign metrics from Amazon Advertising API.

        Note: The Amazon Ads API requires a multi-step report workflow
        (request report -> poll status -> download). This is a simplified
        implementation. For production use, consider the ``amazon-ads``
        Python SDK or Airbyte source-amazon-ads.
        """
        import requests

        logger.info(f"Fetching Amazon Ads data: {start_date} to {end_date}")

        # Step 1: Get access token
        token_url = "https://api.amazon.com/auth/o2/token"
        token_resp = requests.post(token_url, data={
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }, timeout=30)
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]

        # Step 2: Request report (simplified -- real impl needs polling)
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Amazon-Advertising-API-ClientId": self.client_id,
            "Amazon-Advertising-API-Scope": self.profile_id,
        }

        # For now, return empty DataFrame with correct schema
        # Full implementation would request/poll/download the report
        logger.warning(
            "Amazon Ads API connector is a stub. "
            "Use file-drop (data/raw/amazon_ads/) or Airbyte for production."
        )

        return self.normalize(pd.DataFrame(columns=["date", "channel", "spend", "impressions", "clicks"]))

    def test_connection(self) -> bool:
        return bool(self.client_id and self.refresh_token and self.profile_id)
