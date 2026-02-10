"""
Google Ads connector.

Fetches campaign-level spend, impressions, and clicks via the
Google Ads API (``google-ads`` SDK).

Local fallback:
    Place CSV exports from Google Ads Editor in ``data/raw/google_ads/``.

Required env vars:
    GOOGLE_ADS_DEVELOPER_TOKEN
    GOOGLE_ADS_CLIENT_ID
    GOOGLE_ADS_CLIENT_SECRET
    GOOGLE_ADS_REFRESH_TOKEN
    GOOGLE_ADS_CUSTOMER_ID
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from loguru import logger

from connectors.ad_platforms._base import AdPlatformConnector


class GoogleAdsConnector(AdPlatformConnector):
    """Fetch campaign spend from Google Ads API."""

    platform_name = "google_ads"

    def __init__(
        self,
        customer_id: str | None = None,
        developer_token: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        refresh_token: str | None = None,
        **kwargs: Any,
    ):
        self.customer_id = customer_id or os.getenv("GOOGLE_ADS_CUSTOMER_ID", "")
        self.developer_token = developer_token or os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN", "")
        self.client_id = client_id or os.getenv("GOOGLE_ADS_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("GOOGLE_ADS_CLIENT_SECRET", "")
        self.refresh_token = refresh_token or os.getenv("GOOGLE_ADS_REFRESH_TOKEN", "")

    def fetch(
        self,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch campaign metrics from Google Ads API.

        Returns DataFrame with columns: date, channel, spend, impressions, clicks
        """
        try:
            from google.ads.googleads.client import GoogleAdsClient
        except ImportError:
            raise ImportError(
                "google-ads is not installed. Run: pip install google-ads"
            )

        credentials = {
            "developer_token": self.developer_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "use_proto_plus": True,
        }

        client = GoogleAdsClient.load_from_dict(credentials)
        ga_service = client.get_service("GoogleAdsService")

        query = f"""
            SELECT
                segments.date,
                campaign.advertising_channel_type,
                metrics.cost_micros,
                metrics.impressions,
                metrics.clicks
            FROM campaign
            WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
        """

        logger.info(f"Fetching Google Ads data: {start_date} to {end_date}")
        stream = ga_service.search_stream(
            customer_id=self.customer_id,
            query=query,
        )

        records = []
        for batch in stream:
            for row in batch.results:
                records.append({
                    "date": row.segments.date,
                    "channel": f"google_{row.campaign.advertising_channel_type.name.lower()}",
                    "spend": row.metrics.cost_micros / 1_000_000,
                    "impressions": row.metrics.impressions,
                    "clicks": row.metrics.clicks,
                })

        df = pd.DataFrame(records)
        logger.info(f"Fetched {len(df)} rows from Google Ads")
        return self.normalize(df)

    def test_connection(self) -> bool:
        try:
            from google.ads.googleads.client import GoogleAdsClient
            return bool(self.developer_token and self.customer_id)
        except ImportError:
            return False
