"""
TikTok Ads connector.

Fetches campaign-level spend via the TikTok Marketing API.

Local fallback:
    Export from TikTok Ads Manager as CSV -> ``data/raw/tiktok_ads/``

Required env vars:
    TIKTOK_ACCESS_TOKEN
    TIKTOK_ADVERTISER_ID
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from loguru import logger

from connectors.ad_platforms._base import AdPlatformConnector


class TikTokAdsConnector(AdPlatformConnector):
    """Fetch campaign spend from TikTok Marketing API."""

    platform_name = "tiktok"

    def __init__(
        self,
        access_token: str | None = None,
        advertiser_id: str | None = None,
        **kwargs: Any,
    ):
        self.access_token = access_token or os.getenv("TIKTOK_ACCESS_TOKEN", "")
        self.advertiser_id = advertiser_id or os.getenv("TIKTOK_ADVERTISER_ID", "")

    def fetch(
        self,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        import requests

        url = "https://business-api.tiktok.com/open_api/v1.3/report/integrated/get/"
        headers = {"Access-Token": self.access_token}

        logger.info(f"Fetching TikTok Ads data: {start_date} to {end_date}")

        payload = {
            "advertiser_id": self.advertiser_id,
            "report_type": "BASIC",
            "data_level": "AUCTION_CAMPAIGN",
            "dimensions": ["campaign_id", "stat_time_day"],
            "metrics": ["spend", "impressions", "clicks"],
            "start_date": start_date,
            "end_date": end_date,
            "page_size": 1000,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        records = []
        for row in data.get("data", {}).get("list", []):
            dims = row.get("dimensions", {})
            metrics = row.get("metrics", {})
            records.append({
                "date": dims.get("stat_time_day", ""),
                "channel": "tiktok",
                "spend": float(metrics.get("spend", 0)),
                "impressions": int(metrics.get("impressions", 0)),
                "clicks": int(metrics.get("clicks", 0)),
            })

        df = pd.DataFrame(records)
        logger.info(f"Fetched {len(df)} rows from TikTok Ads")
        return self.normalize(df)

    def test_connection(self) -> bool:
        return bool(self.access_token and self.advertiser_id)
