"""
Meta (Facebook / Instagram) Ads connector.

Fetches campaign-level spend via the Facebook Marketing API
(``facebook-business`` SDK).

Local fallback:
    Export from Meta Ads Manager as CSV -> ``data/raw/meta_ads/``

Required env vars:
    META_APP_ID
    META_APP_SECRET
    META_ACCESS_TOKEN
    META_AD_ACCOUNT_ID
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from loguru import logger

from connectors.ad_platforms._base import AdPlatformConnector


class MetaAdsConnector(AdPlatformConnector):
    """Fetch campaign spend from Meta Marketing API."""

    platform_name = "meta"

    def __init__(
        self,
        app_id: str | None = None,
        app_secret: str | None = None,
        access_token: str | None = None,
        ad_account_id: str | None = None,
        **kwargs: Any,
    ):
        self.app_id = app_id or os.getenv("META_APP_ID", "")
        self.app_secret = app_secret or os.getenv("META_APP_SECRET", "")
        self.access_token = access_token or os.getenv("META_ACCESS_TOKEN", "")
        self.ad_account_id = ad_account_id or os.getenv("META_AD_ACCOUNT_ID", "")

    def fetch(
        self,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        try:
            from facebook_business.api import FacebookAdsApi
            from facebook_business.adobjects.adaccount import AdAccount
        except ImportError:
            raise ImportError(
                "facebook-business is not installed. Run: pip install facebook-business"
            )

        FacebookAdsApi.init(self.app_id, self.app_secret, self.access_token)
        account = AdAccount(f"act_{self.ad_account_id}")

        logger.info(f"Fetching Meta Ads data: {start_date} to {end_date}")

        insights = account.get_insights(
            params={
                "time_range": {"since": start_date, "until": end_date},
                "time_increment": 1,  # daily
                "level": "campaign",
            },
            fields=["date_start", "spend", "impressions", "clicks", "campaign_name"],
        )

        records = []
        for row in insights:
            campaign_name = row.get("campaign_name", "")
            # Map to sub-channel based on campaign naming convention
            if "instagram" in campaign_name.lower():
                channel = "meta_instagram"
            else:
                channel = "meta_facebook"

            records.append({
                "date": row["date_start"],
                "channel": channel,
                "spend": float(row.get("spend", 0)),
                "impressions": int(row.get("impressions", 0)),
                "clicks": int(row.get("clicks", 0)),
            })

        df = pd.DataFrame(records)
        logger.info(f"Fetched {len(df)} rows from Meta Ads")
        return self.normalize(df)

    def test_connection(self) -> bool:
        try:
            from facebook_business.api import FacebookAdsApi
            return bool(self.access_token and self.ad_account_id)
        except ImportError:
            return False
