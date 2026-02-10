"""
Ad-platform connectors for Unified-M.

Each connector wraps the platform SDK / API and normalises output to
the canonical MediaSpendInput schema.  All connectors have a file-drop
fallback: place CSV/Parquet in ``data/raw/{platform}/``.

Supported platforms:
  - Google Ads  (google-ads SDK)
  - Meta / Facebook  (facebook-business SDK)
  - TikTok Ads  (Business API)
  - Amazon Ads  (Amazon Advertising API)
"""

from connectors.ad_platforms.google_ads import GoogleAdsConnector
from connectors.ad_platforms.meta_ads import MetaAdsConnector
from connectors.ad_platforms.tiktok_ads import TikTokAdsConnector
from connectors.ad_platforms.amazon_ads import AmazonAdsConnector

__all__ = [
    "GoogleAdsConnector",
    "MetaAdsConnector",
    "TikTokAdsConnector",
    "AmazonAdsConnector",
]
