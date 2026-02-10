"""
PII scanner -- checks DataFrames for columns that might contain
personally identifiable information.

Unified-M is designed for aggregated geo x week data only.
User-level data must never enter the system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from loguru import logger


# Column-name patterns that suggest PII
_PII_BLOCKLIST = [
    "email", "e_mail", "user_id", "userid", "customer_id", "customerid",
    "name", "first_name", "last_name", "full_name", "username",
    "phone", "telephone", "mobile", "cell",
    "ssn", "social_security", "national_id",
    "ip_address", "ip_addr", "ipaddress",
    "address", "street", "zip_code", "zipcode", "postal_code",
    "credit_card", "card_number", "cvv", "account_number",
    "password", "passwd", "secret", "token",
    "dob", "date_of_birth", "birthdate", "birth_date",
    "device_id", "deviceid", "idfa", "gaid", "aaid",
    "cookie_id", "cookieid", "session_id", "sessionid",
]

_PII_PATTERN = re.compile(
    "|".join(re.escape(p) for p in _PII_BLOCKLIST),
    re.IGNORECASE,
)


@dataclass
class PIIScanResult:
    """Result of scanning a DataFrame for PII columns."""

    has_pii: bool = False
    flagged_columns: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "has_pii": self.has_pii,
            "flagged_columns": self.flagged_columns,
            "message": self.message,
        }


def scan_for_pii(
    df: pd.DataFrame,
    source_name: str = "unknown",
) -> PIIScanResult:
    """
    Scan a DataFrame's column names for PII indicators.

    This is a heuristic check on column names only -- it does NOT
    inspect cell values (which would be too slow for large datasets
    and would itself risk exposing PII in logs).

    Args:
        df:          DataFrame to scan.
        source_name: Human-readable name for log messages.

    Returns:
        PIIScanResult indicating whether PII columns were found.
    """
    flagged = []
    for col in df.columns:
        col_lower = col.lower().replace("-", "_").replace(" ", "_")
        if _PII_PATTERN.search(col_lower):
            flagged.append(col)

    if flagged:
        msg = (
            f"[{source_name}] PII-risk columns detected: {flagged}. "
            f"Unified-M operates on aggregated data only. "
            f"Remove user-level columns before ingestion."
        )
        logger.warning(msg)
        return PIIScanResult(has_pii=True, flagged_columns=flagged, message=msg)

    return PIIScanResult(
        has_pii=False,
        flagged_columns=[],
        message=f"[{source_name}] No PII columns detected",
    )
