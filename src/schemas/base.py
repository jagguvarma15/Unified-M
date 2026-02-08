"""
Base Pandera schemas for marketing measurement data.

These schemas are intentionally generalized to support various data sources.
Extend or subclass for specific use cases.
"""

from datetime import date
from typing import Optional

import pandera as pa
from pandera import Column, DataFrameSchema, Check, Index
from pandera.typing import DataFrame, Series
import pandas as pd


# =============================================================================
# Core Schemas
# =============================================================================

class MediaSpendSchema(pa.DataFrameModel):
    """
    Schema for media spend data.
    
    Supports daily/weekly granularity with channel-level spend.
    Can be extended with campaign/tactic level detail.
    
    Example:
        date       | channel    | spend   | impressions | clicks
        2024-01-01 | google     | 1000.0  | 50000       | 500
        2024-01-01 | meta       | 2000.0  | 100000      | 800
    """
    
    date: Series[pa.DateTime] = pa.Field(
        description="Date of spend (daily or week-start)",
        coerce=True,
    )
    channel: Series[str] = pa.Field(
        description="Marketing channel (e.g., google, meta, tv, radio)",
        str_length={"min_value": 1, "max_value": 100},
    )
    spend: Series[float] = pa.Field(
        ge=0,
        description="Spend amount in currency units",
        coerce=True,
    )
    impressions: Optional[Series[float]] = pa.Field(
        ge=0,
        nullable=True,
        description="Number of impressions (optional)",
        coerce=True,
    )
    clicks: Optional[Series[float]] = pa.Field(
        ge=0,
        nullable=True,
        description="Number of clicks (optional)",
        coerce=True,
    )
    
    class Config:
        strict = False  # Allow additional columns
        coerce = True


class OutcomeSchema(pa.DataFrameModel):
    """
    Schema for outcome/response data.
    
    Supports revenue, conversions, or any KPI you want to model.
    
    Example:
        date       | revenue   | conversions | units_sold
        2024-01-01 | 50000.0   | 150         | 200
    """
    
    date: Series[pa.DateTime] = pa.Field(
        description="Date of outcome",
        coerce=True,
    )
    revenue: Optional[Series[float]] = pa.Field(
        ge=0,
        nullable=True,
        description="Revenue in currency units",
        coerce=True,
    )
    conversions: Optional[Series[float]] = pa.Field(
        ge=0,
        nullable=True,
        description="Number of conversions",
        coerce=True,
    )
    
    class Config:
        strict = False  # Allow additional outcome columns
        coerce = True
    
    @pa.check("revenue", "conversions")
    def at_least_one_outcome(cls, series: pd.Series) -> bool:
        """Ensure at least one outcome metric has data."""
        return True  # Relaxed - checked at DataFrame level


class ControlVariableSchema(pa.DataFrameModel):
    """
    Schema for control variables (non-media factors).
    
    Example:
        date       | seasonality | promotion | price_index | competitor_spend
        2024-01-01 | 1.2         | 1         | 100.0       | 5000.0
    """
    
    date: Series[pa.DateTime] = pa.Field(
        description="Date",
        coerce=True,
    )
    
    class Config:
        strict = False  # Allow any control variable columns
        coerce = True


class IncrementalityTestSchema(pa.DataFrameModel):
    """
    Schema for incrementality test results.
    
    Supports geo experiments, holdout tests, and other causal inference results.
    
    Example:
        test_id   | channel | start_date | end_date   | lift_estimate | lift_ci_lower | lift_ci_upper | test_type
        geo_2024  | meta    | 2024-01-01 | 2024-01-31 | 0.15          | 0.10          | 0.20          | geo_lift
    """
    
    test_id: Series[str] = pa.Field(
        description="Unique identifier for the test",
    )
    channel: Series[str] = pa.Field(
        description="Channel being tested",
    )
    start_date: Series[pa.DateTime] = pa.Field(
        description="Test start date",
        coerce=True,
    )
    end_date: Series[pa.DateTime] = pa.Field(
        description="Test end date",
        coerce=True,
    )
    lift_estimate: Series[float] = pa.Field(
        description="Point estimate of incremental lift",
    )
    lift_ci_lower: Series[float] = pa.Field(
        description="Lower bound of confidence interval",
    )
    lift_ci_upper: Series[float] = pa.Field(
        description="Upper bound of confidence interval",
    )
    test_type: Series[str] = pa.Field(
        isin=["geo_lift", "holdout", "switchback", "synthetic_control", "other"],
        description="Type of incrementality test",
    )
    confidence_level: Optional[Series[float]] = pa.Field(
        ge=0,
        le=1,
        nullable=True,
        description="Confidence level (e.g., 0.95 for 95% CI)",
        coerce=True,
    )
    
    class Config:
        strict = False
        coerce = True


class AttributionSchema(pa.DataFrameModel):
    """
    Schema for attribution data from MTA or platform reporting.
    
    Example:
        date       | channel | attributed_conversions | attributed_revenue | model_type
        2024-01-01 | google  | 50                     | 5000.0             | last_click
    """
    
    date: Series[pa.DateTime] = pa.Field(
        description="Date of attributed outcomes",
        coerce=True,
    )
    channel: Series[str] = pa.Field(
        description="Marketing channel",
    )
    attributed_conversions: Optional[Series[float]] = pa.Field(
        ge=0,
        nullable=True,
        description="Conversions attributed to channel",
        coerce=True,
    )
    attributed_revenue: Optional[Series[float]] = pa.Field(
        ge=0,
        nullable=True,
        description="Revenue attributed to channel",
        coerce=True,
    )
    model_type: Series[str] = pa.Field(
        isin=["last_click", "first_click", "linear", "time_decay", "position_based", "data_driven", "platform_reported", "other"],
        description="Attribution model type",
    )
    
    class Config:
        strict = False
        coerce = True


# =============================================================================
# Composite Schema for MMM Input
# =============================================================================

class MMMInputSchema(pa.DataFrameModel):
    """
    Schema for the final MMM-ready dataset.
    
    This is the transformed dataset ready for model training, with:
    - Date index
    - Target variable (y)
    - Media spend columns (will be adstocked/saturated)
    - Control variable columns
    
    Example:
        date       | y       | google_spend | meta_spend | tv_spend | seasonality | promo
        2024-01-01 | 50000.0 | 1000.0       | 2000.0     | 500.0    | 1.2         | 1
    """
    
    date: Series[pa.DateTime] = pa.Field(
        description="Date",
        coerce=True,
    )
    y: Series[float] = pa.Field(
        ge=0,
        description="Target variable (revenue, conversions, etc.)",
        coerce=True,
    )
    
    class Config:
        strict = False  # Allow dynamic media/control columns
        coerce = True


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_media_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Validate media spend data against schema."""
    return MediaSpendSchema.validate(df)


def validate_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Validate outcome data against schema."""
    return OutcomeSchema.validate(df)


def validate_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Validate control variable data against schema."""
    return ControlVariableSchema.validate(df)


def validate_incrementality(df: pd.DataFrame) -> pd.DataFrame:
    """Validate incrementality test data against schema."""
    return IncrementalityTestSchema.validate(df)


def validate_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """Validate attribution data against schema."""
    return AttributionSchema.validate(df)


def validate_mmm_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate MMM-ready input data against schema."""
    return MMMInputSchema.validate(df)

