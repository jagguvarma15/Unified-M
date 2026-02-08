# Data Schemas Reference

This document describes all data schemas used in the Unified-M platform.

## Overview

All data in Unified-M flows through validated schemas using **Pandera**. This ensures data quality and consistency throughout the pipeline.

```python
# Example: Validating data
from schemas import validate_media_spend

validated_df = validate_media_spend(raw_df)  # Raises on invalid data
```

---

## 1. Media Spend Schema

Captures marketing spend across channels.

### Required Fields

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | datetime | Date of spend | 2024-01-15 |
| `channel` | string | Channel identifier | "google_search" |
| `spend` | float | Spend amount (positive) | 5000.00 |

### Optional Fields

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `impressions` | int | Number of impressions | 500000 |
| `clicks` | int | Number of clicks | 2500 |
| `reach` | int | Unique users reached | 150000 |
| `frequency` | float | Avg impressions per user | 3.33 |
| `conversions` | int | Platform-reported conversions | 100 |
| `cpm` | float | Cost per 1000 impressions | 10.00 |
| `cpc` | float | Cost per click | 2.00 |

### Schema Definition

```python
import pandera as pa

MediaSpendSchema = pa.DataFrameSchema({
    "date": pa.Column(pa.DateTime, nullable=False),
    "channel": pa.Column(pa.String, nullable=False),
    "spend": pa.Column(pa.Float, checks=pa.Check.ge(0), nullable=False),
    "impressions": pa.Column(pa.Int, checks=pa.Check.ge(0), nullable=True),
    "clicks": pa.Column(pa.Int, checks=pa.Check.ge(0), nullable=True),
})
```

### Example Data

```
| date       | channel       | spend    | impressions | clicks |
|------------|---------------|----------|-------------|--------|
| 2024-01-01 | google_search | 5000.00  | 450000      | 2250   |
| 2024-01-01 | meta_facebook | 4000.00  | 600000      | 1800   |
| 2024-01-01 | tv_linear     | 10000.00 | 5000000     | NULL   |
| 2024-01-02 | google_search | 5200.00  | 480000      | 2400   |
```

### Channel Naming Conventions

Use consistent, lowercase, underscore-separated names:

| Platform | Recommended Channel Names |
|----------|--------------------------|
| Google | `google_search`, `google_display`, `google_youtube`, `google_shopping` |
| Meta | `meta_facebook`, `meta_instagram`, `meta_audience_network` |
| TV | `tv_linear`, `tv_streaming`, `tv_addressable` |
| Audio | `audio_radio`, `audio_podcast`, `audio_streaming` |
| OOH | `ooh_billboard`, `ooh_transit`, `ooh_digital` |
| Other | `programmatic`, `affiliate`, `influencer`, `direct_mail` |

---

## 2. Outcome Schema

Captures business outcomes (dependent variable).

### Required Fields

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | datetime | Date of outcome | 2024-01-15 |
| `revenue` | float | Revenue amount | 50000.00 |

### Optional Fields

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `conversions` | int | Number of conversions | 500 |
| `orders` | int | Number of orders | 480 |
| `new_customers` | int | New customer count | 150 |
| `returning_customers` | int | Returning customer count | 330 |
| `avg_order_value` | float | Average order value | 104.17 |
| `units_sold` | int | Units sold | 1200 |

### Schema Definition

```python
OutcomeSchema = pa.DataFrameSchema({
    "date": pa.Column(pa.DateTime, nullable=False),
    "revenue": pa.Column(pa.Float, checks=pa.Check.ge(0), nullable=False),
    "conversions": pa.Column(pa.Int, checks=pa.Check.ge(0), nullable=True),
    "orders": pa.Column(pa.Int, checks=pa.Check.ge(0), nullable=True),
})
```

### Example Data

```
| date       | revenue   | conversions | orders | avg_order_value |
|------------|-----------|-------------|--------|-----------------|
| 2024-01-01 | 52000.00  | 520         | 500    | 104.00          |
| 2024-01-02 | 48500.00  | 485         | 470    | 103.19          |
| 2024-01-03 | 55000.00  | 550         | 530    | 103.77          |
```

---

## 3. Control Variables Schema

Captures external factors that influence outcomes.

### Recommended Fields

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | datetime | Date | 2024-01-15 |
| `is_weekend` | int (0/1) | Weekend indicator | 1 |
| `is_holiday` | int (0/1) | Holiday indicator | 0 |
| `is_promo` | int (0/1) | Promotion active | 1 |
| `promo_discount_pct` | float | Discount percentage | 20.0 |

### Optional Fields (Domain-Specific)

| Column | Type | Description |
|--------|------|-------------|
| `temperature_f` | float | Temperature (weather effect) |
| `precipitation_in` | float | Precipitation |
| `consumer_confidence` | float | Economic indicator |
| `competitor_promo` | int (0/1) | Competitor activity |
| `out_of_stock_pct` | float | Inventory issues |
| `price_index` | float | Price changes |

### Schema Definition

```python
ControlVariableSchema = pa.DataFrameSchema({
    "date": pa.Column(pa.DateTime, nullable=False),
    # All other columns flexible - validated at runtime
})
```

### Example Data

```
| date       | is_weekend | is_holiday | is_promo | promo_discount_pct | temperature_f |
|------------|------------|------------|----------|--------------------| --------------|
| 2024-01-01 | 0          | 1          | 1        | 30.0               | 35.2          |
| 2024-01-02 | 0          | 0          | 0        | 0.0                | 38.5          |
| 2024-01-06 | 1          | 0          | 0        | 0.0                | 42.1          |
```

---

## 4. Incrementality Test Schema

Captures results from causal experiments.

### Required Fields

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `test_id` | string | Unique test identifier | "GEO_2024_Q1_TV" |
| `channel` | string | Channel tested | "tv_linear" |
| `start_date` | datetime | Test start date | 2024-01-15 |
| `end_date` | datetime | Test end date | 2024-03-15 |
| `lift_estimate` | float | Point estimate of lift | 0.08 |
| `lift_ci_lower` | float | 95% CI lower bound | 0.03 |
| `lift_ci_upper` | float | 95% CI upper bound | 0.13 |
| `test_type` | string | Type of test | "geo_lift" |

### Optional Fields

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `p_value` | float | Statistical significance | 0.02 |
| `test_spend` | float | Spend during test | 150000.00 |
| `incremental_revenue` | float | Measured incremental revenue | 12000.00 |
| `control_regions` | string | Control group description | "TX,AZ,NV" |
| `treatment_regions` | string | Treatment group description | "CA,OR,WA" |
| `sample_size` | int | Number of observations | 50000 |
| `statistical_power` | float | Power of test | 0.85 |

### Test Types

| Type | Description | Best For |
|------|-------------|----------|
| `geo_lift` | Geographic holdout | TV, Radio, OOH |
| `conversion_lift` | User-level holdout | Digital channels |
| `brand_lift` | Survey-based | Brand metrics |
| `matched_market` | Similar market comparison | Any offline |
| `psa` | Public service announcement holdout | TV, Display |
| `ghost_ads` | Non-rendered ad study | Display |

### Schema Definition

```python
IncrementalityTestSchema = pa.DataFrameSchema({
    "test_id": pa.Column(pa.String, nullable=False, unique=True),
    "channel": pa.Column(pa.String, nullable=False),
    "start_date": pa.Column(pa.DateTime, nullable=False),
    "end_date": pa.Column(pa.DateTime, nullable=False),
    "lift_estimate": pa.Column(pa.Float, nullable=False),
    "lift_ci_lower": pa.Column(pa.Float, nullable=False),
    "lift_ci_upper": pa.Column(pa.Float, nullable=False),
    "test_type": pa.Column(pa.String, nullable=False),
})
```

### Example Data

```
| test_id           | channel       | start_date | end_date   | lift_estimate | lift_ci_lower | lift_ci_upper | test_type       | p_value |
|-------------------|---------------|------------|------------|---------------|---------------|---------------|-----------------|---------|
| GEO_2024_Q1_TV    | tv_linear     | 2024-01-15 | 2024-03-15 | 0.08          | 0.03          | 0.13          | geo_lift        | 0.02    |
| CONV_LIFT_META_Q2 | meta_facebook | 2024-04-01 | 2024-04-28 | 0.15          | 0.10          | 0.20          | conversion_lift | 0.001   |
| CONV_LIFT_GOOG_Q2 | google_search | 2024-05-01 | 2024-05-21 | 0.25          | 0.18          | 0.32          | conversion_lift | 0.0001  |
```

---

## 5. Attribution Schema

Captures touchpoint-level attribution data.

### Required Fields

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | datetime | Date | 2024-01-15 |
| `channel` | string | Channel | "google_search" |
| `attributed_conversions` | float | Attributed conversions | 150.5 |

### Optional Fields

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `attributed_revenue` | float | Attributed revenue | 15050.00 |
| `first_touch_conversions` | float | First-touch model | 120.0 |
| `last_touch_conversions` | float | Last-touch model | 180.0 |
| `linear_conversions` | float | Linear model | 150.0 |
| `time_decay_conversions` | float | Time-decay model | 155.0 |
| `position_based_conversions` | float | Position-based model | 148.0 |
| `data_driven_conversions` | float | Data-driven model | 150.5 |

### Attribution Models

| Model | Description | Bias |
|-------|-------------|------|
| First-touch | All credit to first touchpoint | Over-credits awareness |
| Last-touch | All credit to last touchpoint | Over-credits conversion |
| Linear | Equal credit to all touchpoints | Ignores timing |
| Time-decay | More credit to recent touchpoints | Balance |
| Position-based | 40% first, 40% last, 20% middle | Compromise |
| Data-driven | ML-based (platform specific) | Varies |

### Schema Definition

```python
AttributionSchema = pa.DataFrameSchema({
    "date": pa.Column(pa.DateTime, nullable=False),
    "channel": pa.Column(pa.String, nullable=False),
    "attributed_conversions": pa.Column(pa.Float, checks=pa.Check.ge(0)),
    "attributed_revenue": pa.Column(pa.Float, checks=pa.Check.ge(0), nullable=True),
})
```

### Example Data

```
| date       | channel       | attributed_conversions | attributed_revenue | first_touch | last_touch |
|------------|---------------|------------------------|--------------------| ------------|------------|
| 2024-01-01 | google_search | 150.5                  | 15050.00           | 120.0       | 180.0      |
| 2024-01-01 | meta_facebook | 80.2                   | 8020.00            | 100.0       | 60.0       |
| 2024-01-01 | direct        | 200.0                  | 20000.00           | 50.0        | 350.0      |
```

---

## 6. MMM Input Schema

The final schema for model training (generated by `create_mmm_features`).

### Required Fields

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Date |
| `y` | float | Target variable (revenue/conversions) |
| `{channel}_spend` | float | Spend per channel (one column per channel) |

### Auto-Generated Fields

| Column | Type | Description |
|--------|------|-------------|
| `day_of_week` | int | 0-6 |
| `month` | int | 1-12 |
| `week_of_year` | int | 1-52 |
| `sin_1`, `cos_1` | float | Annual seasonality (1st harmonic) |
| `sin_2`, `cos_2` | float | Semi-annual seasonality (2nd harmonic) |

### Example Data

```
| date       | y        | google_search_spend | meta_facebook_spend | tv_linear_spend | is_weekend | sin_1  | cos_1  |
|------------|----------|---------------------|---------------------|-----------------|------------|--------|--------|
| 2024-01-01 | 52000.00 | 5000.00             | 4000.00             | 10000.00        | 0          | 0.017  | 0.999  |
| 2024-01-02 | 48500.00 | 5200.00             | 4100.00             | 10500.00        | 0          | 0.034  | 0.999  |
```

---

## Data Quality Checks

### Automated Validation

The pipeline automatically checks:

1. **Completeness**: No missing required fields
2. **Type Correctness**: Columns match expected types
3. **Value Ranges**: Spend ≥ 0, dates valid, etc.
4. **Uniqueness**: No duplicate (date, channel) combinations
5. **Temporal Consistency**: Dates are sequential

### Manual Checks Recommended

1. **Channel Coverage**: All channels have data for entire period
2. **Spend Variation**: Sufficient variation for modeling
3. **Outlier Review**: Investigate extreme values
4. **Cross-Source Alignment**: Outcomes align with media timing

---

## File Formats

### Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| Parquet | `.parquet` | Recommended for all data |
| CSV | `.csv` | Human-readable, interchange |
| JSON | `.json` | Configuration, small data |

### Parquet Best Practices

```python
# Writing with compression
df.to_parquet(
    "data.parquet",
    compression="snappy",  # Good balance of speed/size
    index=False,
)

# Reading efficiently
df = pd.read_parquet(
    "data.parquet",
    columns=["date", "channel", "spend"],  # Only needed columns
)
```

---

## Integration Examples

### Loading from Google Ads API

```python
from google.ads.googleads.client import GoogleAdsClient

def fetch_google_ads_data(client, customer_id, start_date, end_date):
    """Fetch spend data from Google Ads API."""
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
    
    response = client.service.google_ads.search(
        customer_id=customer_id,
        query=query,
    )
    
    records = []
    for row in response:
        records.append({
            "date": row.segments.date,
            "channel": f"google_{row.campaign.advertising_channel_type.name.lower()}",
            "spend": row.metrics.cost_micros / 1_000_000,
            "impressions": row.metrics.impressions,
            "clicks": row.metrics.clicks,
        })
    
    return pd.DataFrame(records)
```

### Loading from Data Warehouse (BigQuery)

```python
from google.cloud import bigquery

def fetch_from_bigquery(project_id, query):
    """Fetch data from BigQuery."""
    client = bigquery.Client(project=project_id)
    return client.query(query).to_dataframe()

# Example query
query = """
SELECT
    date,
    channel,
    SUM(spend) as spend,
    SUM(impressions) as impressions
FROM `project.dataset.media_spend`
WHERE date BETWEEN '2024-01-01' AND '2024-12-31'
GROUP BY date, channel
"""
```
