# Unified-M Architecture

This document describes the high-level architecture of the Unified Marketing Measurement (UMM) platform.

## System Overview

Unified-M is a **batch-first architecture** designed for:
- **High-latency pipeline**: Heavy computation (training, reconciliation) runs periodically
- **Low-latency serving**: Precomputed results served via fast API and UI

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HIGH-LATENCY PIPELINE                           │
│  (GitHub Actions - Weekly/Nightly)                                      │
│                                                                         │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌───────┐  ┌──────────────┐ │
│  │ Ingest  │→ │ Validate │→ │ Transform │→ │ Train │→ │  Reconcile   │ │
│  └─────────┘  └──────────┘  └───────────┘  └───────┘  └──────────────┘ │
│       ↓                                                      ↓          │
│  Raw Data                                              ┌──────────┐     │
│  (Parquet)                                             │ Optimize │     │
│                                                        └──────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                           Precomputed Outputs
                           (Parquet + JSON)
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         LOW-LATENCY SERVING                             │
│                                                                         │
│  ┌──────────────────┐              ┌──────────────────────────────────┐ │
│  │    FastAPI       │    ←────→    │        Streamlit UI              │ │
│  │  (REST API)      │              │   (Interactive Dashboard)        │ │
│  └──────────────────┘              └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Ingestion Layer (`src/ingestion/`)

Handles loading data from various sources:

```python
from ingestion import DataIngestion, ParquetLoader, CSVLoader

# Load from different sources
loader = DataIngestion()
df = loader.load("s3://bucket/media_spend.parquet")
```

**Supported Sources:**
- Local files (Parquet, CSV)
- Cloud storage (S3, GCS via fsspec)
- Databases (via DuckDB)

### 2. Schema Validation (`src/schemas/`)

Uses **Pandera** for declarative data validation:

```python
from schemas import MediaSpendSchema, validate_media_spend

# Validate data against schema
validated_df = validate_media_spend(raw_df)
```

**Schemas:**
- `MediaSpendSchema`: date, channel, spend, impressions, clicks
- `OutcomeSchema`: date, revenue, conversions
- `ControlVariableSchema`: date + flexible control columns
- `IncrementalityTestSchema`: test results with confidence intervals
- `AttributionSchema`: touchpoint-level attribution data

### 3. Transform Layer (`src/transforms/`)

Applies marketing science transformations:

#### Adstock (Carryover Effects)
```python
from transforms import geometric_adstock, weibull_adstock

# Model how advertising effect persists over time
adstocked = geometric_adstock(spend, alpha=0.7, l_max=8)
```

**Mathematical Formula (Geometric):**
```
adstock[t] = x[t] + α × adstock[t-1]
```

Where:
- `α` (alpha): Decay rate (0-1), higher = longer carryover
- `l_max`: Maximum lag periods to consider

#### Saturation (Diminishing Returns)
```python
from transforms import hill_saturation

# Model diminishing returns at high spend
saturated = hill_saturation(adstocked, K=1000, S=1.5)
```

**Mathematical Formula (Hill Function):**
```
y = x^S / (K^S + x^S)
```

Where:
- `K`: Half-saturation point (spend at 50% max effect)
- `S`: Steepness parameter (S>1 = more S-shaped)

### 4. MMM Model (`src/mmm/`)

Bayesian Marketing Mix Model using PyMC-Marketing:

```python
from mmm import UnifiedMMM

mmm = UnifiedMMM(
    date_col="date",
    target_col="revenue",
    media_cols=["google_spend", "meta_spend", "tv_spend"],
)
results = mmm.fit(df, n_samples=2000, n_chains=4)
```

**Model Structure:**
```
y[t] = intercept + Σ(β_i × saturate(adstock(x_i[t]))) + controls + ε
```

Where:
- `y[t]`: Outcome (revenue/conversions) at time t
- `β_i`: Channel coefficient (effect size)
- `x_i[t]`: Media spend for channel i at time t
- `controls`: Seasonality, trends, external factors
- `ε`: Error term

### 5. Reconciliation Engine (`src/reconciliation/`)

Fuses multiple measurement signals:

```python
from reconciliation import ReconciliationEngine

engine = ReconciliationEngine(
    mmm_weight=0.5,
    incrementality_weight=0.3,
    attribution_weight=0.2,
)
result = engine.reconcile(
    mmm_results=mmm_output,
    incrementality_tests=test_df,
    attribution_data=attr_df,
)
```

**Fusion Methods:**

1. **Weighted Average**: Simple weighted combination
   ```
   lift = w_mmm × lift_mmm + w_incr × lift_incr + w_attr × lift_attr
   ```

2. **Bayesian Fusion**: MMM as prior, tests as likelihood
   ```
   posterior ∝ prior(MMM) × likelihood(test_data)
   ```

### 6. Budget Optimizer (`src/optimization/`)

Finds optimal budget allocation:

```python
from optimization import BudgetOptimizer

optimizer = BudgetOptimizer(
    response_curves=curves,  # From MMM
    total_budget=100000,
)
result = optimizer.optimize()
```

**Optimization Problem:**
```
maximize: Σ response_i(spend_i)
subject to: Σ spend_i = total_budget
            min_i ≤ spend_i ≤ max_i
```

Uses scipy.optimize (SLSQP) for constrained optimization.

### 7. API Layer (`src/api/`)

FastAPI endpoints for serving results:

```
GET /api/v1/contributions    → Channel contribution decomposition
GET /api/v1/reconciliation   → Reconciled lift estimates
GET /api/v1/optimization     → Budget recommendations
GET /api/v1/response-curves  → Saturation curves
GET /api/v1/scenarios        → Scenario comparisons
```

### 8. UI Dashboard (`ui/`)

Streamlit-based interactive dashboard:

- **Dashboard**: Key metrics and summary charts
- **Contributions**: Channel decomposition waterfall
- **Reconciliation**: Lift estimates with confidence intervals
- **Optimization**: Current vs. recommended allocation
- **Response Curves**: Interactive saturation curves

## Data Flow

### Training Pipeline (Batch)

```
1. INGEST
   ├── Read raw data (Parquet/CSV)
   └── Output: data/validated/*.parquet

2. VALIDATE
   ├── Apply Pandera schemas
   └── Output: Validated data + quality report

3. TRANSFORM
   ├── Pivot media spend to wide format
   ├── Apply adstock transformations
   ├── Apply saturation transformations
   ├── Add seasonality features (Fourier)
   └── Output: data/transformed/mmm_input.parquet

4. TRAIN
   ├── Fit Bayesian MMM (PyMC)
   ├── Extract posterior distributions
   ├── Compute channel contributions
   └── Output: models/mmm_model.pkl, data/outputs/contributions.parquet

5. RECONCILE
   ├── Load MMM results
   ├── Load incrementality test results
   ├── Load attribution data (optional)
   ├── Apply fusion method
   └── Output: data/outputs/reconciliation.json

6. OPTIMIZE
   ├── Build response curves from MMM
   ├── Run constrained optimization
   └── Output: data/outputs/optimization.json, response_curves.json
```

### Serving Pipeline (Real-time)

```
Request → FastAPI → Read Precomputed Artifacts → Response
                           ↑
                    (Parquet/JSON files)
```

No heavy computation at serving time!

## Directory Structure

```
unified-m/
├── src/
│   ├── __init__.py           # Package metadata
│   ├── cli.py                # CLI commands (Typer)
│   ├── config.py             # Configuration management
│   ├── api/                  # FastAPI application
│   │   ├── __init__.py
│   │   └── app.py
│   ├── ingestion/            # Data loading
│   │   ├── __init__.py
│   │   └── loaders.py
│   ├── schemas/              # Data validation
│   │   ├── __init__.py
│   │   └── base.py
│   ├── transforms/           # Feature engineering
│   │   ├── __init__.py
│   │   ├── adstock.py
│   │   ├── saturation.py
│   │   └── features.py
│   ├── mmm/                  # Marketing Mix Model
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── evaluation.py
│   │   └── decomposition.py
│   ├── reconciliation/       # Signal fusion
│   │   ├── __init__.py
│   │   ├── fusion.py
│   │   └── calibration.py
│   └── optimization/         # Budget allocation
│       ├── __init__.py
│       ├── allocator.py
│       └── scenarios.py
├── ui/
│   └── app.py               # Streamlit dashboard
├── data/
│   ├── raw/                 # Input data
│   ├── validated/           # Schema-validated
│   ├── transformed/         # Model-ready features
│   └── outputs/             # Results (JSON, Parquet)
├── models/                  # Trained model artifacts
├── scripts/                  # Runnable demos (reproducible, CI-friendly)
├── tests/                    # Test suite
├── docs/                    # Documentation
└── .github/workflows/       # CI/CD pipelines
```

## Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Data Storage | Parquet | Columnar, compressed, fast |
| Query Engine | DuckDB | In-process OLAP, SQL support |
| Data Processing | Polars/Pandas | Fast transforms |
| Validation | Pandera | Declarative schemas |
| MMM Engine | PyMC-Marketing | Bayesian inference, uncertainty |
| Optimization | scipy.optimize | Proven constrained optimization |
| API | FastAPI | Fast, async, auto-docs |
| UI | Streamlit | Rapid dashboard development |
| Orchestration | GitHub Actions | GitOps, no infrastructure |
| Visualization | Plotly | Interactive charts |

## Scaling Considerations

### Current Design (Single Machine)
- Suitable for: <1M rows of data, <10 channels
- Training time: 10-30 minutes
- Serving latency: <100ms (precomputed)

### Future Scaling Options
1. **Data Volume**: Move to Spark/Dask for distributed transforms
2. **Model Training**: Use PyMC with GPU (JAX backend)
3. **Multi-tenant**: Add tenant isolation to API/storage
4. **Real-time**: Add streaming ingestion (Kafka)

## Security Considerations

- API authentication: Add OAuth2/API keys for production
- Data encryption: Use encrypted storage (S3 SSE)
- Access control: RBAC for multi-user scenarios
- Audit logging: Track data access and model runs
