# Unified-M

**Unified Marketing Measurement Platform**

An end-to-end Unified Marketing Measurement platform that fuses strategic MMM, tactical attribution, and incrementality experiments into a single decision layer, producing consistent channel-level and tactic-level lift with calibrated confidence intervals, stable budget recommendations, and scenario simulations.

## Architecture

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

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Unified-M.git
cd Unified-M

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"
```

### Generate Demo Data

```bash
python -m cli generate-demo --days 365
```

### Run the Pipeline

```bash
# Run full pipeline
python -m cli run-pipeline

# Or run individual steps
python -m cli ingest
python -m cli validate
python -m cli transform
python -m cli train
python -m cli reconcile
python -m cli optimize
```

### Start the UI

```bash
# Start API server
python -m cli serve --port 8000

# Start Streamlit UI (in another terminal)
python -m cli ui --port 8501
```

## Project Structure

```
unified-m/
├── data/
│   ├── raw/              # Raw input data (Parquet)
│   ├── validated/        # Validated data
│   ├── transformed/      # Model-ready features
│   └── outputs/          # Results (JSON, Parquet)
├── models/               # Trained model artifacts
├── src/
│   ├── schemas/          # Pandera data schemas
│   ├── ingestion/        # Data loaders
│   ├── transforms/       # Adstock, saturation, features
│   ├── mmm/              # PyMC-Marketing wrapper
│   ├── reconciliation/   # Fusion logic
│   ├── optimization/     # Budget allocator
│   ├── api/              # FastAPI endpoints
│   ├── config.py         # Configuration
│   └── cli.py            # CLI commands
├── ui/                   # Streamlit dashboard
├── notebooks/            # Exploration notebooks
├── tests/                # Test suite
├── .github/workflows/    # GitHub Actions pipeline
└── pyproject.toml        # Dependencies
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Storage** | Parquet files (S3/GCS compatible) |
| **Query Engine** | DuckDB |
| **Transforms** | Polars + Python |
| **Validation** | Pandera |
| **MMM** | PyMC-Marketing |
| **Orchestration** | GitHub Actions |
| **Optimization** | scipy.optimize |
| **API** | FastAPI |
| **UI** | Streamlit |

## Features

### Marketing Mix Modeling (MMM)
- Bayesian regression with PyMC-Marketing
- Automatic adstock & saturation transformations
- Posterior uncertainty quantification
- Channel contribution decomposition

### Reconciliation
- Fuses MMM + incrementality tests + attribution
- Weighted average or Bayesian fusion
- Calibration factors from experiments
- Confidence scoring

### Budget Optimization
- Response curve-based allocation
- Constrained optimization (min/max per channel)
- Scenario planning & comparison
- Efficiency frontier analysis

### API & UI
- RESTful API for all outputs
- Interactive Streamlit dashboard
- Real-time scenario simulation
- Export capabilities

## Data Schemas

### Media Spend
```
date | channel | spend | impressions | clicks
```

### Outcomes
```
date | revenue | conversions
```

### Incrementality Tests
```
test_id | channel | start_date | end_date | lift_estimate | lift_ci_lower | lift_ci_upper | test_type
```

## Pipeline Schedule

The GitHub Actions pipeline runs:
- **Weekly** (Sunday 2 AM UTC) - Full model retrain
- **On-demand** - Manual trigger via workflow_dispatch

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/

# Type checking
mypy src/
```

## Configuration

Create `config.yaml` for custom settings:

```yaml
project_name: "My MMM Project"
environment: "production"

storage:
  raw_path: "s3://my-bucket/raw/"
  outputs_path: "s3://my-bucket/outputs/"

mmm:
  n_samples: 2000
  n_chains: 4
  adstock_max_lag: 12

reconciliation:
  mmm_weight: 0.5
  incrementality_weight: 0.3
  attribution_weight: 0.2

optimization:
  method: "SLSQP"
  min_channel_budget_pct: 0.05
  max_channel_budget_pct: 0.5
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.
