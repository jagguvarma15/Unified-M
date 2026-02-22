# Unified-M

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue.svg)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-18.3-blue.svg)](https://react.dev/)
[![Bun](https://img.shields.io/badge/Bun-1.3-black.svg)](https://bun.sh/)
[![uv](https://img.shields.io/badge/uv-managed-ffc021.svg)](https://docs.astral.sh/uv/)

**Unified Marketing Measurement Platform**

An end-to-end framework that fuses Marketing Mix Modeling (MMM), incrementality tests, and attribution data into a single source of truth for channel-level lift with calibrated uncertainty, producing stable budget recommendations and scenario simulations. Built for local-first deployment with a modern React dashboard, pluggable model backends, and full audit trails.

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

**Backend (recommended: [uv](https://docs.astral.sh/uv/))**

```bash
# Clone the repository
git clone https://github.com/jagguvarma15/Unified-M.git
cd Unified-M

# Install uv (if needed): curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies from lockfile (reproducible)
uv sync

# Optional: install a specific Python version
uv python install 3.11
```

**Pip fallback**

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

**UI (requires [Bun](https://bun.sh))**

```bash
cd ui && bun install
```

### Generate Demo Data & Run Pipeline

```bash
# Generate synthetic data and run the full pipeline
uv run python -m cli demo

# Or run the pipeline with your own data
uv run python -m cli run \
  --media-spend data/gold/media_spend.parquet \
  --outcomes data/gold/outcomes.parquet \
  --model builtin \
  --target revenue
```

Using the Makefile: `make dev` (demo) | `make run` (full pipeline) | `make serve` (API).

### Start the UI

```bash
# Terminal 1: API server
uv run python -m cli serve --port 8000

# Terminal 2: React dev server
cd ui && bun dev
# Or: uv run python -m cli ui
```

The UI will be available at `http://localhost:5173` and proxies API requests to the backend.

## Project Structure

```
unified-m/
├── data/
│   ├── raw/              # Raw input data (Parquet)
│   ├── processed/        # Validated & transformed data
│   └── outputs/          # Results (JSON, Parquet)
├── runs/                 # Versioned pipeline artifacts
├── src/
│   ├── core/             # Contracts, artifacts, base models, exceptions
│   ├── connectors/       # Data loaders (Parquet, CSV, DuckDB)
│   ├── models/           # MMM backends (builtin, pymc, registry)
│   ├── pipeline/         # End-to-end runner
│   ├── transforms/       # Adstock, saturation, features
│   ├── reconciliation/   # Fusion engine (MMM + tests + attribution)
│   ├── optimization/     # Budget allocator & scenarios
│   ├── server/           # FastAPI application
│   ├── config.py         # Configuration management
│   └── cli.py            # CLI commands
├── ui/                   # React + TypeScript dashboard
│   ├── src/
│   │   ├── pages/        # Dashboard, Contributions, Optimization, etc.
│   │   ├── components/   # Reusable UI components
│   │   └── lib/          # API client, utilities
│   └── package.json      # Bun dependencies
├── scripts/              # Runnable demos (data, transforms, evaluation)
├── tests/                # Test suite
└── pyproject.toml        # Python dependencies
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.10+ |
| **Data Storage** | Parquet files (local or S3/GCS compatible) |
| **Query Engine** | DuckDB (optional) |
| **Validation** | Pydantic |
| **MMM Backends** | Built-in (Ridge), PyMC-Marketing (optional) |
| **Optimization** | scipy.optimize |
| **API** | FastAPI + Uvicorn |
| **UI** | React 18 + TypeScript + Vite |
| **UI Runtime** | Bun |
| **Styling** | Tailwind CSS |
| **Charts** | Recharts |

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
- RESTful API for all outputs (FastAPI)
- Modern React dashboard with TypeScript
- Real-time data visualization with Recharts
- Responsive design with Tailwind CSS
- Fast development with Bun + Vite

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
# Install dependencies (includes [dependency-groups] dev)
uv sync

# Run tests
uv run pytest tests/ -v --cov=src

# Lint & type-check
uv run ruff check src/
uv run mypy src/

# Update lockfile after changing pyproject.toml
uv lock
```

Or use the Makefile: `make install-dev`, `make test`, `make check`, `make lock`.

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
